# infrastructure/terraform/main.tf
# IaC for the full AWS stack supporting the fintech agentic system
#
# Dual audit strategy (AWS recommended for regulated workloads):
#   - CloudTrail: API-level audit — captures all Bedrock, S3, DynamoDB API calls
#   - DynamoDB:   Application-level audit — captures business decisions (risk scores,
#                 routing decisions, agent rationale, retrieval sources)
#
# Resources created:
#   - CloudTrail trail for API-level audit logging
#   - DynamoDB table for application-level audit logs
#   - S3 buckets for documents + CloudTrail logs
#   - OpenSearch Serverless collection for regulatory corpus
#   - SNS topic for deployment alerts
#   - IAM roles for AgentCore Runtime
#   - VPC endpoints for private connectivity

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.30"
    }
  }

  backend "s3" {
    bucket         = "fintech-agentic-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "terraform-lock"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "fintech-agentic-system"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# ── Variables ────────────────────────────────────────────────────────────────

variable "aws_region" {
  default = "us-east-1"
}

variable "environment" {
  default = "production"
}

variable "account_id" {
  description = "AWS account ID"
  type        = string
}

# ── DynamoDB: Audit Log ──────────────────────────────────────────────────────

resource "aws_dynamodb_table" "audit_log" {
  name         = "pipeline-audit-log"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "document_id"
  range_key    = "timestamp"

  attribute {
    name = "document_id"
    type = "S"
  }

  attribute {
    name = "timestamp"
    type = "S"
  }

  attribute {
    name = "client_id"
    type = "S"
  }

  global_secondary_index {
    name            = "client-index"
    hash_key        = "client_id"
    range_key       = "timestamp"
    projection_type = "ALL"
  }

  point_in_time_recovery {
    enabled = true   # required for regulatory compliance
  }

  server_side_encryption {
    enabled = true
  }
}

# ── S3: Document Storage ─────────────────────────────────────────────────────

resource "aws_s3_bucket" "documents" {
  bucket = "fintech-agentic-documents-${var.account_id}"
}

resource "aws_s3_bucket_versioning" "documents" {
  bucket = aws_s3_bucket.documents.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "documents" {
  bucket = aws_s3_bucket.documents.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "documents" {
  bucket = aws_s3_bucket.documents.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ── CloudTrail: API-Level Audit (AWS recommended) ────────────────────────────
# Captures all Bedrock InvokeModel/Converse calls, S3 access, DynamoDB writes,
# IAM activity. This is the infrastructure audit layer — "who called what API, when."
# The application-level audit (risk scores, routing decisions) lives in DynamoDB.

resource "aws_s3_bucket" "cloudtrail_logs" {
  bucket = "fintech-agentic-cloudtrail-${var.account_id}"
}

resource "aws_s3_bucket_policy" "cloudtrail_logs" {
  bucket = aws_s3_bucket.cloudtrail_logs.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AWSCloudTrailAclCheck"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action   = "s3:GetBucketAcl"
        Resource = aws_s3_bucket.cloudtrail_logs.arn
      },
      {
        Sid    = "AWSCloudTrailWrite"
        Effect = "Allow"
        Principal = {
          Service = "cloudtrail.amazonaws.com"
        }
        Action   = "s3:PutObject"
        Resource = "${aws_s3_bucket.cloudtrail_logs.arn}/AWSLogs/${var.account_id}/*"
        Condition = {
          StringEquals = {
            "s3:x-amz-acl" = "bucket-owner-full-control"
          }
        }
      },
    ]
  })
}

resource "aws_s3_bucket_server_side_encryption_configuration" "cloudtrail_logs" {
  bucket = aws_s3_bucket.cloudtrail_logs.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "cloudtrail_logs" {
  bucket = aws_s3_bucket.cloudtrail_logs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_cloudtrail" "agentic_system" {
  name                       = "fintech-agentic-trail"
  s3_bucket_name             = aws_s3_bucket.cloudtrail_logs.id
  include_global_service_events = true
  is_multi_region_trail      = false
  enable_log_file_validation = true    # tamper-proof — required for regulated environments

  # log Bedrock model invocations as data events
  event_selector {
    read_write_type           = "All"
    include_management_events = true

    data_resource {
      type   = "AWS::BedrockRuntime::Model"
      values = ["arn:aws:bedrock:${var.aws_region}::foundation-model/*"]
    }
  }

  # log S3 data events for the document bucket
  event_selector {
    read_write_type           = "All"
    include_management_events = false

    data_resource {
      type   = "AWS::S3::Object"
      values = ["${aws_s3_bucket.documents.arn}/"]
    }
  }

  # log DynamoDB data events for the audit table
  event_selector {
    read_write_type           = "All"
    include_management_events = false

    data_resource {
      type   = "AWS::DynamoDB::Table"
      values = [aws_dynamodb_table.audit_log.arn]
    }
  }

  depends_on = [aws_s3_bucket_policy.cloudtrail_logs]
}

# ── SNS: Deployment Alerts ───────────────────────────────────────────────────

resource "aws_sns_topic" "deployment_alerts" {
  name = "ml-deployment-alerts"
}

# ── IAM: AgentCore Runtime Role ──────────────────────────────────────────────

resource "aws_iam_role" "agentcore_runtime" {
  name = "AgentCoreRuntimeRole"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "bedrock.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "agentcore_permissions" {
  name = "agentcore-permissions"
  role = aws_iam_role.agentcore_runtime.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:Converse",
        ]
        Resource = "arn:aws:bedrock:${var.aws_region}::foundation-model/*"
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:PutItem",
          "dynamodb:GetItem",
          "dynamodb:Query",
        ]
        Resource = aws_dynamodb_table.audit_log.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
        ]
        Resource = "${aws_s3_bucket.documents.arn}/*"
      },
      {
        Effect   = "Allow"
        Action   = ["sns:Publish"]
        Resource = aws_sns_topic.deployment_alerts.arn
      },
    ]
  })
}

# ── Outputs ──────────────────────────────────────────────────────────────────

output "audit_table_name" {
  value = aws_dynamodb_table.audit_log.name
}

output "document_bucket" {
  value = aws_s3_bucket.documents.id
}

output "sns_topic_arn" {
  value = aws_sns_topic.deployment_alerts.arn
}

output "agentcore_role_arn" {
  value = aws_iam_role.agentcore_runtime.arn
}

output "cloudtrail_name" {
  value = aws_cloudtrail.agentic_system.name
}

output "cloudtrail_bucket" {
  value = aws_s3_bucket.cloudtrail_logs.id
}
