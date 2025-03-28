# fine_tuning/data_pipeline.py
"""
Synthetic training data generation for the credit risk assessment model.

Takes a directory of credit/KYC documents, uses Claude 3 Sonnet to generate
instruction-response pairs, then filters for quality. Output is JSONL suitable
for qlora_train.py.

Usage:
    python fine_tuning/data_pipeline.py \
        --input-dir /path/to/credit/docs \
        --output training_data.jsonl \
        --num-examples 8000
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')

bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

# prompt templates for different training task types
# mixing these gives the fine-tuned model broader coverage than
# just risk scoring — it also learns to extract and explain
TASK_TEMPLATES = {
    'risk_assessment': {
        'weight': 0.50,
        'system': (
            'You are a credit risk analyst. Given a document excerpt, produce a '
            'JSON risk assessment with risk_score (0.0-1.0), risk_factors (list), '
            'and risk_summary (2 sentences).'
        ),
        'instruction_prefix': 'Assess the credit risk of this document excerpt:\n\n',
    },
    'factor_extraction': {
        'weight': 0.25,
        'system': (
            'You are a financial analyst. Extract all risk-relevant factors from '
            'the document excerpt. Return a JSON object with factors (list of strings) '
            'and overall_sentiment (POSITIVE, NEGATIVE, MIXED).'
        ),
        'instruction_prefix': 'Extract risk factors from this financial document:\n\n',
    },
    'compliance_check': {
        'weight': 0.25,
        'system': (
            'You are a compliance officer. Review the document excerpt for any '
            'red flags, missing disclosures, or regulatory concerns. Return JSON '
            'with flags (list of {issue, severity}), and compliant (boolean).'
        ),
        'instruction_prefix': 'Review this document for compliance issues:\n\n',
    },
}


def extract_text_from_file(filepath: Path) -> str:
    """Read text content from a document. Supports .txt and .json for now."""
    suffix = filepath.suffix.lower()
    if suffix == '.txt':
        return filepath.read_text(encoding='utf-8')
    elif suffix == '.json':
        data = json.loads(filepath.read_text(encoding='utf-8'))
        return data.get('text', data.get('content', json.dumps(data)))
    elif suffix == '.jsonl':
        lines = filepath.read_text(encoding='utf-8').strip().split('\n')
        return '\n'.join(json.loads(l).get('text', '') for l in lines[:20])
    else:
        logger.warning(f'Skipping unsupported file type: {filepath}')
        return ''


def chunk_document(text: str, chunk_size: int = 800, overlap: int = 100) -> list:
    """Split text into chunks for synthetic example generation."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(' '.join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def generate_synthetic_example(chunk: str, task_type: str) -> dict:
    """
    Use Claude 3 Sonnet to generate a training example from a doc chunk.
    Returns {instruction, response} or None if generation failed.
    """
    template = TASK_TEMPLATES[task_type]

    try:
        response = bedrock.converse(
            modelId='anthropic.claude-3-sonnet-20240229-v1:0',
            system=[{'text': template['system']}],
            messages=[{
                'role': 'user',
                'content': [{'text': template['instruction_prefix'] + chunk}],
            }],
            inferenceConfig={'maxTokens': 512, 'temperature': 0.3},
        )

        answer = response['output']['message']['content'][0]['text']

        # validate it's parseable JSON — reject bad generations
        json.loads(answer)

        return {
            'instruction': template['instruction_prefix'] + chunk,
            'response': answer,
            'task_type': task_type,
            'input_tokens': response['usage']['inputTokens'],
            'output_tokens': response['usage']['outputTokens'],
        }

    except (json.JSONDecodeError, KeyError) as e:
        # sonnet sometimes returns markdown-wrapped JSON, skip these
        logger.debug(f'Bad generation for {task_type}: {e}')
        return None
    except Exception as e:
        logger.warning(f'Generation failed: {e}')
        return None


def select_task_type() -> str:
    """Weighted random selection of task type."""
    r = random.random()
    cumulative = 0.0
    for task, config in TASK_TEMPLATES.items():
        cumulative += config['weight']
        if r <= cumulative:
            return task
    return 'risk_assessment'  # fallback


def run_pipeline(args):
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f'Input directory not found: {input_dir}')

    # collect all document files
    files = list(input_dir.glob('**/*.txt')) + \
            list(input_dir.glob('**/*.json')) + \
            list(input_dir.glob('**/*.jsonl'))
    logger.info(f'Found {len(files)} source documents in {input_dir}')

    if not files:
        logger.error('No supported files found (.txt, .json, .jsonl)')
        return

    # chunk all documents
    all_chunks = []
    for f in files:
        text = extract_text_from_file(f)
        if text:
            chunks = chunk_document(text)
            all_chunks.extend(chunks)
    logger.info(f'Generated {len(all_chunks)} chunks from source documents')

    # shuffle and generate
    random.shuffle(all_chunks)

    examples = []
    failures = 0
    target = args.num_examples

    for i, chunk in enumerate(all_chunks):
        if len(examples) >= target:
            break

        task_type = select_task_type()
        result = generate_synthetic_example(chunk, task_type)

        if result:
            examples.append(result)
            if len(examples) % 100 == 0:
                logger.info(f'Generated {len(examples)}/{target} examples ({failures} failures so far)')
        else:
            failures += 1

    # write output
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    logger.info(f'Wrote {len(examples)} examples to {output_path}')
    logger.info(f'Failures: {failures} ({100*failures/(len(examples)+failures):.1f}%)')

    # quick stats
    task_counts = {}
    total_tokens = 0
    for ex in examples:
        task_counts[ex['task_type']] = task_counts.get(ex['task_type'], 0) + 1
        total_tokens += ex.get('input_tokens', 0) + ex.get('output_tokens', 0)

    logger.info(f'Task distribution: {task_counts}')
    logger.info(f'Total tokens used: {total_tokens:,} (~${total_tokens * 0.000003:.2f} at Sonnet rates)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic training data')
    parser.add_argument('--input-dir', required=True, help='Directory of source documents')
    parser.add_argument('--output', default='training_data.jsonl', help='Output JSONL path')
    parser.add_argument('--num-examples', type=int, default=8000)
    args = parser.parse_args()

    run_pipeline(args)
