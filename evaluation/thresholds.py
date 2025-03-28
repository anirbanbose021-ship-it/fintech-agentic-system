# evaluation/thresholds.py
"""
Configurable RAGAS evaluation thresholds.

Production thresholds are non-negotiable — they're the deployment gate.
Override only in non-production environments for faster iteration.
"""

# Production thresholds — these block deployment if not met
PRODUCTION = {
    'faithfulness':      0.85,  # hallucination guard — the one that matters most
    'answer_relevancy':  0.80,
    'context_precision': 0.75,
    'context_recall':    0.80,
}

# Staging — slightly relaxed for faster iteration during development
# still strict enough to catch real regressions
STAGING = {
    'faithfulness':      0.80,
    'answer_relevancy':  0.75,
    'context_precision': 0.70,
    'context_recall':    0.75,
}

# Development — catch only catastrophic failures
DEV = {
    'faithfulness':      0.70,
    'answer_relevancy':  0.60,
    'context_precision': 0.50,
    'context_recall':    0.60,
}


def get_thresholds(env: str = 'production') -> dict:
    """Get thresholds for the given environment."""
    envs = {
        'production': PRODUCTION,
        'staging': STAGING,
        'dev': DEV,
        'development': DEV,
    }
    thresholds = envs.get(env.lower())
    if thresholds is None:
        raise ValueError(f'Unknown environment: {env}. Use: {list(envs.keys())}')
    return thresholds
