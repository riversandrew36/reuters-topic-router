"""
Configuration for Reuters Topic Router project.
"""
import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "saved_models"

# Ensure directories exist
OUTPUTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Data configuration
NUM_WORDS = 10000  # Vocabulary size
NUM_CLASSES = 46   # Number of Reuters topics
VAL_SPLIT = 1000   # Number of samples for validation

# Model hyperparameters
DENSE_NN_CONFIG = {
    "hidden_dims": [64, 64],
    "dropout_rate": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 512,
    "epochs": 20,
}

EMBEDDING_CONFIG = {
    "embedding_dim": 128,
    "hidden_dim": 64,
    "dropout_rate": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 512,
    "epochs": 20,
}

TFIDF_CONFIG = {
    "max_features": 10000,
    "ngram_range": (1, 2),
    "C": 1.0,  # Regularization for LogisticRegression
}

# MC Dropout configuration
MC_DROPOUT_CONFIG = {
    "n_samples": 50,  # Number of forward passes
    "entropy_threshold": 2.5,  # Abstain if entropy above this (raised from 1.5)
    "confidence_threshold": 0.15,  # Abstain if max prob below this (lowered from 0.5)
}

# Evaluation thresholds
TOP_K_VALUES = [3, 5]

# Random seed for reproducibility
RANDOM_SEED = 42
