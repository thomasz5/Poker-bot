"""Configuration settings for the poker bot project."""

import os
from pathlib import Path
from typing import Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODEL_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# Database Configuration
DATABASE_CONFIG = {
    "url": os.getenv("DATABASE_URL", "postgresql://username:password@localhost:5432/poker_bot"),
    "pool_size": 10,
    "max_overflow": 20,
}

REDIS_CONFIG = {
    "url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    "decode_responses": True,
}

# Training Configuration
TRAINING_CONFIG = {
    "batch_size": 512,
    "learning_rate": 1e-4,
    "epochs": 100,
    "validation_split": 0.2,
    "device": "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu",
    "num_workers": int(os.getenv("MAX_WORKERS", "4")),
}

# Neural Network Architecture
MODEL_CONFIG = {
    "input_dim": 500,
    "hidden_dim": 512,
    "num_heads": 8,
    "num_layers": 6,
    "dropout": 0.1,
}

# Game Configuration
GAME_CONFIG = {
    "num_players": 6,
    "starting_stack": 200,  # Big blinds
    "small_blind": 1,
    "big_blind": 2,
    "ante": 0,
}

# Feature Engineering
FEATURE_CONFIG = {
    "position_features": 6,
    "stack_features": 6,
    "betting_features": 20,
    "board_features": 60,  # 52 cards + texture features
    "opponent_features": 200,
    "meta_features": 20,
    "total_features": 312,  # Sum of above
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    "rotation": "1 day",
    "retention": "30 days",
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": os.getenv("DEBUG", "True").lower() == "true",
}

# External Services
EXTERNAL_SERVICES = {
    "wandb_api_key": os.getenv("WANDB_API_KEY"),
    "piosolver_api_key": os.getenv("PIOSOLVER_API_KEY"),
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary."""
    return {
        "database": DATABASE_CONFIG,
        "redis": REDIS_CONFIG,
        "training": TRAINING_CONFIG,
        "model": MODEL_CONFIG,
        "game": GAME_CONFIG,
        "features": FEATURE_CONFIG,
        "logging": LOGGING_CONFIG,
        "api": API_CONFIG,
        "external": EXTERNAL_SERVICES,
        "paths": {
            "project_root": PROJECT_ROOT,
            "data_dir": DATA_DIR,
            "model_dir": MODEL_DIR,
            "logs_dir": LOGS_DIR,
        }
    } 