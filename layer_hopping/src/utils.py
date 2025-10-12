"""
Utility functions for multi-exit YOLOS experiments
"""

import yaml
import torch
from pathlib import Path
import logging
from typing import Dict, Any


def load_config(config_path: str = "configs/experiment_config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: Dict[str, Any], experiment_name: str) -> logging.Logger:
    """Setup logging for experiments"""
    log_dir = Path(config['paths']['logs'])
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{experiment_name}.log"

    logging.basicConfig(
        level=config['logging']['level'],
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(experiment_name)
    logger.info(f"Logging initialized for {experiment_name}")
    logger.info(f"Log file: {log_file}")

    return logger


def get_device() -> torch.device:
    """Get available device (CUDA or CPU)"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def create_output_dir(base_dir: str) -> Path:
    """Create output directory for experiment"""
    output_dir = Path(base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
