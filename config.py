# config.py

import yaml
from dataclasses import dataclass, field
from typing import Any, Dict
import torch

@dataclass
class TrainingConfig:
    # I/O
    out_dir: str = "out"
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False
    always_save_checkpoint: bool = True
    init_from: str = "scratch"

    # wandb logging
    wandb_log: bool = False
    wandb_project: str = "owt"
    wandb_run_name: str = "gpt2"

    # data
    dataset: str = "openwebtext"
    gradient_accumulation_steps: int = 40
    batch_size: int = 12
    block_size: int = 1024

    # model
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False

    # optimizer
    learning_rate: float = 6e-4
    max_iters: int = 600000
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # learning rate decay
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 600000
    min_lr: float = 6e-5

    # distributed
    backend: str = "nccl"

    # system
    device: str = "cuda"
    dtype: str = "float16"
    compile: bool = True

    # catch‐all for anything else
    extra: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_yaml(path: str) -> "TrainingConfig":
        """Load defaults from a YAML file into a config instance."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return TrainingConfig(**data)

    def update_from_dict(self, override: Dict[str, Any]) -> None:
        """Override individual fields from a dictionary (e.g. parsed args)."""
        for k, v in override.items():
            if not hasattr(self, k):
                raise ValueError(f"Unknown config key: {k}")
            setattr(self, k, v)
