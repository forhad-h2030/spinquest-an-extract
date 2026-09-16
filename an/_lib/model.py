"""Multiclass classifier architectures + checkpoint loading."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class ParticleClassifierMulticlass(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout_rate: float = 0.3,
        flat: bool = False,   # True = constant width; False = halve each layer
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        h = hidden_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout_rate))
            in_dim = h
            if not flat:
                h = max(h // 2, 8)

        layers.append(nn.Linear(in_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)  # [B, K]


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout_rate: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.block(x) + x)


class ParticleResNetMulticlass(nn.Module):
    """Production checkpoint's architecture (model_type="resnet", num_blocks=4)."""
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        num_blocks: int = 4,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout_rate) for _ in range(num_blocks)]
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.classifier(x)


def build_model_from_ckpt(ckpt: dict) -> nn.Module:
    """Pick the architecture matching a checkpoint's own model_type/cfg."""
    input_dim = int(ckpt["input_dim"])
    num_classes = int(ckpt["num_classes"])
    cfg = ckpt.get("cfg", {})

    if ckpt.get("model_type") == "resnet":
        return ParticleResNetMulticlass(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=int(cfg.get("hidden_dim", 512)),
            num_blocks=int(cfg.get("num_layers", 4)),
            dropout_rate=float(cfg.get("dropout_rate", 0.1)),
        )
    return ParticleClassifierMulticlass(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=int(cfg.get("hidden_dim", 512)),
        num_layers=int(cfg.get("num_layers", 4)),
        dropout_rate=float(cfg.get("dropout_rate", 0.3)),
        flat=bool(cfg.get("flat", False)),
    )


def load_checkpoint(ckpt_path: Path, device: str = "cpu"):
    """Returns (model, scaler, class_names, feature_names) -- feature_names
    is the checkpoint's own training feature list (see features.py)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not (isinstance(ckpt, dict) and "state_dict" in ckpt and "input_dim" in ckpt):
        raise RuntimeError(f"{ckpt_path} is not a multiclass payload checkpoint "
                           f"(need state_dict/input_dim/num_classes/class_names).")
    if ckpt.get("model_type") == "transformer":
        raise RuntimeError(f"{ckpt_path}: model_type='transformer' not supported here.")

    model = build_model_from_ckpt(ckpt)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()

    class_names = list(ckpt["class_names"])
    feature_names = ckpt.get("feature_names")
    return model, ckpt.get("scaler", None), class_names, feature_names


def apply_scaler(x: np.ndarray, scaler: Optional[dict]) -> np.ndarray:
    if scaler is None:
        return x
    mu = np.asarray(scaler["mean"], dtype=np.float32).reshape(-1)
    sd = np.asarray(scaler["std"], dtype=np.float32).reshape(-1)
    return ((x - mu) / sd).astype(np.float32)


@torch.no_grad()
def softmax_probs(model: nn.Module, x: np.ndarray, device: str = "cpu",
                   batch_size: int = 8192) -> np.ndarray:
    """(N, K) softmax probabilities for an already-scaled feature matrix."""
    import torch.nn.functional as F
    out = []
    for s in range(0, len(x), batch_size):
        xb = torch.tensor(x[s:s + batch_size], dtype=torch.float32, device=device)
        out.append(F.softmax(model(xb), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)
