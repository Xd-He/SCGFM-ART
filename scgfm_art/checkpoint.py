from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import SCGFMARTConfig, SCGFMARTModel


def model_config_from_checkpoint(payload: dict[str, Any]) -> SCGFMARTConfig:
    config = dict(payload.get("model_config", {}))
    return SCGFMARTConfig(**config)


def save_checkpoint(
    model: SCGFMARTModel,
    path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    epoch: int = 0,
    train_config: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variant": model.implementation_version,
            "model_config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "optimizer": (
                None if optimizer is None else optimizer.state_dict()
            ),
            "scaler": (
                None if scaler is None else scaler.state_dict()
            ),
            "epoch": int(epoch),
            "train_config": train_config or {},
            "history": history or [],
        },
        target,
    )


def load_checkpoint(
    path: str | Path,
    device: str | torch.device,
) -> tuple[SCGFMARTModel, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        payload.get("variant")
        not in SCGFMARTModel.compatible_checkpoint_versions
    ):
        raise ValueError(
            "Expected one of "
            f"{sorted(SCGFMARTModel.compatible_checkpoint_versions)}, "
            f"got {payload.get('variant')!r}."
        )
    model = SCGFMARTModel(
        model_config_from_checkpoint(payload), device=device
    ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload

