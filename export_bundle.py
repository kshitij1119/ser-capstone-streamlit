"""Save a trained notebook classifier and normalization stats for Streamlit inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from torch import nn


def export_bundle(
    model: nn.Module,
    stats: tuple[torch.Tensor, torch.Tensor],
    *,
    encoder_id: str,
    labels: Sequence[str],
    output_dir: str | Path = "model_bundle",
    sample_rate: int = 16_000,
    max_seconds: float = 5.0,
    backbone_key: str = "",
    speaker_normalization: bool = False,
) -> Path:
    """Write a CPU-safe classifier bundle and JSON metadata.

    The single-clip Streamlit app requires speaker_normalization=False because a
    new speaker's reference embedding cannot be estimated from one utterance.
    """
    if speaker_normalization:
        raise ValueError(
            "Export with speaker_normalization=False for single-utterance inference. "
            "Retrain the head without speaker normalization first."
        )
    if len(labels) != 8:
        raise ValueError(f"Expected the notebook's 8 labels; received {len(labels)}.")
    if len(stats) != 2:
        raise ValueError("stats must be the (feature_mean, feature_std) pair from fit_head().")

    feature_mean, feature_std = stats
    if feature_mean.shape != feature_std.shape or feature_mean.ndim != 2:
        raise ValueError("Feature statistics must have equal 2-D (layers, features) shapes.")
    if not torch.isfinite(feature_mean).all() or not torch.isfinite(feature_std).all():
        raise ValueError("Feature statistics contain NaN or infinite values.")

    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    checkpoint = {
        "head_state": state,
        "feature_mean": feature_mean.detach().float().cpu().contiguous(),
        "feature_std": feature_std.detach().float().cpu().clamp_min(1e-6).contiguous(),
    }
    torch.save(checkpoint, folder / "bundle.pt")

    metadata = {
        "format_version": 1,
        "backbone_key": backbone_key,
        "encoder_id": encoder_id,
        "labels": list(labels),
        "sample_rate": int(sample_rate),
        "max_seconds": float(max_seconds),
        "n_layers": int(feature_mean.shape[0]),
        "feature_dim": int(feature_mean.shape[1]),
        "head_hidden": 256,
        "pooling": "temporal mean and standard deviation per hidden layer",
        "speaker_normalization": False,
        "input_standardization": "per-clip z-score after trim/crop/peak-normalization",
    }
    (folder / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return folder

