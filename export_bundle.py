"""Export an ensemble of capstone classifier heads for Streamlit inference."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import nn


def export_bundle(
    members: Sequence[dict[str, Any]], *, labels: Sequence[str], output_dir: str | Path,
    sample_rate: int = 16_000, max_seconds: float = 5.0,
    speaker_normalization: bool = False, results: dict[str, Any] | None = None,
) -> Path:
    """Serialize fitted (model, stats, encoder metadata) entries and run results.

    Each member contains model, stats=(mean,std), backbone_key, encoder_id, and seed.
    Single-utterance inference requires speaker_normalization=False.
    """
    if speaker_normalization:
        raise ValueError("Export single-utterance models with speaker normalization disabled.")
    if len(labels) != 8 or not members:
        raise ValueError("Expected eight emotion labels and at least one ensemble member.")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    first_shape = None
    for item in members:
        model: nn.Module = item["model"]
        mean, std = item["stats"]
        if mean.ndim != 2 or mean.shape != std.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Each member needs finite matching (layers, features) normalization tensors.")
        first_shape = first_shape or tuple(mean.shape)
        if tuple(mean.shape) != first_shape:
            raise ValueError("All encoder members must produce the same feature shape.")
        records.append({
            "backbone_key": str(item["backbone_key"]),
            "encoder_id": str(item["encoder_id"]),
            "seed": int(item["seed"]),
            "head_state": {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
            "feature_mean": mean.detach().float().cpu().contiguous(),
            "feature_std": std.detach().float().cpu().clamp_min(1e-6).contiguous(),
        })
    torch.save({"members": records}, folder / "bundle.pt")
    metadata = {
        "format_version": 2,
        "labels": list(labels),
        "sample_rate": int(sample_rate),
        "max_seconds": float(max_seconds),
        "n_layers": int(first_shape[0]),
        "feature_dim": int(first_shape[1]),
        "head_hidden": 256,
        "encoder_ids": list(dict.fromkeys(x["encoder_id"] for x in records)),
        "member_count": len(records),
        "seed_count": len(set(x["seed"] for x in records)),
        "pooling": "temporal mean and standard deviation per hidden layer",
        "speaker_normalization": False,
        "input_standardization": "per-clip z-score after trim/crop/peak-normalization",
    }
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    if results is not None:
        (folder / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float)+"\n", encoding="utf-8")
    return folder

