"""Bundle loading and audio inference for the SER Streamlit application."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from transformers import AutoModel


class BundleError(RuntimeError):
    """Raised when a model bundle is absent, incomplete, or incompatible."""


class Head(nn.Module):
    """Matches the layer-weighted MLP classifier used in the capstone notebook."""

    def __init__(self, n_layers: int, dim: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.mlp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.layer_logits, dim=0)
        return self.mlp((weights[None, :, None] * x).sum(dim=1))


class SERPredictor:
    """Load a saved head and its matching pretrained speech encoder."""

    def __init__(self, bundle_path: str | Path, metadata_path: str | Path):
        self.bundle_path = Path(bundle_path)
        self.metadata_path = Path(metadata_path)
        if not self.bundle_path.is_file() or not self.metadata_path.is_file():
            raise BundleError(
                "Model bundle is incomplete. Expected bundle.pt and metadata.json in model_bundle/."
            )

        try:
            metadata: dict[str, Any] = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            # weights_only prevents arbitrary Python object deserialization.
            checkpoint = torch.load(self.bundle_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise BundleError(f"Could not read model bundle: {exc}") from exc

        required = {"encoder_id", "labels", "sample_rate", "max_seconds", "n_layers", "feature_dim"}
        missing = required - metadata.keys()
        if missing:
            raise BundleError(f"Bundle metadata is missing fields: {', '.join(sorted(missing))}")
        if not isinstance(checkpoint, dict) or not {
            "head_state", "feature_mean", "feature_std"
        }.issubset(checkpoint):
            raise BundleError("bundle.pt must contain head_state, feature_mean, and feature_std.")

        self.encoder_id = str(metadata["encoder_id"])
        self.labels = list(metadata["labels"])
        self.sample_rate = int(metadata["sample_rate"])
        self.max_seconds = float(metadata["max_seconds"])
        n_layers = int(metadata["n_layers"])
        feature_dim = int(metadata["feature_dim"])
        hidden = int(metadata.get("head_hidden", 256))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not self.labels or self.sample_rate <= 0 or self.max_seconds <= 0:
            raise BundleError("Bundle metadata contains invalid labels or audio settings.")
        if len(checkpoint["feature_mean"].shape) != 2:
            raise BundleError("feature_mean must have shape (layers, pooled_features).")
        if checkpoint["feature_mean"].shape != checkpoint["feature_std"].shape:
            raise BundleError("feature_mean and feature_std shapes do not match.")

        self.feature_mean = checkpoint["feature_mean"].float().to(self.device)
        self.feature_std = checkpoint["feature_std"].float().clamp_min(1e-6).to(self.device)
        self.head = Head(n_layers, feature_dim, len(self.labels), hidden=hidden).to(self.device)
        try:
            self.head.load_state_dict(checkpoint["head_state"], strict=True)
        except Exception as exc:
            raise BundleError(f"Classifier weights do not match bundle metadata: {exc}") from exc
        self.head.eval()

        try:
            self.encoder = AutoModel.from_pretrained(self.encoder_id).to(self.device).eval()
        except Exception as exc:
            raise BundleError(
                f"Could not load pretrained encoder `{self.encoder_id}`. "
                "Check Cloud logs, network access, and available memory."
            ) from exc

    def _prepare_waveform(self, audio_bytes: bytes) -> tuple[np.ndarray, float]:
        if not audio_bytes:
            raise ValueError("The uploaded audio file is empty.")
        try:
            waveform, source_rate = sf.read(
                io.BytesIO(audio_bytes), dtype="float32", always_2d=True
            )
        except Exception as exc:
            raise ValueError("Could not decode this WAV file. Please upload a valid PCM WAV recording.") from exc

        if waveform.size == 0 or source_rate <= 0:
            raise ValueError("The WAV file contains no usable audio samples.")
        waveform = waveform.mean(axis=1, dtype=np.float32)
        if not np.isfinite(waveform).all():
            raise ValueError("The recording contains invalid numeric samples.")
        if source_rate != self.sample_rate:
            waveform = librosa.resample(
                waveform, orig_sr=int(source_rate), target_sr=self.sample_rate
            ).astype(np.float32, copy=False)

        original = waveform
        trimmed, _ = librosa.effects.trim(waveform, top_db=30)
        waveform = trimmed if len(trimmed) > int(0.2 * self.sample_rate) else original
        waveform = waveform[: int(self.max_seconds * self.sample_rate)]

        peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
        if peak < 1e-7:
            raise ValueError("The recording is silent or too quiet to analyze.")
        waveform = waveform / (peak + 1e-8)
        minimum_samples = int(0.5 * self.sample_rate)
        if len(waveform) < minimum_samples:
            waveform = np.pad(waveform, (0, minimum_samples - len(waveform)))

        duration = min(len(waveform) / self.sample_rate, self.max_seconds)
        # Match the notebook's embed() preprocessing after Clips(): per-clip z-score.
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-7)
        return waveform.astype(np.float32, copy=False), duration

    @torch.inference_mode()
    def predict(self, audio_bytes: bytes) -> tuple[np.ndarray, float]:
        waveform, duration = self._prepare_waveform(audio_bytes)
        inputs = torch.from_numpy(waveform).unsqueeze(0).to(self.device)
        output = self.encoder(inputs, output_hidden_states=True)
        hidden_states = output.hidden_states
        if hidden_states is None or len(hidden_states) != self.head.layer_logits.numel():
            raise BundleError("Encoder hidden-state count differs from the exported classifier.")

        # Shape after stacking: (layers, batch, frames, hidden_dim).
        h = torch.stack(hidden_states, dim=0)[:, 0].float()
        features = torch.cat([h.mean(dim=1), h.std(dim=1)], dim=-1).unsqueeze(0)
        if features.shape[1:] != self.feature_mean.shape:
            raise BundleError(
                f"Encoder produced feature shape {tuple(features.shape[1:])}; "
                f"bundle expects {tuple(self.feature_mean.shape)}."
            )
        normalized = (features - self.feature_mean) / self.feature_std
        probabilities = torch.softmax(self.head(normalized), dim=-1)[0]
        return probabilities.cpu().numpy(), duration

