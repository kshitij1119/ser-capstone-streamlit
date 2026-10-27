"""Bundle loading and memory-conscious inference for the SER Streamlit app."""
from __future__ import annotations

import gc
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
    """Raised when a model bundle is missing or incompatible."""


class Head(nn.Module):
    """Layer-weighted MLP head matching the capstone notebook."""
    def __init__(self, n_layers: int, dim: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(n_layers))
        self.mlp = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(dim, hidden), nn.GELU(),
            nn.Dropout(0.4), nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.layer_logits, dim=0)
        return self.mlp((weights[None, :, None] * x).sum(dim=1))


class SERPredictor:
    """Load an ensemble bundle and infer with one encoder resident at a time."""
    def __init__(self, bundle_path: str | Path, metadata_path: str | Path):
        bundle_path, metadata_path = Path(bundle_path), Path(metadata_path)
        if not bundle_path.is_file() or not metadata_path.is_file():
            raise BundleError("Expected bundle.pt and metadata.json in model_bundle/.")
        try:
            self.metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
            checkpoint = torch.load(bundle_path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise BundleError(f"Could not read the model bundle: {exc}") from exc

        self.labels = list(self.metadata.get("labels", []))
        self.sample_rate = int(self.metadata.get("sample_rate", 0))
        self.max_seconds = float(self.metadata.get("max_seconds", 0))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if len(self.labels) != 8 or self.sample_rate <= 0 or self.max_seconds <= 0:
            raise BundleError("Bundle metadata has invalid labels or audio settings.")
        records = checkpoint.get("members") if isinstance(checkpoint, dict) else None
        if not isinstance(records, list) or not records:
            raise BundleError("bundle.pt must contain a non-empty `members` ensemble.")

        self.members: list[dict[str, Any]] = []
        for record in records:
            required = {"backbone_key", "encoder_id", "seed", "head_state", "feature_mean", "feature_std"}
            if not isinstance(record, dict) or not required.issubset(record):
                raise BundleError("Each ensemble member must include encoder, seed, head, and statistics.")
            mean, std = record["feature_mean"].float(), record["feature_std"].float()
            if mean.ndim != 2 or mean.shape != std.shape:
                raise BundleError("Member feature statistics must have matching (layers, features) shapes.")
            head = Head(mean.shape[0], mean.shape[1], len(self.labels), int(self.metadata.get("head_hidden", 256)))
            try:
                head.load_state_dict(record["head_state"], strict=True)
            except Exception as exc:
                raise BundleError(f"Classifier weights are incompatible: {exc}") from exc
            head.eval().to(self.device)
            self.members.append({
                "backbone_key": str(record["backbone_key"]),
                "encoder_id": str(record["encoder_id"]),
                "seed": int(record["seed"]),
                "head": head,
                "mean": mean.to(self.device),
                "std": std.clamp_min(1e-6).to(self.device),
            })
        self.encoder_ids = list(dict.fromkeys(m["encoder_id"] for m in self.members))

        if self.metadata.get("speaker_normalization"):
            raise BundleError(
                "This app requires a bundle trained without speaker normalization, "
                "because it classifies one utterance without speaker calibration."
            )

    def _prepare_waveform(self, audio_bytes: bytes) -> tuple[np.ndarray, float]:
        if not audio_bytes:
            raise ValueError("The audio file is empty.")
        try:
            wave, source_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
        except Exception as exc:
            raise ValueError("Could not decode the recording. Upload or record a valid WAV file.") from exc
        if wave.size == 0 or source_rate <= 0:
            raise ValueError("The recording contains no usable samples.")
        wave = wave.mean(axis=1, dtype=np.float32)
        if not np.isfinite(wave).all():
            raise ValueError("The recording contains invalid numeric samples.")
        if source_rate != self.sample_rate:
            wave = librosa.resample(wave, orig_sr=int(source_rate), target_sr=self.sample_rate).astype(np.float32)
        original = wave
        trimmed, _ = librosa.effects.trim(wave, top_db=30)
        wave = trimmed if len(trimmed) > int(0.2 * self.sample_rate) else original
        wave = wave[: int(self.max_seconds * self.sample_rate)]
        peak = float(np.max(np.abs(wave))) if wave.size else 0.0
        if peak < 1e-7:
            raise ValueError("The recording is silent or too quiet to analyze.")
        wave = wave / (peak + 1e-8)
        minimum = int(0.5 * self.sample_rate)
        if len(wave) < minimum:
            wave = np.pad(wave, (0, minimum - len(wave)))
        duration = min(len(wave) / self.sample_rate, self.max_seconds)
        # Match the notebook's embedding path: z-score after trim, crop, and peak normalization.
        wave = (wave - wave.mean()) / (wave.std() + 1e-7)
        return wave.astype(np.float32, copy=False), duration

    @torch.inference_mode()
    def predict(self, audio_bytes: bytes) -> tuple[np.ndarray, float, dict[str, np.ndarray]]:
        wave, duration = self._prepare_waveform(audio_bytes)
        inputs = torch.from_numpy(wave).unsqueeze(0).to(self.device)
        all_member_probs: list[np.ndarray] = []
        per_encoder: dict[str, np.ndarray] = {}

        # Load and release each large encoder sequentially to reduce peak Cloud RAM.
        for encoder_id in self.encoder_ids:
            try:
                encoder = AutoModel.from_pretrained(encoder_id).to(self.device).eval()
            except Exception as exc:
                raise BundleError(
                    f"Could not load `{encoder_id}`. Check Hugging Face access and Cloud memory."
                ) from exc
            output = encoder(inputs, output_hidden_states=True)
            hidden_states = output.hidden_states
            encoder_members = [m for m in self.members if m["encoder_id"] == encoder_id]
            if hidden_states is None:
                raise BundleError(f"Encoder `{encoder_id}` returned no hidden states.")
            layer_features = torch.stack(hidden_states, dim=0)[:, 0].float()
            features = torch.cat([layer_features.mean(dim=1), layer_features.std(dim=1)], dim=-1).unsqueeze(0)
            member_probs = []
            for member in encoder_members:
                if features.shape[1:] != member["mean"].shape:
                    raise BundleError(f"Feature shape does not match bundle for `{encoder_id}`.")
                normalized = (features - member["mean"]) / member["std"]
                probs = torch.softmax(member["head"](normalized), dim=-1)[0]
                member_probs.append(probs.cpu().numpy())
            encoder_mean = np.mean(member_probs, axis=0)
            per_encoder[encoder_id] = encoder_mean
            all_member_probs.extend(member_probs)
            del encoder, output, hidden_states, layer_features, features
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Average seed-level member probabilities, matching the notebook ensemble rule.
        return np.mean(all_member_probs, axis=0), duration, per_encoder

