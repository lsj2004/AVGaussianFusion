from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch


def _validate_stereo_pair(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, got {pred.shape} and {target.shape}"
        )
    if pred.ndim != 2 or pred.shape[0] != 2:
        raise ValueError(f"pred and target must be stereo tensors shaped (2, T), got {pred.shape}")
    if pred.numel() == 0:
        raise ValueError("pred and target must be non-empty")
    if not torch.is_floating_point(pred) or not torch.is_floating_point(target):
        raise ValueError("pred and target must be floating-point tensors")
    return pred.detach().cpu().float(), target.detach().cpu().float()


def _eval_mag(wav: torch.Tensor) -> torch.Tensor:
    return torch.stft(
        wav,
        n_fft=512,
        hop_length=160,
        win_length=400,
        window=torch.hamming_window(400, device=wav.device),
        pad_mode="constant",
        return_complex=True,
    ).abs()


def compute_mag_distance(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred, target = _validate_stereo_pair(pred, target)
    pred_l, target_l = _eval_mag(pred[0]), _eval_mag(target[0])
    pred_r, target_r = _eval_mag(pred[1]), _eval_mag(target[1])
    distance = (pred_l - target_l).abs().mean() + (pred_r - target_r).abs().mean()
    return float(distance.item())


def _analytic_envelope(channel: torch.Tensor) -> torch.Tensor:
    spectrum = torch.fft.fft(channel)
    multiplier = torch.zeros_like(spectrum)
    n = channel.numel()
    if n % 2 == 0:
        multiplier[0] = 1
        multiplier[n // 2] = 1
        multiplier[1 : n // 2] = 2
    else:
        multiplier[0] = 1
        multiplier[1 : (n + 1) // 2] = 2
    return torch.fft.ifft(spectrum * multiplier).abs()


def compute_env_distance(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred, target = _validate_stereo_pair(pred, target)
    left = torch.sqrt(torch.mean((_analytic_envelope(target[0]) - _analytic_envelope(pred[0])) ** 2))
    right = torch.sqrt(torch.mean((_analytic_envelope(target[1]) - _analytic_envelope(pred[1])) ** 2))
    return float((left + right).item())


def compute_lre(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred, target = _validate_stereo_pair(pred, target)
    pred_lr_ratio = 10 * torch.log10(
        (pred[0].pow(2).sum() + 1e-5) / (pred[1].pow(2).sum() + 1e-5)
    )
    target_lr_ratio = 10 * torch.log10(
        (target[0].pow(2).sum() + 1e-5) / (target[1].pow(2).sum() + 1e-5)
    )
    return float(torch.abs(pred_lr_ratio - target_lr_ratio).item())


def compute_dpam(pred: torch.Tensor, target: torch.Tensor, sample_rate: int) -> float:
    pred, target = _validate_stereo_pair(pred, target)
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    try:
        import cdpam
    except Exception as error:  # pragma: no cover - exact missing dependency varies by env
        raise RuntimeError(f"cdpam is unavailable: {error}") from error

    with tempfile.TemporaryDirectory(prefix="avfusion-dpam-") as tmpdir:
        tmp = Path(tmpdir)
        pred_path = tmp / "pred.wav"
        target_path = tmp / "target.wav"
        sf.write(pred_path, pred.T.numpy(), sample_rate)
        sf.write(target_path, target.T.numpy(), sample_rate)
        wav_ref = cdpam.load_audio(str(target_path))
        wav_out = cdpam.load_audio(str(pred_path))
        original_torch_load = torch.load

        def _torch_load_cdpam_compat(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = _torch_load_cdpam_compat
        try:
            loss_fn = cdpam.CDPAM()
        finally:
            torch.load = original_torch_load
        value = loss_fn.forward(wav_ref, wav_out).detach().cpu().numpy()[0]
        return float(value)


def compute_audiogs_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    sample_rate: int,
    include_dpam: bool = True,
) -> dict[str, Any]:
    pred, target = _validate_stereo_pair(pred, target)
    metrics: dict[str, Any] = {
        "MAG": compute_mag_distance(pred, target),
        "ENV": compute_env_distance(pred, target),
        "LRE": compute_lre(pred, target),
        "RTE": None,
        "RTE_available": False,
        "RTE_error": "RT60 estimator integration is not configured for this evaluator",
    }
    if include_dpam:
        try:
            metrics["DPAM"] = compute_dpam(pred, target, sample_rate)
            metrics["DPAM_available"] = True
            metrics["DPAM_error"] = None
        except RuntimeError as error:
            metrics["DPAM"] = None
            metrics["DPAM_available"] = False
            metrics["DPAM_error"] = str(error)
    return metrics
