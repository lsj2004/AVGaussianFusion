from __future__ import annotations

import torch
import torch.nn.functional as F


def _stft_magnitude(
    audio: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: torch.Tensor,
) -> torch.Tensor:
    return torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
    ).abs().square().clamp_min(1e-7).sqrt()


def stft_magnitude_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 128,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}"
        )
    if pred.ndim != 2:
        raise ValueError(f"audio tensors must be 2-D, got {pred.ndim} dimensions")
    if pred.shape[-1] < n_fft:
        raise ValueError(
            f"audio length {pred.shape[-1]} is shorter than n_fft={n_fft}"
        )

    window = torch.hann_window(n_fft, device=pred.device, dtype=pred.dtype)
    pred_mag = torch.stft(
        pred,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    ).abs()
    target_mag = torch.stft(
        target,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    ).abs()
    return F.l1_loss(torch.log1p(pred_mag), torch.log1p(target_mag))


def audiogs_mono_diff_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    diff_weight: float = 2.0,
    use_log_mag_loss: bool = False,
    lre_loss_weight: float = 0.0,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}"
        )
    if pred.ndim != 2 or pred.shape[0] != 2:
        raise ValueError(f"audio tensors must have shape (2, samples), got {tuple(pred.shape)}")
    if pred.shape[-1] < n_fft:
        raise ValueError(
            f"audio length {pred.shape[-1]} is shorter than n_fft={n_fft}"
        )
    if win_length <= 0 or hop_length <= 0 or n_fft <= 0:
        raise ValueError("STFT sizes must be positive")
    if win_length > n_fft:
        raise ValueError(f"win_length={win_length} cannot exceed n_fft={n_fft}")
    if diff_weight < 0:
        raise ValueError(f"diff_weight must be nonnegative, got {diff_weight}")
    if lre_loss_weight < 0:
        raise ValueError(f"lre_loss_weight must be nonnegative, got {lre_loss_weight}")

    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    window = torch.hamming_window(win_length, device=pred.device, dtype=pred.dtype)

    pred_mono = pred[0:1] + pred[1:2]
    pred_diff = pred[0:1] - pred[1:2]
    target_mono = target[0:1] + target[1:2]
    target_diff = target[0:1] - target[1:2]

    pred_mono_mag = _stft_magnitude(pred_mono, n_fft, hop_length, win_length, window)
    target_mono_mag = _stft_magnitude(target_mono, n_fft, hop_length, win_length, window)
    pred_diff_mag = _stft_magnitude(pred_diff, n_fft, hop_length, win_length, window)
    target_diff_mag = _stft_magnitude(target_diff, n_fft, hop_length, win_length, window)

    if use_log_mag_loss:
        pred_mono_mag = torch.log(pred_mono_mag.clamp_min(1e-7))
        target_mono_mag = torch.log(target_mono_mag.clamp_min(1e-7))
        pred_diff_mag = torch.log(pred_diff_mag.clamp_min(1e-7))
        target_diff_mag = torch.log(target_diff_mag.clamp_min(1e-7))

    loss = F.mse_loss(pred_mono_mag, target_mono_mag)
    loss = loss + float(diff_weight) * F.mse_loss(pred_diff_mag, target_diff_mag)
    if lre_loss_weight > 0:
        eps = pred.new_tensor(1e-5)
        pred_l_energy = pred[0].square().sum()
        pred_r_energy = pred[1].square().sum()
        target_l_energy = target[0].square().sum()
        target_r_energy = target[1].square().sum()
        pred_lre = 10.0 * torch.log10((pred_l_energy + eps) / (pred_r_energy + eps))
        target_lre = 10.0 * torch.log10((target_l_energy + eps) / (target_r_energy + eps))
        lre_loss = F.l1_loss(
            torch.nan_to_num(pred_lre, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(target_lre, nan=0.0, posinf=0.0, neginf=0.0),
        )
        loss = loss + float(lre_loss_weight) * lre_loss
    return loss
