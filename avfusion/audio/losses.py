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


def _stft_complex(
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
    )


def _validate_stft_scale(scale: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(scale) != 3:
        raise ValueError(f"STFT scale must be (n_fft, hop_length, win_length), got {scale!r}")
    n_fft, hop_length, win_length = (int(scale[0]), int(scale[1]), int(scale[2]))
    if n_fft <= 0 or hop_length <= 0 or win_length <= 0:
        raise ValueError("STFT sizes must be positive")
    if win_length > n_fft:
        raise ValueError(f"win_length={win_length} cannot exceed n_fft={n_fft}")
    return n_fft, hop_length, win_length


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
    mr_stft_scales: tuple[tuple[int, int, int], ...] | None = None,
    phase_loss_weight: float = 0.0,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(
            f"shape mismatch: pred={tuple(pred.shape)} target={tuple(target.shape)}"
        )
    if pred.ndim != 2 or pred.shape[0] != 2:
        raise ValueError(f"audio tensors must have shape (2, samples), got {tuple(pred.shape)}")
    scales = (
        tuple(_validate_stft_scale(scale) for scale in mr_stft_scales)
        if mr_stft_scales is not None
        else ((_validate_stft_scale((n_fft, hop_length, win_length))),)
    )
    max_n_fft = max(scale[0] for scale in scales)
    if pred.shape[-1] < max_n_fft:
        raise ValueError(
            f"audio length {pred.shape[-1]} is shorter than n_fft={max_n_fft}"
        )
    if diff_weight < 0:
        raise ValueError(f"diff_weight must be nonnegative, got {diff_weight}")
    if lre_loss_weight < 0:
        raise ValueError(f"lre_loss_weight must be nonnegative, got {lre_loss_weight}")
    if phase_loss_weight < 0:
        raise ValueError(f"phase_loss_weight must be nonnegative, got {phase_loss_weight}")

    pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

    pred_mono = pred[0:1] + pred[1:2]
    pred_diff = pred[0:1] - pred[1:2]
    target_mono = target[0:1] + target[1:2]
    target_diff = target[0:1] - target[1:2]

    scale_losses = []
    for scale_n_fft, scale_hop, scale_win in scales:
        window = torch.hamming_window(scale_win, device=pred.device, dtype=pred.dtype)
        pred_mono_spec = _stft_complex(pred_mono, scale_n_fft, scale_hop, scale_win, window)
        target_mono_spec = _stft_complex(target_mono, scale_n_fft, scale_hop, scale_win, window)
        pred_diff_spec = _stft_complex(pred_diff, scale_n_fft, scale_hop, scale_win, window)
        target_diff_spec = _stft_complex(target_diff, scale_n_fft, scale_hop, scale_win, window)
        pred_mono_mag = pred_mono_spec.abs().square().clamp_min(1e-7).sqrt()
        target_mono_mag = target_mono_spec.abs().square().clamp_min(1e-7).sqrt()
        pred_diff_mag = pred_diff_spec.abs().square().clamp_min(1e-7).sqrt()
        target_diff_mag = target_diff_spec.abs().square().clamp_min(1e-7).sqrt()

        if use_log_mag_loss:
            pred_mono_mag = torch.log(pred_mono_mag.clamp_min(1e-7))
            target_mono_mag = torch.log(target_mono_mag.clamp_min(1e-7))
            pred_diff_mag = torch.log(pred_diff_mag.clamp_min(1e-7))
            target_diff_mag = torch.log(target_diff_mag.clamp_min(1e-7))

        scale_loss = F.mse_loss(pred_mono_mag, target_mono_mag)
        scale_loss = scale_loss + float(diff_weight) * F.mse_loss(pred_diff_mag, target_diff_mag)
        if phase_loss_weight > 0:
            pred_spec = _stft_complex(pred, scale_n_fft, scale_hop, scale_win, window)
            target_spec = _stft_complex(target, scale_n_fft, scale_hop, scale_win, window)
            phase_delta = torch.angle(pred_spec) - torch.angle(target_spec)
            phase_loss = (1.0 - torch.cos(phase_delta)).mean()
            scale_loss = scale_loss + float(phase_loss_weight) * phase_loss
        scale_losses.append(scale_loss)

    loss = torch.stack(scale_losses).mean()
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
