from __future__ import annotations

import torch
import torch.nn.functional as F


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
