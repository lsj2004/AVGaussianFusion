from __future__ import annotations

import torch
from torch import Tensor, nn


class JointAudioHead(nn.Module):
    def __init__(self, num_points: int, top_k: int | None = None):
        super().__init__()
        if num_points <= 0:
            raise ValueError(f"num_points must be positive, got {num_points}")
        if top_k is not None and top_k < 2:
            raise ValueError(f"top_k must be at least 2, got {top_k}")
        if num_points < 2:
            raise ValueError("at least two points are required for route weighting")
        self.num_points = int(num_points)
        self.top_k = min(int(top_k), num_points) if top_k is not None else num_points
        gain_init = torch.linspace(1e-3, 2e-3, num_points).reshape(num_points, 1)
        self.audio_opacity = nn.Parameter(torch.zeros(num_points, 1))
        self.mono_gain = nn.Parameter(gain_init.clone())
        self.diff_gain = nn.Parameter(gain_init.clone())
        self.delay_offset = nn.Parameter(torch.zeros(num_points, 1))
        self.attenuation_logit = nn.Parameter(torch.zeros(num_points, 1))

    @property
    def active_count(self) -> int:
        return self.top_k

    def forward(self, state: dict[str, Tensor], source_audio: Tensor) -> Tensor:
        if source_audio.ndim != 2 or source_audio.shape[0] != 2:
            raise ValueError(f"source_audio must have shape (2, samples), got {tuple(source_audio.shape)}")
        xyz = state["xyz"]
        opacity = state["opacity"]
        velocity = state["velocity"]
        scores = (opacity + self.audio_opacity).reshape(-1)
        indices = torch.argsort(scores, descending=True, stable=True)[: self.top_k]
        selected_opacity = scores.index_select(0, indices).reshape(-1, 1)
        weights = torch.softmax(selected_opacity, dim=0)
        selected_xyz = xyz.index_select(0, indices)
        selected_velocity = velocity.index_select(0, indices)
        distance = selected_xyz.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        attenuation = torch.sigmoid(self.attenuation_logit.index_select(0, indices)) / distance
        delay_weight = torch.sigmoid(self.delay_offset.index_select(0, indices))
        velocity_projection = selected_velocity.mean(dim=-1, keepdim=True)
        velocity_factor = 1.0 + 0.01 * torch.tanh(velocity_projection)
        weights = weights * attenuation * delay_weight * velocity_factor
        weights = weights / weights.sum().clamp_min(1e-6)
        mono = torch.tanh((weights * self.mono_gain.index_select(0, indices)).sum())
        diff = torch.tanh((weights * self.diff_gain.index_select(0, indices)).sum())
        mono_source = source_audio.to(mono).mean(dim=0, keepdim=True)
        left = mono_source * (1 + mono + diff)
        right = mono_source * (1 + mono - diff)
        return torch.cat([left, right], dim=0)


class SpectralJointAudioHead(nn.Module):
    """AudioGS-style frequency-mask head for Route B ablations.

    This keeps the AVFusion Gaussian carrier but moves audio rendering into the
    STFT domain: selected Gaussians aggregate frequency-dependent mono/diff
    masks, which modulate the source magnitude before iSTFT reconstruction.
    """

    def __init__(
        self,
        num_points: int,
        num_frequency_bins: int = 257,
        top_k: int | None = None,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
    ):
        super().__init__()
        if num_points <= 0:
            raise ValueError(f"num_points must be positive, got {num_points}")
        if num_points < 2:
            raise ValueError("at least two points are required for route weighting")
        if top_k is not None and top_k < 2:
            raise ValueError(f"top_k must be at least 2, got {top_k}")
        if n_fft <= 0 or hop_length <= 0 or win_length <= 0:
            raise ValueError("STFT sizes must be positive")
        expected_bins = n_fft // 2 + 1
        if int(num_frequency_bins) != expected_bins:
            raise ValueError(
                f"num_frequency_bins must match n_fft // 2 + 1 ({expected_bins}), "
                f"got {num_frequency_bins}"
            )
        self.num_points = int(num_points)
        self.num_frequency_bins = int(num_frequency_bins)
        self.top_k = min(int(top_k), num_points) if top_k is not None else num_points
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.audio_opacity = nn.Parameter(torch.zeros(num_points, 1))
        point_init = torch.linspace(1e-3, 2e-3, num_points).reshape(num_points, 1)
        freq_init = torch.linspace(0.5, 1.0, self.num_frequency_bins).reshape(1, self.num_frequency_bins)
        self.mono_mask = nn.Parameter(point_init * freq_init)
        self.diff_mask = nn.Parameter(0.5 * point_init * freq_init)
        self.distance_logit = nn.Parameter(torch.zeros(num_points, self.num_frequency_bins))
        self.velocity_scale = nn.Parameter(torch.zeros(num_points, 1))

    @property
    def active_count(self) -> int:
        return self.top_k

    def _select_weights(self, state: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        xyz = state["xyz"]
        opacity = state["opacity"]
        velocity = state["velocity"]
        scores = (opacity + self.audio_opacity).reshape(-1)
        indices = torch.argsort(scores, descending=True, stable=True)[: self.top_k]
        selected_scores = scores.index_select(0, indices).reshape(-1, 1)
        weights = torch.softmax(selected_scores, dim=0)
        selected_xyz = xyz.index_select(0, indices)
        selected_velocity = velocity.index_select(0, indices)
        distance = selected_xyz.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        attenuation = torch.sigmoid(self.distance_logit.index_select(0, indices)) / distance
        velocity_projection = selected_velocity.mean(dim=-1, keepdim=True)
        velocity_factor = 1.0 + 0.01 * torch.tanh(
            velocity_projection * (1.0 + self.velocity_scale.index_select(0, indices))
        )
        weights = weights * attenuation * velocity_factor
        weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-6)
        return indices, weights

    def forward(self, state: dict[str, Tensor], source_audio: Tensor) -> Tensor:
        if source_audio.ndim != 2 or source_audio.shape[0] != 2:
            raise ValueError(f"source_audio must have shape (2, samples), got {tuple(source_audio.shape)}")
        if source_audio.shape[-1] < self.n_fft:
            raise ValueError(
                f"source audio length {source_audio.shape[-1]} is shorter than n_fft={self.n_fft}"
            )
        device = self.mono_mask.device
        dtype = self.mono_mask.dtype
        source_audio = source_audio.to(device=device, dtype=dtype)
        window = torch.hamming_window(self.win_length, device=device, dtype=dtype)
        source_spec = torch.stft(
            source_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        source_magnitude = source_spec.abs()
        source_phase = torch.angle(source_spec)
        carrier_magnitude = source_magnitude.mean(dim=0)

        indices, weights = self._select_weights(state)
        mono = torch.tanh((weights * self.mono_mask.index_select(0, indices)).sum(dim=0))
        diff = torch.tanh((weights * self.diff_mask.index_select(0, indices)).sum(dim=0))
        mono = mono.reshape(-1, 1)
        diff = diff.reshape(-1, 1)
        left_mag = torch.relu(carrier_magnitude * (1.0 + mono + diff))
        right_mag = torch.relu(carrier_magnitude * (1.0 + mono - diff))
        left_spec = torch.polar(left_mag, source_phase[0])
        right_spec = torch.polar(right_mag, source_phase[1])
        left = torch.istft(
            left_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            length=source_audio.shape[-1],
        )
        right = torch.istft(
            right_spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            length=source_audio.shape[-1],
        )
        return torch.stack([left, right], dim=0)
