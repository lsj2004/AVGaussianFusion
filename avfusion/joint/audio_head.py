from __future__ import annotations

import torch
import torch.nn.functional as F
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


class AudioGSMaskedSpectralHead(nn.Module):
    """AudioGS-like mono/diff TF mask head over the shared Gaussian carrier.

    The original AudioGS model learns spherical-harmonic mono/diff fields on
    audio Gaussian points and renders masks in the STFT domain. This head keeps
    AVFusion's shared FTGS++ points, but adopts the same core audio synthesis:
    source mid magnitude -> mono mask -> diff envelope -> L/R magnitude -> iSTFT.
    """

    def __init__(
        self,
        num_points: int,
        num_frequency_bins: int = 257,
        top_k: int | None = None,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        sh_rand_init_std: float = 0.1,
        freq_atten_alpha: float = 1.0,
        diff_lr_sign: float = 1.0,
        sh_degree: int = 3,
        carrier_indices: Tensor | None = None,
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
        if int(sh_degree) not in {1, 3}:
            raise ValueError(f"sh_degree must be 1 or 3, got {sh_degree}")
        self.num_points = int(num_points)
        self.num_frequency_bins = int(num_frequency_bins)
        self.top_k = min(int(top_k), num_points) if top_k is not None else num_points
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.freq_atten_alpha = float(freq_atten_alpha)
        self.diff_lr_sign = float(diff_lr_sign)
        self.sh_degree = int(sh_degree)
        self.sh_basis_dim = 16 if self.sh_degree == 3 else 4
        if carrier_indices is not None:
            carrier_indices = torch.as_tensor(carrier_indices, dtype=torch.long)
            if carrier_indices.ndim != 1:
                raise ValueError("carrier_indices must be a 1D tensor")
            if carrier_indices.numel() != self.top_k:
                raise ValueError(
                    f"carrier_indices must contain active_count={self.top_k} entries, "
                    f"got {carrier_indices.numel()}"
                )
            if carrier_indices.numel() and (
                int(carrier_indices.min()) < 0 or int(carrier_indices.max()) >= self.num_points
            ):
                raise ValueError("carrier_indices contains an out-of-range Gaussian index")
        else:
            carrier_indices = torch.empty(0, dtype=torch.long)
        self.register_buffer("carrier_indices", carrier_indices, persistent=True)

        self.audio_opacity = nn.Parameter(torch.zeros(self.top_k, 1))
        self.rotation = nn.Parameter(torch.zeros(self.top_k, 3))
        self.freq_atten_logit = nn.Parameter(torch.zeros(self.top_k, num_frequency_bins))
        self.velocity_scale = nn.Parameter(torch.zeros(self.top_k, 1))
        self.mono_bias = nn.Parameter(torch.zeros(num_frequency_bins))
        self.diff_bias = nn.Parameter(torch.full((num_frequency_bins,), -2.0))
        self.mono_sh = nn.Parameter(
            torch.randn(self.top_k, num_frequency_bins, self.sh_basis_dim) * float(sh_rand_init_std)
        )
        self.diff_sh = nn.Parameter(
            torch.randn(self.top_k, num_frequency_bins, self.sh_basis_dim) * float(sh_rand_init_std)
        )

    @property
    def active_count(self) -> int:
        return self.top_k

    def _direction_features(self, xyz: Tensor, rotation: Tensor) -> tuple[Tensor, Tensor]:
        direction = F.normalize(xyz + torch.tanh(rotation), dim=-1, eps=1e-6)
        ones = torch.ones(direction.shape[0], 1, device=xyz.device, dtype=xyz.dtype)
        x, y, z = direction.unbind(dim=-1)
        degree1 = [ones.reshape(-1), x, y, z]
        if self.sh_degree == 1:
            features = torch.stack(degree1, dim=-1)
        else:
            features = torch.stack(
                [
                    *degree1,
                    x * y,
                    y * z,
                    2.0 * z.square() - x.square() - y.square(),
                    x * z,
                    x.square() - y.square(),
                    y * (3.0 * x.square() - y.square()),
                    x * y * z,
                    y * (4.0 * z.square() - x.square() - y.square()),
                    z * (2.0 * z.square() - 3.0 * x.square() - 3.0 * y.square()),
                    x * (4.0 * z.square() - x.square() - y.square()),
                    z * (x.square() - y.square()),
                    x * (x.square() - 3.0 * y.square()),
                ],
                dim=-1,
            )
        distance = xyz.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        return features, distance

    @staticmethod
    def combine_mono_diff_magnitudes(
        mono_mag: Tensor,
        diff_envelope: Tensor,
        diff_lr_sign: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        diff = diff_envelope.clamp(0.0, 1.0)
        left = mono_mag * (1.0 + float(diff_lr_sign) * diff)
        right = mono_mag * (1.0 - float(diff_lr_sign) * diff)
        return left.clamp_min(0.0), right.clamp_min(0.0)

    def _render_masks(self, state: dict[str, Tensor], source_magnitude: Tensor) -> tuple[Tensor, Tensor]:
        xyz = state["xyz"]
        opacity = state["opacity"]
        velocity = state["velocity"]
        visual_scores = opacity.reshape(-1)
        if self.carrier_indices.numel():
            indices = self.carrier_indices.to(device=xyz.device)
        else:
            indices = torch.argsort(visual_scores, descending=True, stable=True)[: self.top_k]
        local_indices = torch.arange(indices.numel(), device=xyz.device)
        selected_audio_opacity = self.audio_opacity.index_select(0, local_indices)
        scores = visual_scores.index_select(0, indices) + selected_audio_opacity.reshape(-1)

        selected_xyz = xyz.index_select(0, indices)
        selected_rotation = self.rotation.index_select(0, local_indices)
        features, distance = self._direction_features(selected_xyz, selected_rotation)
        point_weight = torch.softmax(scores, dim=0).reshape(-1, 1)
        freq_atten = torch.sigmoid(self.freq_atten_logit.index_select(0, local_indices))
        distance_atten = distance.pow(-self.freq_atten_alpha)
        velocity_projection = velocity.index_select(0, indices).mean(dim=-1, keepdim=True)
        velocity_factor = 1.0 + 0.01 * torch.tanh(
            velocity_projection * (1.0 + self.velocity_scale.index_select(0, local_indices))
        )
        weights = point_weight * freq_atten * distance_atten * velocity_factor
        weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-6)

        mono_fields = torch.einsum("pc,pfc->pf", features, self.mono_sh.index_select(0, local_indices))
        diff_fields = torch.einsum("pc,pfc->pf", features, self.diff_sh.index_select(0, local_indices))
        mono_logits = (weights * mono_fields).sum(dim=0) + self.mono_bias
        diff_logits = (weights * diff_fields).sum(dim=0) + self.diff_bias

        mono_mask = torch.sigmoid(mono_logits).reshape(-1, 1)
        diff_mask = torch.sigmoid(diff_logits).reshape(-1, 1)
        return mono_mask.expand_as(source_magnitude), diff_mask.expand_as(source_magnitude)

    def forward(self, state: dict[str, Tensor], source_audio: Tensor) -> Tensor:
        if source_audio.ndim != 2 or source_audio.shape[0] != 2:
            raise ValueError(f"source_audio must have shape (2, samples), got {tuple(source_audio.shape)}")
        if source_audio.shape[-1] < self.n_fft:
            raise ValueError(
                f"source audio length {source_audio.shape[-1]} is shorter than n_fft={self.n_fft}"
            )
        device = self.mono_sh.device
        dtype = self.mono_sh.dtype
        source_audio = source_audio.to(device=device, dtype=dtype)
        window = torch.hamming_window(self.win_length, device=device, dtype=dtype)
        spec_l = torch.stft(
            source_audio[0:1],
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )[0]
        spec_r = torch.stft(
            source_audio[1:2],
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )[0]
        mag_l = spec_l.abs()
        mag_r = spec_r.abs()
        source_magnitude = torch.nan_to_num(0.5 * (mag_l + mag_r), nan=0.0, posinf=0.0, neginf=0.0)
        mono_mask, diff_mask = self._render_masks(state, source_magnitude)
        mono_mag = torch.nan_to_num(mono_mask * source_magnitude, nan=0.0, posinf=0.0, neginf=0.0)
        left_mag, right_mag = self.combine_mono_diff_magnitudes(
            mono_mag,
            diff_mask,
            diff_lr_sign=self.diff_lr_sign,
        )
        left_spec = torch.polar(left_mag, torch.nan_to_num(torch.angle(spec_l), nan=0.0, posinf=0.0, neginf=0.0))
        right_spec = torch.polar(right_mag, torch.nan_to_num(torch.angle(spec_r), nan=0.0, posinf=0.0, neginf=0.0))
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
        return torch.nan_to_num(torch.stack([left, right], dim=0), nan=0.0, posinf=0.0, neginf=0.0)
