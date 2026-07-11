from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)


class DualBranchAudioUNet(nn.Module):
    """AudioGS dual-branch TF renderer for mono/diff masks.

    This mirrors the local AudioGS renderer structure: a mono branch consumes
    source magnitude, SH response, and inverse distance; a diff branch consumes
    the directional diff response plus optional cues. The branches merge through
    a shared encoder/decoder and emit mono/diff TF masks.
    """

    def __init__(self, use_groupnorm: bool = True, diff_in_channels: int = 1):
        super().__init__()
        self.diff_in_channels = int(diff_in_channels)
        self.enc1 = self._make_layer(3, 64, use_groupnorm)
        self.enc2 = self._make_layer(64, 128, use_groupnorm)
        self.enc3 = self._make_layer(128, 256, use_groupnorm)
        self.enc4 = self._make_layer(256, 512, use_groupnorm)
        self.diff_enc1 = self._make_layer(self.diff_in_channels, 64, use_groupnorm)
        self.maxpool = nn.MaxPool2d(2)
        self.upconv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = self._make_layer(512, 256, use_groupnorm)
        self.upconv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = self._make_layer(256, 128, use_groupnorm)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = self._make_layer(128, 64, use_groupnorm)
        self.upconv1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = self._make_layer(128, 64, use_groupnorm)
        self.out_mono = nn.Conv2d(64, 1, 1)
        self.out_diff = nn.Conv2d(64, 1, 1)

    @staticmethod
    def _norm(num_channels: int, use_groupnorm: bool) -> nn.Module:
        if use_groupnorm:
            return nn.GroupNorm(num_groups=8 if num_channels >= 8 else 1, num_channels=num_channels)
        return nn.BatchNorm2d(num_channels)

    @classmethod
    def _make_layer(cls, in_channels: int, out_channels: int, use_groupnorm: bool) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            cls._norm(out_channels, use_groupnorm),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            cls._norm(out_channels, use_groupnorm),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _resize_like(x: Tensor, ref: Tensor) -> Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, mono_features: Tensor, diff_features: Tensor) -> tuple[Tensor, Tensor]:
        e1_mono = self.enc1(mono_features)
        e1_pool_mono = self.maxpool(e1_mono)
        e1_diff = self.diff_enc1(diff_features)
        e1_pool_diff = self.maxpool(e1_diff)
        e1_pool = 0.5 * (e1_pool_mono + e1_pool_diff)

        e2 = self.enc2(e1_pool)
        e2_pool = self.maxpool(e2)
        e3 = self.enc3(e2_pool)
        e3_pool = self.maxpool(e3)
        e4 = self.enc4(e3_pool)

        d4_up = self._resize_like(self.upconv4(e4), e3)
        d4 = self.dec4(torch.cat([d4_up, e3], dim=1))
        d3_up = self._resize_like(self.upconv3(d4), e2)
        d3 = self.dec3(torch.cat([d3_up, e2], dim=1))
        d2_up = self._resize_like(self.upconv2(d3), e1_mono)
        d2 = self.dec2(torch.cat([d2_up, e1_mono], dim=1))
        d1_up = self._resize_like(self.upconv1(d2), e1_mono)
        d1 = self.dec1(torch.cat([d1_up, e1_mono], dim=1))

        mono_mask = F.softplus(self.out_mono(d1)) + 0.1
        diff_mask = torch.tanh(self.out_diff(d1))
        return mono_mask, diff_mask


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

    def forward(
        self,
        state: dict[str, Tensor],
        source_audio: Tensor,
        camera_w2c: Tensor | None = None,
    ) -> Tensor:
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

    def forward(
        self,
        state: dict[str, Tensor],
        source_audio: Tensor,
        camera_w2c: Tensor | None = None,
    ) -> Tensor:
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
        sample_rate: int = 16000,
        sh_rand_init_std: float = 0.1,
        freq_atten_alpha: float = 1.0,
        diff_lr_sign: float = 1.0,
        sh_degree: int = 3,
        use_geom_phase: bool = True,
        use_ear_distance_attenuation: bool = True,
        renderer_type: str = "direct",
        use_groupnorm: bool = True,
        use_stereo_cues: bool = False,
        diff_use_inv_distance: bool = False,
        diff_use_side_mag: bool = False,
        head_radius: float = 0.0875,
        sound_speed: float = 343.0,
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
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if head_radius < 0:
            raise ValueError(f"head_radius must be nonnegative, got {head_radius}")
        if sound_speed <= 0:
            raise ValueError(f"sound_speed must be positive, got {sound_speed}")
        expected_bins = n_fft // 2 + 1
        if int(num_frequency_bins) != expected_bins:
            raise ValueError(
                f"num_frequency_bins must match n_fft // 2 + 1 ({expected_bins}), "
                f"got {num_frequency_bins}"
            )
        if int(sh_degree) not in {1, 3}:
            raise ValueError(f"sh_degree must be 1 or 3, got {sh_degree}")
        if renderer_type not in {"direct", "unet"}:
            raise ValueError(f"renderer_type must be direct or unet, got {renderer_type!r}")
        self.num_points = int(num_points)
        self.num_frequency_bins = int(num_frequency_bins)
        self.top_k = min(int(top_k), num_points) if top_k is not None else num_points
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.sample_rate = int(sample_rate)
        self.freq_atten_alpha = float(freq_atten_alpha)
        self.diff_lr_sign = float(diff_lr_sign)
        self.sh_degree = int(sh_degree)
        self.use_geom_phase = bool(use_geom_phase)
        self.use_ear_distance_attenuation = bool(use_ear_distance_attenuation)
        self.renderer_type = str(renderer_type)
        self.use_stereo_cues = bool(use_stereo_cues)
        self.diff_use_inv_distance = bool(diff_use_inv_distance)
        self.diff_use_side_mag = bool(diff_use_side_mag)
        self.head_radius = float(head_radius)
        self.sound_speed = float(sound_speed)
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
        rotation_init = torch.zeros(self.top_k, 4)
        rotation_init[:, 0] = 1.0
        self.rotation = nn.Parameter(rotation_init)
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
        self.phase_residual = nn.Parameter(torch.zeros(self.top_k, num_frequency_bins))
        diff_in_channels = 1
        if self.use_stereo_cues:
            diff_in_channels += 1
        if self.diff_use_inv_distance:
            diff_in_channels += 1
        if self.diff_use_side_mag:
            diff_in_channels += 1
        self.renderer = (
            DualBranchAudioUNet(
                use_groupnorm=bool(use_groupnorm),
                diff_in_channels=diff_in_channels,
            )
            if self.renderer_type == "unet"
            else None
        )
        frequencies = torch.linspace(
            0.0,
            0.5 * float(self.sample_rate),
            self.num_frequency_bins,
        )
        self.register_buffer("frequencies_hz", frequencies, persistent=False)

    @property
    def active_count(self) -> int:
        return self.top_k

    @staticmethod
    def _axis_angle_to_quaternion(axis_angle: Tensor) -> Tensor:
        angle = axis_angle.norm(dim=-1, keepdim=True)
        half_angle = 0.5 * angle
        small = angle < 1e-8
        scale = torch.where(
            small,
            0.5 - angle.square() / 48.0,
            torch.sin(half_angle) / angle.clamp_min(1e-8),
        )
        return torch.cat([torch.cos(half_angle), axis_angle * scale], dim=-1)

    @staticmethod
    def _normalize_quaternion(quaternion: Tensor) -> Tensor:
        return F.normalize(quaternion, dim=-1, eps=1e-8)

    @classmethod
    def _rotate_by_quaternion(cls, vector: Tensor, quaternion: Tensor) -> Tensor:
        q = cls._normalize_quaternion(quaternion)
        q_w = q[..., :1]
        q_xyz = q[..., 1:]
        uv = torch.cross(q_xyz, vector, dim=-1)
        uuv = torch.cross(q_xyz, uv, dim=-1)
        return vector + 2.0 * (q_w * uv + uuv)

    @staticmethod
    def _world_to_camera_xyz(xyz: Tensor, camera_w2c: Tensor | None) -> Tensor:
        if camera_w2c is None:
            return xyz
        w2c = torch.as_tensor(camera_w2c, device=xyz.device, dtype=xyz.dtype)
        if w2c.ndim == 3:
            if w2c.shape[0] != 1:
                raise ValueError(
                    "AudioGSMaskedSpectralHead expects a single camera pose per audio render, "
                    f"got batch={w2c.shape[0]}"
                )
            w2c = w2c[0]
        if w2c.shape != (4, 4):
            raise ValueError(f"camera_w2c must have shape (4, 4) or (1, 4, 4), got {tuple(w2c.shape)}")
        rotation = w2c[:3, :3]
        translation = w2c[:3, 3]
        return xyz @ rotation.T + translation.reshape(1, 3)

    def _sh_features(self, direction: Tensor) -> Tensor:
        x, y, z = direction.unbind(dim=-1)
        features = [
            torch.full_like(x, SH_C0),
            -SH_C1 * y,
            SH_C1 * z,
            -SH_C1 * x,
        ]
        if self.sh_degree == 1:
            return torch.stack(features, dim=-1)
        features.extend(
            [
                SH_C2[0] * x * y,
                SH_C2[1] * y * z,
                SH_C2[2] * (2.0 * z.square() - x.square() - y.square()),
                SH_C2[3] * x * z,
                SH_C2[4] * (x.square() - y.square()),
                SH_C3[0] * y * (3.0 * x.square() - y.square()),
                SH_C3[1] * x * y * z,
                SH_C3[2] * y * (4.0 * z.square() - x.square() - y.square()),
                SH_C3[3] * z * (2.0 * z.square() - 3.0 * x.square() - 3.0 * y.square()),
                SH_C3[4] * x * (4.0 * z.square() - x.square() - y.square()),
                SH_C3[5] * z * (x.square() - y.square()),
                SH_C3[6] * x * (x.square() - 3.0 * y.square()),
            ]
        )
        return torch.stack(features, dim=-1)

    def _direction_features(
        self,
        xyz: Tensor,
        rotation: Tensor,
        camera_w2c: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        camera_xyz = self._world_to_camera_xyz(xyz, camera_w2c)
        camera_direction = F.normalize(camera_xyz, dim=-1, eps=1e-6)
        local_direction = F.normalize(
            self._rotate_by_quaternion(camera_direction, rotation),
            dim=-1,
            eps=1e-6,
        )
        features = self._sh_features(local_direction)
        distance = camera_xyz.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        return features, distance

    def _geometry_phase_delta(self, camera_direction: Tensor, weights: Tensor) -> Tensor:
        if not self.use_geom_phase:
            return torch.zeros(
                self.num_frequency_bins,
                1,
                device=weights.device,
                dtype=weights.dtype,
            )
        if camera_direction.ndim != 2 or camera_direction.shape[-1] != 3:
            raise ValueError(f"camera_direction must have shape (points, 3), got {tuple(camera_direction.shape)}")
        if weights.ndim != 2 or weights.shape[-1] != self.num_frequency_bins:
            raise ValueError(
                f"weights must have shape (points, {self.num_frequency_bins}), got {tuple(weights.shape)}"
            )
        itd_seconds = float(self.head_radius) * camera_direction[:, 0:1] / float(self.sound_speed)
        phase_per_point = 2.0 * torch.pi * itd_seconds * self.frequencies_hz.to(weights).reshape(1, -1)
        phase = (weights * phase_per_point).sum(dim=0)
        return phase.reshape(-1, 1)

    def _ear_distance_gains(self, camera_xyz: Tensor, weights: Tensor) -> tuple[Tensor, Tensor]:
        ones = torch.ones(
            self.num_frequency_bins,
            1,
            device=weights.device,
            dtype=weights.dtype,
        )
        if not self.use_ear_distance_attenuation or self.head_radius <= 0:
            return ones, ones
        if camera_xyz.ndim != 2 or camera_xyz.shape[-1] != 3:
            raise ValueError(f"camera_xyz must have shape (points, 3), got {tuple(camera_xyz.shape)}")
        if weights.ndim != 2 or weights.shape[-1] != self.num_frequency_bins:
            raise ValueError(
                f"weights must have shape (points, {self.num_frequency_bins}), got {tuple(weights.shape)}"
            )
        left_ear = camera_xyz.new_tensor([self.head_radius, 0.0, 0.0])
        right_ear = camera_xyz.new_tensor([-self.head_radius, 0.0, 0.0])
        center_distance = camera_xyz.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        left_distance = (camera_xyz - left_ear).norm(dim=-1, keepdim=True).clamp_min(1e-4)
        right_distance = (camera_xyz - right_ear).norm(dim=-1, keepdim=True).clamp_min(1e-4)
        left_per_point = (center_distance / left_distance).pow(self.freq_atten_alpha)
        right_per_point = (center_distance / right_distance).pow(self.freq_atten_alpha)
        left_gain = (weights * left_per_point).sum(dim=0).reshape(-1, 1)
        right_gain = (weights * right_per_point).sum(dim=0).reshape(-1, 1)
        mean_gain = (0.5 * (left_gain + right_gain)).clamp_min(1e-6)
        return left_gain / mean_gain, right_gain / mean_gain

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        rotation_key = prefix + "rotation"
        if rotation_key in state_dict:
            rotation = state_dict[rotation_key]
            if isinstance(rotation, torch.Tensor) and rotation.ndim == 2 and rotation.shape[-1] == 3:
                state_dict = dict(state_dict)
                state_dict[rotation_key] = self._axis_angle_to_quaternion(torch.tanh(rotation))
        phase_key = prefix + "phase_residual"
        if phase_key not in state_dict:
            state_dict = dict(state_dict)
            state_dict[phase_key] = self.phase_residual.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def combine_mono_diff_magnitudes(
        mono_mag: Tensor,
        diff_envelope: Tensor,
        diff_lr_sign: float = 1.0,
    ) -> tuple[Tensor, Tensor]:
        sign = -1.0 if float(diff_lr_sign) < 0.0 else 1.0
        left = torch.relu(mono_mag * (1.0 + sign * diff_envelope))
        right = torch.relu(mono_mag * (1.0 - sign * diff_envelope))
        return left, right

    def _render_masks(
        self,
        state: dict[str, Tensor],
        source_magnitude: Tensor,
        ild_spec: Tensor | None = None,
        side_spec: Tensor | None = None,
        camera_w2c: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
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
        camera_xyz = self._world_to_camera_xyz(selected_xyz, camera_w2c)
        camera_direction = F.normalize(camera_xyz, dim=-1, eps=1e-6)
        features, distance = self._direction_features(
            selected_xyz,
            selected_rotation,
            camera_w2c=camera_w2c,
        )
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
        if self.renderer_type == "unet":
            mono_mask, diff_mask = self._render_unet_masks(
                mono_fields=mono_fields,
                diff_fields=diff_fields,
                weights=weights,
                distance=distance,
                source_magnitude=source_magnitude,
                ild_spec=ild_spec,
                side_spec=side_spec,
            )
        else:
            mono_logits = (weights * mono_fields).sum(dim=0) + self.mono_bias
            diff_logits = (weights * diff_fields).sum(dim=0) + self.diff_bias
            mono_mask = torch.sigmoid(mono_logits).reshape(-1, 1).expand_as(source_magnitude)
            diff_mask = torch.sigmoid(diff_logits).reshape(-1, 1).expand_as(source_magnitude)
        residual_phase = (
            weights * self.phase_residual.index_select(0, local_indices)
        ).sum(dim=0).reshape(-1, 1)
        phase_delta = self._geometry_phase_delta(camera_direction, weights) + residual_phase
        left_gain, right_gain = self._ear_distance_gains(camera_xyz, weights)
        return (
            mono_mask,
            diff_mask,
            phase_delta.expand_as(source_magnitude),
            left_gain.expand_as(source_magnitude),
            right_gain.expand_as(source_magnitude),
        )

    def _render_unet_masks(
        self,
        mono_fields: Tensor,
        diff_fields: Tensor,
        weights: Tensor,
        distance: Tensor,
        source_magnitude: Tensor,
        ild_spec: Tensor | None = None,
        side_spec: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.renderer is None:
            raise RuntimeError("renderer_type='unet' requires a DualBranchAudioUNet renderer")
        if source_magnitude.ndim != 2:
            raise ValueError(f"source_magnitude must have shape (freq, time), got {tuple(source_magnitude.shape)}")
        freq_bins, time_bins = source_magnitude.shape
        if freq_bins != self.num_frequency_bins:
            raise ValueError(
                f"source_magnitude freq bins must be {self.num_frequency_bins}, got {freq_bins}"
            )
        mono_tf = (weights * mono_fields).sum(dim=0).reshape(freq_bins, 1).expand(freq_bins, time_bins)
        diff_tf = (weights * diff_fields).sum(dim=0).reshape(freq_bins, 1).expand(freq_bins, time_bins)
        inv_distance = (weights * distance.reciprocal()).sum(dim=0).reshape(freq_bins, 1).expand(freq_bins, time_bins)
        mono_features = torch.stack(
            [source_magnitude, mono_tf, inv_distance],
            dim=0,
        ).unsqueeze(0)
        diff_channels = [diff_tf]
        if self.use_stereo_cues:
            diff_channels.append(torch.zeros_like(diff_tf) if ild_spec is None else ild_spec)
        if self.diff_use_inv_distance:
            diff_channels.append(inv_distance)
        if self.diff_use_side_mag:
            diff_channels.append(torch.zeros_like(diff_tf) if side_spec is None else side_spec)
        diff_features = torch.stack(diff_channels, dim=0).unsqueeze(0)
        mono_features = torch.nan_to_num(mono_features, nan=0.0, posinf=0.0, neginf=0.0)
        diff_features = torch.nan_to_num(diff_features, nan=0.0, posinf=0.0, neginf=0.0)
        mono_mask, diff_mask = self.renderer(mono_features, diff_features)
        return mono_mask[0, 0], diff_mask[0, 0]

    def forward(
        self,
        state: dict[str, Tensor],
        source_audio: Tensor,
        camera_w2c: Tensor | None = None,
    ) -> Tensor:
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
        eps = source_magnitude.new_tensor(1e-8)
        ild_spec = torch.log(mag_l.clamp_min(eps)) - torch.log(mag_r.clamp_min(eps))
        ild_spec = torch.nan_to_num(ild_spec, nan=0.0, posinf=0.0, neginf=0.0)
        side_spec = torch.nan_to_num((mag_l - mag_r).abs(), nan=0.0, posinf=0.0, neginf=0.0)
        mono_mask, diff_mask, phase_delta, left_gain, right_gain = self._render_masks(
            state,
            source_magnitude,
            ild_spec=ild_spec,
            side_spec=side_spec,
            camera_w2c=camera_w2c,
        )
        mono_mag = torch.nan_to_num(mono_mask * source_magnitude, nan=0.0, posinf=0.0, neginf=0.0)
        left_mag, right_mag = self.combine_mono_diff_magnitudes(
            mono_mag,
            diff_mask,
            diff_lr_sign=self.diff_lr_sign,
        )
        mid_phase = torch.nan_to_num(
            torch.angle(0.5 * (spec_l + spec_r)),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        left_spec = torch.polar(left_mag * left_gain, mid_phase + phase_delta)
        right_spec = torch.polar(right_mag * right_gain, mid_phase - phase_delta)
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
