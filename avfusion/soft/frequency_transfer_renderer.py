from __future__ import annotations

import torch
from torch import Tensor, nn


class FrequencyTransferRenderer(nn.Module):
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        min_transfer: float = 0.05,
        max_transfer: float = 4.0,
        use_phase_delay: bool = False,
        use_geometry_diff_head: bool = False,
        geometry_diff_scale: float = 0.25,
    ):
        super().__init__()
        if n_fft <= 0 or hop_length <= 0 or win_length <= 0:
            raise ValueError("STFT sizes must be positive")
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length)
        self.min_transfer = float(min_transfer)
        self.max_transfer = float(max_transfer)
        self.use_phase_delay = bool(use_phase_delay)
        self.use_geometry_diff_head = bool(use_geometry_diff_head)
        self.geometry_diff_scale = float(geometry_diff_scale)
        if self.use_geometry_diff_head:
            self.geometry_diff_head = nn.Linear(6, 1)
            nn.init.zeros_(self.geometry_diff_head.weight)
            nn.init.zeros_(self.geometry_diff_head.bias)

    @property
    def num_frequency_bins(self) -> int:
        return self.n_fft // 2 + 1

    @staticmethod
    def _world_to_camera_xyz(xyz: Tensor, camera_w2c: Tensor | None) -> Tensor:
        if camera_w2c is None:
            return xyz
        w2c = torch.as_tensor(camera_w2c, device=xyz.device, dtype=xyz.dtype)
        if w2c.ndim == 3:
            w2c = w2c[0]
        if w2c.shape != (4, 4):
            raise ValueError(f"camera_w2c must have shape (4, 4) or (1, 4, 4), got {tuple(w2c.shape)}")
        rot = w2c[:3, :3]
        trans = w2c[:3, 3]
        return xyz @ rot.T + trans

    def _render_transfer(
        self,
        state: dict[str, Tensor],
        camera_w2c: Tensor | None = None,
        source_ild: Tensor | None = None,
        source_side_mag: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        xyz = state["xyz"]
        opacity = state["opacity"]
        audio_opacity = state["audio_opacity"]
        mono_response = state["mono_response"]
        diff_response = state["diff_response"]
        diff_directional_response = state.get(
            "diff_directional_response",
            torch.zeros(
                diff_response.shape[0],
                3,
                diff_response.shape[-1],
                device=diff_response.device,
                dtype=diff_response.dtype,
            ),
        )
        side_response = state.get("side_response", torch.zeros_like(mono_response))
        distance_decay = state["distance_decay"]
        phase_delay = state["phase_delay"]
        if mono_response.shape[-1] != self.num_frequency_bins:
            raise ValueError(
                f"acoustic field frequency bins {mono_response.shape[-1]} do not match "
                f"renderer frequency bins {self.num_frequency_bins}"
            )
        if (
            diff_response.shape != mono_response.shape
            or diff_directional_response.shape != (mono_response.shape[0], 3, mono_response.shape[-1])
            or side_response.shape != mono_response.shape
            or distance_decay.shape != mono_response.shape
        ):
            raise ValueError("acoustic transfer attributes must share shape (N, F)")

        camera_xyz = self._world_to_camera_xyz(xyz, camera_w2c)
        distance = camera_xyz.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        direction = camera_xyz / distance
        side = direction[:, 0:1]
        front = direction[:, 2:3]
        inv_distance = 1.0 / distance
        activity = torch.softmax((opacity + audio_opacity).reshape(-1), dim=0).reshape(-1, 1)
        attenuation = torch.exp(-torch.nn.functional.softplus(distance_decay) * torch.log1p(distance))

        mono = 1.0 + (activity * torch.tanh(mono_response) * attenuation).sum(dim=0)
        directional_diff = (torch.tanh(diff_directional_response) * direction.unsqueeze(-1)).sum(dim=1)
        diff = (activity * torch.tanh(diff_response + directional_diff) * side * attenuation).sum(dim=0).clamp(-0.95, 0.95)
        side_keep = 1.0 + (activity * torch.tanh(side_response) * attenuation).sum(dim=0)
        mid_transfer = mono.clamp(self.min_transfer, self.max_transfer)
        side_transfer = (mono * side_keep).clamp(self.min_transfer, self.max_transfer)
        geometry_diff_delta = None
        if self.use_geometry_diff_head:
            if source_ild is None:
                source_ild = mono_response.new_zeros(self.num_frequency_bins)
            if source_side_mag is None:
                source_side_mag = mono_response.new_zeros(self.num_frequency_bins)
            freq = torch.linspace(-1.0, 1.0, self.num_frequency_bins, device=mono_response.device, dtype=mono_response.dtype)
            side_feature = (activity * side).sum(dim=0).expand_as(freq)
            front_feature = (activity * front).sum(dim=0).expand_as(freq)
            inv_distance_feature = (activity * inv_distance).sum(dim=0).expand_as(freq)
            features = torch.stack(
                [
                    side_feature,
                    front_feature,
                    inv_distance_feature,
                    source_ild.to(device=mono_response.device, dtype=mono_response.dtype),
                    source_side_mag.to(device=mono_response.device, dtype=mono_response.dtype),
                    freq,
                ],
                dim=-1,
            )
            geometry_diff_delta = self.geometry_diff_scale * torch.tanh(self.geometry_diff_head(features).squeeze(-1))
            diff = (diff + geometry_diff_delta).clamp(-0.95, 0.95)
        diff_transfer = (mono * diff).clamp(-self.max_transfer, self.max_transfer)
        left_transfer = (mid_transfer + diff_transfer).clamp(self.min_transfer, self.max_transfer)
        right_transfer = (mid_transfer - diff_transfer).clamp(self.min_transfer, self.max_transfer)
        phase = (activity * torch.tanh(phase_delay)).sum(dim=0)
        return mid_transfer, side_transfer, diff_transfer, left_transfer, right_transfer, phase, geometry_diff_delta

    def forward(
        self,
        state: dict[str, Tensor],
        source_audio: Tensor,
        camera_w2c: Tensor | None = None,
        return_debug: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if source_audio.ndim != 2 or source_audio.shape[0] != 2:
            raise ValueError(f"source_audio must have shape (2, samples), got {tuple(source_audio.shape)}")
        if source_audio.shape[-1] < self.n_fft:
            raise ValueError(f"source audio length {source_audio.shape[-1]} is shorter than n_fft={self.n_fft}")
        device = state["xyz"].device
        dtype = state["xyz"].dtype
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
        mid_spec = 0.5 * (spec_l + spec_r)
        side_spec = 0.5 * (spec_l - spec_r)
        source_ild = None
        source_side_mag = None
        if self.use_geometry_diff_head:
            eps = source_audio.new_tensor(1e-7)
            source_ild = (torch.log(spec_l.abs().clamp_min(eps)) - torch.log(spec_r.abs().clamp_min(eps))).mean(dim=-1)
            source_side_mag = side_spec.abs().mean(dim=-1)
        mid_transfer, side_transfer, diff_transfer, left_transfer, right_transfer, phase, geometry_diff_delta = self._render_transfer(
            state,
            camera_w2c=camera_w2c,
            source_ild=source_ild,
            source_side_mag=source_side_mag,
        )
        mid_tf = mid_transfer.reshape(-1, 1).expand_as(mid_spec)
        side_tf = side_transfer.reshape(-1, 1).expand_as(side_spec)
        diff_tf = diff_transfer.reshape(-1, 1).expand_as(mid_spec)
        left_tf = left_transfer.reshape(-1, 1).expand_as(mid_spec)
        right_tf = right_transfer.reshape(-1, 1).expand_as(mid_spec)
        phase_tf = phase.reshape(-1, 1).expand_as(mid_spec)
        if self.use_phase_delay:
            mid_spec = mid_spec * torch.exp(1j * phase_tf)
            side_spec = side_spec * torch.exp(-1j * phase_tf)
        mid_out = mid_spec * mid_tf
        side_out = side_spec * side_tf + mid_spec * diff_tf
        left_spec = mid_out + side_out
        right_spec = mid_out - side_out
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
        pred = torch.nan_to_num(torch.stack([left, right], dim=0), nan=0.0, posinf=0.0, neginf=0.0)
        if not return_debug:
            return pred
        return pred, {
            "mid_transfer": mid_transfer,
            "side_transfer": side_transfer,
            "diff_transfer": diff_transfer,
            "geometry_diff_delta": geometry_diff_delta
            if geometry_diff_delta is not None
            else torch.zeros_like(diff_transfer),
            "left_transfer": left_transfer,
            "right_transfer": right_transfer,
            "phase": phase,
            "mid_transfer_tf": mid_tf,
            "side_transfer_tf": side_tf,
            "diff_transfer_tf": diff_tf,
            "left_transfer_tf": left_tf,
            "right_transfer_tf": right_tf,
        }
