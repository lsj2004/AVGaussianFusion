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
    ) -> tuple[Tensor, Tensor, Tensor]:
        xyz = state["xyz"]
        opacity = state["opacity"]
        audio_opacity = state["audio_opacity"]
        mono_response = state["mono_response"]
        diff_response = state["diff_response"]
        distance_decay = state["distance_decay"]
        phase_delay = state["phase_delay"]
        if mono_response.shape[-1] != self.num_frequency_bins:
            raise ValueError(
                f"acoustic field frequency bins {mono_response.shape[-1]} do not match "
                f"renderer frequency bins {self.num_frequency_bins}"
            )
        if diff_response.shape != mono_response.shape or distance_decay.shape != mono_response.shape:
            raise ValueError("acoustic transfer attributes must share shape (N, F)")

        camera_xyz = self._world_to_camera_xyz(xyz, camera_w2c)
        distance = camera_xyz.norm(dim=-1, keepdim=True).clamp_min(1e-4)
        direction = camera_xyz / distance
        side = direction[:, 0:1]
        activity = torch.softmax((opacity + audio_opacity).reshape(-1), dim=0).reshape(-1, 1)
        attenuation = torch.exp(-torch.nn.functional.softplus(distance_decay) * torch.log1p(distance))

        mono = 1.0 + (activity * torch.tanh(mono_response) * attenuation).sum(dim=0)
        diff = (activity * torch.tanh(diff_response) * side * attenuation).sum(dim=0).clamp(-0.95, 0.95)
        left_transfer = (mono * (1.0 + diff)).clamp(self.min_transfer, self.max_transfer)
        right_transfer = (mono * (1.0 - diff)).clamp(self.min_transfer, self.max_transfer)
        phase = (activity * torch.tanh(phase_delay)).sum(dim=0)
        return left_transfer, right_transfer, phase

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
        mid_mag = mid_spec.abs()
        mid_phase = torch.nan_to_num(torch.angle(mid_spec), nan=0.0, posinf=0.0, neginf=0.0)
        left_transfer, right_transfer, phase = self._render_transfer(state, camera_w2c=camera_w2c)
        left_tf = left_transfer.reshape(-1, 1).expand_as(mid_mag)
        right_tf = right_transfer.reshape(-1, 1).expand_as(mid_mag)
        phase_tf = phase.reshape(-1, 1).expand_as(mid_mag)
        if self.use_phase_delay:
            left_phase = mid_phase + phase_tf
            right_phase = mid_phase - phase_tf
        else:
            left_phase = mid_phase
            right_phase = mid_phase
        left_spec = torch.polar(mid_mag * left_tf, left_phase)
        right_spec = torch.polar(mid_mag * right_tf, right_phase)
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
            "left_transfer": left_transfer,
            "right_transfer": right_transfer,
            "phase": phase,
            "left_transfer_tf": left_tf,
            "right_transfer_tf": right_tf,
        }
