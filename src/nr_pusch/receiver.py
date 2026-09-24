"""Four-user PUSCH receiver with Sionna LMMSE and NR transport decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sionna.phy.mapping import Demapper
from sionna.phy.channel import time_lag_discrete_time_channel
from sionna.phy.mimo import StreamManagement
from sionna.phy.nr import PUSCHReceiver, PUSCHTransmitter, TBDecoder
from sionna.phy.ofdm import LMMSEEqualizer, LinearDetector

from .config import TxSettings
from .transmitter import NrPuschTx


class DftSOfdmMimoDetector(torch.nn.Module):
    """LMMSE-detect the spread symbols, then undo per-symbol DFT spreading."""

    def __init__(self, transmitter: PUSCHTransmitter, stream_management: StreamManagement):
        super().__init__()
        self._resource_grid = transmitter.resource_grid
        self._equalizer = LMMSEEqualizer(self._resource_grid, stream_management, device=transmitter.device)
        self._demapper = Demapper(
            "maxlog",
            "qam",
            transmitter._num_bits_per_symbol,
            device=transmitter.device,
        )

        # Reuse Sionna's resource-grid data indexing so demapper ordering stays
        # consistent with its PUSCH layer mapper and TB decoder.
        extractor = LinearDetector(
            "lmmse", "symbol", "maxlog", self._resource_grid, stream_management,
            "qam", transmitter._num_bits_per_symbol, device=transmitter.device,
        )
        indices = extractor._data_ind
        if not torch.all(indices == indices[0:1, 0:1]).item():
            raise ValueError("DFT-s-OFDM currently requires identical data RE positions for all users")
        ordered_indices = indices[0, 0]
        fft_size = self._resource_grid.num_effective_subcarriers
        symbol_indices = torch.div(ordered_indices, fft_size, rounding_mode="floor")
        counts = torch.bincount(symbol_indices, minlength=self._resource_grid.num_ofdm_symbols)
        active_counts = counts[counts > 0]
        if active_counts.numel() == 0 or not torch.all(active_counts == fft_size).item():
            raise ValueError(
                "DFT-s-OFDM receiver requires each data-bearing OFDM symbol to contain "
                "one complete effective-subcarrier allocation"
            )
        if not torch.all(symbol_indices[1:] >= symbol_indices[:-1]).item():
            raise ValueError("Sionna data RE indices are not ordered by OFDM symbol")
        self._num_spread_symbols = int(active_counts.numel())
        self._fft_size = int(fft_size)
        self._num_data_symbols = int(ordered_indices.numel())
        self._num_bits_per_symbol = transmitter._num_bits_per_symbol

    def forward(
        self,
        y: torch.Tensor,
        h_hat: torch.Tensor,
        err_var: torch.Tensor,
        no: torch.Tensor,
    ) -> torch.Tensor:
        x_hat, no_eff = self._equalizer(y, h_hat, err_var, no)
        batch, num_tx, num_streams, num_data = x_hat.shape
        expected = self._num_spread_symbols * self._fft_size
        if num_data != expected:
            raise ValueError(
                f"PUSCH 数据符号数 {num_data} 与 DFT-s-OFDM 资源映射预期 {expected} 不符"
            )

        x_hat = x_hat.reshape(batch, num_tx, num_streams, self._num_spread_symbols, self._fft_size)
        no_eff = no_eff.reshape_as(x_hat.real)
        # Tx applies a unitary forward DFT per data-bearing OFDM symbol.
        x_hat = torch.fft.ifft(x_hat, dim=-1, norm="ortho")
        # The unitary inverse DFT produces correlated noise when subcarrier
        # variances differ. Average variance per spread symbol for LLR scaling.
        no_eff = no_eff.mean(dim=-1, keepdim=True).expand_as(x_hat.real)
        x_hat = x_hat.reshape(batch, num_tx, num_streams, self._num_data_symbols)
        no_eff = no_eff.reshape_as(x_hat.real)
        return self._demapper(x_hat, no_eff)


class DftSOfdmDmrsEstimator(torch.nn.Module):
    """Estimate static slot taps from the configured DFT-s-OFDM OCC pilots."""

    def __init__(self, pilot_grid: torch.Tensor, resource_grid, l_min: int, l_max: int):
        super().__init__()
        # [user, num_subcarriers] template from the actual DFT-s-OFDM mapper.
        if pilot_grid.ndim != 2 or pilot_grid.shape[0] != 4:
            raise ValueError("DMRS 模板必须为 [4 users, num_subcarriers]")
        self.register_buffer("_pilot_grid", pilot_grid)
        self._num_subcarriers = pilot_grid.shape[-1]
        self._num_ofdm_symbols = resource_grid.num_ofdm_symbols
        self._l_min = l_min
        self._l_max = l_max
        self._num_taps = l_max - l_min + 1
        mask = resource_grid.pilot_pattern.mask[0, 0]
        dmrs_symbols = torch.where(mask.sum(dim=-1) > 0)[0]
        if dmrs_symbols.numel() != 1:
            raise ValueError("初始 DFT-s-OFDM DMRS 估计要求恰好一个 DMRS OFDM 符号")
        self._dmrs_symbol = int(dmrs_symbols.item())

        support_groups: dict[tuple[int, ...], list[int]] = {}
        for user in range(4):
            support = tuple(torch.where(pilot_grid[user].abs() > 0)[0].tolist())
            if len(support) < 4 or len(support) % 2:
                raise ValueError("每个 DFT-s-OFDM DMRS comb 至少需要 4 个且为偶数个 pilot RE")
            support_groups.setdefault(support, []).append(user)
        if len(support_groups) != 2 or any(len(users) != 2 for users in support_groups.values()):
            raise ValueError("当前 DMRS 估计要求四个用户组成两组共享 comb 的正交端口")

        self._pairs: list[tuple[int, int, str, str, str]] = []
        q = torch.arange(self._num_subcarriers, dtype=torch.float32, device=pilot_grid.device)
        q = q - self._num_subcarriers // 2
        lags = torch.arange(l_min, l_max + 1, dtype=torch.float32, device=pilot_grid.device)
        frequency_basis = torch.exp((-2j * torch.pi / self._num_subcarriers) * q[:, None] * lags[None, :])
        self.register_buffer("_frequency_basis", frequency_basis)
        for support_tuple, users in support_groups.items():
            support = torch.tensor(support_tuple, dtype=torch.long, device=pilot_grid.device)
            base, partner = users
            ratio = pilot_grid[partner, support] / pilot_grid[base, support]
            ratio_pairs = ratio.reshape(-1, 2)
            if not torch.allclose(ratio_pairs[:, 0], torch.ones_like(ratio_pairs[:, 0])):
                raise ValueError("DMRS OCC pair must start with a +1 cover chip")
            if not torch.allclose(ratio_pairs[:, 1], -torch.ones_like(ratio_pairs[:, 1])):
                raise ValueError("DMRS OCC pair must alternate +1/-1 for despreading")
            design = torch.cat(
                (
                    pilot_grid[base, support, None] * frequency_basis[support],
                    pilot_grid[partner, support, None] * frequency_basis[support],
                ),
                dim=-1,
            )
            covariance = torch.linalg.pinv(design.mH @ design)
            pair_id = len(self._pairs)
            design_name = f"_design_{pair_id}"
            support_name = f"_support_{pair_id}"
            covariance_name = f"_covariance_{pair_id}"
            self.register_buffer(design_name, design)
            self.register_buffer(support_name, support)
            self.register_buffer(covariance_name, covariance)
            self._pairs.append((base, partner, design_name, support_name, covariance_name))

    def forward(self, y: torch.Tensor, no: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, num_rx, num_rx_ant, _, _ = y.shape
        h_hat = torch.zeros(
            (batch, num_rx, num_rx_ant, 4, 1, self._num_ofdm_symbols, self._num_subcarriers),
            dtype=y.dtype,
            device=y.device,
        )
        no = torch.as_tensor(no, dtype=y.real.dtype, device=y.device)
        if no.ndim == 2 and tuple(no.shape) == (batch, num_rx_ant):
            no = no.unsqueeze(1)
        no = torch.broadcast_to(no, (batch, num_rx, num_rx_ant))
        err_var = torch.zeros_like(h_hat.real)
        y_pilot = y[:, 0, :, self._dmrs_symbol, :]
        for base, partner, design_name, support_name, covariance_name in self._pairs:
            design = getattr(self, design_name).to(y.device)
            support = getattr(self, support_name).to(y.device)
            covariance = getattr(self, covariance_name).to(y.device)
            observations = y_pilot.index_select(-1, support)
            rhs = observations.permute(2, 0, 1).reshape(support.numel(), -1)
            taps = torch.linalg.lstsq(design, rhs).solution
            taps = taps.reshape(2 * self._num_taps, batch, num_rx_ant).permute(1, 2, 0)
            for user, tap_slice in ((base, slice(0, self._num_taps)), (partner, slice(self._num_taps, 2 * self._num_taps))):
                user_taps = taps[..., tap_slice]
                h_freq = user_taps @ self._frequency_basis.to(y.device).T
                h_hat[:, 0, :, user, 0, :, :] = h_freq.unsqueeze(-2).expand(
                    -1, -1, self._num_ofdm_symbols, -1
                )
                cov_user = covariance[tap_slice, tap_slice]
                basis = self._frequency_basis.to(y.device)
                frequency_error = torch.einsum(
                    "nl,lm,nm->n", basis, cov_user, basis.conj()
                ).real.clamp_min(0.0)
                err_var[:, 0, :, user, 0, :, :] = (
                    no[:, 0, :, None] * frequency_error[None, None, :]
                ).unsqueeze(-2).expand(-1, -1, self._num_ofdm_symbols, -1)
        return h_hat, err_var


@dataclass
class RxResult:
    bits: torch.Tensor  # [batch, user, transport_block_bit]
    crc_status: torch.Tensor  # [batch, user], True means CRC pass
    metadata: dict[str, Any]


class NrPuschRx:
    """Decode time-domain captures or simulated samples for four PUSCH users.

    ``channel_estimator="dmrs"`` estimates the initial static-slot profile
    from its orthogonal transform-precoded DMRS. ``"perfect"`` consumes the
    simulated CDL taps as a reference/upper-bound mode.
    """

    def __init__(
        self,
        settings: TxSettings,
        *,
        channel_estimator: str = "dmrs",
        l_min: int = -6,
        max_delay_spread_s: float = 3e-6,
        num_decoder_iterations: int = 20,
        device: str | None = None,
    ) -> None:
        settings.validate()
        if settings.pusch.waveform != "dft_s_ofdm":
            raise NotImplementedError("当前接收机仅支持仓库配置的 DFT-s-OFDM PUSCH")
        if channel_estimator not in {"perfect", "dmrs"}:
            raise ValueError("channel_estimator 仅支持 perfect 或 dmrs")
        if num_decoder_iterations < 1:
            raise ValueError("num_decoder_iterations 必须大于 0")
        if max_delay_spread_s <= 0:
            raise ValueError("max_delay_spread_s 必须大于 0")

        self.settings = settings
        self.device = device
        self.l_min = l_min
        tx = PUSCHTransmitter(
            settings.to_sionna_configs(),
            return_bits=False,
            output_domain="freq",
            device=device,
        )
        self.sample_rate_hz = int(tx.resource_grid.fft_size * tx.resource_grid.subcarrier_spacing)
        _, l_max = time_lag_discrete_time_channel(self.sample_rate_hz, max_delay_spread_s)
        stream_management = StreamManagement(np.ones((1, len(settings.users)), dtype=bool), 1)
        detector = DftSOfdmMimoDetector(tx, stream_management)
        if channel_estimator == "dmrs":
            template_tx = NrPuschTx(settings, device=device)
            zero_bits = torch.zeros(
                (1, len(settings.users), int(template_tx.transport_block_size)),
                dtype=torch.float32,
                device=template_tx.device,
            )
            pilot_grid = template_tx.generate(batch_size=1, bits=zero_bits).frequency_grid[0, :, 0]
            mask = template_tx._tx_freq.pilot_pattern.mask[0, 0]
            dmrs_symbol = int(torch.where(mask.sum(dim=-1) > 0)[0].item())
            pilot_grid = pilot_grid[:, dmrs_symbol, :]
            estimator = DftSOfdmDmrsEstimator(
                pilot_grid,
                tx.resource_grid,
                l_min=l_min,
                l_max=l_max,
            )
        else:
            estimator = "perfect"
        tb_decoder = TBDecoder(
            tx._tb_encoder,
            num_bp_iter=num_decoder_iterations,
            device=device,
        )
        self._receiver = PUSCHReceiver(
            tx,
            channel_estimator=estimator,
            mimo_detector=detector,
            tb_decoder=tb_decoder,
            return_tb_crc_status=True,
            stream_management=stream_management,
            input_domain="time",
            l_min=l_min,
            device=device,
        )
        self.channel_estimator = channel_estimator

    def receive(
        self,
        iq: torch.Tensor,
        noise_variance: torch.Tensor | float,
        *,
        channel_taps: torch.Tensor | None = None,
    ) -> RxResult:
        """Decode `[batch, rx_antenna, sample]` received complex IQ samples."""
        if iq.ndim != 3 or iq.shape[1] != 4 or not iq.is_complex():
            raise ValueError("iq 形状必须为复数 [batch, 4 rx_antennas, samples]")
        if self.channel_estimator == "perfect":
            if channel_taps is None:
                raise ValueError("perfect CSI 模式要求提供 CDL channel_taps")
            if channel_taps.ndim != 5 or channel_taps.shape[:3] != (iq.shape[0], 4, 4):
                raise ValueError("channel_taps 形状必须为 [batch, 4 users, 4 rx_antennas, time, taps]")
            # Sionna PUSCHReceiver perfect time-domain CSI axis order:
            # [batch, num_rx=1, rx_ant, num_tx=user, tx_ant=1, time, tap].
            h = channel_taps.permute(0, 2, 1, 3, 4).unsqueeze(1).unsqueeze(4)
        else:
            h = None
        y = iq.unsqueeze(1)
        no = torch.as_tensor(noise_variance, dtype=iq.real.dtype, device=iq.device)
        if self.device is not None:
            y = y.to(self.device)
            no = no.to(self.device)
            if h is not None:
                h = h.to(self.device)
        bits, crc_status = self._receiver(y, no, h)
        metadata = {
            "waveform": self.settings.pusch.waveform,
            "detector": "Sionna LMMSE + inverse DFT spreading + max-log QAM demapper",
            "decoder": "Sionna NR TBDecoder",
            "channel_estimator": "perfect CDL CSI" if self.channel_estimator == "perfect" else "DFT-s-OFDM DMRS OCC LS",
            "crc_status_axes": ["batch", "user"],
            "bits_axes": ["batch", "user", "transport_block_bit"],
            "sample_rate_hz": self.sample_rate_hz,
        }
        return RxResult(bits=bits, crc_status=crc_status, metadata=metadata)
