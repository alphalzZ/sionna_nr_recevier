"""Sionna 38.901 CDL channel for configurable multi-antenna PUSCH UEs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from sionna.phy.channel import (
    ApplyOFDMChannel,
    ApplyTimeChannel,
    cir_to_time_channel,
    time_lag_discrete_time_channel,
)
from sionna.phy.channel.tr38901 import AntennaArray, CDL

from .channel_config import ChannelSettings
from .device import use_device


@dataclass
class ChannelResult:
    """Channelized IQ and intermediate values with documented tensor axes."""

    iq: torch.Tensor  # [batch, rx_antenna, sample]
    per_user_iq: torch.Tensor  # [batch, user, rx_antenna, sample]
    channel_taps: torch.Tensor  # [batch, user, rx_antenna, tx_antenna, sample, tap]
    sample_rate_hz: int
    metadata: dict[str, Any]


@dataclass
class FrequencyChannelResult:
    """Frequency-domain channel output and CSI with explicit tensor axes."""

    grid: torch.Tensor  # [batch, num_rx=1, rx_antenna, ofdm_symbol, fft_bin]
    per_user_grid: torch.Tensor  # [batch, user, rx_antenna, ofdm_symbol, fft_bin]
    channel_frequency_response: torch.Tensor  # [batch, 1, rx_ant, user, tx_ant, symbol, fft]
    metadata: dict[str, Any]


class NrPuschCdlChannel:
    """Apply independent uplink CDL realizations and sum UE waveforms."""

    def __init__(
        self,
        settings: ChannelSettings,
        *,
        device: str | None = None,
    ) -> None:
        settings.validate()
        device = use_device(device)
        self.settings = settings
        self.device = device
        c = settings.channel
        a = settings.antennas
        self.ut_array = AntennaArray(
            num_rows=a.tx_num_rows,
            num_cols=a.tx_num_cols,
            polarization="single",
            polarization_type=a.polarization_type,
            antenna_pattern=a.antenna_pattern,
            carrier_frequency=c.carrier_frequency_hz,
            device=device,
        )
        self.bs_array = AntennaArray(
            num_rows=a.rx_num_rows,
            num_cols=a.rx_num_cols,
            polarization=a.polarization,
            polarization_type=a.polarization_type,
            antenna_pattern=a.antenna_pattern,
            carrier_frequency=c.carrier_frequency_hz,
            device=device,
        )
        self._cdl = self._make_cdl()
        self._apply_ofdm_channel = ApplyOFDMChannel(device=device)

    def apply_frequency(self, frequency_grid: torch.Tensor, resource_grid) -> FrequencyChannelResult:
        """Apply CDL through the same finite-tap response as the time path.

        ``frequency_grid`` uses Sionna's ``[batch, user, tx_ant, symbol, fft]``
        layout. CDL is sampled once per OFDM symbol, converted to the configured
        discrete sinc taps, then transformed to the frequency grid. This keeps
        batch processing efficient while matching the time path's tap support.
        The single-tap OFDM model still assumes a sufficient cyclic prefix.
        """
        tx_antennas = self.settings.antennas.tx_num_rows * self.settings.antennas.tx_num_cols
        if (frequency_grid.ndim != 5 or frequency_grid.shape[1] < 1
                or frequency_grid.shape[2] != tx_antennas):
            raise ValueError(
                f"frequency_grid 形状必须为 [batch, user, {tx_antennas} tx antenna, symbols, fft_size]"
            )
        if not frequency_grid.is_complex():
            raise ValueError("frequency_grid 必须为复数张量")
        if frequency_grid.shape[-2:] != (resource_grid.num_ofdm_symbols, resource_grid.fft_size):
            raise ValueError("frequency_grid 的 symbol/fft 维度与 resource_grid 不一致")
        if self.device is not None:
            frequency_grid = frequency_grid.to(self.device)

        batch_size = frequency_grid.shape[0]
        num_users = frequency_grid.shape[1]
        num_symbols = resource_grid.num_ofdm_symbols
        fft_size = resource_grid.fft_size
        num_rx_antennas = self.settings.antennas.rx_num_rows * self.settings.antennas.rx_num_cols
        sample_rate_hz = float(resource_grid.bandwidth)
        l_min, l_max = time_lag_discrete_time_channel(
            sample_rate_hz, self.settings.channel.max_delay_spread_s
        )
        num_taps = l_max - l_min + 1
        if num_taps > fft_size:
            raise ValueError("CDL 离散 taps 数量不能超过 OFDM FFT size")

        # The CDL batch axis represents independent UE links. Sample only once
        # per OFDM symbol, avoiding the many Nyquist-rate time samples needed
        # by ApplyTimeChannel.
        a, tau = self._cdl(
            batch_size * num_users,
            num_symbols,
            1.0 / resource_grid.ofdm_symbol_duration,
        )
        h_time = cir_to_time_channel(
            sample_rate_hz,
            a,
            tau,
            l_min,
            l_max,
            normalize=self.settings.channel.normalize_channel,
        )
        # Equivalent to Sionna time_to_ofdm_channel for one CIR sample per
        # OFDM symbol: move negative lags to the end before the FFT. Avoid
        # expanding the taps to every Nyquist-rate sample in a batch.
        h_padded = torch.nn.functional.pad(h_time, (0, fft_size - num_taps))
        h_flat = torch.fft.fftshift(
            torch.fft.fft(torch.roll(h_padded, shifts=l_min, dims=-1), dim=-1),
            dim=-1,
        )
        h_freq = h_flat.reshape(
            batch_size, num_users, 1, num_rx_antennas, 1, tx_antennas,
            num_symbols, fft_size,
        ).squeeze(2).permute(0, 3, 2, 1, 4, 5, 6).contiguous()

        y = self._apply_ofdm_channel(frequency_grid, h_freq)
        # Retain per-UE contributions for diagnostics, using the same channel
        # tensor as the summed Sionna ApplyOFDMChannel result.
        x_expanded = frequency_grid[:, None, None, :, :, :, :]
        user_contributions = (h_freq * x_expanded).sum(dim=4).squeeze(1)
        per_user_grid = user_contributions.permute(0, 2, 1, 3, 4).contiguous()
        metadata = {
            "model": self.settings.channel.model,
            "standard": "3GPP TR 38.901 CDL",
            "domain": "frequency",
            "direction": self.settings.channel.direction,
            "grid_axes": ["batch", "num_rx", "rx_antenna", "ofdm_symbol", "fft_bin"],
            "channel_axes": ["batch", "num_rx", "rx_antenna", "user", "tx_antenna", "ofdm_symbol", "fft_bin"],
            "num_users": num_users,
            "num_rx_antennas": num_rx_antennas,
            "num_tx_antennas_per_user": tx_antennas,
            "num_ofdm_symbols": num_symbols,
            "fft_size": fft_size,
            "subcarrier_spacing_hz": float(resource_grid.subcarrier_spacing),
            "ofdm_symbol_duration_s": float(resource_grid.ofdm_symbol_duration),
            "cyclic_prefix_assumption": "sufficient; frequency-domain channel excludes ISI",
            "frequency_response_source": "Sionna discrete sinc taps with time-domain lag support",
            "time_lag_min": l_min,
            "time_lag_max": l_max,
            "delay_spread_s": self.settings.channel.delay_spread_s,
            "carrier_frequency_hz": self.settings.channel.carrier_frequency_hz,
        }
        return FrequencyChannelResult(
            grid=y,
            per_user_grid=per_user_grid,
            channel_frequency_response=h_freq,
            metadata=metadata,
        )

    def apply(
        self,
        iq: torch.Tensor,
        sample_rate_hz: int,
    ) -> ChannelResult:
        """Apply CDL to ``[batch, user, tx_antenna, sample]`` IQ.

        UE links are independent; their received antenna contributions sum
        over the air. The output includes the channel filter tail.
        """
        tx_antennas = self.settings.antennas.tx_num_rows * self.settings.antennas.tx_num_cols
        if iq.ndim != 4 or iq.shape[1] < 1 or iq.shape[2] != tx_antennas:
            raise ValueError(
                f"iq 形状必须为 [batch, user, {tx_antennas} tx antenna, samples]，"
                f"实际为 {tuple(iq.shape)}"
            )
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz 必须大于 0")
        if not iq.is_complex():
            raise ValueError("iq 必须为复数张量")
        if self.device is not None:
            iq = iq.to(self.device)

        batch_size, num_users, _, num_samples = iq.shape
        channel = self._cdl
        l_min, l_max = time_lag_discrete_time_channel(
            float(sample_rate_hz), self.settings.channel.max_delay_spread_s
        )
        num_taps = l_max - l_min + 1
        num_channel_steps = num_samples + num_taps - 1

        # Flatten batch and UE axes so each UE is a distinct CDL link.
        tx = iq.reshape(batch_size * num_users, 1, tx_antennas, num_samples)
        a, tau = channel(batch_size * num_users, num_channel_steps, float(sample_rate_hz))
        h_time = cir_to_time_channel(
            float(sample_rate_hz),
            a,
            tau,
            l_min,
            l_max,
            normalize=self.settings.channel.normalize_channel,
        )
        apply_time_channel = ApplyTimeChannel(
            num_time_samples=num_samples,
            l_tot=num_taps,
            device=self.device,
        )
        rx = apply_time_channel(tx, h_time)
        num_rx_antennas = self.settings.antennas.rx_num_rows * self.settings.antennas.rx_num_cols
        per_user_iq = rx[:, 0, :, :].reshape(batch_size, num_users, num_rx_antennas, -1)
        channel_taps = h_time[:, 0, :, 0, :, :, :].reshape(
            batch_size, num_users, num_rx_antennas, tx_antennas, num_channel_steps, num_taps
        )
        output = per_user_iq.sum(dim=1)
        metadata = {
            "model": self.settings.channel.model,
            "standard": "3GPP TR 38.901 CDL",
            "direction": self.settings.channel.direction,
            "iq_axes": ["batch", "rx_antenna", "sample"],
            "per_user_iq_axes": ["batch", "user", "rx_antenna", "sample"],
            "num_users": num_users,
            "tx_antennas_per_user": tx_antennas,
            "channel_tap_axes": ["batch", "user", "rx_antenna", "tx_antenna", "sample", "tap"],
            "num_rx_antennas": num_rx_antennas,
            "delay_spread_s": self.settings.channel.delay_spread_s,
            "carrier_frequency_hz": self.settings.channel.carrier_frequency_hz,
            "sample_rate_hz": int(sample_rate_hz),
            "time_lag_min": l_min,
            "time_lag_max": l_max,
            "channel_tail_samples": num_taps - 1,
        }
        return ChannelResult(
            iq=output,
            per_user_iq=per_user_iq,
            channel_taps=channel_taps,
            sample_rate_hz=int(sample_rate_hz),
            metadata=metadata,
        )

    def _make_cdl(self) -> CDL:
        c = self.settings.channel
        return CDL(
            model=c.model,
            delay_spread=c.delay_spread_s,
            carrier_frequency=c.carrier_frequency_hz,
            ut_array=self.ut_array,
            bs_array=self.bs_array,
            direction=c.direction,
            min_speed=c.min_speed_mps,
            max_speed=c.max_speed_mps,
            normalize_delays=c.normalize_delays,
            device=self.device,
        )
