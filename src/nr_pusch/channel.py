"""Sionna 38.901 CDL channel application for four single-antenna PUSCH UEs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from sionna.phy.channel import ApplyTimeChannel, cir_to_time_channel, time_lag_discrete_time_channel
from sionna.phy.channel.tr38901 import AntennaArray, CDL

from .channel_config import ChannelSettings


@dataclass
class ChannelResult:
    """Channelized IQ and intermediate values with documented tensor axes."""

    iq: torch.Tensor  # [batch, rx_antenna, sample]
    per_user_iq: torch.Tensor  # [batch, user, rx_antenna, sample]
    channel_taps: torch.Tensor  # [batch, user, rx_antenna, sample, tap]
    sample_rate_hz: int
    metadata: dict[str, Any]


class NrPuschCdlChannel:
    """Apply independent uplink CDL realizations and sum four UE waveforms."""

    num_users = 4
    tx_antennas_per_user = 1

    def __init__(
        self,
        settings: ChannelSettings,
        *,
        device: str | None = None,
    ) -> None:
        settings.validate()
        self.settings = settings
        self.device = device
        c = settings.channel
        a = settings.antennas
        self.ut_array = AntennaArray(
            num_rows=1,
            num_cols=1,
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

    def apply(
        self,
        iq: torch.Tensor,
        sample_rate_hz: int,
    ) -> ChannelResult:
        """Apply CDL to `[batch, 4 users, 1 tx antenna, sample]` IQ.

        Each UE gets an independently drawn CDL realization. The returned
        `iq` sums the four users over the air; `per_user_iq` is retained for
        tests and interference diagnostics. The output includes the channel
        filter tail (`num_taps - 1`) as standard linear convolution does.
        """
        if iq.ndim != 4 or iq.shape[1] != self.num_users or iq.shape[2] != 1:
            raise ValueError(
                "iq 形状必须为 [batch, 4 users, 1 tx antenna, samples]，"
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
        tx = iq[:, :, 0, :].reshape(batch_size * num_users, 1, 1, num_samples)
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
        channel_taps = h_time[:, 0, :, 0, 0, :, :].reshape(
            batch_size, num_users, num_rx_antennas, num_channel_steps, num_taps
        )
        output = per_user_iq.sum(dim=1)
        metadata = {
            "model": self.settings.channel.model,
            "standard": "3GPP TR 38.901 CDL",
            "direction": self.settings.channel.direction,
            "iq_axes": ["batch", "rx_antenna", "sample"],
            "per_user_iq_axes": ["batch", "user", "rx_antenna", "sample"],
            "num_users": num_users,
            "tx_antennas_per_user": 1,
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
