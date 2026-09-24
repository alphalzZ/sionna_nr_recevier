"""Reproducible complex AWGN injection for time-domain IQ captures."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class NoisySignal:
    iq: torch.Tensor  # [batch, rx_antenna, sample]
    noise_variance: torch.Tensor  # [batch, 1, rx_antenna]


def add_awgn(iq: torch.Tensor, snr_db: float, *, seed: int = 0) -> NoisySignal:
    """Add complex AWGN at a measured per-RX-antenna signal-to-noise ratio.

    Signal power is measured independently for each batch and receive antenna
    across the available samples, including any CDL channel-filter tail.
    ``noise_variance`` follows Sionna's variance-per-complex-sample convention.
    """
    if iq.ndim != 3 or iq.shape[1] != 4 or not iq.is_complex():
        raise ValueError("iq 形状必须为复数 [batch, 4 rx_antennas, samples]")
    if not torch.isfinite(torch.tensor(snr_db)).item():
        raise ValueError("snr_db 必须为有限数值")
    power = iq.abs().square().mean(dim=-1)
    if torch.any(power <= 0).item():
        raise ValueError("每个接收天线的信号功率必须大于 0")
    noise_variance = power * (10.0 ** (-float(snr_db) / 10.0))
    generator = torch.Generator(device=iq.device)
    generator.manual_seed(seed)
    real_dtype = iq.real.dtype
    noise_shape = iq.shape
    real = torch.randn(noise_shape, dtype=real_dtype, device=iq.device, generator=generator)
    imag = torch.randn(noise_shape, dtype=real_dtype, device=iq.device, generator=generator)
    noise = torch.complex(real, imag) * torch.sqrt(noise_variance.unsqueeze(-1) / 2.0)
    return NoisySignal(iq=iq + noise, noise_variance=noise_variance.unsqueeze(1))
