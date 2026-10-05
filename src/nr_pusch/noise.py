"""Reproducible complex AWGN injection for time-domain IQ captures."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

@dataclass
class NoisySignal:
    iq: torch.Tensor  # [batch, rx_antenna, sample]
    noise_variance: torch.Tensor  # [batch, 1, rx_antenna]


@dataclass
class NoisyResourceGrid:
    grid: torch.Tensor  # [batch, num_rx, rx_antenna, ofdm_symbol, fft_bin]
    noise_variance: torch.Tensor  # [batch, num_rx, rx_antenna]


def add_awgn(iq: torch.Tensor, snr_db: float, *, seed: int = 0) -> NoisySignal:
    """Add complex AWGN at a measured per-RX-antenna signal-to-noise ratio.

    Signal power is measured independently for each batch and receive antenna
    across the available samples, including any CDL channel-filter tail.
    ``noise_variance`` follows Sionna's variance-per-complex-sample convention.
    """
    if iq.ndim != 3 or iq.shape[1] < 1 or not iq.is_complex():
        raise ValueError("iq 形状必须为复数 [batch, rx_antennas, samples]")
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


def add_awgn_resource_grid(
    grid: torch.Tensor,
    snr_db: float,
    *,
    seed: int = 0,
) -> NoisyResourceGrid:
    """Add complex AWGN to ``[batch, rx, rx_ant, symbol, fft_bin]`` grids."""
    if grid.ndim != 5 or grid.shape[1] < 1 or grid.shape[2] < 1 or not grid.is_complex():
        raise ValueError("grid 形状必须为复数 [batch, num_rx, rx_antennas, symbols, fft_bins]")
    if not torch.isfinite(torch.tensor(snr_db)).item():
        raise ValueError("snr_db 必须为有限数值")
    power = grid.abs().square().mean(dim=(-1, -2))
    if torch.any(power <= 0).item():
        raise ValueError("每个接收天线的资源网格平均功率必须大于 0")
    noise_variance = power * (10.0 ** (-float(snr_db) / 10.0))
    generator = torch.Generator(device=grid.device)
    generator.manual_seed(seed)
    real = torch.randn(grid.shape, dtype=grid.real.dtype, device=grid.device, generator=generator)
    imag = torch.randn(grid.shape, dtype=grid.real.dtype, device=grid.device, generator=generator)
    scale = torch.sqrt(noise_variance[..., None, None] / 2.0)
    noise = torch.complex(real, imag) * scale
    return NoisyResourceGrid(grid=grid + noise, noise_variance=noise_variance)


def beam_noise_covariance(
    weights: torch.Tensor,
    array_variance: float,
    post_variance: float,
) -> torch.Tensor:
    """Return ``sigma_a² WᴴW + sigma_v² I`` for four receive beams."""
    _validate_beam_weights(weights)
    array_variance = _validate_variance(array_variance, "array_variance")
    post_variance = _validate_variance(post_variance, "post_variance")
    gram = weights.conj().transpose(0, 1) @ weights
    identity = torch.eye(4, dtype=weights.dtype, device=weights.device)
    return array_variance * gram + post_variance * identity


def add_correlated_awgn(
    signal: torch.Tensor,
    weights: torch.Tensor,
    *,
    array_noise_variance: float,
    post_noise_variance: float,
    beam_axis: int,
    seed: int,
) -> torch.Tensor:
    """Add shared element noise after ``Wᴴ`` plus independent beam-chain noise.

    Independent standard complex-normal element and post-combiner draws are
    generated in a fixed order from ``seed``. Reusing the seed across noise
    ratios therefore reuses the element-noise draw and the post-noise draw.
    """
    if signal.ndim < 1 or not signal.is_complex():
        raise ValueError("signal 必须是至少一维复数张量")
    _validate_beam_weights(weights)
    array_noise_variance = _validate_variance(array_noise_variance, "array_noise_variance")
    post_noise_variance = _validate_variance(post_noise_variance, "post_noise_variance")
    if isinstance(beam_axis, bool) or not isinstance(beam_axis, int):
        raise ValueError("beam_axis 必须为整数")
    if not -signal.ndim <= beam_axis < signal.ndim:
        raise ValueError("beam_axis 超出 signal 维度")
    beam_axis %= signal.ndim
    if signal.shape[beam_axis] != 4:
        raise ValueError("beam_axis 长度必须为 4")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed 必须为非负整数")
    if array_noise_variance == 0.0 and post_noise_variance == 0.0:
        return signal

    weights = weights.to(device=signal.device, dtype=signal.dtype)
    if not torch.isfinite(signal).all().item():
        raise ValueError("signal 必须为有限复数")
    if not torch.isfinite(weights).all().item():
        raise ValueError("weights 必须为有限复数")
    other_shape = tuple(size for index, size in enumerate(signal.shape) if index != beam_axis)
    real_dtype = signal.real.dtype
    generator = torch.Generator(device=signal.device)
    generator.manual_seed(seed)

    array_noise = 0.0
    if array_noise_variance > 0.0:
        element_shape = (*other_shape, weights.shape[0])
        element_draw = _standard_complex_normal(
            element_shape,
            dtype=real_dtype,
            device=signal.device,
            generator=generator,
        )
        array_noise = torch.einsum("nb,...n->...b", weights.conj(), element_draw)
        array_noise = array_noise * math.sqrt(array_noise_variance)

    post_noise = 0.0
    if post_noise_variance > 0.0:
        post_shape = (*other_shape, 4)
        post_noise = _standard_complex_normal(
            post_shape,
            dtype=real_dtype,
            device=signal.device,
            generator=generator,
        ) * math.sqrt(post_noise_variance)
    total_noise = (array_noise + post_noise).movedim(-1, beam_axis)
    return signal + total_noise


def _validate_beam_weights(weights: torch.Tensor) -> None:
    if not isinstance(weights, torch.Tensor) or weights.ndim != 2 or weights.shape[0] < 1 or weights.shape[1] != 4:
        raise ValueError("weights 必须为 [element,4] 复数张量")
    if not weights.is_complex():
        raise ValueError("weights 必须为复数张量")
    if not torch.isfinite(weights).all().item():
        raise ValueError("weights 必须为有限复数")


def _validate_variance(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} 必须为有限非负数")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} 必须为有限非负数")
    return result


def _standard_complex_normal(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    real = torch.randn(shape, dtype=dtype, device=device, generator=generator)
    imag = torch.randn(shape, dtype=dtype, device=device, generator=generator)
    return torch.complex(real, imag) / math.sqrt(2.0)
