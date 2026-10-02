"""Sionna-backed multi-user PUSCH waveform generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sionna.phy.nr import PUSCHTransmitter
from sionna.phy.ofdm import OFDMModulator

from .config import TxSettings
from .device import use_device


@dataclass
class TxResult:
    """Generated samples and payload bits, with stable user-axis semantics."""

    iq: torch.Tensor
    frequency_grid: torch.Tensor
    bits: torch.Tensor
    sample_rate_hz: int
    metadata: dict[str, Any]


class NrPuschTx:
    """Generate separate UE waveforms with configured layer and antenna axes.

    The returned UE axis is not summed over the air interface. Each UE signal is
    kept separate so a channel model or captured receiver can combine them.
    """

    def __init__(self, settings: TxSettings, *, device: str | None = None):
        settings.validate()
        self.settings = settings
        self.configs = settings.to_sionna_configs()
        device = use_device(device)
        self._tx_freq = PUSCHTransmitter(
            self.configs,
            return_bits=False,
            output_domain="freq",
            device=device,
        )
        # Sionna resolves `None` to its configured default (often CUDA when
        # available); generate input bits and run added waveform processing there.
        self.device = self._tx_freq.device
        if settings.pusch.waveform == "cp_ofdm":
            self._tx_time = PUSCHTransmitter(
                self.configs,
                return_bits=False,
                output_domain="time",
                device=self.device,
            )
            self._ofdm_modulator = None
        else:
            self._tx_time = None
            self._ofdm_modulator = OFDMModulator(
                self._tx_freq.resource_grid.cyclic_prefix_length,
                device=self.device,
            )

    @property
    def transport_block_size(self) -> int:
        sizes = {cfg.tb_size for cfg in self.configs}
        if len(sizes) != 1:
            raise RuntimeError("Sionna configurations produced unequal transport block sizes")
        return sizes.pop()

    @property
    def sample_rate_hz(self) -> int:
        grid = self._tx_freq.resource_grid
        return int(grid.fft_size * grid.subcarrier_spacing)

    def generate(
        self,
        batch_size: int = 1,
        *,
        seed: int = 0,
        bits: torch.Tensor | None = None,
    ) -> TxResult:
        if batch_size < 1:
            raise ValueError("batch_size 必须大于 0")
        expected = (batch_size, len(self.settings.users), self.transport_block_size)
        if bits is None:
            generator = torch.Generator(device=self.device or "cpu")
            generator.manual_seed(seed)
            bits = torch.randint(
                0, 2, expected, generator=generator, device=self.device,
                dtype=torch.int32,
            ).to(torch.float32)
        else:
            if tuple(bits.shape) != expected:
                raise ValueError(f"bits 形状应为 {expected}，实际为 {tuple(bits.shape)}")
            bits = bits.to(device=self.device, dtype=torch.float32)

        frequency_grid = self._tx_freq(bits)
        if self.settings.pusch.waveform == "dft_s_ofdm":
            frequency_grid = self._transform_precoding(frequency_grid)
            frequency_grid = self._map_transform_precoded_dmrs(frequency_grid)
            iq = self._ofdm_modulator(frequency_grid)
        else:
            iq = self._tx_time(bits)
        metadata = {
            "waveform": self.settings.pusch.waveform,
            "users": [u.name for u in self.settings.users],
            "iq_axes": ["batch", "user", "tx_antenna", "sample"],
            "num_users": len(self.settings.users),
            "num_layers_per_user": self.settings.pusch.num_layers,
            "num_tx_antennas": self.settings.pusch.num_antenna_ports,
            "total_streams": len(self.settings.users) * self.settings.pusch.num_layers,
            "frequency_grid_axes": ["batch", "user", "tx_antenna", "symbol", "subcarrier"],
            "bits_axes": ["batch", "user", "tb_bit"],
            "sionna_output_domain": "time",
            "frequency_grid_stage": (
                "post DFT-s-OFDM transform precoding" if self.settings.pusch.waveform == "dft_s_ofdm"
                else "Sionna CP-OFDM resource grid"
            ),
            "configured_mcs": {"table": self.settings.pusch.mcs_table, "index": self.settings.pusch.mcs_index},
            "sionna_effective_mcs": {
                "table": self.settings.effective_sionna_mcs()[0],
                "index": self.settings.effective_sionna_mcs()[1],
            },
            "target_code_rate": float(self.configs[0].tb.target_coderate.item()),
            "transport_block_size_bits": self.transport_block_size,
            "sample_rate_hz": self.sample_rate_hz,
        }
        return TxResult(
            iq=iq,
            frequency_grid=frequency_grid,
            bits=bits,
            sample_rate_hz=self.sample_rate_hz,
            metadata=metadata,
        )

    def _transform_precoding(self, grid: torch.Tensor) -> torch.Tensor:
        """Apply unitary DFT precoding to each data-bearing PUSCH symbol."""
        output = grid.clone()
        mask = self._tx_freq.pilot_pattern.mask
        for user_index, cfg in enumerate(self.configs):
            start = (cfg.n_start_bwp - cfg.carrier.n_start_grid) * 12
            size = cfg.n_size_bwp * 12
            stop = start + size
            if start < 0 or stop > grid.shape[-1]:
                raise ValueError("BWP subcarrier allocation lies outside the Sionna resource grid")
            for symbol_index in range(grid.shape[-2]):
                # DMRS symbols are reserved by Sionna's pilot mask and are left
                # intact; DFT precoding applies to the QAM data symbols.
                if torch.any(mask[user_index, 0, symbol_index, start:stop] != 0).item():
                    continue
                data = grid[:, user_index, :, symbol_index, start:stop]
                output[:, user_index, :, symbol_index, start:stop] = torch.fft.fft(
                    data, dim=-1, norm="ortho"
                )
        return output

    def _map_transform_precoded_dmrs(self, grid: torch.Tensor) -> torch.Tensor:
        """Replace native CP-OFDM pilots with low-PAPR type-1 DMRS per layer.

        MATLAB TX fixtures use comb-first port order; MATLAB RX captures use
        Sionna's native OCC-first order. The profile selects the convention.
        """
        if self.settings.pusch.dmrs_config_type != 1:
            raise NotImplementedError("DFT-s-OFDM DMRS currently supports config type 1 only")

        output = grid.clone()
        beta = self.settings.pusch.dmrs_beta
        for user_index, cfg in enumerate(self.configs):
            start = (cfg.n_start_bwp - cfg.carrier.n_start_grid) * 12
            num_subcarriers = cfg.n_size_bwp * 12
            stop = start + num_subcarriers
            if num_subcarriers % 2:
                raise ValueError("DFT-s-OFDM type-1 DMRS requires an even BWP subcarrier count")

            n_id_rs = cfg.carrier.n_cell_id if cfg.dmrs.n_id is None else cfg.dmrs.n_id
            if isinstance(n_id_rs, (tuple, list)):
                n_id_rs = n_id_rs[0]
            group = int(n_id_rs) % 30
            m_zc = num_subcarriers // 2
            n_zc = _largest_prime_below(m_zc)
            q = int(np.floor(n_zc * (group + 1) / 31 + 0.5))
            n = torch.arange(m_zc, dtype=torch.float64, device=grid.device)
            n = torch.remainder(n, n_zc)
            phase = -torch.pi * q * n * (n + 1) / n_zc
            sequence = torch.polar(torch.full_like(phase, beta), phase).to(dtype=grid.dtype)
            frequency_cover = torch.where(
                torch.arange(m_zc, device=grid.device) % 2 == 0, 1.0, -1.0
            )
            for symbol_index in cfg.dmrs_symbol_indices:
                dmrs_layers = torch.zeros(
                    (cfg.num_layers, num_subcarriers), dtype=grid.dtype, device=grid.device
                )
                l_prime = cfg.dmrs_symbol_indices.index(symbol_index) % cfg.dmrs.length
                for layer, port in enumerate(cfg.dmrs.dmrs_port_set):
                    if self.settings.pusch.dft_s_dmrs_port_order == "native":
                        delta = int(cfg.dmrs.deltas[layer])
                        alternating = int(cfg.dmrs.w_f[1, layer]) == -1
                    else:
                        delta = port % 2
                        alternating = (port // 2) % 2 == 1
                    time_cover = cfg.dmrs.w_t[l_prime, layer]
                    dmrs_layers[layer, delta::2] = sequence * (
                        frequency_cover if alternating else 1.0
                    ) * time_cover
                if cfg.precoding == "codebook":
                    matrix = torch.as_tensor(
                        cfg.precoding_matrix, device=grid.device, dtype=grid.dtype
                    )
                    dmrs_antennas = matrix @ dmrs_layers
                else:
                    dmrs_antennas = dmrs_layers
                output[:, user_index, :, symbol_index, start:stop] = dmrs_antennas
        return output


def _largest_prime_below(value: int) -> int:
    """Return the largest prime strictly smaller than ``value``."""
    for candidate in range(value - 1, 1, -1):
        if all(candidate % divisor for divisor in range(2, int(candidate**0.5) + 1)):
            return candidate
    raise ValueError(f"No prime sequence length available below {value}")
