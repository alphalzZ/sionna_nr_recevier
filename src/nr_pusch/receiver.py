"""Four-user PUSCH receiver with Sionna LMMSE and NR transport decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sionna.phy.mapping import (
    Constellation,
    Demapper,
    LLRs2SymbolLogits,
    SymbolLogits2Moments,
)
from sionna.phy.channel import time_lag_discrete_time_channel
from sionna.phy.fec.scrambling import Descrambler, Scrambler
from sionna.phy.fec.ldpc import LDPC5GDecoder
from sionna.phy.mimo import KBestDetector as FlatKBestDetector
from sionna.phy.mimo import StreamManagement, lmmse_matrix, whiten_channel
from sionna.phy.nr import PUSCHReceiver, PUSCHTransmitter, TBDecoder
from sionna.phy.ofdm import (
    LMMSEEqualizer,
    LinearDetector,
)

from .config import TxSettings
from .device import use_device
from .transmitter import NrPuschTx

_MIN_NOISE_VARIANCE = 1e-12


def _validate_scrambling_sequences(
    sequences: torch.Tensor, num_users: int, num_code_bits: int
) -> None:
    if tuple(sequences.shape) != (num_users, num_code_bits):
        raise ValueError(
            f"scrambling_sequences 形状应为 ({num_users}, {num_code_bits})，"
            f"实际为 {tuple(sequences.shape)}"
        )
    if not bool(torch.all((sequences == 0) | (sequences == 1))):
        raise ValueError("scrambling_sequences 只能包含 0/1")


def _use_explicit_scrambling(
    tb_decoder: TBDecoder,
    sequences: torch.Tensor,
    num_users: int,
    num_code_bits: int,
) -> None:
    """Replace an NR TB decoder's RNTI-derived descrambling sequence."""
    _validate_scrambling_sequences(sequences, num_users, num_code_bits)
    scrambler = Scrambler(
        sequence=sequences.to(dtype=torch.float32),
        binary=False,
        device=tb_decoder.device,
    )
    tb_decoder._descrambler = Descrambler(
        scrambler, binary=False, device=tb_decoder.device
    )


def _use_explicit_tx_scrambling(
    transmitter: NrPuschTx,
    sequences: torch.Tensor,
    num_users: int,
    num_code_bits: int,
) -> None:
    """Replace an NR TB encoder's RNTI-derived scrambling sequence."""
    _validate_scrambling_sequences(sequences, num_users, num_code_bits)
    transmitter._tx_freq._tb_encoder._scrambler = Scrambler(
        sequence=sequences.to(dtype=torch.float32),
        binary=True,
        device=transmitter.device,
    )

class _LdpcSoftFeedback(torch.nn.Module):
    """Return rate-matched LDPC posterior-minus-channel extrinsic LLRs."""

    def __init__(self, tb_decoder: TBDecoder, num_iterations: int, device: str | None):
        super().__init__()
        self._encoder = tb_decoder._tb_encoder
        self._descrambler = tb_decoder._descrambler
        self._output_perm_inv = tb_decoder._output_perm_inv
        self._output_perm = torch.argsort(self._output_perm_inv)
        self._decoder = LDPC5GDecoder(
            encoder=self._encoder.ldpc_encoder,
            num_iter=num_iterations,
            hard_out=False,
            return_infobits=False,
            device=device,
        )
        self._cw_length = int(self._encoder.n)
        self._num_cbs = int(self._encoder.num_cbs)
        self._ldpc_n = int(self._encoder.ldpc_encoder.n)
        self._num_fillers = self._ldpc_n * self._num_cbs - int(
            self._encoder.cw_lengths_sum
        )

    def forward(self, llr: torch.Tensor) -> torch.Tensor:
        """Map scrambled rate-matched LLRs [batch,user,n] to same-order extrinsic."""
        if llr.ndim != 3 or llr.shape[-1] != self._cw_length:
            raise ValueError(
                f"LDPC feedback expects [batch,user,{self._cw_length}] LLRs"
            )
        llr = llr.float().clamp(-20.0, 20.0)
        llr_cb_order = self._descrambler(llr) if self._descrambler is not None else llr
        if self._num_fillers:
            llr_cb_order = torch.cat(
                (
                    llr_cb_order,
                    torch.zeros(
                        *llr_cb_order.shape[:-1],
                        self._num_fillers,
                        dtype=llr_cb_order.dtype,
                        device=llr_cb_order.device,
                    ),
                ),
                dim=-1,
            )
        llr_cb_order = torch.index_select(
            llr_cb_order, -1, self._output_perm_inv.to(llr.device)
        )
        llr_cb = llr_cb_order.reshape(
            *llr.shape[:2], self._num_cbs, self._ldpc_n
        )
        posterior = self._decoder(llr_cb)
        extrinsic_cb = posterior - llr_cb.clamp(-20.0, 20.0)
        extrinsic = extrinsic_cb.reshape(*llr.shape[:2], -1)
        extrinsic = torch.index_select(extrinsic, -1, self._output_perm.to(llr.device))
        extrinsic = extrinsic[..., : self._cw_length]
        extrinsic = self._descrambler(extrinsic) if self._descrambler is not None else extrinsic
        return extrinsic.clamp(-20.0, 20.0)



class DftSOfdmMimoDetector(torch.nn.Module):
    """Detect spread symbols, undo DFT spreading, then produce QAM LLRs."""

    def __init__(self, transmitter: PUSCHTransmitter, stream_management: StreamManagement,
                 method: str = "lmmse", parameter: int | None = None):
        super().__init__()
        self._resource_grid = transmitter.resource_grid
        self.method = method
        self.parameter = parameter
        bps = transmitter._num_bits_per_symbol
        if method == "lmmse":
            self._detector = LMMSEEqualizer(
                self._resource_grid, stream_management, device=transmitter.device
            )
        else:
            raise ValueError(f"不支持的 MIMO detector: {method}")
        self._demapper = Demapper(
            "app",
            "qam",
            bps,
            device=transmitter.device,
        )
        self._constellation = Constellation(
            "qam", bps, device=transmitter.device
        )

        # Reuse Sionna's resource-grid data indexing so demapper ordering stays
        # consistent with its PUSCH layer mapper and TB decoder.
        extractor = LinearDetector("lmmse", "symbol", "app", self._resource_grid,
                                   stream_management, "qam", bps, device=transmitter.device)
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
        self.register_buffer(
            "_data_symbol_indices",
            torch.unique(symbol_indices, sorted=True),
        )
        self._num_data_symbols = int(ordered_indices.numel())
        self._num_bits_per_symbol = bps
        self.last_llr: torch.Tensor | None = None

    def forward(
        self,
        y: torch.Tensor,
        h_hat: torch.Tensor,
        err_var: torch.Tensor,
        no: torch.Tensor,
    ) -> torch.Tensor:
        if self.method == "lmmse":
            x_hat, no_eff = self._detector(y, h_hat, err_var, no)
        else:
            x_hat = self._detector(y, h_hat, err_var, no)
        if not x_hat.is_complex():
            # Sionna hard_out=True returns QAM point indices for symbol output.
            x_hat = self._constellation.points[x_hat.to(torch.long)]
        batch, num_tx, num_streams, num_data = x_hat.shape
        expected = self._num_spread_symbols * self._fft_size
        if num_data != expected:
            raise ValueError(
                f"PUSCH 数据符号数 {num_data} 与 DFT-s-OFDM 资源映射预期 {expected} 不符"
            )

        x_hat = x_hat.reshape(batch, num_tx, num_streams, self._num_spread_symbols, self._fft_size)
        # Tx applies a unitary forward DFT per data-bearing OFDM symbol.
        x_hat = torch.fft.ifft(x_hat, dim=-1, norm="ortho")
        # The unitary inverse DFT produces correlated noise when subcarrier
        # variances differ. Average variance per spread symbol for LLR scaling.
        if self.method == "lmmse":
            no_eff = no_eff.reshape_as(x_hat.real).mean(dim=-1, keepdim=True).expand_as(x_hat.real)
        else:
            no_eff = torch.as_tensor(no, dtype=x_hat.real.dtype, device=x_hat.device)
            if no_eff.ndim >= 2:
                no_eff = no_eff.mean(dim=tuple(range(1, no_eff.ndim)))
            no_eff = no_eff.reshape(batch, 1, 1, 1, 1).expand_as(x_hat.real)
        x_hat = x_hat.reshape(batch, num_tx, num_streams, self._num_data_symbols)
        no_eff = no_eff.reshape_as(x_hat.real)
        llr = self._demapper(x_hat, no_eff)
        self.last_llr = llr
        return llr


class DftSOfdmKBestDetector(DftSOfdmMimoDetector):
    """LMMSE pre-equalize, despread, then K-best detect each time sample."""

    def __init__(
        self,
        transmitter: PUSCHTransmitter,
        stream_management: StreamManagement,
        k: int | None = None,
    ) -> None:
        super().__init__(transmitter, stream_management, method="lmmse")
        self.k = 64 if k is None else k
        self._flat_detector = FlatKBestDetector(
            "symbol",
            num_streams=stream_management._num_tx * stream_management._num_streams_per_tx,
            k=self.k,
            constellation_type="qam",
            num_bits_per_symbol=int(torch.as_tensor(self._num_bits_per_symbol).reshape(-1)[0]),
            hard_out=True,
            device=transmitter.device,
        )

    def forward(
        self,
        y: torch.Tensor,
        h_hat: torch.Tensor,
        err_var: torch.Tensor,
        no: torch.Tensor,
    ) -> torch.Tensor:
        """Return ordered PUSCH LLRs after per-sample spatial K-best search."""
        batch, num_rx, num_rx_ant, num_symbols, fft_size = y.shape
        if num_rx != 1 or fft_size != self._fft_size:
            raise ValueError("K-best DFT-s-OFDM 仅支持单接收端口和当前资源网格尺寸")

        # Match Sionna's LMMSEEqualizer covariance construction. All four UE
        # streams are desired streams, so only thermal noise and CSI error enter S.
        y_eff = self._detector._removed_nulled_scs(y)
        y_dt = y_eff.permute(0, 1, 3, 4, 2)
        h_dt = h_hat.permute(0, 1, 5, 6, 2, 3, 4).flatten(-2)
        err_dt = torch.broadcast_to(err_var, h_hat.shape)
        err_dt = err_dt.permute(0, 1, 5, 6, 2, 3, 4).flatten(-2)
        no = torch.as_tensor(no, dtype=y.real.dtype, device=y.device)
        if no.ndim == 1:
            no = no.unsqueeze(1)
        no = torch.broadcast_to(no, (batch, num_rx, num_rx_ant))
        no_dt = no[:, :, None, None, :].expand(
            batch, num_rx, num_symbols, fft_size, num_rx_ant
        )
        err_power = err_dt.sum(dim=-1)
        covariance = torch.diag_embed(no_dt + err_power).to(y.dtype)

        y_white, h_white = whiten_channel(
            y_dt, h_dt, covariance, return_s=False
        )
        equalizer_matrix = lmmse_matrix(h_white, s=None)
        gram = equalizer_matrix @ h_white
        diagonal = torch.diagonal(gram, dim1=-2, dim2=-1)
        normalized_filter = equalizer_matrix / diagonal.unsqueeze(-1)
        x_hat = (equalizer_matrix @ y_white.unsqueeze(-1)).squeeze(-1)
        x_hat = x_hat / diagonal
        effective_channel = normalized_filter @ h_white
        post_eq_noise_covariance = normalized_filter @ normalized_filter.mH

        # [batch, data-bearing OFDM symbol, subcarrier, UE]
        x_freq = x_hat[:, 0].index_select(1, self._data_symbol_indices)
        effective_channel = effective_channel[:, 0].index_select(
            1, self._data_symbol_indices
        )
        post_eq_noise_covariance = post_eq_noise_covariance[:, 0].index_select(
            1, self._data_symbol_indices
        )

        # IDFT the LMMSE output and retain the zero-lag part of its effective
        # channel as the small per-sample MIMO matrix. Remaining frequency
        # variation becomes colored residual ISI in the K-best noise covariance.
        z_time = torch.fft.ifft(x_freq.permute(0, 1, 3, 2), dim=-1, norm="ortho")
        z_time = z_time.permute(0, 1, 3, 2)
        h_zero_lag = effective_channel.mean(dim=2)
        channel_residual = effective_channel - h_zero_lag.unsqueeze(2)
        isi_covariance = (channel_residual @ channel_residual.mH).mean(dim=2)
        noise_covariance = post_eq_noise_covariance.mean(dim=2)
        covariance_time = noise_covariance + isi_covariance

        regularization = covariance_time.diagonal(dim1=-2, dim2=-1).real.mean(dim=-1)
        regularization = regularization.clamp_min(1e-4) * 1e-5 + 1e-7
        eye = torch.eye(num_rx_ant, dtype=covariance_time.dtype, device=y.device)
        covariance_time = (covariance_time + covariance_time.mH) * 0.5
        covariance_time = covariance_time + regularization[..., None, None] * eye

        flat_y = z_time.reshape(-1, num_rx_ant)
        flat_h = h_zero_lag.unsqueeze(2).expand(
            batch,
            self._num_spread_symbols,
            self._fft_size,
            num_rx_ant,
            num_rx_ant,
        ).reshape(-1, num_rx_ant, num_rx_ant)
        flat_covariance = covariance_time.unsqueeze(2).expand(
            batch,
            self._num_spread_symbols,
            self._fft_size,
            num_rx_ant,
            num_rx_ant,
        ).reshape(-1, num_rx_ant, num_rx_ant)
        symbol_indices = self._flat_detector(flat_y, flat_h, flat_covariance)
        symbols = self._constellation.points[symbol_indices.to(torch.long)]
        symbols = symbols.reshape(
            batch, self._num_spread_symbols, self._fft_size, num_rx_ant
        )
        bits_per_symbol = int(torch.as_tensor(self._num_bits_per_symbol).reshape(-1)[0])
        information_matrix = h_zero_lag.mH @ torch.linalg.solve(
            covariance_time, h_zero_lag
        )
        post_detection_covariance = torch.linalg.inv(information_matrix)
        post_detection_variance = torch.diagonal(
            post_detection_covariance, dim1=-2, dim2=-1
        ).real.clamp_min(1e-7)
        post_detection_variance = post_detection_variance.unsqueeze(2).expand(
            batch, self._num_spread_symbols, self._fft_size, num_rx_ant
        )
        symbols = symbols.permute(0, 3, 1, 2).reshape(
            batch, num_rx_ant, 1, self._num_data_symbols
        )
        post_detection_variance = post_detection_variance.permute(0, 3, 1, 2).reshape(
            batch, num_rx_ant, 1, self._num_data_symbols
        )
        llr = self._demapper(symbols, post_detection_variance)
        self.last_llr = llr
        return llr


class DftSOfdmEpDetector(DftSOfdmMimoDetector):
    """EP detection on time-domain spatial symbols after frequency LMMSE.

    The LMMSE front-end removes most frequency-selective ISI. The residual
    frequency variation is represented by a colored Gaussian covariance, and
    an EP factor graph then detects the four spatial QAM symbols at each
    DFT-s-OFDM time sample. EP sites are updated in parallel across users.
    """

    def __init__(
        self,
        transmitter: PUSCHTransmitter,
        stream_management: StreamManagement,
        iterations: int | None = None,
        damping: float = 0.5,
    ) -> None:
        super().__init__(transmitter, stream_management, method="lmmse")
        self.iterations = 10 if iterations is None else iterations
        if self.iterations < 1:
            raise ValueError("EP iteration count must be positive")
        if not 0.0 < damping <= 1.0:
            raise ValueError("EP damping must be in (0, 1]")
        self.damping = damping
        self._points = self._constellation.points

    def forward(
        self,
        y: torch.Tensor,
        h_hat: torch.Tensor,
        err_var: torch.Tensor,
        no: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_rx, num_rx_ant, num_symbols, fft_size = y.shape
        if num_rx != 1 or fft_size != self._fft_size:
            raise ValueError("EP DFT-s-OFDM 仅支持单接收端口和当前资源网格尺寸")

        # Use the same LMMSE whitening and posterior channel as the K-best path.
        y_eff = self._detector._removed_nulled_scs(y)
        y_dt = y_eff.permute(0, 1, 3, 4, 2)
        h_dt = h_hat.permute(0, 1, 5, 6, 2, 3, 4).flatten(-2)
        err_dt = torch.broadcast_to(err_var, h_hat.shape)
        err_dt = err_dt.permute(0, 1, 5, 6, 2, 3, 4).flatten(-2)
        no = torch.as_tensor(no, dtype=y.real.dtype, device=y.device)
        if no.ndim == 1:
            no = no.unsqueeze(1)
        no = torch.broadcast_to(no, (batch, num_rx, num_rx_ant))
        no_dt = no[:, :, None, None, :].expand(
            batch, num_rx, num_symbols, fft_size, num_rx_ant
        )
        covariance_fd = torch.diag_embed(no_dt + err_dt.sum(dim=-1)).to(y.dtype)
        y_white, h_white = whiten_channel(y_dt, h_dt, covariance_fd, return_s=False)
        equalizer = lmmse_matrix(h_white, s=None)
        gram = equalizer @ h_white
        diagonal = torch.diagonal(gram, dim1=-2, dim2=-1)
        normalized_filter = equalizer / diagonal.unsqueeze(-1)
        x_hat = (equalizer @ y_white.unsqueeze(-1)).squeeze(-1) / diagonal
        effective_channel = normalized_filter @ h_white
        post_eq_noise = normalized_filter @ normalized_filter.mH

        x_freq = x_hat[:, 0].index_select(1, self._data_symbol_indices)
        effective_channel = effective_channel[:, 0].index_select(
            1, self._data_symbol_indices
        )
        post_eq_noise = post_eq_noise[:, 0].index_select(
            1, self._data_symbol_indices
        )
        # IDFT the equalized stream vector; the mean channel is its zero-lag
        # spatial component, while frequency variation forms residual ISI.
        z_time = torch.fft.ifft(x_freq.permute(0, 1, 3, 2), dim=-1, norm="ortho")
        z_time = z_time.permute(0, 1, 3, 2)
        h_flat = effective_channel.mean(dim=2)
        h_residual = effective_channel - h_flat.unsqueeze(2)
        cov_time = post_eq_noise.mean(dim=2) + (h_residual @ h_residual.mH).mean(dim=2)
        cov_time = (cov_time + cov_time.mH) * 0.5
        scale = cov_time.diagonal(dim1=-2, dim2=-1).real.mean(dim=-1).clamp_min(1e-6)
        eye_rx = torch.eye(num_rx_ant, dtype=y.dtype, device=y.device)
        cov_time = cov_time + (scale * 1e-6)[..., None, None] * eye_rx

        # Whiten each effective 4x4 spatial system. Leading dimensions combine
        # batch, data-bearing OFDM symbols, and time samples.
        flat_y = z_time.reshape(-1, num_rx_ant)
        flat_h = h_flat.unsqueeze(2).expand(
            batch, self._num_spread_symbols, self._fft_size, num_rx_ant, num_rx_ant
        ).reshape(-1, num_rx_ant, num_rx_ant)
        flat_cov = cov_time.unsqueeze(2).expand(
            batch, self._num_spread_symbols, self._fft_size, num_rx_ant, num_rx_ant
        ).reshape(-1, num_rx_ant, num_rx_ant)
        chol = torch.linalg.cholesky(flat_cov)
        yw = torch.linalg.solve_triangular(chol, flat_y.unsqueeze(-1), upper=False).squeeze(-1)
        hw = torch.linalg.solve_triangular(chol, flat_h, upper=False)

        num_users = hw.shape[-1]
        eye_u = torch.eye(num_users, dtype=y.dtype, device=y.device)
        gram_w = hw.mH @ hw
        rhs = (hw.mH @ yw.unsqueeze(-1)).squeeze(-1)
        site_precision = torch.ones(
            (flat_y.shape[0], num_users), dtype=y.real.dtype, device=y.device
        )
        site_natural = torch.zeros_like(rhs)
        points = self._points.to(device=y.device, dtype=y.dtype)
        points = points.reshape(1, 1, -1)
        eps = 1e-6

        for _ in range(self.iterations):
            precision = gram_w + torch.diag_embed(site_precision.to(y.dtype))
            precision = precision + eps * eye_u
            posterior_cov = torch.linalg.inv(precision)
            posterior_mean = (posterior_cov @ (rhs + site_natural).unsqueeze(-1)).squeeze(-1)
            posterior_var = torch.diagonal(posterior_cov, dim1=-2, dim2=-1).real.clamp_min(eps)

            # Remove each site's Gaussian message, project the cavity onto QAM,
            # then damp the new Gaussian site parameters.
            cavity_precision = (1.0 / posterior_var - site_precision).clamp_min(eps)
            cavity_var = 1.0 / cavity_precision
            cavity_mean = (posterior_mean / posterior_var - site_natural) / cavity_precision
            logits = -torch.abs(points - cavity_mean.unsqueeze(-1)).square() / cavity_var.unsqueeze(-1)
            probabilities = torch.softmax(logits, dim=-1)
            discrete_mean = (probabilities * points).sum(dim=-1)
            discrete_var = (
                probabilities * torch.abs(points - discrete_mean.unsqueeze(-1)).square()
            ).sum(dim=-1).real.clamp_min(eps)
            proposed_precision = (1.0 / discrete_var - cavity_precision).clamp_min(eps)
            proposed_natural = discrete_mean / discrete_var - cavity_mean * cavity_precision
            site_precision = (1.0 - self.damping) * site_precision + self.damping * proposed_precision
            site_natural = (1.0 - self.damping) * site_natural + self.damping * proposed_natural

        precision = gram_w + torch.diag_embed(site_precision.to(y.dtype)) + eps * eye_u
        posterior_cov = torch.linalg.inv(precision)
        posterior_mean = (posterior_cov @ (rhs + site_natural).unsqueeze(-1)).squeeze(-1)
        posterior_var = torch.diagonal(posterior_cov, dim1=-2, dim2=-1).real.clamp_min(eps)
        means = posterior_mean.reshape(
            batch, self._num_spread_symbols, self._fft_size, num_users
        ).permute(0, 3, 1, 2).reshape(batch, num_users, 1, self._num_data_symbols)
        variances = posterior_var.reshape(
            batch, self._num_spread_symbols, self._fft_size, num_users
        ).permute(0, 3, 1, 2).reshape(batch, num_users, 1, self._num_data_symbols)
        llr = self._demapper(means, variances)
        self.last_llr = llr
        return llr


class DftSOfdmMmsePicDetector(DftSOfdmMimoDetector):
    """Iterative frequency-domain PIC with soft time-domain symbol feedback."""

    def __init__(
        self,
        transmitter: PUSCHTransmitter,
        stream_management: StreamManagement,
        *,
        num_iterations: int = 4,
        damping: float = 0.25,
    ) -> None:
        super().__init__(transmitter, stream_management, method="lmmse")
        if num_iterations < 1:
            raise ValueError("MMSE-PIC 至少需要一次软干扰抵消迭代")
        if not 0.0 < damping <= 1.0:
            raise ValueError("MMSE-PIC damping 必须位于 (0, 1]")
        self.num_iterations = num_iterations
        self.damping = damping
        bps = int(torch.as_tensor(self._num_bits_per_symbol).reshape(-1)[0])
        self._llrs_to_logits = LLRs2SymbolLogits(bps, device=transmitter.device)
        self._symbol_moments = SymbolLogits2Moments(
            constellation=self._constellation,
            device=transmitter.device,
        )

    def forward(
        self,
        y: torch.Tensor,
        h_hat: torch.Tensor,
        err_var: torch.Tensor,
        no: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_rx, num_rx_ant, num_symbols, fft_size = y.shape
        if num_rx != 1 or fft_size != self._fft_size:
            raise ValueError("MMSE-PIC DFT-s-OFDM 仅支持单接收端口和当前资源网格尺寸")
        users = h_hat.shape[3]
        bps = int(torch.as_tensor(self._num_bits_per_symbol).reshape(-1)[0])
        data_symbols = self._data_symbol_indices

        # Iteration zero: Sionna's joint per-subcarrier LMMSE equalizer,
        # inverse DFT, and soft QAM demapping.
        llr = super().forward(y, h_hat, err_var, no)

        y_eff = self._detector._removed_nulled_scs(y)
        y_data = y_eff[:, 0].index_select(2, data_symbols).permute(0, 2, 3, 1)
        h_data = torch.broadcast_to(h_hat, h_hat.shape)[:, 0, :, :, 0]
        h_data = h_data.index_select(3, data_symbols).permute(0, 3, 4, 1, 2)
        err_data = torch.broadcast_to(err_var, h_hat.shape)[:, 0, :, :, 0]
        err_data = err_data.index_select(3, data_symbols).permute(0, 3, 4, 1, 2)
        no = torch.as_tensor(no, dtype=y.real.dtype, device=y.device)
        if no.ndim == 3 and no.shape[1] == 1:
            no = no.squeeze(1)
        no = torch.broadcast_to(no, (batch, num_rx_ant))
        thermal_noise = no[:, None, None, :].expand(
            batch, self._num_spread_symbols, fft_size, num_rx_ant
        )

        for _ in range(self.num_iterations):
            bit_llrs = llr.reshape(
                batch, users, 1, self._num_data_symbols, bps
            )
            symbol_logits = self._llrs_to_logits(bit_llrs)
            soft_mean, soft_variance = self._symbol_moments(symbol_logits)
            soft_mean = soft_mean.reshape(
                batch, users, self._num_spread_symbols, fft_size
            )
            soft_variance = soft_variance.reshape(
                batch, users, self._num_spread_symbols, fft_size
            ).clamp_min(0.0)

            soft_frequency = torch.fft.fft(soft_mean, dim=-1, norm="ortho")
            variance_frequency = soft_variance.mean(dim=-1).permute(0, 2, 1)
            variance_frequency = variance_frequency.unsqueeze(2).expand(
                batch, self._num_spread_symbols, fft_size, users
            )

            # Parallel interference cancellation for every desired UE.
            h_soft = h_data * soft_frequency.permute(0, 2, 3, 1).unsqueeze(-2)
            sum_soft_interference = h_soft.sum(dim=-1)
            y_cancelled = (
                y_data.unsqueeze(-1)
                - sum_soft_interference.unsqueeze(-1)
                + h_data * soft_frequency.permute(0, 2, 3, 1).unsqueeze(-2)
            )
            second_moment = soft_mean.abs().square() + soft_variance
            second_moment = second_moment.mean(dim=-1).permute(0, 2, 1)
            csi_noise = (
                err_data * second_moment.unsqueeze(2).unsqueeze(3)
            ).sum(dim=-1)
            residual_interference = (
                h_data.abs().square() * variance_frequency.unsqueeze(-2)
            )
            residual_per_user = (
                thermal_noise.unsqueeze(-1)
                + csi_noise.unsqueeze(-1)
                + residual_interference.sum(dim=-1, keepdim=True)
                - residual_interference
            )
            channel_norm = h_data.abs().square().sum(dim=-2).clamp_min(1e-9)
            matched = (h_data.conj() * y_cancelled).sum(dim=-2)
            x_hat_fd = matched / channel_norm
            no_eff_fd = (
                h_data.abs().square() * residual_per_user
            ).sum(dim=-2) / channel_norm.square()

            x_hat_td = torch.fft.ifft(
                x_hat_fd.permute(0, 3, 1, 2), dim=-1, norm="ortho"
            )
            no_eff_td = no_eff_fd.permute(0, 3, 1, 2).mean(
                dim=-1, keepdim=True
            ).expand_as(x_hat_td.real)
            llr_new = self._demapper(
                x_hat_td.reshape(batch, users, 1, self._num_data_symbols),
                no_eff_td.reshape(batch, users, 1, self._num_data_symbols),
            )
            llr = (1.0 - self.damping) * llr + self.damping * llr_new

        self.last_llr = llr
        return llr


class DftSOfdmSoftMmsePicDetector(DftSOfdmMimoDetector):
    """Soft PIC with rate-matched LDPC extrinsic feedback between iterations."""

    def __init__(
        self,
        transmitter: PUSCHTransmitter,
        stream_management: StreamManagement,
        tb_decoder: TBDecoder,
        *,
        num_feedback_iterations: int = 1,
        damping: float = 0.25,
        num_decoder_iterations: int = 20,
        device: str | None = None,
    ) -> None:
        super().__init__(transmitter, stream_management, method="lmmse")
        if (
            isinstance(num_feedback_iterations, bool)
            or not isinstance(num_feedback_iterations, int)
            or num_feedback_iterations < 1
        ):
            raise ValueError("soft-mmse-pic 外反馈次数必须为正整数")
        if not 0.0 < damping <= 1.0:
            raise ValueError("soft-mmse-pic damping 必须位于 (0, 1]")
        self.num_feedback_iterations = num_feedback_iterations
        self.damping = damping
        self._feedback = _LdpcSoftFeedback(
            tb_decoder, num_decoder_iterations, device
        )
        bps = int(torch.as_tensor(self._num_bits_per_symbol).reshape(-1)[0])
        self._llrs_to_logits = LLRs2SymbolLogits(bps, device=transmitter.device)
        self._symbol_moments = SymbolLogits2Moments(
            constellation=self._constellation, device=transmitter.device
        )

    def forward(
        self,
        y: torch.Tensor,
        h_hat: torch.Tensor,
        err_var: torch.Tensor,
        no: torch.Tensor,
    ) -> torch.Tensor:
        batch, num_rx, num_rx_ant, _, fft_size = y.shape
        if num_rx != 1 or fft_size != self._fft_size:
            raise ValueError("soft-mmse-pic 仅支持单接收端口和当前资源网格尺寸")
        users = h_hat.shape[3]
        bps = int(torch.as_tensor(self._num_bits_per_symbol).reshape(-1)[0])
        data_symbols = self._data_symbol_indices
        llr = super().forward(y, h_hat, err_var, no)

        y_eff = self._detector._removed_nulled_scs(y)
        y_data = y_eff[:, 0].index_select(2, data_symbols).permute(0, 2, 3, 1)
        h_data = torch.broadcast_to(h_hat, h_hat.shape)[:, 0, :, :, 0]
        h_data = h_data.index_select(3, data_symbols).permute(0, 3, 4, 1, 2)
        err_data = torch.broadcast_to(err_var, h_hat.shape)[:, 0, :, :, 0]
        err_data = err_data.index_select(3, data_symbols).permute(0, 3, 4, 1, 2)
        no = torch.as_tensor(no, dtype=y.real.dtype, device=y.device)
        if no.ndim == 3 and no.shape[1] == 1:
            no = no.squeeze(1)
        no = torch.broadcast_to(no, (batch, num_rx_ant))
        thermal_noise = no[:, None, None, :].expand(
            batch, self._num_spread_symbols, fft_size, num_rx_ant
        )
        prior = torch.zeros_like(llr.reshape(batch, users, -1))

        for _ in range(self.num_feedback_iterations):
            extrinsic = self._feedback(llr.squeeze(2))
            prior = (
                (1.0 - self.damping) * prior + self.damping * extrinsic
            ).clamp(-20.0, 20.0)
            cancellation_llr = (llr.squeeze(2) + prior).clamp(-20.0, 20.0)
            symbol_logits = self._llrs_to_logits(
                cancellation_llr.reshape(
                    batch, users, 1, self._num_data_symbols, bps
                )
            )
            soft_mean, soft_variance = self._symbol_moments(symbol_logits)
            soft_mean = soft_mean.reshape(
                batch, users, self._num_spread_symbols, fft_size
            )
            soft_variance = soft_variance.reshape(
                batch, users, self._num_spread_symbols, fft_size
            ).clamp_min(0.0)
            soft_frequency = torch.fft.fft(soft_mean, dim=-1, norm="ortho")
            variance_frequency = soft_variance.mean(dim=-1).permute(0, 2, 1)
            variance_frequency = variance_frequency.unsqueeze(2).expand(
                batch, self._num_spread_symbols, fft_size, users
            )
            h_soft = h_data * soft_frequency.permute(0, 2, 3, 1).unsqueeze(-2)
            y_cancelled = (
                y_data.unsqueeze(-1) - h_soft.sum(dim=-1, keepdim=True) + h_soft
            )
            second_moment = (soft_mean.abs().square() + soft_variance)
            second_moment = second_moment.mean(dim=-1).permute(0, 2, 1)
            csi_noise = (
                err_data * second_moment.unsqueeze(2).unsqueeze(3)
            ).sum(dim=-1)
            residual_interference = (
                h_data.abs().square() * variance_frequency.unsqueeze(-2)
            )
            other_variance = (
                residual_interference.sum(dim=-1, keepdim=True)
                - residual_interference
            ).clamp_min(0.0)
            residual_noise = (
                thermal_noise.unsqueeze(-1)
                + csi_noise.unsqueeze(-1)
                + other_variance
            ).clamp_min(_MIN_NOISE_VARIANCE)

            weighted_channel = h_data.abs().square() / residual_noise
            gain = weighted_channel.sum(dim=-2).clamp_min(1e-9)
            estimate_fd = (
                h_data.conj() * y_cancelled / residual_noise
            ).sum(dim=-2) / gain
            variance_fd = 1.0 / gain
            estimate_td = torch.fft.ifft(
                estimate_fd.permute(0, 3, 1, 2), dim=-1, norm="ortho"
            )
            variance_td = variance_fd.permute(0, 3, 1, 2).mean(
                dim=-1, keepdim=True
            ).expand_as(estimate_td.real)
            llr = self._demapper(
                estimate_td.reshape(batch, users, 1, self._num_data_symbols),
                variance_td.reshape(batch, users, 1, self._num_data_symbols),
            )
            if not torch.isfinite(llr).all().item():
                raise RuntimeError("soft-mmse-pic produced non-finite detector LLRs")

        self.last_llr = llr
        return llr


class DftSOfdmLmmseSicDetector(DftSOfdmMimoDetector):
    """Decode users strongest-first and cancel CRC-verified reconstructions."""

    def __init__(
        self,
        transmitter: PUSCHTransmitter,
        stream_management: StreamManagement,
        settings: TxSettings,
        *,
        num_decoder_iterations: int,
        device: str | None,
        scrambling_sequences: torch.Tensor | None = None,
    ) -> None:
        super().__init__(transmitter, stream_management, method="lmmse")
        self._sic_decoder = TBDecoder(
            transmitter._tb_encoder,
            num_bp_iter=num_decoder_iterations,
            device=device,
        )
        self._reencoder = NrPuschTx(settings, device=device)
        self._num_users = len(settings.users)
        if scrambling_sequences is not None:
            sequences = scrambling_sequences.to(device=transmitter.device)
            num_code_bits = int(transmitter._tb_encoder.n)
            _use_explicit_scrambling(
                self._sic_decoder,
                sequences,
                self._num_users,
                num_code_bits,
            )
            _use_explicit_tx_scrambling(
                self._reencoder,
                sequences,
                self._num_users,
                num_code_bits,
            )
        self._transport_block_size = self._reencoder.transport_block_size
        self.last_user_order: list[int] = []
        self.last_crc_status: torch.Tensor | None = None

    def forward(
        self,
        y: torch.Tensor,
        h_hat: torch.Tensor,
        err_var: torch.Tensor,
        no: torch.Tensor,
    ) -> torch.Tensor:
        residual_y = y.clone()
        residual_h = h_hat.clone()
        residual_err = torch.broadcast_to(err_var, h_hat.shape).clone()
        llr_final = None

        # Strongest estimated channel is decoded first. Only a TB with passing
        # CRC is reconstructed and subtracted, avoiding SIC error propagation.
        user_power = h_hat.abs().square().sum(dim=(1, 2, 4, 5, 6)).mean(dim=0)
        user_order = torch.argsort(user_power, descending=True).tolist()
        self.last_user_order = user_order
        crc_by_user = torch.zeros(
            (y.shape[0], self._num_users), dtype=torch.bool, device=y.device
        )
        for user in user_order:
            llr = super().forward(residual_y, residual_h, residual_err, no)
            if llr_final is None:
                llr_final = torch.zeros_like(llr)
            llr_final[:, user] = llr[:, user]

            decoded_bits, crc_status = self._sic_decoder(llr)
            # Sionna may retain a singleton transport-block axis on CRC status
            # (for example [batch, user, 1]); flatten the selected UE to keep
            # cancellation masks aligned with the batch axis.
            cancel_mask = crc_status[:, user].reshape(-1)
            crc_by_user[:, user] = cancel_mask
            if not torch.any(cancel_mask).item():
                continue

            tx_bits = torch.zeros(
                (y.shape[0], self._num_users, self._transport_block_size),
                dtype=torch.float32,
                device=y.device,
            )
            decoded_user_bits = decoded_bits[:, user].reshape(
                y.shape[0], self._transport_block_size
            )
            tx_bits[:, user] = decoded_user_bits.to(dtype=tx_bits.dtype)
            reconstructed_grid = self._reencoder.generate(
                batch_size=y.shape[0], bits=tx_bits
            ).frequency_grid[:, user, 0]

            h_user = h_hat[:, :, :, user, 0, :, :]
            contribution = h_user * reconstructed_grid[:, None, None, :, :]
            mask_y = cancel_mask.reshape(-1, 1, 1, 1, 1)
            residual_y = residual_y - mask_y * contribution

            mask_h = cancel_mask.reshape(-1, 1, 1, 1, 1, 1)
            residual_h[:, :, :, user, :, :, :] *= ~mask_h
            residual_err[:, :, :, user, :, :, :] *= ~mask_h

        if llr_final is None:
            raise RuntimeError("SIC detector has no users to process")
        self.last_crc_status = crc_by_user
        self.last_llr = llr_final
        return llr_final


def _build_dmrs_frequency_basis(
    num_subcarriers: int,
    l_min: int,
    l_max: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Build the centered-FFT frequency response basis for discrete taps."""
    centered_bins = (
        torch.arange(num_subcarriers, dtype=torch.float32, device=device)
        - num_subcarriers // 2
    )
    lags = torch.arange(l_min, l_max + 1, dtype=torch.float32, device=device)
    return torch.exp(
        (-2j * torch.pi / num_subcarriers) * centered_bins[:, None] * lags[None, :]
    )



class DftSOfdmDmrsEstimator(torch.nn.Module):
    """Fit frequency-domain OCC DMRS to a finite-tap channel basis.

    Both time-IQ and frequency-grid receivers call this block after OFDM
    demodulation. Sionna's stock PUSCH LS estimator uses its native pilot
    sequence, while this transmitter maps a transform-precoded low-PAPR DMRS.
    The fitted taps are only interpolation parameters; input and output are
    both frequency-domain resource grids and CSI.
    """

    def __init__(
        self,
        pilot_grid: torch.Tensor,
        resource_grid,
        l_min: int,
        l_max: int,
        estimate_delay: bool = False,
        tap_power_prior: torch.Tensor | None = None,
    ):
        super().__init__()
        self._estimate_delay = estimate_delay
        self.last_channel_estimate: torch.Tensor | None = None
        self.last_offsets: torch.Tensor | None = None
        # [user, num_subcarriers] template from the actual DFT-s-OFDM mapper.
        if pilot_grid.ndim != 2 or pilot_grid.shape[0] != 4:
            raise ValueError("DMRS 模板必须为 [4 users, num_subcarriers]")
        self.register_buffer("_pilot_grid", pilot_grid)
        self._num_subcarriers = pilot_grid.shape[-1]
        self._num_ofdm_symbols = resource_grid.num_ofdm_symbols
        self._num_taps = l_max - l_min + 1
        if tap_power_prior is not None:
            tap_power_prior = torch.as_tensor(tap_power_prior, device=pilot_grid.device)
            if tap_power_prior.is_complex() or tuple(tap_power_prior.shape) != (self._num_taps,):
                raise ValueError("DMRS tap-power prior 必须是长度等于候选 taps 的实向量")
            tap_power_prior = tap_power_prior.to(dtype=pilot_grid.real.dtype)
            if not torch.isfinite(tap_power_prior).all().item() or not torch.all(tap_power_prior > 0).item():
                raise ValueError("DMRS tap-power prior 必须全部为正有限值")
        self.register_buffer("_tap_power_prior", tap_power_prior)
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
        frequency_basis = _build_dmrs_frequency_basis(
            self._num_subcarriers, l_min, l_max, device=pilot_grid.device
        )
        self.register_buffer("_frequency_basis", frequency_basis)
        self.register_buffer(
            "_frequency_bins",
            torch.arange(self._num_subcarriers, dtype=torch.float32, device=pilot_grid.device)
            - self._num_subcarriers // 2,
        )
        for support_tuple, users in support_groups.items():
            support = torch.tensor(support_tuple, dtype=torch.long, device=pilot_grid.device)
            if support.numel() < 2 * self._num_taps:
                raise ValueError(
                    "DMRS pilot RE 数量不足以联合拟合两个用户的候选信道 taps；"
                    "请减小 max_delay_spread_s 或增加 DMRS 资源"
                )
            base, partner = users
            ratio = pilot_grid[partner, support] / pilot_grid[base, support]
            ratio_pairs = ratio.reshape(-1, 2)
            if not torch.allclose(ratio_pairs[:, 0], torch.ones_like(ratio_pairs[:, 0])):
                raise ValueError("DMRS OCC pair must start with a +1 cover chip")
            if not torch.allclose(ratio_pairs[:, 1], -torch.ones_like(ratio_pairs[:, 1])):
                raise ValueError("DMRS OCC pair must alternate +1/-1 for despreading")
            pair_id = len(self._pairs)
            support_name = f"_support_{pair_id}"
            self.register_buffer(support_name, support)
            design = torch.cat(
                (
                    pilot_grid[base, support, None] * frequency_basis[support],
                    pilot_grid[partner, support, None] * frequency_basis[support],
                ),
                dim=-1,
            )
            covariance = torch.linalg.pinv(design.mH @ design)
            design_name = f"_design_{pair_id}"
            covariance_name = f"_covariance_{pair_id}"
            self.register_buffer(design_name, design)
            self.register_buffer(covariance_name, covariance)
            self._pairs.append((base, partner, design_name, support_name, covariance_name))


    def _fit_pair(
        self, design: torch.Tensor, rhs: torch.Tensor, no: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Fit two OCC users and return taps plus optional posterior covariance."""
        batch, num_rx_ant = no.shape[0], no.shape[-1]
        num_coefficients = 2 * self._num_taps
        if self._tap_power_prior is None:
            taps = torch.linalg.lstsq(design, rhs).solution
            return (
                taps.reshape(num_coefficients, batch, num_rx_ant).permute(1, 2, 0),
                None,
            )

        # A R Aᴴ has a large null space when the pilot count exceeds the
        # candidate tap count. At high SNR its regularizing σ²I eigenvalues
        # approach float32 precision, so solve the posterior in complex128.
        work_dtype = torch.complex128 if design.dtype == torch.complex64 else design.dtype
        work_design = design.to(dtype=work_dtype)
        work_rhs = rhs.to(dtype=work_dtype)
        sigma2 = no[:, 0, :].reshape(-1).to(dtype=work_design.real.dtype)
        if not torch.isfinite(sigma2).all().item() or torch.any(sigma2 < 0).item():
            raise ValueError("DMRS LMMSE noise variance 必须为非负有限值")
        tap_power = torch.cat((self._tap_power_prior, self._tap_power_prior)).to(
            dtype=work_design.real.dtype, device=design.device
        )
        tap_power_complex = tap_power.to(dtype=work_dtype)
        a_r = work_design * tap_power_complex[None, :]
        a_r_ah = a_r @ work_design.mH
        identity = torch.eye(
            design.shape[0], dtype=work_dtype, device=design.device
        )
        taps_flat = torch.zeros(
            (num_coefficients, batch * num_rx_ant),
            dtype=work_dtype,
            device=design.device,
        )
        posterior_flat = torch.zeros(
            (batch * num_rx_ant, num_coefficients, num_coefficients),
            dtype=work_dtype,
            device=design.device,
        )
        zero_indices = torch.where(sigma2 == 0)[0]
        if zero_indices.numel():
            taps_flat[:, zero_indices] = torch.linalg.lstsq(
                work_design, work_rhs.index_select(1, zero_indices)
            ).solution
        noisy_indices = torch.where(sigma2 > 0)[0]
        if noisy_indices.numel():
            sigma2_noisy = sigma2.index_select(0, noisy_indices)
            system = a_r_ah.unsqueeze(0) + sigma2_noisy[:, None, None] * identity
            rhs_noisy = (
                work_rhs.index_select(1, noisy_indices).transpose(0, 1).unsqueeze(-1)
            )
            solved_rhs = torch.linalg.solve(system, rhs_noisy)
            a_r_h = a_r.mH
            taps_flat[:, noisy_indices] = torch.matmul(
                a_r_h.unsqueeze(0), solved_rhs
            ).squeeze(-1).transpose(0, 1)
            solved_a_r = torch.linalg.solve(
                system, a_r.unsqueeze(0).expand(noisy_indices.numel(), -1, -1)
            )
            posterior = torch.diag(tap_power_complex).unsqueeze(0) - torch.matmul(
                a_r_h.unsqueeze(0), solved_a_r
            )
            posterior_flat[noisy_indices] = posterior
        return (
            taps_flat.reshape(num_coefficients, batch, num_rx_ant)
            .permute(1, 2, 0)
            .to(dtype=design.dtype),
            posterior_flat.reshape(batch, num_rx_ant, num_coefficients, num_coefficients)
            .to(dtype=design.dtype),
        )

    @staticmethod
    def _estimate_offsets(h_freq: torch.Tensor) -> torch.Tensor:
        """Per-antenna bulk delay from an FFT peak with parabolic refinement.

        ``h_freq`` is ``[batch, rx_antenna, subcarrier]``. The band is mapped to
        the delay domain, the strongest tap is located, and the samples around
        it are interpolated for sub-sample accuracy. The returned delay is in
        samples and is signed around the band centre.
        """
        num_subcarriers = h_freq.shape[-1]
        power = torch.fft.fft(h_freq, n=num_subcarriers, dim=-1).abs().square()
        index = power.argmax(dim=-1)
        left = power.gather(-1, ((index - 1) % num_subcarriers).unsqueeze(-1)).squeeze(-1)
        center = power.gather(-1, index.unsqueeze(-1)).squeeze(-1)
        right = power.gather(-1, ((index + 1) % num_subcarriers).unsqueeze(-1)).squeeze(-1)
        curvature = left - 2 * center + right
        refine = torch.where(
            curvature.abs() > 0,
            0.5 * (left - right) / curvature,
            torch.zeros_like(curvature),
        ).clamp(-0.5, 0.5)
        estimate = index.to(h_freq.real.dtype) + refine
        return torch.where(
            estimate > num_subcarriers / 2, estimate - num_subcarriers, estimate
        )

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
        self.last_offsets = None
        offsets = (
            torch.empty((batch, 4, num_rx_ant), dtype=y.real.dtype, device=y.device)
            if self._estimate_delay
            else None
        )
        y_pilot = y[:, 0, :, self._dmrs_symbol, :]
        for base, partner, design_name, support_name, covariance_name in self._pairs:
            design = getattr(self, design_name).to(y.device)
            support = getattr(self, support_name).to(y.device)
            covariance = getattr(self, covariance_name).to(y.device)
            observations = y_pilot.index_select(-1, support)
            rhs = observations.permute(2, 0, 1).reshape(support.numel(), -1)
            taps, posterior = self._fit_pair(design, rhs, no)
            for user, tap_slice in (
                (base, slice(0, self._num_taps)),
                (partner, slice(self._num_taps, 2 * self._num_taps)),
            ):
                user_taps = taps[..., tap_slice]
                h_freq = user_taps @ self._frequency_basis.to(y.device).T
                if offsets is not None:
                    offsets[:, user] = self._estimate_offsets(h_freq)
                frequency_basis = self._frequency_basis.to(y.device)
                if posterior is None:
                    covariance_user = covariance[tap_slice, tap_slice]
                    frequency_error = torch.einsum(
                        "nl,lm,nm->n", frequency_basis, covariance_user,
                        frequency_basis.conj()
                    ).real.clamp_min(0.0)
                    estimation_error = no[:, 0, :, None] * frequency_error[None, None, :]
                else:
                    covariance_user = posterior[:, :, tap_slice, tap_slice]
                    estimation_error = torch.einsum(
                        "nl,balm,nm->ban", frequency_basis, covariance_user,
                        frequency_basis.conj()
                    ).real.clamp_min(0.0)
                h_hat[:, 0, :, user, 0, :, :] = h_freq.unsqueeze(-2).expand(
                    -1, -1, self._num_ofdm_symbols, -1
                )
                estimation_error = estimation_error.unsqueeze(-2).expand(
                    -1, -1, self._num_ofdm_symbols, -1
                )
                err_var[:, 0, :, user, 0, :, :] = estimation_error
        self.last_channel_estimate = h_hat.detach()
        self.last_offsets = offsets
        return h_hat, err_var


@dataclass
class RxResult:
    bits: torch.Tensor  # [batch, user, transport_block_bit]
    crc_status: torch.Tensor  # [batch, user], True means CRC pass
    constellation: torch.Tensor  # [batch, user, data_symbol], soft QAM estimates
    metadata: dict[str, Any]


class _CbCrcProbe(torch.nn.Module):
    """Record the per-code-block CRC verdict the TB decoder otherwise drops."""

    def __init__(self, decoder: torch.nn.Module, sink: list[torch.Tensor]) -> None:
        super().__init__()
        self._decoder = decoder
        self._sink = sink

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self._decoder(inputs)
        self._sink.append(output[1].detach())
        return output


class NrPuschRx:
    """Decode time-domain captures or simulated samples for four PUSCH users.

    ``channel_estimator="dmrs"`` estimates the initial static-slot profile
    with OCC least squares. ``"dmrs-lmmse"`` applies a CDL tap-power prior;
    ``"perfect"`` consumes simulated CDL CSI as an upper bound.
    """

    def __init__(
        self,
        settings: TxSettings,
        *,
        channel_estimator: str = "dmrs",
        dmrs_tap_power_prior: torch.Tensor | None = None,
        l_min: int = -6,
        max_delay_spread_s: float = 3e-6,
        num_decoder_iterations: int = 20,
        estimate_delay: bool = False,
        track_cb_crc: bool = False,
        detector: str = "lmmse",
        detector_parameter: int | None = None,
        detector_damping: float = 0.25,
        input_domain: str = "time",
        device: str | None = None,
        scrambling_sequences: torch.Tensor | None = None,
    ) -> None:
        settings.validate()
        if settings.pusch.waveform != "dft_s_ofdm":
            raise NotImplementedError("当前接收机仅支持仓库配置的 DFT-s-OFDM PUSCH")
        if channel_estimator not in {"perfect", "dmrs", "dmrs-lmmse"}:
            raise ValueError("channel_estimator 仅支持 perfect、dmrs 或 dmrs-lmmse")
        if channel_estimator == "dmrs-lmmse" and dmrs_tap_power_prior is None:
            raise ValueError("dmrs-lmmse 模式必须提供 dmrs_tap_power_prior")
        if channel_estimator != "dmrs-lmmse" and dmrs_tap_power_prior is not None:
            raise ValueError("dmrs_tap_power_prior 仅能用于 dmrs-lmmse 模式")
        if input_domain not in {"time", "frequency"}:
            raise ValueError("input_domain 仅支持 time 或 frequency")
        if num_decoder_iterations < 1:
            raise ValueError("num_decoder_iterations 必须大于 0")
        if max_delay_spread_s <= 0:
            raise ValueError("max_delay_spread_s 必须大于 0")
        if not 0.0 < detector_damping <= 1.0:
            raise ValueError("detector_damping 必须位于 (0, 1]")
        self.settings = settings
        device = use_device(device)
        self.device = device
        self.input_domain = input_domain
        self.l_min = l_min
        tx = PUSCHTransmitter(
            settings.to_sionna_configs(),
            return_bits=False,
            output_domain="freq",
            device=device,
        )
        self.sample_rate_hz = int(tx.resource_grid.fft_size * tx.resource_grid.subcarrier_spacing)
        self._num_ofdm_symbols = tx.resource_grid.num_ofdm_symbols
        self._num_bits_per_symbol = int(torch.as_tensor(tx._num_bits_per_symbol).reshape(-1)[0])
        self._fft_size = tx.resource_grid.fft_size
        _, l_max = time_lag_discrete_time_channel(self.sample_rate_hz, max_delay_spread_s)
        stream_management = StreamManagement(np.ones((1, len(settings.users)), dtype=bool), 1)
        tb_decoder = TBDecoder(
            tx._tb_encoder,
            num_bp_iter=num_decoder_iterations,
            device=device,
        )
        if scrambling_sequences is not None:
            _use_explicit_scrambling(
                tb_decoder,
                scrambling_sequences.to(device=device),
                len(settings.users),
                int(tx._tb_encoder.n),
            )
        self._cb_crc_status: list[torch.Tensor] = []
        if detector == "soft-mmse-pic":
            detector_block = DftSOfdmSoftMmsePicDetector(
                tx,
                stream_management,
                tb_decoder,
                num_feedback_iterations=1 if detector_parameter is None else detector_parameter,
                damping=detector_damping,
                num_decoder_iterations=num_decoder_iterations,
                device=device,
            )
        elif detector == "lmmse-sic":
            detector_block = DftSOfdmLmmseSicDetector(
                tx,
                stream_management,
                settings,
                num_decoder_iterations=num_decoder_iterations,
                device=device,
                scrambling_sequences=scrambling_sequences,
            )
        elif detector == "k-best":
            detector_block = DftSOfdmKBestDetector(
                tx, stream_management, k=detector_parameter
            )
        elif detector == "mmse-pic":
            detector_block = DftSOfdmMmsePicDetector(
                tx,
                stream_management,
                num_iterations=4 if detector_parameter is None else detector_parameter,
                damping=detector_damping,
            )
        elif detector == "ep":
            detector_block = DftSOfdmEpDetector(
                tx,
                stream_management,
                iterations=10 if detector_parameter is None else detector_parameter,
                damping=detector_damping,
            )
        else:
            detector_block = DftSOfdmMimoDetector(
                tx, stream_management, detector, detector_parameter
            )
        if channel_estimator in {"dmrs", "dmrs-lmmse"}:
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
                estimate_delay=estimate_delay,
                tap_power_prior=dmrs_tap_power_prior,
            )
        else:
            estimator = "perfect"
        if track_cb_crc and tb_decoder._cb_crc_decoder is not None:
            tb_decoder._cb_crc_decoder = _CbCrcProbe(
                tb_decoder._cb_crc_decoder, self._cb_crc_status
            )
        self._receiver = PUSCHReceiver(
            tx,
            channel_estimator=estimator,
            mimo_detector=detector_block,
            tb_decoder=tb_decoder,
            return_tb_crc_status=True,
            stream_management=stream_management,
            input_domain="time" if input_domain == "time" else "freq",
            l_min=l_min if input_domain == "time" else None,
            device=device,
        )
        self.channel_estimator = channel_estimator
        self.detector = detector
        self.detector_parameter = detector_parameter
        if detector == "k-best" and detector_parameter is None:
            self.detector_parameter = 64
        elif detector == "mmse-pic" and detector_parameter is None:
            self.detector_parameter = 4
        elif detector == "soft-mmse-pic" and detector_parameter is None:
            self.detector_parameter = 1
        elif detector == "ep" and detector_parameter is None:
            self.detector_parameter = 10
        self.detector_damping = detector_damping
        self.estimate_delay = estimate_delay
        self._estimator = estimator
        self.track_cb_crc = track_cb_crc


    def receive(
        self,
        iq: torch.Tensor,
        noise_variance: torch.Tensor | float,
        *,
        channel_taps: torch.Tensor | None = None,
    ) -> RxResult:
        """Decode `[batch, rx_antenna, sample]` received complex IQ samples."""
        if self.input_domain != "time":
            raise ValueError("该接收机使用 frequency 输入；请调用 receive_frequency_grid")
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
        return self._decode(y, no, h)

    def receive_frequency_grid(
        self,
        grid: torch.Tensor,
        noise_variance: torch.Tensor | float,
        *,
        channel_frequency_response: torch.Tensor | None = None,
    ) -> RxResult:
        """Decode ``[batch, 1, rx_ant, OFDM symbol, fft_bin]`` receive grids."""
        if self.input_domain != "frequency":
            raise ValueError("该接收机使用 time 输入；请调用 receive")
        if (
            grid.ndim != 5
            or grid.shape[1:3] != (1, 4)
            or grid.shape[-2:] != (self._num_ofdm_symbols, self._fft_size)
            or not grid.is_complex()
        ):
            raise ValueError(
                "grid 形状必须为复数 [batch, 1, 4 rx_antennas, configured_symbols, fft_size]"
            )
        if self.channel_estimator == "perfect":
            expected = (
                grid.shape[0], 1, 4, len(self.settings.users), 1,
                self._num_ofdm_symbols, self._fft_size,
            )
            if channel_frequency_response is None:
                raise ValueError("perfect CSI 频域模式要求提供 channel_frequency_response")
            if tuple(channel_frequency_response.shape) != expected:
                raise ValueError(
                    f"channel_frequency_response 形状应为 {expected}，"
                    f"实际为 {tuple(channel_frequency_response.shape)}"
                )
            h = channel_frequency_response
        else:
            h = None
        y = grid
        no = torch.as_tensor(noise_variance, dtype=grid.real.dtype, device=grid.device)
        if self.device is not None:
            y = y.to(self.device)
            no = no.to(self.device)
            if h is not None:
                h = h.to(self.device)
        return self._decode(y, no, h)

    def _decode(
        self,
        y: torch.Tensor,
        no: torch.Tensor,
        h: torch.Tensor | None,
    ) -> RxResult:
        # Sionna's LMMSE equalizer whitens by the inverse noise covariance.
        # An exactly noiseless capture with a perfect LS estimate gives a zero
        # covariance, which produces NaN LLRs and can decode as an all-zero TB.
        # Keep zero-noise captures numerically well-defined with a tiny floor.
        no = no.clamp_min(_MIN_NOISE_VARIANCE)
        bits, crc_status = self._receiver(y, no, h)
        mimo_detector = self._receiver._mimo_detector
        llr = getattr(mimo_detector, "last_llr", None)
        if llr is None:
            raise RuntimeError("MIMO detector did not retain its soft outputs for analysis")
        if not torch.isfinite(llr).all().item():
            raise RuntimeError("MIMO detector produced non-finite LLRs; CRC status is not trustworthy")
        symbol_logits = LLRs2SymbolLogits(
            self._num_bits_per_symbol, device=self.device
        )(llr.reshape(*llr.shape[:-1], -1, self._num_bits_per_symbol))
        constellation, _ = SymbolLogits2Moments(
            constellation=mimo_detector._constellation, device=self.device
        )(symbol_logits)
        constellation = constellation.squeeze(2)
        cb_crc_status = None
        if self.track_cb_crc and self._cb_crc_status:
            latest = self._cb_crc_status[-1]
            # The CRC decoder returns one flag per code block with a trailing
            # singleton axis; flatten it so each entry is a plain boolean.
            cb_crc_status = [
                latest[0, user].reshape(-1).detach().cpu().tolist()
                for user in range(len(self.settings.users))
            ]
            self._cb_crc_status.clear()
        delay_metadata = {}
        if (
            self.channel_estimator in {"dmrs", "dmrs-lmmse"}
            and getattr(self._estimator, "last_offsets", None) is not None
        ):
            offsets = self._estimator.last_offsets[0].detach().cpu().tolist()
            delay_metadata["estimated_bulk_delay_samples"] = offsets[-1]
            delay_metadata["estimated_bulk_delay_samples_by_user"] = {
                user.name: offsets[index]
                for index, user in enumerate(self.settings.users)
            }
        sic_metadata = {}
        if self.detector == "lmmse-sic":
            sic_detector = self._receiver._mimo_detector
            sic_metadata = {
                "sic_user_order": [self.settings.users[i].name for i in sic_detector.last_user_order],
                "sic_crc_before_cancel": (
                    sic_detector.last_crc_status.detach().cpu().tolist()
                    if sic_detector.last_crc_status is not None else None
                ),
            }
        metadata = {
            "waveform": self.settings.pusch.waveform,
            "input_domain": self.input_domain,
            "detector": self.detector,
            "detector_parameter": self.detector_parameter,
            "detector_damping": self.detector_damping if self.detector in {"mmse-pic", "soft-mmse-pic", "ep"} else None,
            "detector_feedback_iterations": (
                self._receiver._mimo_detector.num_feedback_iterations
                if self.detector == "soft-mmse-pic" else None
            ),
            "detector_note": (
                "Strongest-first LMMSE-SIC; cancel only after the user's TB CRC passes."
                if self.detector == "lmmse-sic"
                else "LMMSE frequency pre-equalization, IDFT despreading, and per-sample spatial K-best."
                if self.detector == "k-best"
                else "Frequency-domain LMMSE equalization followed by inverse DFT spreading."
                if self.detector == "lmmse"
                else "LDPC extrinsic-feedback soft-MMSE-PIC with frequency-domain cancellation."
                if self.detector == "soft-mmse-pic"
                else "Iterative time-domain soft-symbol PIC with frequency-domain cancellation."
                if self.detector == "mmse-pic"
                else "Frequency LMMSE pre-equalization, IDFT despreading, then damped time-domain spatial EP with Gaussian residual-ISI covariance."
            ),
            "decoder": "Sionna NR TBDecoder",
            "cb_crc_status": cb_crc_status,
            "delay_estimation": self.estimate_delay,
            "channel_estimator": (
                "perfect CDL CSI"
                if self.channel_estimator == "perfect"
                else "DFT-s-OFDM frequency DMRS OCC tap-domain LMMSE"
                if self.channel_estimator == "dmrs-lmmse"
                else "DFT-s-OFDM frequency DMRS OCC LS tap fit"
            ),
            "crc_status_axes": ["batch", "user"],
            "bits_axes": ["batch", "user", "transport_block_bit"],
            "constellation_axes": ["batch", "user", "qam_symbol"],
            "sample_rate_hz": self.sample_rate_hz,
            **delay_metadata,
            **sic_metadata,
        }
        return RxResult(
            bits=bits,
            crc_status=crc_status,
            constellation=constellation,
            metadata=metadata,
        )
