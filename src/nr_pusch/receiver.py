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
from sionna.phy.mimo import KBestDetector as FlatKBestDetector
from sionna.phy.mimo import StreamManagement, lmmse_matrix, whiten_channel
from sionna.phy.nr import PUSCHReceiver, PUSCHTransmitter, TBDecoder
from sionna.phy.ofdm import (
    LMMSEEqualizer,
    LinearDetector,
)

from .config import TxSettings
from .transmitter import NrPuschTx


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
        return self._demapper(x_hat, no_eff)


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
        return self._demapper(symbols, post_detection_variance)


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
        return self._demapper(means, variances)


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
    ) -> None:
        super().__init__(transmitter, stream_management, method="lmmse")
        self._sic_decoder = TBDecoder(
            transmitter._tb_encoder,
            num_bp_iter=num_decoder_iterations,
            device=device,
        )
        self._reencoder = NrPuschTx(settings, device=device)
        self._num_users = len(settings.users)
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
            cancel_mask = crc_status[:, user]
            crc_by_user[:, user] = cancel_mask
            if not torch.any(cancel_mask).item():
                continue

            tx_bits = torch.zeros(
                (y.shape[0], self._num_users, self._transport_block_size),
                dtype=torch.float32,
                device=y.device,
            )
            tx_bits[:, user] = decoded_bits[:, user].to(dtype=tx_bits.dtype)
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
        return llr_final


class DftSOfdmDmrsEstimator(torch.nn.Module):
    """Despread OCC DMRS and interpolate the channel across frequency."""

    def __init__(
        self,
        pilot_grid: torch.Tensor,
        resource_grid,
        l_min: int,
        l_max: int,
        mode: str = "time",
    ):
        super().__init__()
        if mode not in {"time", "frequency"}:
            raise ValueError("DMRS estimator mode must be time or frequency")
        self.mode = mode
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
        self._frequency_pairs: list[tuple[int, int, str, str, str]] = []
        target_bins = torch.arange(self._num_subcarriers, dtype=torch.float32, device=pilot_grid.device)
        if mode == "time":
            q = target_bins - self._num_subcarriers // 2
            lags = torch.arange(l_min, l_max + 1, dtype=torch.float32, device=pilot_grid.device)
            frequency_basis = torch.exp(
                (-2j * torch.pi / self._num_subcarriers) * q[:, None] * lags[None, :]
            )
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
            pair_id = len(self._pairs) + len(self._frequency_pairs)
            support_name = f"_support_{pair_id}"
            self.register_buffer(support_name, support)
            if mode == "time":
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
            else:
                pilot_locations = (support[0::2] + support[1::2]).to(torch.float32) * 0.5
                right = torch.searchsorted(pilot_locations, target_bins).clamp(0, pilot_locations.numel() - 1)
                left = (right - 1).clamp(0, pilot_locations.numel() - 1)
                same = left == right
                denominator = (pilot_locations[right] - pilot_locations[left]).clamp_min(1.0)
                alpha = torch.where(
                    same,
                    torch.zeros_like(target_bins),
                    (target_bins - pilot_locations[left]) / denominator,
                )
                interpolation = torch.zeros(
                    (self._num_subcarriers, pilot_locations.numel()),
                    dtype=torch.float32,
                    device=pilot_grid.device,
                )
                interpolation.scatter_add_(1, left[:, None], (1.0 - alpha)[:, None])
                interpolation.scatter_add_(1, right[:, None], alpha[:, None])
                pilot_name = f"_base_pilot_{pair_id}"
                interpolation_name = f"_interpolation_{pair_id}"
                self.register_buffer(pilot_name, pilot_grid[base, support])
                self.register_buffer(interpolation_name, interpolation)
                self._frequency_pairs.append(
                    (base, partner, support_name, pilot_name, interpolation_name)
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
        y_pilot = y[:, 0, :, self._dmrs_symbol, :]
        if self.mode == "time":
            for base, partner, design_name, support_name, covariance_name in self._pairs:
                design = getattr(self, design_name).to(y.device)
                support = getattr(self, support_name).to(y.device)
                covariance = getattr(self, covariance_name).to(y.device)
                observations = y_pilot.index_select(-1, support)
                rhs = observations.permute(2, 0, 1).reshape(support.numel(), -1)
                taps = torch.linalg.lstsq(design, rhs).solution
                taps = taps.reshape(2 * self._num_taps, batch, num_rx_ant).permute(1, 2, 0)
                for user, tap_slice in (
                    (base, slice(0, self._num_taps)),
                    (partner, slice(self._num_taps, 2 * self._num_taps)),
                ):
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

        for base, partner, support_name, pilot_name, interpolation_name in self._frequency_pairs:
            support = getattr(self, support_name).to(y.device)
            base_pilot = getattr(self, pilot_name).to(y.device)
            interpolation = getattr(self, interpolation_name).to(y.device)
            observations = y_pilot.index_select(-1, support)
            despread = observations / base_pilot[None, None, :]
            first = despread[..., 0::2]
            second = despread[..., 1::2]
            pilot_estimates = ((first + second) * 0.5, (first - second) * 0.5)
            pilot_noise = no[:, 0, :, None] * 0.25 * (
                base_pilot[0::2].abs().square().reciprocal()
                + base_pilot[1::2].abs().square().reciprocal()
            )[None, None, :]
            interpolated_variance = torch.einsum(
                "brp,fp->brf", pilot_noise, interpolation.square()
            )
            for user, pilot_estimate in ((base, pilot_estimates[0]), (partner, pilot_estimates[1])):
                h_freq = torch.einsum("brp,fp->brf", pilot_estimate, interpolation.to(y.dtype))
                h_hat[:, 0, :, user, 0, :, :] = h_freq.unsqueeze(-2).expand(
                    -1, -1, self._num_ofdm_symbols, -1
                )
                err_var[:, 0, :, user, 0, :, :] = interpolated_variance.unsqueeze(-2).expand(
                    -1, -1, self._num_ofdm_symbols, -1
                )
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
        detector: str = "lmmse",
        detector_parameter: int | None = None,
        detector_damping: float = 0.25,
        input_domain: str = "time",
        device: str | None = None,
    ) -> None:
        settings.validate()
        if settings.pusch.waveform != "dft_s_ofdm":
            raise NotImplementedError("当前接收机仅支持仓库配置的 DFT-s-OFDM PUSCH")
        if channel_estimator not in {"perfect", "dmrs"}:
            raise ValueError("channel_estimator 仅支持 perfect 或 dmrs")
        if input_domain not in {"time", "frequency"}:
            raise ValueError("input_domain 仅支持 time 或 frequency")
        if num_decoder_iterations < 1:
            raise ValueError("num_decoder_iterations 必须大于 0")
        if max_delay_spread_s <= 0:
            raise ValueError("max_delay_spread_s 必须大于 0")
        if not 0.0 < detector_damping <= 1.0:
            raise ValueError("detector_damping 必须位于 (0, 1]")

        self.settings = settings
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
        self._fft_size = tx.resource_grid.fft_size
        _, l_max = time_lag_discrete_time_channel(self.sample_rate_hz, max_delay_spread_s)
        stream_management = StreamManagement(np.ones((1, len(settings.users)), dtype=bool), 1)
        if detector == "lmmse-sic":
            detector_block = DftSOfdmLmmseSicDetector(
                tx,
                stream_management,
                settings,
                num_decoder_iterations=num_decoder_iterations,
                device=device,
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
                mode=input_domain,
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
        elif detector == "ep" and detector_parameter is None:
            self.detector_parameter = 10
        self.detector_damping = detector_damping

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
        bits, crc_status = self._receiver(y, no, h)
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
            "detector_damping": self.detector_damping if self.detector in {"mmse-pic", "ep"} else None,
            "detector_note": (
                "Strongest-first LMMSE-SIC; cancel only after the user's TB CRC passes."
                if self.detector == "lmmse-sic"
                else "LMMSE frequency pre-equalization, IDFT despreading, and per-sample spatial K-best."
                if self.detector == "k-best"
                else "Frequency-domain LMMSE equalization followed by inverse DFT spreading."
                if self.detector == "lmmse"
                else "Iterative time-domain soft-symbol PIC with frequency-domain cancellation."
                if self.detector == "mmse-pic"
                else "Frequency LMMSE pre-equalization, IDFT despreading, then damped time-domain spatial EP with Gaussian residual-ISI covariance."
            ),
            "decoder": "Sionna NR TBDecoder",
            "channel_estimator": "perfect CDL CSI" if self.channel_estimator == "perfect" else "DFT-s-OFDM DMRS OCC LS",
            "crc_status_axes": ["batch", "user"],
            "bits_axes": ["batch", "user", "transport_block_bit"],
            "sample_rate_hz": self.sample_rate_hz,
            **sic_metadata,
        }
        return RxResult(bits=bits, crc_status=crc_status, metadata=metadata)
