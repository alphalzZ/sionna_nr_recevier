"""Readers for MATLAB transmitter and receiver HDF5 reference data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class MatlabTxReference:
    """Reference arrays normalized to user-first axes.

    ``frequency_grid`` has shape ``[user, ofdm_symbol, layer, subcarrier]``.
    ``bits`` has shape ``[user, transport_block_bit]``.
    """

    users: tuple[str, ...]
    frequency_grid: np.ndarray
    bits: np.ndarray
    source: Path


@dataclass(frozen=True)
class MatlabRxReference:
    """Receive frequency grid normalized to ``[rx, symbol, subcarrier]`` axes."""

    frequency_grid: np.ndarray
    source: Path
    data_grid: np.ndarray | None = None
    pilot_grid: np.ndarray | None = None


def read_matlab_rx_reference(path: str | Path, num_rx_antennas: int = 4) -> MatlabRxReference:
    """Read a MATLAB ``FreqData/IQdataPdu`` receive grid.

    The split real/imag datasets are stored in MATLAB-facing
    ``[ofdm_symbol, rx_antenna, active_subcarrier]`` order. The returned main
    grid is transposed to ``[rx_antenna, ofdm_symbol, active_subcarrier]`` for
    direct use by :class:`NrPuschRx` after batch and stream axes are added.
    Optional ``data_*`` and ``pilot_*`` datasets are included when both pairs
    exist, so analysis code can inspect the separated data and DMRS arrays.
    MATLAB stores data in ``[symbol, rx, subcarrier]`` order but the supplied
    pilot dataset in ``[rx, subcarrier]`` order.
    """
    source = Path(path)
    grid = _read_split_complex(
        source,
        "FreqData/IQdataPdu_real",
        "FreqData/IQdataPdu_imag",
        ndim=3,
        label="IQdataPdu",
    )
    if grid.shape[1] != num_rx_antennas:
        raise ValueError(
            f"IQdataPdu 接收天线轴应为 {num_rx_antennas}，实际为 {grid.shape[1]}"
        )

    data_real_key, data_imag_key = "FreqData/data_real", "FreqData/data_imag"
    pilot_real_key, pilot_imag_key = "FreqData/pilot_real", "FreqData/pilot_imag"
    data_present = data_real_key in _dataset_names(source) and data_imag_key in _dataset_names(source)
    pilot_present = pilot_real_key in _dataset_names(source) and pilot_imag_key in _dataset_names(source)
    data_grid = (
        _read_split_complex(source, data_real_key, data_imag_key, ndim=3, label="data").transpose(1, 0, 2)
        if data_present else None
    )
    pilot_grid = (
        _read_split_complex(source, pilot_real_key, pilot_imag_key, ndim=2, label="pilot")
        if pilot_present else None
    )
    if data_grid is not None and data_grid.shape[0] != num_rx_antennas:
        raise ValueError("FreqData/data 的接收天线数量与 IQdataPdu 不一致")
    if pilot_grid is not None and pilot_grid.shape[0] != num_rx_antennas:
        raise ValueError("FreqData/pilot 的接收天线数量与 IQdataPdu 不一致")

    return MatlabRxReference(
        frequency_grid=grid.transpose(1, 0, 2),
        source=source,
        data_grid=data_grid,
        pilot_grid=pilot_grid,
    )


def _dataset_names(source: Path) -> set[str]:
    names: set[str] = set()
    with h5py.File(source, "r") as h5:
        h5.visit(names.add)
    return names


def _read_split_complex(
    source: Path,
    real_key: str,
    imag_key: str,
    *,
    ndim: int,
    label: str,
) -> np.ndarray:
    with h5py.File(source, "r") as h5:
        missing = [key for key in (real_key, imag_key) if key not in h5]
        if missing:
            raise ValueError(f"{source} 缺少数据集: {', '.join(missing)}")
        real = np.asarray(h5[real_key], dtype=np.float32)
        imag = np.asarray(h5[imag_key], dtype=np.float32)
    if real.shape != imag.shape or real.ndim != ndim:
        raise ValueError(f"{label} real/imag 应为形状相同的 {ndim} 维数组")
    if not np.isfinite(real).all() or not np.isfinite(imag).all():
        raise ValueError(f"{label} 频域数据包含 NaN 或 Inf")
    return (real + 1j * imag).astype(np.complex64, copy=False)


def read_matlab_tx_reference(path: str | Path, num_users: int = 4) -> MatlabTxReference:
    """Read split real/imag arrays and payload bits from the MATLAB H5 file.

    The provided MATLAB file is exposed by h5py in ``[symbol, layer,
    subcarrier]`` order for its frequency arrays and ``[bit, 1]`` for payloads.
    """
    source = Path(path)
    users = tuple(f"ue{i}" for i in range(num_users))
    grids: list[np.ndarray] = []
    payloads: list[np.ndarray] = []

    with h5py.File(source, "r") as h5:
        for user in users:
            real_key = f"{user}FreqIQ_real"
            imag_key = f"{user}FreqIQ_imag"
            bits_key = f"{user}TxBits"
            missing = [key for key in (real_key, imag_key, bits_key) if key not in h5]
            if missing:
                raise ValueError(f"{source} 缺少数据集: {', '.join(missing)}")

            real = np.asarray(h5[real_key], dtype=np.float64)
            imag = np.asarray(h5[imag_key], dtype=np.float64)
            payload = np.asarray(h5[bits_key], dtype=np.float64).reshape(-1)
            if real.shape != imag.shape or real.ndim != 3:
                raise ValueError(f"{user} 的频域 real/imag 应是相同的三维数组")
            if not np.isfinite(real).all() or not np.isfinite(imag).all():
                raise ValueError(f"{user} 频域参考数据包含 NaN 或 Inf")
            if not np.isfinite(payload).all() or not np.isin(payload, (0.0, 1.0)).all():
                raise ValueError(f"{user} TxBits 必须只包含 0/1")

            grids.append((real + 1j * imag).astype(np.complex128, copy=False))
            payloads.append(payload.astype(np.uint8, copy=False))

    grid_shapes = {grid.shape for grid in grids}
    bit_sizes = {payload.size for payload in payloads}
    if len(grid_shapes) != 1 or len(bit_sizes) != 1:
        raise ValueError("各用户的参考频域数组形状和 payload 长度必须一致")

    return MatlabTxReference(
        users=users,
        frequency_grid=np.stack(grids, axis=0),
        bits=np.stack(payloads, axis=0),
        source=source,
    )
