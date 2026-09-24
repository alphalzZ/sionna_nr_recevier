"""Reader for the supplied MATLAB ``TxTestVector.h5`` fixture."""

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

