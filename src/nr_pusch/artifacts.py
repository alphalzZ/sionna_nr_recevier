"""Portable IQ and run-manifest export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import sionna
import torch

from .config import TxSettings
from .transmitter import TxResult


def save_tx_result(result: TxResult, settings: TxSettings, output: str | Path) -> tuple[Path, Path]:
    output = Path(output)
    if output.suffix.lower() != ".npz":
        raise ValueError("发送 IQ 输出文件必须使用 .npz 后缀")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output.with_suffix(".json")

    iq = result.iq.detach().cpu().numpy().astype(np.complex64, copy=False)
    frequency_grid = result.frequency_grid.detach().cpu().numpy().astype(np.complex64, copy=False)
    bits = result.bits.detach().cpu().numpy().astype(np.uint8, copy=False)
    np.savez_compressed(output, iq=iq, frequency_grid=frequency_grid, bits=bits)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "settings": settings.to_dict(),
        "result": result.metadata,
        "versions": {"sionna": sionna.__version__, "torch": torch.__version__},
        "artifacts": {
            "archive": output.name,
            "arrays": {
                "iq": "iq",
                "frequency_grid": "frequency_grid",
                "payload_bits": "bits",
            },
            "format": "NumPy NPZ",
            "complex_dtype": "complex64",
            "bits_dtype": "uint8",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output, manifest_path
