"""Device selection shared by the Sionna-backed blocks.

Sionna allocates parts of a resource grid (for example the pilot pattern) from
its global ``sionna.phy.config.device`` instead of the per-block device. Asking
for a CPU run on a GPU host therefore fails inside Sionna unless the global
config points at the same device, so every entry point applies the requested
device there before building blocks.
"""

from __future__ import annotations

import sionna.phy
import torch


def resolve_device(requested: str | None) -> str:
    """Normalize ``auto``/``cuda``/``None`` to a device string Sionna accepts."""
    if requested is None or requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("配置请求 CUDA，但当前 PyTorch 环境没有可用 GPU/CUDA")
        return "cuda:0"
    return requested


def use_device(requested: str | None) -> str:
    """Apply ``requested`` to Sionna's global device and return the resolved name."""
    device = resolve_device(requested)
    sionna.phy.config.device = device
    return device
