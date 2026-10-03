"""Validated CDL tap-power prior artifacts for DMRS LMMSE estimation."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch
from sionna.phy.channel import time_lag_discrete_time_channel

from .channel_config import ChannelSettings
from .config import TxSettings

_FORMAT_VERSION = 1


_GATE_SHA256 = re.compile(r"[0-9a-f]{64}")


def dmrs_prior_compatibility(
    tx_settings: TxSettings,
    channel_settings: ChannelSettings,
    *,
    l_min: int,
    max_delay_spread_s: float,
    fft_size: int,
    sample_rate_hz: int,
) -> dict[str, Any]:
    """Return the exact run geometry against which a prior was calibrated."""
    _, l_max = time_lag_discrete_time_channel(sample_rate_hz, max_delay_spread_s)
    tx_pusch = asdict(tx_settings.pusch)
    # MCS changes transport-block coding, not the channel tap-power prior.
    tx_pusch.pop("mcs_table")
    tx_pusch.pop("mcs_index")
    compatibility = {
        "tx_carrier": asdict(tx_settings.carrier),
        "tx_pusch": tx_pusch,
        "channel": asdict(channel_settings),
        "l_min": int(l_min),
        "l_max": int(l_max),
        "max_delay_spread_s": float(max_delay_spread_s),
        "fft_size": int(fft_size),
        "sample_rate_hz": int(sample_rate_hz),
    }
    return json.loads(json.dumps(compatibility))


def _compatibility_without_mcs(compatibility: Any) -> Any:
    if not isinstance(compatibility, dict):
        return compatibility
    normalized = dict(compatibility)
    tx_pusch = compatibility.get("tx_pusch")
    if isinstance(tx_pusch, dict):
        normalized["tx_pusch"] = {
            key: value
            for key, value in tx_pusch.items()
            if key not in {"mcs_table", "mcs_index"}
        }
    return normalized


def prior_registry_path(prior_dir: str | Path, compatibility: dict[str, Any]) -> Path:
    """Return the registry file that holds the prior for this exact run geometry."""
    normalized = _compatibility_without_mcs(compatibility)
    if not isinstance(normalized, dict):
        raise ValueError("DMRS prior 兼容性必须是字典")
    tx_pusch = normalized.get("tx_pusch")
    waveform = tx_pusch.get("waveform") if isinstance(tx_pusch, dict) else None
    if not isinstance(waveform, str) or not waveform:
        raise ValueError("DMRS prior 兼容性缺少 tx_pusch.waveform")
    digest = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return Path(prior_dir) / waveform / f"{digest}.npz"


def default_prior_dir(channel_config_path: str | Path) -> Path:
    """Return the shared prior directory that sits beside a channel configuration."""
    return Path(channel_config_path).resolve().parent / "tap_power_prior"


def resolve_dmrs_tap_power_prior(
    prior_dir: str | Path,
    *,
    expected_compatibility: dict[str, Any],
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, Path, dict[str, Any]]:
    """Look up the accepted prior for one geometry inside a shared prior directory."""
    path = prior_registry_path(prior_dir, expected_compatibility)
    if not path.is_file():
        raise ValueError(f"未找到与当前 TX/CDL/抽头窗口匹配的已通过验证的 DMRS prior: {path}")
    prior, metadata = load_dmrs_tap_power_prior(
        path, expected_compatibility=expected_compatibility, device=device
    )
    marker = metadata.get("accepted_gate_sha256")
    if not isinstance(marker, str) or not _GATE_SHA256.fullmatch(marker):
        raise ValueError(f"DMRS prior {path} 缺少通过验证的 acceptance 标记")
    return prior, path, metadata


def save_dmrs_tap_power_prior(
    path: str | Path,
    tap_power: torch.Tensor | np.ndarray,
    *,
    compatibility: dict[str, Any],
    training_seed: int,
    training_realizations: int,
    accepted_gate_sha256: str | None = None,
) -> Path:
    """Write a portable NPZ prior with explicit provenance and run geometry."""
    output = Path(path)
    values = np.asarray(
        tap_power.detach().cpu().numpy() if isinstance(tap_power, torch.Tensor) else tap_power,
        dtype=np.float64,
    )
    _validate_tap_power(values)
    if training_realizations < 1:
        raise ValueError("training_realizations 必须为正整数")
    if accepted_gate_sha256 is not None and not _GATE_SHA256.fullmatch(accepted_gate_sha256):
        raise ValueError("accepted_gate_sha256 必须是 64 位小写十六进制摘要")
    metadata = {
        "format_version": _FORMAT_VERSION,
        "compatibility": compatibility,
        "training_seed": int(training_seed),
        "training_realizations": int(training_realizations),
    }
    if accepted_gate_sha256 is not None:
        metadata["accepted_gate_sha256"] = accepted_gate_sha256
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        np.savez_compressed(
            stream,
            tap_power=values,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        )
    return output


def publish_accepted_dmrs_prior(
    prior_dir: str | Path,
    candidate_path: str | Path,
    *,
    compatibility: dict[str, Any],
    acceptance_gate: dict[str, Any],
) -> Path:
    """Promote a validated candidate into the shared prior directory atomically."""
    if not isinstance(acceptance_gate, dict) or acceptance_gate.get("passed") is not True:
        raise ValueError("仅 acceptance_gate.passed 为 true 的候选才可发布到共享先验目录")
    tap_power, metadata = load_dmrs_tap_power_prior(
        candidate_path, expected_compatibility=compatibility
    )
    gate_digest = hashlib.sha256(
        json.dumps(acceptance_gate, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    target = prior_registry_path(prior_dir, compatibility)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(f"{target.stem}.{os.getpid()}.tmp")
    try:
        save_dmrs_tap_power_prior(
            staged,
            tap_power,
            compatibility=compatibility,
            training_seed=int(metadata["training_seed"]),
            training_realizations=int(metadata["training_realizations"]),
            accepted_gate_sha256=gate_digest,
        )
        os.replace(staged, target)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return target


def load_dmrs_tap_power_prior(
    path: str | Path,
    *,
    expected_compatibility: dict[str, Any],
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load and strictly validate a prior against the current run geometry."""
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != {"tap_power", "metadata_json"}:
                raise ValueError("DMRS prior NPZ 必须恰好包含 tap_power 与 metadata_json")
            values = np.asarray(archive["tap_power"], dtype=np.float64)
            metadata_raw = archive["metadata_json"].item()
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"无法读取 DMRS tap-power prior {source}: {exc}") from exc
    if not isinstance(metadata_raw, str):
        raise ValueError("DMRS prior metadata_json 必须是 UTF-8 JSON 字符串")
    try:
        metadata = json.loads(metadata_raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"DMRS prior metadata_json 无效: {exc}") from exc
    if not isinstance(metadata, dict) or metadata.get("format_version") != _FORMAT_VERSION:
        raise ValueError("DMRS prior 格式版本不支持")
    compatibility = metadata.get("compatibility")
    if _compatibility_without_mcs(compatibility) != _compatibility_without_mcs(
        expected_compatibility
    ):
        raise ValueError("DMRS tap-power prior 与当前 TX/CDL/抽头窗口/采样配置不兼容")
    if not isinstance(metadata.get("training_seed"), int):
        raise ValueError("DMRS prior 缺少有效 training_seed")
    if not isinstance(metadata.get("training_realizations"), int) or metadata["training_realizations"] < 1:
        raise ValueError("DMRS prior 缺少有效 training_realizations")
    _validate_tap_power(values)
    expected_taps = int(expected_compatibility["l_max"] - expected_compatibility["l_min"] + 1)
    if values.shape != (expected_taps,):
        raise ValueError(f"DMRS prior tap 数应为 {expected_taps}，实际为 {values.shape}")
    prior = torch.as_tensor(values, dtype=torch.float32, device=device)
    return prior, metadata


def _validate_tap_power(values: np.ndarray) -> None:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("DMRS tap-power prior 必须是一维非空向量")
    if not np.isfinite(values).all() or not np.all(values > 0):
        raise ValueError("DMRS tap-power prior 必须全部为正有限值")
