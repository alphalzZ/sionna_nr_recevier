"""Typed project configuration and Sionna PUSCH config construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import tomllib
from typing import Any

from sionna.phy.nr import PUSCHConfig, decode_mcs_index


@dataclass(frozen=True)
class CarrierSettings:
    subcarrier_spacing_khz: int
    cyclic_prefix: str
    n_cell_id: int
    n_size_grid: int
    n_start_grid: int
    frame_number: int
    slot_number: int


@dataclass(frozen=True)
class UserSettings:
    name: str
    n_rnti: int
    dmrs_ports: tuple[int, ...]


@dataclass(frozen=True)
class PuschSettings:
    waveform: str
    mapping_type: str
    symbol_allocation: tuple[int, int]
    n_size_bwp: int
    n_start_bwp: int
    mcs_table: int
    mcs_index: int
    dmrs_config_type: int
    dmrs_type_a_position: int
    dmrs_additional_position: int
    dmrs_num_cdm_groups_without_data: int
    dmrs_beta: float
    num_layers: int = 1
    num_antenna_ports: int = 1
    precoding: str = "non-codebook"
    tpmi: int = 0
    dmrs_length: int = 1
    dft_s_dmrs_port_order: str = "comb-first"


@dataclass(frozen=True)
class TxSettings:
    carrier: CarrierSettings
    pusch: PuschSettings
    users: tuple[UserSettings, ...]

    @classmethod
    def from_toml(cls, path: str | Path) -> "TxSettings":
        with Path(path).open("rb") as f:
            raw = tomllib.load(f)
        carrier = CarrierSettings(**raw["carrier"])
        pusch_raw = dict(raw["pusch"])
        pusch_raw["symbol_allocation"] = tuple(pusch_raw["symbol_allocation"])
        pusch = PuschSettings(**pusch_raw)
        users = tuple(UserSettings(**{**u, "dmrs_ports": tuple(u["dmrs_ports"])}) for u in raw["users"])
        settings = cls(carrier=carrier, pusch=pusch, users=users)
        settings.validate()
        return settings

    def validate(self) -> None:
        p = self.pusch
        if not self.users:
            raise ValueError("至少需要一个用户")
        if not 1 <= p.num_layers <= 4:
            raise ValueError("num_layers 必须为 1..4")
        if len(self.users) * p.num_layers > 8:
            raise ValueError("总流数不能超过 8")
        if p.num_antenna_ports not in {1, 2, 4} or p.num_layers > p.num_antenna_ports:
            raise ValueError("num_antenna_ports 必须是 1、2 或 4，且不少于 num_layers")
        if p.precoding == "non-codebook" and p.num_layers != p.num_antenna_ports:
            raise ValueError("non-codebook 要求 num_layers 等于 num_antenna_ports")
        if len({u.name for u in self.users}) != len(self.users):
            raise ValueError("用户 name 必须唯一")
        if len({u.n_rnti for u in self.users}) != len(self.users):
            raise ValueError("用户 n_rnti 必须互不相同")
        ports = [port for u in self.users for port in u.dmrs_ports]
        if any(len(u.dmrs_ports) != p.num_layers for u in self.users):
            raise ValueError("每个用户 dmrs_ports 数量必须等于 num_layers")
        if len(set(ports)) != len(ports):
            raise ValueError("同一时频资源上的 DMRS ports 必须互不重复")
        if self.pusch.waveform not in {"cp_ofdm", "dft_s_ofdm"}:
            raise ValueError("waveform 仅支持 cp_ofdm 或 dft_s_ofdm")
        if p.dft_s_dmrs_port_order not in {"comb-first", "native"}:
            raise ValueError("dft_s_dmrs_port_order 必须为 comb-first 或 native")
        if self.carrier.subcarrier_spacing_khz not in {15, 30, 60, 120, 240}:
            raise ValueError("subcarrier_spacing_khz 必须是 NR numerology 对应的有效值")
        if self.pusch.n_size_bwp < 1:
            raise ValueError("n_size_bwp 必须大于 0")
        if self.pusch.n_size_bwp > self.carrier.n_size_grid:
            raise ValueError("n_size_bwp 不能大于 n_size_grid")
        if self.pusch.waveform == "dft_s_ofdm":
            if self.pusch.dmrs_config_type != 1:
                raise NotImplementedError("DFT-s-OFDM 当前仅支持 PUSCH DMRS config type 1")
            additional_position = self.pusch.dmrs_additional_position
            if (
                isinstance(additional_position, bool)
                or not isinstance(additional_position, int)
                or additional_position not in {0, 1, 2}
            ):
                raise ValueError("dmrs_additional_position 必须为 0、1 或 2 的整数")
            if self.pusch.dmrs_num_cdm_groups_without_data != 2:
                raise NotImplementedError("DFT-s-OFDM 当前要求两个 type-1 DMRS CDM groups")
            rb_count = self.pusch.n_size_bwp
            for factor in (2, 3, 5):
                while rb_count % factor == 0:
                    rb_count //= factor
            if rb_count != 1:
                raise ValueError("DFT-s-OFDM BWP size must yield a 2/3/5-smooth DFT length")
            self.effective_sionna_mcs()
        self._build_sionna_configs()

    def effective_sionna_mcs(self) -> tuple[int, int]:
        """Return a non-transform Sionna MCS with equivalent Qm and code rate.

        Sionna 2.0.1 does not expose transform-precoding through PUSCHConfig.
        Its TB encoder still accepts the same modulation order and target rate,
        so map the configured transform-precoding MCS to an equivalent native
        MCS for encoding. For table 1/index 20 this resolves to table 1/index 21.
        """
        p = self.pusch
        if p.waveform != "dft_s_ofdm":
            return p.mcs_table, p.mcs_index

        q_target, r_target = decode_mcs_index(
            p.mcs_index,
            table_index=p.mcs_table,
            is_pusch=True,
            transform_precoding=True,
        )
        target_q = int(q_target.item())
        target_r = float(r_target.item())
        candidates: list[tuple[int, int]] = []
        for table in range(1, 5):
            for index in range(29):
                try:
                    q, r = decode_mcs_index(index, table_index=table, is_pusch=True)
                except (AssertionError, ValueError):
                    continue
                if int(q.item()) == target_q and abs(float(r.item()) - target_r) < 1e-12:
                    candidates.append((table, index))
        if not candidates:
            raise ValueError("找不到与 DFT-s-OFDM MCS 调制阶数和码率匹配的 Sionna MCS")
        candidates.sort(key=lambda pair: (pair[0] != p.mcs_table, pair[0], pair[1]))
        return candidates[0]

    def to_sionna_configs(self) -> list[PUSCHConfig]:
        """Build one native PUSCH config per UE."""
        self.validate()
        return self._build_sionna_configs()

    def _build_sionna_configs(self) -> list[PUSCHConfig]:
        configs: list[PUSCHConfig] = []
        for user in self.users:
            cfg = PUSCHConfig()
            c = self.carrier
            cfg.carrier.subcarrier_spacing = c.subcarrier_spacing_khz
            cfg.carrier.cyclic_prefix = c.cyclic_prefix
            cfg.carrier.n_cell_id = c.n_cell_id
            cfg.carrier.n_size_grid = c.n_size_grid
            cfg.carrier.n_start_grid = c.n_start_grid
            cfg.carrier.frame_number = c.frame_number
            cfg.carrier.slot_number = c.slot_number

            p = self.pusch
            cfg.mapping_type = p.mapping_type
            cfg.symbol_allocation = list(p.symbol_allocation)
            cfg.n_size_bwp = p.n_size_bwp
            cfg.n_start_bwp = p.n_start_bwp
            cfg.n_rnti = user.n_rnti
            cfg.num_layers = p.num_layers
            cfg.num_antenna_ports = p.num_antenna_ports
            cfg.precoding = p.precoding
            cfg.tpmi = p.tpmi
            cfg.transform_precoding = False
            cfg.tb.mcs_table, cfg.tb.mcs_index = self.effective_sionna_mcs()
            cfg.dmrs.config_type = p.dmrs_config_type
            cfg.dmrs.type_a_position = p.dmrs_type_a_position
            cfg.dmrs.additional_position = p.dmrs_additional_position
            cfg.dmrs.length = p.dmrs_length
            cfg.dmrs.num_cdm_groups_without_data = p.dmrs_num_cdm_groups_without_data
            cfg.dmrs.dmrs_port_set = list(user.dmrs_ports)
            cfg.check_config()
            if p.waveform == "cp_ofdm" and (
                not math.isfinite(p.dmrs_beta)
                or abs(p.dmrs_beta - cfg.dmrs.beta) > 1e-6
            ):
                raise ValueError(
                    "cp_ofdm 的 dmrs_beta 必须等于原生 Sionna DMRS beta"
                )
            configs.append(cfg)
        return configs

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
