from dataclasses import replace
from pathlib import Path
import unittest

from nr_pusch.channel_config import ChannelSettings
from nr_pusch.config import TxSettings, UserSettings


ROOT = Path(__file__).parents[2]


class TxTopologyConfigTest(unittest.TestCase):
    def setUp(self):
        self.base = TxSettings.from_toml(ROOT / "configs" / "pusch_4ue.toml")

    def test_native_one_four_and_eight_stream_topologies(self):
        for users, layers, ports, length in (
            (1, 1, 1, 1),
            (1, 4, 4, 1),
            (2, 4, 4, 2),
        ):
            with self.subTest(users=users, layers=layers):
                cfg = replace(
                    self.base,
                    pusch=replace(self.base.pusch, waveform="cp_ofdm", num_layers=layers,
                                  num_antenna_ports=ports, dmrs_length=length,
                                  dmrs_beta=2**0.5),
                    users=tuple(UserSettings(f"ue{i}", i + 1,
                                             tuple(range(i * layers, (i + 1) * layers)))
                                for i in range(users)),
                )
                native = cfg.to_sionna_configs()
                self.assertEqual(len(native), users)
                self.assertEqual([list(c.dmrs.dmrs_port_set) for c in native],
                                 [list(u.dmrs_ports) for u in cfg.users])
                self.assertTrue(all(c.num_layers == layers for c in native))

    def test_rejects_invalid_streams_ports_and_dmrs(self):
        cases = (
            ((), self.base.pusch),
            ((UserSettings("ue", 1, (0,)),), replace(self.base.pusch, num_layers=5)),
            (tuple(UserSettings(f"ue{i}", i + 1, (i,)) for i in range(9)),
             self.base.pusch),
            ((UserSettings("ue", 1, (0, 1)),), replace(self.base.pusch, num_layers=2)),
            ((UserSettings("ue", 1, (0,)),), replace(self.base.pusch, num_layers=2,
                                                       num_antenna_ports=2)),
            ((UserSettings("ue", 1, (0, 0)),), replace(self.base.pusch, num_layers=2,
                                                         num_antenna_ports=2)),
            ((UserSettings("ue", 1, (4,)),), self.base.pusch),
            ((UserSettings("ue", 1, (0,)),), replace(self.base.pusch, dmrs_length=2,
                                                       dmrs_additional_position=2)),
        )
        for users, pusch in cases:
            with self.subTest(users=users, pusch=pusch):
                with self.assertRaises((ValueError, TypeError)):
                    replace(self.base, users=users, pusch=pusch).validate()

    def test_codebook_three_layers_on_four_ports(self):
        cfg = replace(self.base,
                      pusch=replace(self.base.pusch, waveform="cp_ofdm", num_layers=3,
                                    num_antenna_ports=4, precoding="codebook",
                                    dmrs_beta=2**0.5),
                      users=(UserSettings("ue", 1, (0, 1, 2)),))
        self.assertEqual(cfg.to_sionna_configs()[0].precoding_matrix.shape, (4, 3))


    def test_cp_dmrs_beta_matches_native_sionna_and_rejects_mismatch(self):
        settings = TxSettings.from_toml(ROOT / "configs" / "pusch_cp_2ue_2layer.toml")
        self.assertEqual(settings.pusch.waveform, "cp_ofdm")
        self.assertEqual(len(settings.users), 2)
        self.assertEqual([cfg.dmrs.beta for cfg in settings.to_sionna_configs()],
                         [settings.pusch.dmrs_beta] * 2)

        invalid = replace(
            settings,
            pusch=replace(settings.pusch, dmrs_beta=settings.pusch.dmrs_beta + 0.01),
        )
        with self.assertRaisesRegex(ValueError, "cp_ofdm 的 dmrs_beta"):
            invalid.validate()

        type2 = replace(
            settings,
            pusch=replace(
                settings.pusch,
                dmrs_config_type=2,
                dmrs_num_cdm_groups_without_data=1,
                dmrs_beta=1.0,
                num_layers=1,
                num_antenna_ports=1,
                precoding="non-codebook",
            ),
            users=(UserSettings("ue0", 1, (0,)),),
        )
        self.assertEqual(type2.to_sionna_configs()[0].dmrs.beta, 1.0)

    def test_new_cdl_profile_matches_cp_tx_topology(self):
        settings = TxSettings.from_toml(ROOT / "configs" / "pusch_cp_2ue_2layer.toml")
        channel = ChannelSettings.from_toml(ROOT / "configs" / "cdl_38_901_2tx_4rx.toml")
        channel.validate_transmitter(settings)
        self.assertEqual(
            (channel.antennas.tx_num_rows * channel.antennas.tx_num_cols,
             channel.antennas.rx_num_rows * channel.antennas.rx_num_cols),
            (2, 4),
        )

    def test_shipped_topology_profiles_keep_unique_native_ports(self):
        for name, users, layers, length in (
            ("pusch_1ue_1tx_4rx.toml", 1, 1, 1),
            ("pusch_1ue_4layer.toml", 1, 4, 1),
            ("pusch_2ue_4layer.toml", 2, 4, 2),
            ("pusch_cp_2ue_2layer.toml", 2, 2, 1),
        ):
            with self.subTest(name=name):
                settings = TxSettings.from_toml(ROOT / "configs" / name)
                self.assertEqual((len(settings.users), settings.pusch.num_layers,
                                  settings.pusch.dmrs_length), (users, layers, length))
                self.assertEqual(len({port for user in settings.users
                                      for port in user.dmrs_ports}), users * layers)

    def test_legacy_profile_ports_and_mcs_remain_unchanged(self):
        self.assertEqual([u.dmrs_ports for u in self.base.users],
                         [(0,), (1,), (2,), (3,)])
        self.assertEqual(self.base.effective_sionna_mcs(), (1, 21))
        self.assertEqual((self.base.pusch.num_layers, self.base.pusch.num_antenna_ports,
                          self.base.pusch.dmrs_length), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
