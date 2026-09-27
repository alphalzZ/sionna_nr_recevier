"""Unit tests for the shared device policy."""

import unittest
from unittest import mock

import torch

from nr_pusch.device import resolve_device, use_device


class ResolveDeviceTest(unittest.TestCase):
    def test_auto_selects_cuda_when_available(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(resolve_device("auto"), "cuda:0")
            self.assertEqual(resolve_device(None), "cuda:0")
            self.assertEqual(resolve_device("cuda"), "cuda:0")

    def test_auto_falls_back_to_cpu_without_cuda(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(resolve_device("auto"), "cpu")
            self.assertEqual(resolve_device(None), "cpu")

    def test_explicit_cuda_without_device_is_rejected(self):
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaises(RuntimeError):
                resolve_device("cuda")

    def test_explicit_device_is_passed_through(self):
        self.assertEqual(resolve_device("cpu"), "cpu")
        self.assertEqual(resolve_device("cuda:1"), "cuda:1")


class UseDeviceTest(unittest.TestCase):
    def test_applies_requested_device_to_sionna_config(self):
        import sionna.phy

        previous = sionna.phy.config.device
        self.addCleanup(setattr, sionna.phy.config, "device", previous)
        resolved = use_device("cpu")
        self.assertEqual(resolved, "cpu")
        self.assertEqual(sionna.phy.config.device, "cpu")


if __name__ == "__main__":
    unittest.main()
