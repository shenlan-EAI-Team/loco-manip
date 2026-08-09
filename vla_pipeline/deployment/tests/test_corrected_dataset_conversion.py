#!/usr/bin/env python3
"""Regression tests for the already-decoded g1_debug dataset contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from inspect_dataset import reconstruct_g1  # noqa: E402


def _full43(body: np.ndarray) -> np.ndarray:
    full = np.zeros(43, dtype=np.float64)
    full[:22] = body[:22]
    full[29:36] = body[22:29]
    return full


class CorrectedDatasetConversionTest(unittest.TestCase):
    def test_reconstruct_preserves_published_hardware_absolute_values(self) -> None:
        feedback = np.linspace(-0.7, 0.7, 29)
        command = np.linspace(0.9, -0.9, 29)
        frame = pd.DataFrame(
            {
                "observation.state": [_full43(feedback)],
                "action.wbc": [_full43(command)],
            }
        )

        result = reconstruct_g1(frame)

        np.testing.assert_array_equal(result["feedback_abs_hw"][0], feedback)
        np.testing.assert_array_equal(result["command_abs_hw"][0], command)
        np.testing.assert_array_equal(result["instant_delta_hw"][0], command - feedback)


if __name__ == "__main__":
    unittest.main()
