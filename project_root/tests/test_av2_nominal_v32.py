from __future__ import annotations

import unittest
from math import radians
from pathlib import Path
import tempfile

import numpy as np

from configs.av2_nominal_1000_v1 import NOMINAL_PROTOCOL as V31_PROTOCOL
from configs.av2_nominal_1000_v32 import (
    NOMINAL_PROTOCOL,
    NOMINAL_PROTOCOL_NAME,
    config_sha256,
    validate_native_noise_contract,
)
from simulation.av2_sensor_ekf import run_av2_sensor_ekfs
from tools.av2_1000.cache import build_sim
from tools.av2_1000.common import stable_seed, write_csv


class AV2NominalV32Tests(unittest.TestCase):
    def test_angular_noise_is_stored_in_native_radians(self) -> None:
        validate_native_noise_contract()
        self.assertEqual(NOMINAL_PROTOCOL.sensors[1].noise_standard_deviations[1][0], "bearing_rad")
        self.assertAlmostEqual(
            NOMINAL_PROTOCOL.sensors[1].noise_standard_deviations[1][1], radians(0.40)
        )
        self.assertAlmostEqual(
            NOMINAL_PROTOCOL.sensors[2].noise_standard_deviations[1][1], radians(0.45)
        )
        self.assertAlmostEqual(
            NOMINAL_PROTOCOL.sensors[3].noise_standard_deviations[0][1], radians(0.80)
        )
        # The frozen V3.1 object is still numerically unchanged.
        self.assertEqual(V31_PROTOCOL.sensors[1].noise_standard_deviations[1], ("bearing_deg", 0.40))

    def test_generated_covariance_matches_corrected_angular_variance(self) -> None:
        timestamps = np.arange(11, dtype=np.int64) * 100_000_000
        seconds = timestamps.astype(np.float64) * 1e-9
        truth = np.column_stack(
            (seconds * 5.0, seconds * 2.0, np.full(11, 5.0), np.full(11, 2.0))
        )
        output = run_av2_sensor_ekfs(
            timestamps,
            truth,
            rng=np.random.default_rng(7),
            protocol=NOMINAL_PROTOCOL,
        )
        self.assertAlmostEqual(output.measurements[0][1].R_actual[1, 1], radians(0.40) ** 2)
        self.assertAlmostEqual(output.measurements[0][2].R_actual[1, 1], radians(0.45) ** 2)
        self.assertAlmostEqual(output.measurements[0][3].R_actual[0, 0], radians(0.80) ** 2)

    def test_rng_stream_is_scoped_by_scenario_and_protocol(self) -> None:
        first = stable_seed(NOMINAL_PROTOCOL_NAME, "scene-a", 100, "measurement")
        second = stable_seed(NOMINAL_PROTOCOL_NAME, "scene-b", 100, "measurement")
        self.assertNotEqual(first, second)
        self.assertEqual(first, stable_seed(NOMINAL_PROTOCOL_NAME, "scene-a", 100, "measurement"))
        self.assertEqual(len(config_sha256()), 64)

    def test_cache_builder_persists_distinct_scenario_scoped_rng_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifests"
            truth = root / "truth"
            output = root / "sim"
            rows = [
                {"scenario_id": "scene-a"},
                {"scenario_id": "scene-b"},
            ]
            write_csv(manifest / "test_200.csv", rows, ["scenario_id"])
            timestamps = np.arange(110, dtype=np.int64) * 100_000_000
            seconds = timestamps.astype(np.float64) * 1e-9
            target = np.column_stack(
                (seconds * 5.0, seconds * 2.0, np.full(110, 5.0), np.full(110, 2.0))
            ).astype(np.float32)
            (truth / "test").mkdir(parents=True)
            for row in rows:
                np.savez(
                    truth / "test" / f"{row['scenario_id']}.npz",
                    timestamps_ns=timestamps,
                    target=target,
                )
            counts = build_sim(
                manifest,
                truth,
                output,
                splits=("test",),
                protocol=NOMINAL_PROTOCOL,
                protocol_name=NOMINAL_PROTOCOL_NAME,
                config_sha256=config_sha256(),
                scenario_scoped_rng=True,
            )
            self.assertEqual(counts, {"test": 6})
            resumed = build_sim(
                manifest,
                truth,
                output,
                splits=("test",),
                protocol=NOMINAL_PROTOCOL,
                protocol_name=NOMINAL_PROTOCOL_NAME,
                config_sha256=config_sha256(),
                scenario_scoped_rng=True,
            )
            self.assertEqual(resumed, {"test": 6})
            with np.load(output / "test" / "scene-a" / "seed_100.npz") as first:
                first_rng = int(first["rng_seed"])
                first_z = first["measurement_z"].copy()
            with np.load(output / "test" / "scene-b" / "seed_100.npz") as second:
                second_rng = int(second["rng_seed"])
                second_z = second["measurement_z"].copy()
            self.assertNotEqual(first_rng, second_rng)
            self.assertFalse(np.array_equal(first_z, second_z))


if __name__ == "__main__":
    unittest.main()
