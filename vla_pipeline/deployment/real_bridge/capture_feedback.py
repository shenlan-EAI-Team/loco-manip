from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import yaml

from configs.g1_o6_config import g1_o6_config
from deployment.common import ACTION_KEYS
from deployment.observation_sources.g1_live import G1LiveObservationSource

from .models import FeedbackSnapshot


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture one read-only synchronized arming snapshot")
    parser.add_argument("--config", default="deployment/config/live_shadow.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    required = {
        "real_hardware_enabled": False,
        "publish_commands": False,
        "shadow_only": True,
        "dry_run": True,
    }
    for key, expected in required.items():
        if config.get(key) is not expected:
            raise RuntimeError(f"read-only snapshot refused: {key}={config.get(key)!r}")

    source = G1LiveObservationSource(config, g1_o6_config)
    try:
        source.start()
        deadline = time.monotonic() + args.timeout_s
        sample = None
        while sample is None and time.monotonic() < deadline:
            sample = source.get_observation()
            if sample is None:
                time.sleep(0.001)
        if sample is None:
            raise TimeoutError("no synchronized read-only observation before deadline")
        groups = {
            key: np.asarray(sample.flat_observation[f"state.{key}"], dtype=np.float64)
            for key in ACTION_KEYS
        }
        metadata = sample.source_metadata["g1"]
        snapshot = FeedbackSnapshot.create(
            groups,
            monotonic_ns=int(sample.monotonic_timestamp * 1e9),
            g1_mode_machine=int(metadata["mode_machine"]),
            g1_mode_pr=int(metadata["mode_pr"]),
            waist=sample.flat_observation["state.waist"],
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(snapshot.as_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(output)
    finally:
        source.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
