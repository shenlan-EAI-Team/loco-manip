from __future__ import annotations

import base64

import cv2
import numpy as np

from deployment.null_sink import NullActionSink
from deployment.observation_sources.g1_live import (
    CameraZmqReader,
    O6ZmqReadOnlyReader,
    parse_live_policy_observation,
    projected_gravity_wxyz,
)
from configs.g1_o6_config import g1_o6_config


def test_projected_gravity_matches_training_transform() -> None:
    np.testing.assert_allclose(
        projected_gravity_wxyz(np.array([1.0, 0.0, 0.0, 0.0])),
        [0.0, 0.0, -1.0],
        atol=1e-7,
    )


def test_camera_legacy_jpeg_keeps_training_rgb_numeric_order() -> None:
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    rgb[..., 0] = 240
    ok, encoded = cv2.imencode(".jpg", rgb)
    assert ok
    decoded = CameraZmqReader._decode_image(base64.b64encode(encoded).decode())
    assert decoded[..., 0].mean() > 230
    assert decoded[..., 2].mean() < 10


def test_o6_requires_atomic_v2_side_payload_in_training_scale() -> None:
    value, metadata = O6ZmqReadOnlyReader._side(
        {
            "side": "left",
            "actual_q": [0, 20, 40, 60, 80, 100],
            "feedback_valid": True,
            "feedback_age_ms": 12.5,
            "timestamp_ns": 123,
        },
        "left",
    )
    np.testing.assert_array_equal(value, [0, 20, 40, 60, 80, 100])
    assert metadata["feedback_valid"] is True
    rejected = False
    try:
        O6ZmqReadOnlyReader._side(
            {
                "side": "right",
                "actual_q": [0, 0, 0, 0, 0, 101],
                "feedback_valid": True,
                "feedback_age_ms": 1,
            },
            "right",
        )
    except ValueError:
        rejected = True
    assert rejected


def test_null_sink_has_no_command_or_ownership_path() -> None:
    sink = NullActionSink()
    sink.record(1.0, {"left_arm": np.zeros(7), "right_arm": np.zeros(7)})
    assert sink.metrics() == {
        "records": 1,
        "command_publish_attempts": 0,
        "control_ownership_requests": 0,
        "real_sdk_objects_created": 0,
    }


def test_live_single_frame_gets_policy_batch_and_time_axes() -> None:
    flat = {
        "state.left_arm": np.zeros(7, dtype=np.float32),
        "state.right_arm": np.zeros(7, dtype=np.float32),
        "state.left_o6": np.zeros(6, dtype=np.float32),
        "state.right_o6": np.zeros(6, dtype=np.float32),
        "state.waist": np.zeros(3, dtype=np.float32),
        "state.projected_gravity": np.array([0, 0, -1], dtype=np.float32),
        "video.ego_view": np.zeros((480, 640, 3), dtype=np.uint8),
        "annotation.human.task_description": "test",
    }
    observation = parse_live_policy_observation(flat, g1_o6_config)
    assert observation["video"]["ego_view"].shape == (1, 1, 480, 640, 3)
    assert observation["state"]["left_arm"].shape == (1, 1, 7)
    assert observation["state"]["left_o6"].shape == (1, 1, 6)
    assert flat["video.ego_view"].shape == (480, 640, 3)


if __name__ == "__main__":
    test_projected_gravity_matches_training_transform()
    test_camera_legacy_jpeg_keeps_training_rgb_numeric_order()
    test_o6_requires_atomic_v2_side_payload_in_training_scale()
    test_null_sink_has_no_command_or_ownership_path()
    test_live_single_frame_gets_policy_batch_and_time_axes()
    print("5 Live Shadow read-only tests passed")
