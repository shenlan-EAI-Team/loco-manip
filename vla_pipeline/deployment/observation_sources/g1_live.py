"""Strict, read-only synchronized observation source for G1 Live Shadow."""

from __future__ import annotations

import base64
from collections import Counter, deque
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Any

import cv2
import msgpack
import numpy as np
from scipy.spatial.transform import Rotation
import zmq

from gr00t.data.utils import parse_observation_gr00t

from deployment.observation_sources.base import ObservationSample, ObservationSource


STATE_DIMS = {
    "left_arm": 7,
    "right_arm": 7,
    "left_o6": 6,
    "right_o6": 6,
    "waist": 3,
    "projected_gravity": 3,
}


def parse_live_policy_observation(
    flat: dict[str, Any], modality_config: dict[str, Any]
) -> dict[str, Any]:
    """Add the single-frame time axis expected by the trained policy."""
    temporal_flat: dict[str, Any] = {}
    for key, value in flat.items():
        if key.startswith("state."):
            array = np.asarray(value)
            if array.ndim != 1:
                raise ValueError(f"live state must be 1-D before batching: {key} {array.shape}")
            temporal_flat[key] = array[None, :]
        elif key.startswith("video."):
            array = np.asarray(value)
            if array.ndim != 3:
                raise ValueError(f"live video must be HWC before batching: {key} {array.shape}")
            temporal_flat[key] = array[None, ...]
        else:
            temporal_flat[key] = value
    return parse_observation_gr00t(temporal_flat, modality_config)


@dataclass(frozen=True)
class TimedDatum:
    source: str
    source_timestamp: float | int | None
    receive_wall_ns: int
    receive_monotonic_ns: int
    sequence: int
    value: Any
    metadata: dict[str, Any]


class LowStateProcessReader:
    """Consume JSON from a separate Python 3.8 DDS subscriber process."""

    def __init__(self, python: str, script: Path, interface: str, topic: str) -> None:
        self.command = [
            python,
            "-u",
            str(script),
            "--interface",
            interface,
            "--topic",
            topic,
        ]
        self.process: subprocess.Popen[str] | None = None
        self.latest: TimedDatum | None = None
        self.errors: deque[str] = deque(maxlen=100)
        self.sequence = 0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._thread.start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

    def _read_stdout(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            try:
                payload = json.loads(line)
                if payload.get("schema") != "g1_lowstate_readonly_v1":
                    continue
                self.sequence += 1
                datum = TimedDatum(
                    source="g1_lowstate",
                    source_timestamp=int(payload["tick"]),
                    receive_wall_ns=int(payload["receive_wall_ns"]),
                    receive_monotonic_ns=int(payload["receive_monotonic_ns"]),
                    sequence=self.sequence,
                    value={
                        key: np.asarray(payload[key], dtype=np.float32)
                        for key in ("left_arm", "right_arm", "waist", "base_quat_wxyz")
                    },
                    metadata={
                        "topic": payload["topic"],
                        "tick": int(payload["tick"]),
                        "mode_machine": int(payload["mode_machine"]),
                        "mode_pr": int(payload["mode_pr"]),
                    },
                )
                with self._lock:
                    self.latest = datum
            except Exception as exc:  # malformed lines are never observations
                self.errors.append(f"lowstate_decode:{type(exc).__name__}:{exc}")

    def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        for line in self.process.stderr:
            self.errors.append(line.rstrip())

    def get_latest(self) -> TimedDatum | None:
        with self._lock:
            return self.latest

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)


class CameraZmqReader:
    """SUB-only decoder for the existing D435i camera server."""

    def __init__(self, endpoint: str) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(endpoint)
        self.endpoint = endpoint
        self.latest: TimedDatum | None = None
        self.sequence = 0
        self.rejected = Counter()
        self.last_source_timestamp: float | None = None

    @staticmethod
    def _decode_image(value: Any) -> np.ndarray:
        if isinstance(value, str):
            encoded = base64.b64decode(value)
            # The server receives RealSense rgb8 and passes that numeric array
            # through OpenCV JPEG encode/decode; the resulting numeric channel
            # order remains the training-time RGB order.
            image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        elif isinstance(value, (bytes, bytearray)):
            image = cv2.imdecode(np.frombuffer(value, np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                image = image[..., ::-1]
        else:
            image = np.asarray(value)
        if image is None:
            raise ValueError("JPEG decode failed")
        image = np.asarray(image)
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise ValueError(f"ego_view expected uint8 (480,640,3), got {image.dtype} {image.shape}")
        return np.ascontiguousarray(image)

    def poll(self) -> TimedDatum | None:
        while True:
            try:
                raw = self.socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            receive_wall_ns = time.time_ns()
            receive_monotonic_ns = time.monotonic_ns()
            try:
                message = msgpack.unpackb(raw, raw=False)
                images = message["images"]
                timestamps = message["timestamps"]
                image = self._decode_image(images["ego_view"])
                source_timestamp = float(timestamps["ego_view"])
                duplicate = source_timestamp == self.last_source_timestamp
                self.last_source_timestamp = source_timestamp
                self.sequence += 1
                self.latest = TimedDatum(
                    source="ego_view",
                    source_timestamp=source_timestamp,
                    receive_wall_ns=receive_wall_ns,
                    receive_monotonic_ns=receive_monotonic_ns,
                    sequence=self.sequence,
                    value=image,
                    metadata={"endpoint": self.endpoint, "duplicate": duplicate},
                )
            except Exception as exc:
                self.rejected[f"{type(exc).__name__}:{exc}"] += 1
        return self.latest

    def stop(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


class O6ZmqReadOnlyReader:
    """Strict SUB-only dual O6 state reader; no command socket exists."""

    def __init__(self, endpoint: str) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(endpoint)
        self.endpoint = endpoint
        self.latest: TimedDatum | None = None
        self.sequence = 0
        self.rejected = Counter()
        self.last_source_timestamp: int | None = None

    @staticmethod
    def _side(payload: Any, side: str) -> tuple[np.ndarray, dict[str, Any]]:
        if not isinstance(payload, dict):
            raise ValueError(f"missing {side} object")
        if payload.get("side", side) != side:
            raise ValueError(f"{side} side tag mismatch")
        actual = np.asarray(payload.get("actual_q"), dtype=np.float32)
        if actual.shape != (6,) or not np.isfinite(actual).all():
            raise ValueError(f"{side}.actual_q invalid")
        if np.any(actual < 0) or np.any(actual > 100):
            raise ValueError(f"{side}.actual_q outside training range 0..100")
        legacy_valid = bool(payload.get("valid", False))
        feedback_valid = bool(payload.get("feedback_valid", legacy_valid))
        feedback_age_ms = float(payload.get("feedback_age_ms", payload.get("age_ms", -1)))
        if not np.isfinite(feedback_age_ms) or feedback_age_ms < 0:
            raise ValueError(f"{side}.feedback_age_ms invalid")
        return actual, {
            "feedback_valid": feedback_valid,
            "feedback_age_ms": feedback_age_ms,
            "timestamp_ns": int(payload.get("timestamp_ns", 0)),
        }

    def poll(self) -> TimedDatum | None:
        while True:
            try:
                raw = self.socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            receive_wall_ns = time.time_ns()
            receive_monotonic_ns = time.monotonic_ns()
            try:
                message = msgpack.unpackb(raw, raw=False)
                if int(message.get("schema_version", 0)) != 2:
                    raise ValueError("Live Shadow requires atomic dual O6 schema v2")
                left, left_meta = self._side(message.get("left"), "left")
                right, right_meta = self._side(message.get("right"), "right")
                source_timestamp = int(message.get("timestamp_ns", 0))
                duplicate = source_timestamp == self.last_source_timestamp
                self.last_source_timestamp = source_timestamp
                self.sequence += 1
                self.latest = TimedDatum(
                    source="dual_o6",
                    source_timestamp=source_timestamp,
                    receive_wall_ns=receive_wall_ns,
                    receive_monotonic_ns=receive_monotonic_ns,
                    sequence=self.sequence,
                    value={"left_o6": left, "right_o6": right},
                    metadata={
                        "endpoint": self.endpoint,
                        "left": left_meta,
                        "right": right_meta,
                        "duplicate": duplicate,
                    },
                )
            except Exception as exc:
                self.rejected[f"{type(exc).__name__}:{exc}"] += 1
        return self.latest

    def stop(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


def projected_gravity_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("base quaternion must be finite wxyz shape (4,)")
    norm = float(np.linalg.norm(quaternion))
    if not 0.95 <= norm <= 1.05:
        raise ValueError(f"base quaternion norm invalid: {norm}")
    rotation = Rotation.from_quat(quaternion / norm, scalar_first=True)
    return rotation.inv().apply([0.0, 0.0, -1.0]).astype(np.float32)


class G1LiveObservationSource(ObservationSource):
    """Synchronize real read-only sources and fail closed on every bad sample."""

    def __init__(self, config: dict[str, Any], modality_config: dict[str, Any]) -> None:
        self.config = config
        self.modality_config = modality_config
        self._validate_security_config()
        script = Path(__file__).with_name("g1_lowstate_stdout.py")
        self.lowstate = LowStateProcessReader(
            str(config["unitree_reader_python"]),
            script,
            str(config["g1_wired_interface"]),
            str(config["unitree_lowstate_topic"]),
        )
        self.camera = CameraZmqReader(str(config["camera_endpoint"]))
        self.o6 = O6ZmqReadOnlyReader(
            f"tcp://{config['g1_host']}:{int(config['o6_state_port'])}"
        )
        self.counters = Counter()
        self.last_camera_sequence = 0
        self.frame_index = 0
        self.started = False
        self.camera_clock_offset_ns: int | None = None
        self.receive_intervals: dict[str, list[float]] = {
            "camera": [], "g1": [], "o6": []
        }
        self._last_receive: dict[str, int] = {}

    def _validate_security_config(self) -> None:
        required = {
            "real_hardware_enabled": False,
            "publish_commands": False,
            "shadow_only": True,
            "dry_run": True,
        }
        for key, expected in required.items():
            if self.config.get(key) is not expected:
                raise RuntimeError(f"Live Shadow requires {key}={expected}")
        if int(self.config.get("command_publish_attempt_limit", -1)) != 0:
            raise RuntimeError("command publish attempt limit must be zero")
        if int(self.config.get("control_ownership_request_limit", -1)) != 0:
            raise RuntimeError("control ownership request limit must be zero")

    def start(self) -> None:
        self.lowstate.start()
        deadline = time.monotonic() + float(self.config["startup_timeout_s"])
        while time.monotonic() < deadline:
            self.camera.poll()
            self.o6.poll()
            if self.lowstate.get_latest() and self.camera.latest and self.o6.latest:
                print("REAL ROBOT CONNECTED", flush=True)
                self.started = True
                return
            if self.lowstate.process and self.lowstate.process.poll() is not None:
                raise RuntimeError(
                    "read-only LowState subprocess exited: "
                    + "; ".join(self.lowstate.errors)
                )
            time.sleep(0.01)
        missing = []
        if self.lowstate.get_latest() is None:
            missing.append("DDS rt/lowstate")
        if self.camera.latest is None:
            missing.append("D435i ZMQ")
        if self.o6.latest is None:
            missing.append("dual O6 ZMQ")
        raise TimeoutError("Live observation startup timed out; missing " + ", ".join(missing))

    def _track_receive(self, key: str, datum: TimedDatum) -> None:
        previous = self._last_receive.get(key)
        if previous is not None and datum.receive_monotonic_ns != previous:
            self.receive_intervals[key].append((datum.receive_monotonic_ns - previous) / 1e9)
        self._last_receive[key] = datum.receive_monotonic_ns

    def _reject(self, reason: str) -> None:
        self.counters[f"skipped.{reason}"] += 1

    def get_observation(self) -> ObservationSample | None:
        construct_start_ns = time.monotonic_ns()
        camera = self.camera.poll()
        o6 = self.o6.poll()
        g1 = self.lowstate.get_latest()
        if camera is None or g1 is None or o6 is None:
            self._reject("missing_source")
            return None
        if camera.sequence == self.last_camera_sequence:
            return None
        self.last_camera_sequence = camera.sequence
        self.counters["camera_candidates"] += 1
        for key, datum in (("camera", camera), ("g1", g1), ("o6", o6)):
            self._track_receive(key, datum)

        now_ns = time.monotonic_ns()
        ages_ms = {
            "camera": (now_ns - camera.receive_monotonic_ns) / 1e6,
            "g1": (now_ns - g1.receive_monotonic_ns) / 1e6,
            "o6_transport": (now_ns - o6.receive_monotonic_ns) / 1e6,
            "left_o6_feedback": float(o6.metadata["left"]["feedback_age_ms"])
            + (now_ns - o6.receive_monotonic_ns) / 1e6,
            "right_o6_feedback": float(o6.metadata["right"]["feedback_age_ms"])
            + (now_ns - o6.receive_monotonic_ns) / 1e6,
        }
        camera_source_wall_ns = int(float(camera.source_timestamp) * 1e9)
        observed_camera_clock_delta_ns = camera.receive_wall_ns - camera_source_wall_ns
        if self.camera_clock_offset_ns is None:
            self.camera_clock_offset_ns = observed_camera_clock_delta_ns
        ages_ms["camera_source_clock_corrected"] = max(
            0.0,
            (observed_camera_clock_delta_ns - self.camera_clock_offset_ns) / 1e6,
        )
        if ages_ms["camera"] > float(self.config["camera_stale_timeout_ms"]):
            self._reject("stale_camera")
            return None
        if ages_ms["camera_source_clock_corrected"] > float(
            self.config["camera_stale_timeout_ms"]
        ):
            self._reject("stale_camera_source_timestamp")
            return None
        if ages_ms["g1"] > float(self.config["g1_stale_timeout_ms"]):
            self._reject("stale_g1")
            return None
        for side in ("left", "right"):
            required = bool(self.config[f"require_{side}_o6"])
            if required and not bool(o6.metadata[side]["feedback_valid"]):
                self._reject(f"invalid_{side}_o6")
                return None
            if required and ages_ms[f"{side}_o6_feedback"] > float(
                self.config["o6_stale_timeout_ms"]
            ):
                self._reject(f"stale_{side}_o6")
                return None

        receive_times = np.array(
            [camera.receive_monotonic_ns, g1.receive_monotonic_ns, o6.receive_monotonic_ns]
        )
        cross_modal_skew_ms = float((receive_times.max() - receive_times.min()) / 1e6)
        if cross_modal_skew_ms > float(self.config["timestamp_tolerance_ms"]):
            self._reject("cross_modal_skew")
            return None

        try:
            gravity = projected_gravity_wxyz(g1.value["base_quat_wxyz"])
            state = {
                "left_arm": g1.value["left_arm"],
                "right_arm": g1.value["right_arm"],
                "left_o6": o6.value["left_o6"],
                "right_o6": o6.value["right_o6"],
                "waist": g1.value["waist"],
                "projected_gravity": gravity,
            }
            for key, dimension in STATE_DIMS.items():
                value = np.asarray(state[key], dtype=np.float32)
                if value.shape != (dimension,) or not np.isfinite(value).all():
                    raise ValueError(f"{key} invalid shape/value: {value.shape}")
            image = np.asarray(camera.value)
            flat: dict[str, Any] = {
                **{f"state.{key}": value for key, value in state.items()},
                "video.ego_view": image,
                "annotation.human.task_description": self.config["task_prompt"],
            }
            observation = parse_live_policy_observation(flat, self.modality_config)
        except Exception as exc:
            self._reject(f"invalid_observation.{type(exc).__name__}")
            return None

        self.counters["accepted"] += 1
        self.counters["camera_duplicate"] += int(bool(camera.metadata.get("duplicate")))
        self.counters["o6_duplicate"] += int(bool(o6.metadata.get("duplicate")))
        sample = ObservationSample(
            episode_index=0,
            frame_index=self.frame_index,
            dataset_timestamp=float(camera.source_timestamp or time.time()),
            monotonic_timestamp=now_ns / 1e9,
            observation=observation,
            flat_observation=flat,
            source_metadata={
                "source_timestamps": {
                    "camera_wall_s": camera.source_timestamp,
                    "g1_tick": g1.source_timestamp,
                    "o6_monotonic_ns": o6.source_timestamp,
                },
                "receive_wall_ns": {
                    "camera": camera.receive_wall_ns,
                    "g1": g1.receive_wall_ns,
                    "o6": o6.receive_wall_ns,
                },
                "receive_monotonic_ns": {
                    "camera": camera.receive_monotonic_ns,
                    "g1": g1.receive_monotonic_ns,
                    "o6": o6.receive_monotonic_ns,
                },
                "age_ms": ages_ms,
                "cross_modal_skew_ms": cross_modal_skew_ms,
                "observation_construct_ms": (time.monotonic_ns() - construct_start_ns) / 1e6,
                "g1": g1.metadata,
                "o6": o6.metadata,
                "camera": camera.metadata,
            },
        )
        self.frame_index += 1
        return sample

    def diagnostics(self) -> dict[str, Any]:
        frequency = {}
        jitter_ms = {}
        for key, intervals in self.receive_intervals.items():
            values = np.asarray(intervals, dtype=np.float64)
            frequency[key] = float(1.0 / values.mean()) if len(values) else 0.0
            jitter_ms[key] = float(values.std() * 1000) if len(values) else 0.0
        return {
            "counters": dict(self.counters),
            "frequency_hz": frequency,
            "interval_jitter_ms": jitter_ms,
            "camera_rejected": dict(self.camera.rejected),
            "o6_rejected": dict(self.o6.rejected),
            "lowstate_errors": list(self.lowstate.errors),
            "camera_wall_clock_offset_ms": (
                self.camera_clock_offset_ns / 1e6
                if self.camera_clock_offset_ns is not None
                else None
            ),
        }

    def stop(self) -> None:
        # Closing readers and inference only; no hold/zero/stop command exists.
        self.lowstate.stop()
        self.camera.stop()
        self.o6.stop()
        self.started = False
