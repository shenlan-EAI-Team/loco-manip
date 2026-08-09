import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "control/envs/g1/utils/command_sender.py"
)


class MotorCommand:
    def __init__(self):
        self.mode = 0
        self.q = 0.0
        self.dq = 0.0
        self.tau = 0.0
        self.kp = 0.0
        self.kd = 0.0


class LowCommand:
    def __init__(self):
        self.head = [0, 0]
        self.level_flag = 0
        self.gpio = 0
        self.mode_machine = 0
        self.mode_pr = 0
        self.motor_cmd = [MotorCommand() for _ in range(35)]
        self.crc = 0


class Publisher:
    instances = []

    def __init__(self, topic, message_type):
        self.topic = topic
        self.message_type = message_type
        self.records = []
        self.instances.append(self)

    def Init(self):
        return None

    def Write(self, message):
        self.records.append(message)


def module(name: str, **values):
    value = ModuleType(name)
    for key, item in values.items():
        setattr(value, key, item)
    return value


def load_sender(monkeypatch):
    fake_modules = {
        "unitree_sdk2py": module("unitree_sdk2py"),
        "unitree_sdk2py.core": module("unitree_sdk2py.core"),
        "unitree_sdk2py.core.channel": module(
            "unitree_sdk2py.core.channel", ChannelPublisher=Publisher
        ),
        "unitree_sdk2py.idl": module("unitree_sdk2py.idl"),
        "unitree_sdk2py.idl.default": module(
            "unitree_sdk2py.idl.default",
            unitree_hg_msg_dds__HandCmd_=LowCommand,
            unitree_hg_msg_dds__LowCmd_=LowCommand,
        ),
        "unitree_sdk2py.idl.unitree_hg": module("unitree_sdk2py.idl.unitree_hg"),
        "unitree_sdk2py.idl.unitree_hg.msg": module("unitree_sdk2py.idl.unitree_hg.msg"),
        "unitree_sdk2py.idl.unitree_hg.msg.dds_": module(
            "unitree_sdk2py.idl.unitree_hg.msg.dds_",
            HandCmd_=LowCommand,
            LowCmd_=LowCommand,
        ),
        "unitree_sdk2py.utils": module("unitree_sdk2py.utils"),
        "unitree_sdk2py.utils.crc": module(
            "unitree_sdk2py.utils.crc", CRC=lambda: SimpleNamespace(Crc=lambda _: 123)
        ),
    }
    for name, value in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, value)
    spec = importlib.util.spec_from_file_location("tested_command_sender", MODULE_PATH)
    loaded = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(loaded)
    return loaded.BodyCommandSender


def test_real_body_command_sender_is_hard_disabled_before_publisher_construction(
    monkeypatch,
) -> None:
    Publisher.instances.clear()
    sender_class = load_sender(monkeypatch)
    config = {
        "ENV_TYPE": "real",
        "ROBOT_TYPE": "g1_29dof",
        "NUM_MOTORS": 29,
        "NUM_JOINTS": 29,
        "MOTOR_KP": [1.0] * 29,
        "MOTOR_KD": [0.1] * 29,
        "WeakMotorJointIndex": {},
        "JOINT2MOTOR": list(range(29)),
        "MOTOR2JOINT": list(range(29)),
        "DEFAULT_MOTOR_ANGLES": [0.0] * 29,
        "UNITREE_LEGGED_CONST": {
            "PosStopF": 2146000000.0,
            "VelStopF": 16000.0,
            "MODE_MACHINE": 5,
            "MODE_PR": 0,
        },
    }
    with pytest.raises(RuntimeError, match="only permitted rt/lowcmd writer"):
        sender_class(config)
    assert Publisher.instances == []
