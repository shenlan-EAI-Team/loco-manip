from types import ModuleType, SimpleNamespace
import sys

import numpy as np

from decoupled_wbc.control.real_safe.lowcmd_guard.unitree_backend import UnitreeLowCmdWriter

from .test_lowcmd_guard_core import guard_config, ready_core


class Motor:
    def __init__(self):
        self.mode = 99
        self.q = 99.0
        self.dq = 99.0
        self.kp = 99.0
        self.kd = 99.0
        self.tau = 99.0


class Message:
    def __init__(self):
        self.mode_pr = 99
        self.mode_machine = 99
        self.motor_cmd = [Motor() for _ in range(35)]
        self.crc = 0


class Publisher:
    instances = []

    def __init__(self, topic, message_type):
        self.topic = topic
        self.message_type = message_type
        self.messages = []
        self.closed = False
        self.instances.append(self)

    def Init(self):
        return None

    def Write(self, message):
        self.messages.append(message)

    def Close(self):
        self.closed = True


def module(name, **items):
    value = ModuleType(name)
    for key, item in items.items():
        setattr(value, key, item)
    return value


def install_fake_sdk(monkeypatch):
    modules = {
        "unitree_sdk2py": module("unitree_sdk2py"),
        "unitree_sdk2py.core": module("unitree_sdk2py.core"),
        "unitree_sdk2py.core.channel": module(
            "unitree_sdk2py.core.channel", ChannelPublisher=Publisher
        ),
        "unitree_sdk2py.idl": module("unitree_sdk2py.idl"),
        "unitree_sdk2py.idl.default": module(
            "unitree_sdk2py.idl.default", unitree_hg_msg_dds__LowCmd_=Message
        ),
        "unitree_sdk2py.idl.unitree_hg": module("unitree_sdk2py.idl.unitree_hg"),
        "unitree_sdk2py.idl.unitree_hg.msg": module("unitree_sdk2py.idl.unitree_hg.msg"),
        "unitree_sdk2py.idl.unitree_hg.msg.dds_": module(
            "unitree_sdk2py.idl.unitree_hg.msg.dds_", LowCmd_=Message
        ),
        "unitree_sdk2py.utils": module("unitree_sdk2py.utils"),
        "unitree_sdk2py.utils.crc": module(
            "unitree_sdk2py.utils.crc", CRC=lambda: SimpleNamespace(Crc=lambda _: 0xA5A5)
        ),
    }
    for name, value in modules.items():
        monkeypatch.setitem(sys.modules, name, value)


def test_real_writer_serializes_all_35_slots_and_crc(monkeypatch) -> None:
    Publisher.instances.clear()
    install_fake_sdk(monkeypatch)
    command = ready_core(guard_config()).prepared_command
    assert command is not None

    writer = UnitreeLowCmdWriter()
    assert len(Publisher.instances) == 1
    assert Publisher.instances[0].topic == "rt/lowcmd"
    writer.write(command)

    message = Publisher.instances[0].messages[0]
    assert message.mode_pr == 0
    assert message.mode_machine == 5
    assert message.crc == 0xA5A5
    for index in range(29):
        motor = message.motor_cmd[index]
        assert motor.mode == 1
        assert motor.q == command.q[index]
        assert motor.dq == 0.0
        assert motor.kp == command.kp[index]
        assert motor.kd == command.kd[index]
        assert motor.tau == 0.0
    for index in range(29, 35):
        motor = message.motor_cmd[index]
        assert (motor.mode, motor.q, motor.dq, motor.kp, motor.kd, motor.tau) == (
            0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    np.testing.assert_array_equal(command.q, ready_core(guard_config()).prepared_command.q)
    writer.close()
    assert Publisher.instances[0].closed is True
