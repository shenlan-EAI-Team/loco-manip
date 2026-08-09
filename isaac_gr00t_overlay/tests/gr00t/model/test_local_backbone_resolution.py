# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

from gr00t.model.gr00t_n1d7.gr00t_n1d7 import get_backbone_cls
import pytest


def test_get_backbone_cls_resolves_local_qwen3_vl_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))

    backbone_cls = get_backbone_cls(SimpleNamespace(model_name=str(tmp_path)))

    assert backbone_cls.__name__ == "Qwen3Backbone"


def test_get_backbone_cls_rejects_other_local_model_type(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bert"}))

    with pytest.raises(ValueError, match="Unsupported model name"):
        get_backbone_cls(SimpleNamespace(model_name=str(tmp_path)))
