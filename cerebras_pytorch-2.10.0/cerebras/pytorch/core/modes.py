# Copyright 2016-2023 Cerebras Systems
# SPDX-License-Identifier: BSD-3-Clause

from cerebras.appliance.pb.workflow.appliance.common.common_config_pb2 import (
    ExecutionModeKey,
)

TRAIN = "train"
EVAL = "eval"
TRAIN_AND_EVAL = "train_and_eval"
INFERENCE = "inference"
EVAL_ALL = "eval_all"


def get_modes():
    return (TRAIN, EVAL, TRAIN_AND_EVAL, INFERENCE, EVAL_ALL)


def is_valid(mode):
    return mode in get_modes()


def map_mode_to_modekey(mode: str) -> int:
    """Maps PyTorch mode string to appliance mode key."""
    if mode == TRAIN:
        return ExecutionModeKey.EMK_TRAIN
    elif mode == EVAL:
        return ExecutionModeKey.EMK_EVAL
    else:
        raise ValueError(f"Unsupported mode: {mode}")
