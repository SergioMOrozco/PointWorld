# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load a pretrained PointWorld checkpoint directly into BaseModel for live inference,
bypassing Trainer/Tester (no DDP, no optimizer, no WebDataset dataloader needed).

Mirrors the dim-inference pattern training/trainer.py uses when inference_only=True and
data_info_dict is None (see Trainer.__init__, training/trainer.py:82-104), and the
checkpoint-contract application pattern used by both Trainer and Tester on load.
"""

import argparse

import torch

from arguments import parse_args
from pointworld.base import BaseModel
from pointworld.checkpoint_contract import apply_model_contract_to_args, read_checkpoint_contract

SO101_NORM_STATS_PATH = "stats/so101"
SO101_DOMAIN = "so101"


def load_pointworld_model(
    model_path: str, device: str = "cuda"
) -> tuple[BaseModel, argparse.Namespace]:
    args = parse_args(skip_command_line=True)
    args.device = device
    args.distributed = False
    args.disable_compile = True

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    context = f"checkpoint '{model_path}'"
    model_contract, _data_contract = read_checkpoint_contract(checkpoint, context=context)
    apply_model_contract_to_args(
        args, model_contract, context=context, explicit_cli_dests=set()
    )

    # Override the checkpoint's own droid/behavior norm_stats_path + domains: SO101 is a new,
    # unseen-at-training-time domain aliased to droid's stats (see stats/so101/norm_stats.json).
    # Must happen before BaseModel construction below, which reads these in _init_norm_stats().
    args.norm_stats_path = SO101_NORM_STATS_PATH
    args.domains = [SO101_DOMAIN]

    state = checkpoint["model"]
    data_info_dict = {
        "scene_features_dim": int(
            state["scene_feature_encoder.scene_raw_feat_proj.weight"].shape[1]
        ),
        "robot_features_dim": int(state["robot_proj.fc1.weight"].shape[1]),
    }

    model = BaseModel(args, data_info_dict, rank=0, cpu_pg=None)
    model.load_state_dict(state)
    model.to(args.device)
    model.eval()

    return model, args
