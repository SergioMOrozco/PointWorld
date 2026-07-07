# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify build_live_sample() end-to-end on a synthetic RGBD frame + a real SO101 joint
trajectory (no physical camera/robot needed), then feed the result straight into the model.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from online_eval.model_loading import load_pointworld_model
from online_eval.sample_builder import RawCameraFrame, build_live_sample
from online_eval.so101_robot import build_so101_robot_sampler

T = 11


def make_synthetic_frame(h_raw: int = 480, w_raw: int = 640) -> RawCameraFrame:
    rng = np.random.RandomState(0)
    rgb = rng.randint(0, 255, (h_raw, w_raw, 3), dtype=np.uint8)
    # A simple planar "tabletop" depth: roughly 0.5m in front of the camera, small variation.
    depth = np.full((h_raw, w_raw), 0.6, dtype=np.float32)
    depth += rng.uniform(-0.02, 0.02, size=depth.shape).astype(np.float32)
    intrinsic = np.array(
        [[600.0, 0, w_raw / 2], [0, 600.0, h_raw / 2], [0, 0, 1]], dtype=np.float32
    )
    extrinsic = np.eye(4, dtype=np.float32)
    return RawCameraFrame(rgb=rgb, depth=depth, intrinsic=intrinsic, extrinsic_world_to_cam=extrinsic)


def main(model_path: str, device: str) -> None:
    model, args = load_pointworld_model(model_path, device=device)
    sampler = build_so101_robot_sampler(args, device="cpu")

    frame = make_synthetic_frame()

    rng = np.random.RandomState(1)
    joint_traj = np.zeros((T, len(sampler.joint_names)), dtype=np.float32)
    joint_traj[:, sampler.joint_names.index("gripper")] = np.linspace(0.0, 1.0, T)
    joint_traj[:, sampler.joint_names.index("shoulder_pan")] = np.linspace(0.0, 0.3, T)
    gripper_traj = np.linspace(0.0, 1.0, T).astype(np.float32)

    sample = build_live_sample(
        now_frame=frame,
        joint_trajectory=joint_traj,
        gripper_trajectory=gripper_traj,
        sampler=sampler,
        args=args,
        device=device,
        sample_key="synthetic-0",
    )

    print("sample keys:", sorted(sample.keys()))
    for key in ("scene_flows", "scene_features", "scene_exists", "robot_flows", "robot_features", "robot_exists"):
        v = sample[key]
        print(f"  {key}: shape={tuple(v.shape)} dtype={v.dtype}")

    with torch.no_grad():
        out = model(sample, training=False)

    assert torch.isfinite(out["scene_flows"]).all(), "predicted scene_flows has non-finite values"
    assert torch.isfinite(out["confidence"]).all(), "confidence has non-finite values"
    print("pred scene_flows shape:", tuple(out["scene_flows"].shape))
    print("confidence shape:", tuple(out["confidence"].shape))
    print("OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path", type=str, default="pretrained_checkpoints/small-droid/model-best.pt"
    )
    parser.add_argument("--device", type=str, default="cuda")
    cli_args = parser.parse_args()
    main(cli_args.model_path, cli_args.device)
