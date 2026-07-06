# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verification step 2 (plan): load a pretrained checkpoint via load_pointworld_model, build an
all-synthetic data_dict of the right shapes/dtypes tagged domain "so101", and confirm
BaseModel.forward runs without the unknown-domain assertion or norm-stats errors, returning
finite scene_flows/confidence. Doesn't require a camera or robot -- pure shape/contract check.
"""

import argparse

import torch

from online_eval.model_loading import load_pointworld_model

T = 11  # CONTEXT_HORIZON(1) + PRED_HORIZON(10), see pointworld/base.py
H, W = 180, 320


def main(model_path: str, device: str, num_scene_points: int) -> None:
    model, args = load_pointworld_model(model_path, device=device)
    print(f"scene_features_dim={model.data_info_dict['scene_features_dim']}")
    print(f"robot_features_dim={model.data_info_dict['robot_features_dim']}")

    Ns = num_scene_points
    Nr = args.max_robot_points
    Ds = model.data_info_dict["scene_features_dim"]
    Fr = model.data_info_dict["robot_features_dim"]

    data_dict = {
        "scene_flows": torch.randn(1, T, Ns, 3, device=device),
        "scene_features": torch.randn(1, 1, Ns, Ds, device=device),
        "scene_exists": torch.ones(1, T, Ns, dtype=torch.bool, device=device),
        "robot_flows": torch.randn(1, T, Nr, 3, device=device),
        "robot_features": torch.randn(1, T, Nr, Fr, device=device),
        "robot_exists": torch.ones(1, T, Nr, dtype=torch.bool, device=device),
        "cam0_initial_rgb": torch.randint(0, 255, (1, H, W, 3), dtype=torch.uint8, device=device),
        "cam0_initial_depth": torch.rand(1, H, W, device=device) * 2.0 + 0.3,
        "cam0_intrinsic": torch.tensor(
            [[300.0, 0, W / 2], [0, 300.0, H / 2], [0, 0, 1]], device=device
        ).unsqueeze(0),
        "cam0_extrinsic": torch.eye(4, device=device).unsqueeze(0),
        "__domain__": ["so101"],
    }

    with torch.no_grad():
        out = model(data_dict, training=False)

    assert out["scene_flows"].shape == (1, T, Ns, 3)
    assert torch.isfinite(out["scene_flows"]).all()
    assert torch.isfinite(out["confidence"]).all()
    print(f"pred scene_flows shape: {tuple(out['scene_flows'].shape)}  all finite: True")
    print(f"confidence shape: {tuple(out['confidence'].shape)}  all finite: True")
    print("OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path", type=str, default="pretrained_checkpoints/small-droid/model-best.pt"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_scene_points", type=int, default=800)
    cli_args = parser.parse_args()
    main(cli_args.model_path, cli_args.device, cli_args.num_scene_points)
