# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verification step 1 (plan): standalone RobotSampler/SO101 sanity check.

Sweeps the gripper joint through its URDF limits (plus a few other joints), dumps the
presampled+FK'd gripper point cloud into a plain viser.ViserServer with a frame slider, and
prints per-frame bounding-box stats. No camera/model/checkpoint needed.

Open http://localhost:<port> and step through the slider to confirm visually: the two point
clusters should look like a parallel-jaw gripper, and the jaw should visibly open/close as the
slider moves (this mirrors the numeric bbox-extent check already done in
online_eval/tools/inspect_so101_urdf.py, which showed extent growing 0.067 -> 0.137m across the
sweep -- this script is the visual counterpart of that same check).
"""

import argparse
import time

import numpy as np
import torch
import viser

from online_eval.so101_robot import build_so101_robot_sampler

NUM_FRAMES = 20


def main(port: int, max_robot_points: int) -> None:
    args = argparse.Namespace(max_robot_points=max_robot_points)
    sampler = build_so101_robot_sampler(args, device="cpu")
    joint_names = sampler.joint_names
    n_joints = len(joint_names)

    gripper_idx = joint_names.index("gripper")
    shoulder_idx = joint_names.index("shoulder_pan")

    joint_traj = np.zeros((NUM_FRAMES, n_joints), dtype=np.float32)
    joint_traj[:, gripper_idx] = np.linspace(-0.17, 1.74, NUM_FRAMES)
    joint_traj[:, shoulder_idx] = np.linspace(-0.5, 0.5, NUM_FRAMES)

    joint_values = {
        name: torch.as_tensor(joint_traj[:, i], dtype=torch.float32) for i, name in enumerate(joint_names)
    }
    points, colors, _normals = sampler.compute_points(joint_values)
    points = points.numpy()
    colors = colors.numpy()

    for f in range(NUM_FRAMES):
        extent = points[f].max(axis=0) - points[f].min(axis=0)
        print(f"frame={f:2d} gripper={joint_traj[f, gripper_idx]:+.3f} bbox_extent={extent}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    print(f"[viser] open http://localhost:{port}")

    frame_slider = server.gui.add_slider("frame", min=0, max=NUM_FRAMES - 1, step=1, initial_value=0)
    point_cloud_handle = server.scene.add_point_cloud(
        "/so101_gripper", points=points[0], colors=colors[0], point_size=0.003
    )

    @frame_slider.on_update
    def _(_) -> None:
        f = int(frame_slider.value)
        point_cloud_handle.points = points[f]
        point_cloud_handle.colors = colors[f]

    print("Server running -- Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--max_robot_points", type=int, default=500)
    cli_args = parser.parse_args()
    main(cli_args.port, cli_args.max_robot_points)
