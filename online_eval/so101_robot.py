# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SO101 RobotSampler configuration for the live/online PointWorld evaluation.

See online_eval/tools/inspect_so101_urdf.py and check_mesh_link_matching.py for how this
configuration was derived and verified: SO101's gripper-region meshes ('wrist_roll_follower'
on gripper_link, 'moving_jaw' on moving_jaw_so101_v1_link) don't match RobotSampler's default
GRIPPER_KEYWORDS, and its default mesh-to-link matching (full 4x4 Frobenius norm) misattaches
them due to CAD-typical rotational offsets in their <visual><origin> tags -- both are fixed in
robot_sampler.py (extended GRIPPER_KEYWORDS, opt-in mesh_link_match_mode='translation'),
verified not to change Franka/R1Pro's existing behavior.
"""

from pathlib import Path

from robot_sampler import RobotSampler

SO101_URDF_PATH = str(
    Path(__file__).resolve().parent.parent / "assets" / "so101" / "so101_new_calib.urdf"
)
SO101_GRIPPER_LINKS = ["gripper_link", "moving_jaw_so101_v1_link"]
SO101_PRESAMPLE_SEED = 0


def build_so101_robot_sampler(args, device: str = "cpu") -> RobotSampler:
    """Construct and presample() a RobotSampler for the SO101 gripper.

    presample() is called once here with a fixed seed (mirrors how the existing
    droid/behavior eval dataloader always presamples with the same seed per rank -- see
    dataset_components/robot.py's droid path) and the resulting point set is reused for every
    subsequent compute_points() call; it is not re-randomized per inference tick.
    """
    sampler = RobotSampler(
        urdf_path=SO101_URDF_PATH,
        gripper_only=True,
        device=device,
        link_whitelist=SO101_GRIPPER_LINKS,
        mesh_link_match_mode="translation",
    )
    sampler.presample(num_points=args.max_robot_points, seed=SO101_PRESAMPLE_SEED)
    return sampler
