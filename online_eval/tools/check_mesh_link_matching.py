# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare full-4x4-Frobenius mesh-to-link matching (current robot_sampler.py behavior)
against translation-only matching, for every URDF in assets/, to check whether switching
the metric would change any attachment for the existing production robots (Franka, R1Pro)
before applying it as a fix for SO101.
"""

import numpy as np
import torch

from pointworld.urdfpy_compat import ensure_urdfpy_numpy_compat

ensure_urdfpy_numpy_compat()

from robot_sampler import RobotSampler, get_mesh_name  # noqa: E402

URDFS = {
    "franka+robotiq (droid)": "assets/franka_description/franka_panda_robotiq_2f85.urdf",
    "r1pro (behavior)": "assets/r1pro/urdf/r1pro.urdf",
    "so101": "assets/so101/so101_new_calib.urdf",
}


def check(name: str, urdf_path: str) -> None:
    print(f"\n=== {name}: {urdf_path} ===")
    sampler = RobotSampler(urdf_path=urdf_path, gripper_only=False, device="cpu")

    reference_cfg = sampler.joint_defaults.copy()
    fk_ref = sampler.robot_urdf.visual_trimesh_fk(cfg=reference_cfg)
    zero_cfg = {jn: torch.zeros(1, device=sampler.device, dtype=sampler.dtype) for jn in sampler.joint_names}
    link_tf_ref = sampler.chain.forward_kinematics(zero_cfg)

    link_mats = {ln: tf.get_matrix()[0].cpu().numpy() for ln, tf in link_tf_ref.items()}

    n_meshes = 0
    n_disagree = 0
    for i, mesh in enumerate(fk_ref):
        mesh_name = get_mesh_name(mesh, i)
        mesh_T = fk_ref[mesh]
        n_meshes += 1

        best_full, best_full_link = float("inf"), None
        best_trans, best_trans_link = float("inf"), None
        for link_name, link_mat in link_mats.items():
            err_full = np.linalg.norm(link_mat - mesh_T, ord="fro")
            err_trans = np.linalg.norm(link_mat[:3, 3] - mesh_T[:3, 3])
            if err_full < best_full:
                best_full, best_full_link = err_full, link_name
            if err_trans < best_trans:
                best_trans, best_trans_link = err_trans, link_name

        if best_full_link != best_trans_link:
            n_disagree += 1
            print(f"  DISAGREE mesh={mesh_name!r:40s} full->{best_full_link!r:25s} trans->{best_trans_link!r}")

    print(f"  Total meshes: {n_meshes}, disagreements: {n_disagree}")


if __name__ == "__main__":
    for name, path in URDFS.items():
        check(name, path)
