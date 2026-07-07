# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One-off inspection script for the SO101 URDF.

Prints every visual mesh's source filename + parent link name (from urdfpy), and the joint
name order RobotSampler actually uses for forward kinematics (from pytorch_kinematics). Run
this once when bringing up a new robot asset to confirm assumptions before wiring it into
online_eval/so101_robot.py -- don't hardcode names from reading the URDF XML alone, since
joint traversal order depends on pytorch_kinematics's URDF tree walk.
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pointworld.urdfpy_compat import ensure_urdfpy_numpy_compat

ensure_urdfpy_numpy_compat()

import urdfpy  # noqa: E402

from robot_sampler import RobotSampler  # noqa: E402

DEFAULT_URDF = "assets/so101/so101_new_calib.urdf"


def inspect_urdf(urdf_path: str) -> None:
    robot = urdfpy.URDF.load(urdf_path)

    print(f"=== Links & visual meshes ({urdf_path}) ===")
    for link in robot.links:
        for visual in link.visuals:
            mesh_filename = None
            if visual.geometry is not None and visual.geometry.mesh is not None:
                mesh_filename = visual.geometry.mesh.filename
            print(f"  link={link.name!r:35s} mesh_filename={mesh_filename!r}")

    print("\n=== Joints (URDF declaration order) ===")
    for joint in robot.joints:
        print(f"  name={joint.name!r:20s} type={joint.joint_type!r:10s} parent={joint.parent!r:20s} child={joint.child!r}")


def inspect_robot_sampler_joint_order(urdf_path: str) -> "RobotSampler":
    print("\n=== RobotSampler.joint_names (actual FK order, non-fixed joints) ===")
    sampler = RobotSampler(
        urdf_path=urdf_path,
        gripper_only=True,
        device="cpu",
        link_whitelist=["gripper_link", "moving_jaw_so101_v1_link"],
        mesh_link_match_mode="translation",
    )
    print(f"  joint_names = {sampler.joint_names}")

    print("\n=== sampler._mesh_to_link (ground truth mesh->link mapping) ===")
    for mesh_name, link_name in sorted(sampler._mesh_to_link.items(), key=lambda kv: kv[1]):
        print(f"  mesh={mesh_name!r:40s} -> link={link_name!r}")

    print("\n=== presample() mesh coverage (link_whitelist restricted) ===")
    sampler.presample(num_points=500, seed=0)
    for mesh_name, points in sampler._presampled_points.items():
        link_name = sampler._mesh_to_link.get(mesh_name)
        print(f"  mesh={mesh_name!r:40s} link={link_name!r:30s} n_points={points.shape[0]}")

    return sampler


def diagnose_mismatch(sampler: RobotSampler, target_mesh_name: str) -> None:
    import numpy as np

    reference_cfg = sampler.joint_defaults.copy()
    fk_ref = sampler.robot_urdf.visual_trimesh_fk(cfg=reference_cfg)
    zero_cfg = {jn: __import__("torch").zeros(1, device=sampler.device, dtype=sampler.dtype) for jn in sampler.joint_names}
    link_tf_ref = sampler.chain.forward_kinematics(zero_cfg)

    mesh_T = None
    for i, mesh in enumerate(fk_ref):
        from robot_sampler import get_mesh_name
        if get_mesh_name(mesh, i) == target_mesh_name:
            mesh_T = fk_ref[mesh]
            break
    assert mesh_T is not None, f"mesh {target_mesh_name} not found"

    print(f"\n=== Diagnosing '{target_mesh_name}' ===")
    print(f"mesh_T (from urdfpy visual_trimesh_fk):\n{np.round(mesh_T, 4)}")

    dists = []
    for link_name, link_tf in link_tf_ref.items():
        link_mat = link_tf.get_matrix()[0].cpu().numpy()
        err = np.linalg.norm(link_mat - mesh_T, ord="fro")
        dists.append((err, link_name, link_mat))
    dists.sort(key=lambda x: x[0])
    print("Closest links by Frobenius norm:")
    for err, link_name, link_mat in dists[:6]:
        print(f"  err={err:.6f}  link={link_name!r}")
        print(f"    {np.round(link_mat, 4)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf_path", type=str, default=DEFAULT_URDF)
    args = parser.parse_args()

    urdf_path = str(Path(args.urdf_path).resolve())
    inspect_urdf(urdf_path)
    sampler = inspect_robot_sampler_joint_order(urdf_path)
    diagnose_mismatch(sampler, "wrist_roll_follower_so101_v1.stl_15")
    diagnose_mismatch(sampler, "moving_jaw_so101_v1.stl_16")
