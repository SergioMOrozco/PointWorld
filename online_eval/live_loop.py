# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live/online PointWorld evaluation on a RealSense camera + LeRobot SO101 leader/follower rig.

This is the integration point for real hardware. The control loop itself (buffer -> window ->
sample -> model forward -> viser visualization) is fully implemented and verified (see
online_eval/tools/verify_*.py and --dry_run below); the two hardware-facing callables
(`read_camera_frame`, `read_leader_state`) are where your actual pyrealsense2 / LeRobot SO101
driver calls plug in -- marked TODO below, since real hardware wasn't available to test this
against directly. Everything upstream of those two functions has been verified against
synthetic data standing in for them.
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# Allow running this file directly (`python online_eval/live_loop.py`), not just as
# `python -m online_eval.live_loop` -- the latter puts the repo root on sys.path automatically,
# the former does not (sys.path[0] becomes this file's own directory instead).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from online_eval.model_loading import load_pointworld_model
from online_eval.rolling_buffer import N_STEPS_PER_MODEL_STEP, RollingBuffer
from online_eval.sample_builder import RawCameraFrame, backproject_scene, build_live_sample
from online_eval.so101_robot import build_so101_robot_sampler
from visualization.prediction_viz.config import PredictionVisualizerConfig
from visualization.prediction_viz.sample import build_sample_from_dictionary
from visualization.prediction_viz.visualizer import PredictionVisualizer
from visualization.viser_tools.visualization_utils import CameraObservation

SO101_URDF_PATH = "assets/so101/so101_new_calib.urdf"


# --------------------------------------------------------------------------------------- #
# Hardware-facing callables -- TODO: replace with real pyrealsense2 / LeRobot SO101 calls.
# --------------------------------------------------------------------------------------- #
def read_camera_frame_TODO(pipeline) -> RawCameraFrame:
    """Grab one RGBD frame from the RealSense camera.

    TODO(user): implement using pyrealsense2, e.g.:
        frames = pipeline.wait_for_frames()
        color = np.asanyarray(frames.get_color_frame().get_data())          # (H,W,3) uint8, BGR
        depth_raw = np.asanyarray(frames.get_depth_frame().get_data())      # (H,W) uint16, mm
        depth_m = depth_raw.astype(np.float32) / 1000.0                    # convert mm -> meters
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    intrinsic/extrinsic: intrinsic comes from the RealSense stream profile; extrinsic
    (world_to_cam, i.e. camera-to-robot-base calibration) should already be available per your
    "hardware fully working" setup -- load it once at startup rather than every frame.
    """
    raise NotImplementedError("Plug in your RealSense (pyrealsense2) capture here.")


def read_leader_state_TODO(so101_leader) -> tuple[np.ndarray, float]:
    """Read the current SO101 leader arm's joint positions + gripper position.

    TODO(user): implement using your LeRobot SO101 leader driver, e.g.:
        obs = so101_leader.get_observation()
        joint_positions = np.array([obs[name] for name in ARM_JOINT_ORDER], dtype=np.float32)
        gripper_position = float(obs["gripper"])
    joint_positions must be in the SAME order as sampler.joint_names (see
    online_eval/tools/inspect_so101_urdf.py to confirm this order, and convert your driver's
    joint-value convention/units to match the URDF's radians if needed).
    """
    raise NotImplementedError("Plug in your LeRobot SO101 leader driver read here.")


# --------------------------------------------------------------------------------------- #
# Dry-run stand-ins, for exercising the full control loop without physical hardware.
# --------------------------------------------------------------------------------------- #
def _dry_run_camera_frame(t: float) -> RawCameraFrame:
    h_raw, w_raw = 480, 640
    rng = np.random.RandomState(0)
    rgb = rng.randint(0, 255, (h_raw, w_raw, 3), dtype=np.uint8)
    depth = np.full((h_raw, w_raw), 0.6, dtype=np.float32)
    intrinsic = np.array([[600.0, 0, w_raw / 2], [0, 600.0, h_raw / 2], [0, 0, 1]], dtype=np.float32)
    extrinsic = np.eye(4, dtype=np.float32)
    return RawCameraFrame(rgb=rgb, depth=depth, intrinsic=intrinsic, extrinsic_world_to_cam=extrinsic)


def _make_dry_run_leader_reader(joint_names: list[str]) -> Callable[[float], tuple[np.ndarray, float]]:
    def _read(t: float) -> tuple[np.ndarray, float]:
        joints = np.zeros(len(joint_names), dtype=np.float32)
        joints[joint_names.index("shoulder_pan")] = 0.3 * np.sin(0.5 * t)
        gripper = 0.5 + 0.5 * np.sin(1.0 * t)
        joints[joint_names.index("gripper")] = gripper
        return joints, float(gripper)

    return _read


def run_live_loop(
    model_path: str,
    device: str,
    viewer_port: int,
    n_steps_per_model_step: int,
    dry_run: bool,
    max_iterations: Optional[int] = None,
    tick_hz: float = 10.0,
) -> None:
    model, args = load_pointworld_model(model_path, device=device)
    sampler = build_so101_robot_sampler(args, device="cpu")

    viz_config = PredictionVisualizerConfig(viewer_port=viewer_port)
    visualizer = PredictionVisualizer(viz_config, urdf_path=SO101_URDF_PATH)

    buffer = RollingBuffer(n_steps_per_model_step=n_steps_per_model_step)
    live_session = None

    if dry_run:
        leader_reader = _make_dry_run_leader_reader(sampler.joint_names)

    iteration = 0
    dt = 1.0 / tick_hz
    t0_wall = time.monotonic()
    while max_iterations is None or iteration < max_iterations:
        now = time.monotonic() - t0_wall

        if dry_run:
            frame = _dry_run_camera_frame(now)
            joint_positions, gripper_position = leader_reader(now)
        else:
            frame = read_camera_frame_TODO(pipeline=None)
            joint_positions, gripper_position = read_leader_state_TODO(so101_leader=None)

        buffer.push(frame, joint_positions, gripper_position, timestamp=now)
        window = buffer.try_get_window()

        if window is not None:
            sample = build_live_sample(
                now_frame=window.t0_frame,
                joint_trajectory=window.joint_trajectory,
                gripper_trajectory=window.gripper_trajectory,
                sampler=sampler,
                args=args,
                device=device,
                sample_key=f"live-{window.t0_timestamp:.3f}",
            )

            import torch

            with torch.no_grad():
                out = model(sample, training=False)
            pred_scene_flows = out["scene_flows"][0].detach().cpu().numpy()

            sample_np = {}
            for key, value in sample.items():
                if hasattr(value, "detach"):
                    sample_np[key] = value[0].detach().cpu().numpy()
                elif isinstance(value, list):
                    sample_np[key] = value[0]
                else:
                    sample_np[key] = value

            viz_sample = build_sample_from_dictionary(
                sample_dict=sample_np, predictions={"scene_flows": pred_scene_flows}
            )
            result = visualizer.visualize(
                viz_sample, launch_viewer=(live_session is None), live_session=live_session
            )
            live_session = result.get("live_session", live_session)

            # "Actual future" overlay: backproject the buffer's now-frame (once real time has
            # caught up to it) directly onto the same live server. visualize() resets the scene
            # graph each call (see SceneBuilder.populate_existing), so this must be re-added
            # every iteration, after visualize().
            if live_session is not None:
                actual_points, actual_colors, _actual_normals, _rgb, _depth, _intr = backproject_scene(
                    window.now_frame
                )
                try:
                    live_session.server.scene.add_point_cloud(
                        "/actual_future",
                        points=actual_points,
                        colors=actual_colors,
                        point_size=0.004,
                    )
                except Exception as exc:  # pragma: no cover - viser server may not be ready yet
                    print(f"[live_loop] failed to add actual_future overlay: {exc}")

        iteration += 1
        time.sleep(max(0.0, dt))

    if live_session is not None:
        live_session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_path", type=str, default="pretrained_checkpoints/small-droid/model-best.pt"
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--viewer_port", type=int, default=8080)
    parser.add_argument("--n_steps_per_model_step", type=int, default=N_STEPS_PER_MODEL_STEP)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Use synthetic camera/leader data instead of real hardware (structural test).",
    )
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--tick_hz", type=float, default=10.0)
    cli_args = parser.parse_args()

    if not cli_args.dry_run:
        raise SystemExit(
            "Real-hardware mode is not implemented yet -- read_camera_frame_TODO / "
            "read_leader_state_TODO in this file need your RealSense/LeRobot driver calls "
            "plugged in first. Run with --dry_run to exercise the rest of the loop."
        )

    run_live_loop(
        model_path=cli_args.model_path,
        device=cli_args.device,
        viewer_port=cli_args.viewer_port,
        n_steps_per_model_step=cli_args.n_steps_per_model_step,
        dry_run=cli_args.dry_run,
        max_iterations=cli_args.max_iterations,
        tick_hz=cli_args.tick_hz,
    )
