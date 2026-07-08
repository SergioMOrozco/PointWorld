# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live/online PointWorld evaluation on a RealSense camera + LeRobot SO101 leader/follower rig.

This is the integration point for real hardware. The control loop itself (buffer -> window ->
sample -> model forward -> viser visualization) is fully implemented and verified (see
online_eval/tools/verify_*.py and --dry_run below). Real hardware is driven through teleop.py's
TeleopPointCloudSystem, which already handles RealSense capture and SO101 forward kinematics
internally -- this file's job is just to adapt what TeleopPointCloudSystem.step() returns
(per-camera datapoints + per-link robot point clouds) into the buffer/sample-builder pipeline.
"""

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

# Allow running this file directly (`python online_eval/live_loop.py`), not just as
# `python -m online_eval.live_loop` -- the latter puts the repo root on sys.path automatically,
# the former does not (sys.path[0] becomes this file's own directory instead).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from online_eval.model_loading import load_pointworld_model
from online_eval.rolling_buffer import N_STEPS_PER_MODEL_STEP, RollingBuffer
from online_eval.sample_builder import backproject_scene_points, build_live_sample, raw_camera_frame_from_datapoint
from online_eval.so101_robot import SO101_GRIPPER_LINKS, SO101_URDF_PATH, RobotLinkNormalEstimator
from visualization.prediction_viz.config import PredictionVisualizerConfig
from visualization.prediction_viz.sample import build_sample_from_dictionary
from visualization.prediction_viz.visualizer import PredictionVisualizer


# --------------------------------------------------------------------------------------- #
# Dry-run stand-ins, for exercising the full control loop without physical hardware. Shaped
# like TeleopPointCloudSystem.step()'s real output (see teleop.py) so the rest of the loop
# below is identical whether or not --dry_run is set.
# --------------------------------------------------------------------------------------- #
def _dry_run_datapoint(t: float) -> dict:
    h_raw, w_raw = 480, 640
    rng = np.random.RandomState(0)
    color_bgr = rng.randint(0, 255, (h_raw, w_raw, 3), dtype=np.uint8)
    depth_scale = 0.001
    depth_raw = np.full((h_raw, w_raw), 0.6 / depth_scale, dtype=np.uint16)
    color_intrinsics = np.array(
        [[600.0, 0, w_raw / 2], [0, 600.0, h_raw / 2], [0, 0, 1]], dtype=np.float32
    )
    X_WC = np.eye(4, dtype=np.float32)  # cam-to-world; identity is fine for structural testing
    return {
        "serial": "dry_run",
        "color": color_bgr,
        "depth": depth_raw,
        "depth_colormap": None,
        "depth_scale": depth_scale,
        "max_depth": 10.0,
        "X_WC": X_WC,
        "color_intrinsics": color_intrinsics,
        "obj_mask": None,
    }


def _dry_run_robot_link_pcds(t: float) -> dict[str, np.ndarray]:
    """Small synthetic per-link point sets that actually move over time, so
    RobotLinkNormalEstimator's rigid-registration path (see so101_robot.py) gets exercised
    rather than trivially rotating by the identity every tick."""
    rng = np.random.RandomState(1)
    local_points = {name: rng.uniform(-0.02, 0.02, size=(40, 3)).astype(np.float32) for name in SO101_GRIPPER_LINKS}
    angle = 0.3 * np.sin(0.5 * t)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    translation = np.array([0.5, 0.0, 0.3 + 0.05 * np.sin(0.7 * t)], dtype=np.float32)
    return {name: (pts @ R.T) + translation[None, :] for name, pts in local_points.items()}


def _make_dry_run_gripper_reader():
    def _read(t: float) -> float:
        return 0.5 + 0.5 * np.sin(1.0 * t)

    return _read


def run_live_loop(
    model_path: str,
    device: str,
    viewer_port: int,
    n_steps_per_model_step: int,
    dry_run: bool,
    max_iterations: Optional[int] = None,
    tick_hz: float = 10.0,
    viz_hz: float = 2.0,
    extrinsic_json: str = "extrinsic_calibration.json",
    realsense_serials: Optional[list] = None,
) -> None:
    model, args = load_pointworld_model(model_path, device=device)
    robot_normal_estimator = RobotLinkNormalEstimator(link_names=SO101_GRIPPER_LINKS)

    viz_config = PredictionVisualizerConfig(viewer_port=viewer_port)
    visualizer = PredictionVisualizer(viz_config, urdf_path=SO101_URDF_PATH)

    buffer = RollingBuffer(n_steps_per_model_step=n_steps_per_model_step)
    live_session = None

    system = None
    if dry_run:
        gripper_reader = _make_dry_run_gripper_reader()
    else:
        from lerobot_playground.hardware_config import TeleopSystemConfig

        from teleop import TeleopPointCloudSystem

        config = replace(
            TeleopSystemConfig(),
            extrinsic_json=extrinsic_json,
            tune=False,
            publish_to_foxglove=False,
            display_point_cloud_viewer=False,
        )
        if realsense_serials is not None:
            config = replace(config, realsense_serials=tuple(realsense_serials))
        system = TeleopPointCloudSystem(config)
        system.connect()

    iteration = 0
    dt = 1.0 / tick_hz
    # Buffer-feeding (fast, matches the real camera/robot rate) is decoupled from
    # inference+visualization (slow): visualize(live_stream=True) below updates the viewer's
    # existing point-cloud/mesh/GUI handles in place rather than tearing down and rebuilding the
    # scene each call (see PredictionVisualizer._apply_new_sample), but running inference plus
    # the data-prep it does (voxel upsampling, flow-timeline rebuilds) on every tick is still
    # wasted work when nothing's changed enough to matter. Only run it every ~1/viz_hz seconds
    # instead.
    viz_period = 1.0 / viz_hz
    last_viz_time = -float("inf")
    t0_wall = time.monotonic()
    try:
        while max_iterations is None or iteration < max_iterations:
            now = time.monotonic() - t0_wall

            if dry_run:
                datapoint = _dry_run_datapoint(now)
                robot_link_pcds = _dry_run_robot_link_pcds(now)
                gripper_position = gripper_reader(now)
            else:
                actions, datapoints, _scene_pcd, _robot_pcd, robot_link_pcds = system.step()
                datapoint = datapoints[0]
                # RANGE_0_100-normalized motor value (see lerobot's SO101Leader config) -> [0, 1]
                # openness, matching the droid-domain right_gripper_open convention (norm_stats
                # are aliased from stats/droid -- see online_eval/model_loading.py).
                gripper_position = float(actions[0]["gripper.pos"]) / 100.0

            frame = raw_camera_frame_from_datapoint(datapoint)
            buffer.push(frame, robot_link_pcds, gripper_position, timestamp=now)
            window = buffer.try_get_window()

            if window is not None and (now - last_viz_time) >= viz_period:
                last_viz_time = now
                print(f"[live_loop] update @ t={now:.2f}s (iteration {iteration})")
                sample = build_live_sample(
                    now_frame=window.t0_frame,
                    link_pcds_trajectory=window.robot_link_pcds_trajectory,
                    gripper_trajectory=window.gripper_trajectory,
                    robot_normal_estimator=robot_normal_estimator,
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
                    viz_sample,
                    launch_viewer=(live_session is None),
                    live_session=live_session,
                    live_stream=True,
                )
                live_session = result.get("live_session", live_session)

                # "Actual future" overlay: backproject the buffer's now-frame (once real time has
                # caught up to it) directly onto the same live server. visualize() resets the scene
                # graph each call (see SceneBuilder.populate_existing), so this must be re-added
                # every iteration, after visualize().
                #
                # backproject_scene_points returns points in raw world coordinates, but the main
                # visualization is rendered in build_live_sample's center_shift'd frame (scene_flows/
                # robot_flows/cam*_extrinsic were all re-centered around the first scene+robot points'
                # mean -- see dataset_components/transforms.py::center_shift). Without correcting for
                # that, this overlay would show up as a second copy of the scene, offset by the shift
                # amount. sample["__shift_amount__"] is defined so that
                # shifted_point == raw_point + shift_amount (verified against center_shift directly;
                # do not flip this sign without re-checking).
                if live_session is not None:
                    actual_points, actual_colors, _rgb, _depth, _intr = backproject_scene_points(
                        window.now_frame
                    )
                    if "__shift_amount__" in sample_np:
                        actual_points = actual_points + np.asarray(
                            sample_np["__shift_amount__"], dtype=actual_points.dtype
                        ).reshape(1, 3)
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
    finally:
        if system is not None:
            system.close()
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
        help="Use synthetic datapoints/robot_link_pcds instead of real hardware (structural test).",
    )
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--tick_hz", type=float, default=10.0)
    parser.add_argument(
        "--viz_hz",
        type=float,
        default=2.0,
        help="How often to actually run inference + push new data into the viewer (updates are "
        "in-place, not a full rebuild, but inference plus the visualization data-prep it "
        "triggers still isn't free -- keep this well below tick_hz).",
    )
    parser.add_argument(
        "--extrinsic-json",
        type=str,
        default="extrinsic_calibration.json",
        help="Camera extrinsics JSON, passed through to TeleopSystemConfig (cwd, "
        "LEROBOT_PLAYGROUND_EXTRINSIC_JSON, or src/ next to package).",
    )
    parser.add_argument(
        "--realsense-serial",
        dest="realsense_serials",
        action="append",
        default=None,
        metavar="SERIAL",
        help="RealSense device serial (repeat flag once per camera, order matches extrinsics "
        "JSON). Omit to use defaults from TeleopSystemConfig.",
    )
    cli_args = parser.parse_args()

    run_live_loop(
        model_path=cli_args.model_path,
        device=cli_args.device,
        viewer_port=cli_args.viewer_port,
        n_steps_per_model_step=cli_args.n_steps_per_model_step,
        dry_run=cli_args.dry_run,
        max_iterations=cli_args.max_iterations,
        tick_hz=cli_args.tick_hz,
        viz_hz=cli_args.viz_hz,
        extrinsic_json=cli_args.extrinsic_json,
        realsense_serials=cli_args.realsense_serials,
    )
