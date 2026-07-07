# `online_eval/`

Live/online PointWorld evaluation on real hardware: a RealSense RGB-D camera + a LeRobot SO101
leader/follower arm. This is a standalone package that builds inference samples directly from a
live sensor stream instead of reading pre-converted WebDataset shards — it does not modify
anything in the existing training/eval pipeline (`train.py`, `eval.py`, `dataset_components/`).

Everything here runs on the `main` (train/eval) checkout; dataset *conversion* concerns (how
`droid`/`behavior` WDS shards were built, capture frequency, etc.) live on the `data` branch and
are outside this package's scope — where those facts matter (see "Open calibration items" below),
this code can only approximate them.

## Status

The control loop (buffer → window → sample → model forward → viser visualization) is fully
implemented and verified against synthetic data (`--dry_run`, plus the `tools/verify_*.py`
scripts). **Not yet connected to physical hardware.** Two callables in `live_loop.py` are stubs
that raise `NotImplementedError`:

- `read_camera_frame_TODO` — needs your `pyrealsense2` capture code.
- `read_leader_state_TODO` — needs your LeRobot SO101 leader-arm read code.

Everything downstream of those two functions (buffering, sample assembly, model forward,
visualization) has been exercised end-to-end with synthetic stand-ins and should not need to
change when you wire in real hardware — see "Plugging in real hardware" below.

## Data flow

```
read_camera_frame_TODO()  ──┐
                             ├─► RollingBuffer.push()  (every tick, ~tick_hz)
read_leader_state_TODO()  ──┘         │
                                       ▼
                          RollingBuffer.try_get_window()
                          (delayed t=0 frame + real t=0..10 joint/gripper trajectory)
                                       │
                                       ▼
                          sample_builder.build_live_sample()
                          (RGBD → world point cloud, robot FK, dataset_components pipeline)
                                       │
                                       ▼
                              BaseModel.forward()  (model_loading.load_pointworld_model)
                                       │
                                       ▼
                    visualization.prediction_viz.PredictionVisualizer  (viser, localhost:<port>)
                                       │
                                       ▼
                    "actual future" overlay (backproject_scene_points on the real "now" frame,
                     added to the same live viser session for visual predicted-vs-actual check)
```

`live_loop.py::run_live_loop` is the driver that wires all of the above together in a loop.

## Files

- **`live_loop.py`** — the entry point / control loop. Ticks at `tick_hz`, feeds the rolling
  buffer every tick, but only runs inference + rebuilds the viser scene every `1/viz_hz` seconds
  (inference + `visualizer.visualize()` do a full scene teardown/rebuild — see
  `visualization/viser_flow/scene_builder.py` — so running it every tick makes the viewer visibly
  "reload" continuously). Contains the two hardware TODO stubs and their `--dry_run` synthetic
  replacements. Run directly with `python online_eval/live_loop.py` (it patches `sys.path` itself
  so this works without `-m`).

- **`rolling_buffer.py`** — `RollingBuffer` / `BufferedWindow`. Implements the "delayed buffer"
  trick (see Design decisions below): ingests `(camera frame, joint state, gripper state,
  timestamp)` ticks and, once enough history exists, yields an 11-step window where t=0 is a
  frame from `PRED_HORIZON * n_steps_per_model_step` ticks ago and t=1..10 is the **real**
  (not extrapolated) leader trajectory between then and now.

- **`sample_builder.py`** — `build_live_sample()`: turns one `BufferedWindow` into a batched
  tensor dict ready for `BaseModel.forward`. Backprojects the RGBD frame to a world-frame point
  cloud (`backproject_scene_points`), FKs the robot gripper across all T=11 steps
  (`build_robot_geometry`, via `RobotSampler`), then reuses
  `dataset_components/pipeline.py::apply_release_pipeline_to_sample` (the same single-sample
  transform pipeline the offline WDS path uses) for grid-sampling, point-count capping,
  centering, and feature gathering — with `skip_scene_sampling=True` since the scene was already
  downsampled beforehand (see the normal-estimation ordering note below). Also exposes
  `backproject_scene_points`, reused by `live_loop.py` for the "actual future" overlay.

- **`model_loading.py`** — `load_pointworld_model()`: loads a pretrained checkpoint straight into
  `BaseModel` for inference, bypassing `Trainer`/`Tester` (no DDP, no optimizer, no WebDataset
  dataloader). Applies the checkpoint's model contract to `args` (mirrors `Trainer`/`Tester`'s
  load-time contract application), then overrides `norm_stats_path`/`domains` to point at the
  aliased `stats/so101/norm_stats.json` domain (see Design decisions).

- **`so101_robot.py`** — `build_so101_robot_sampler()`: constructs and `presample()`s a
  gripper-only `RobotSampler` for the SO101 URDF (`assets/so101/so101_new_calib.urdf`), with two
  SO101-specific fixes to `robot_sampler.py` (extended `GRIPPER_KEYWORDS`, opt-in
  `mesh_link_match_mode="translation"`) needed because SO101's mesh filenames and CAD-typical
  `<visual><origin>` rotational offsets don't match `RobotSampler`'s defaults — verified not to
  change Franka/R1Pro's existing behavior (see `tools/check_mesh_link_matching.py`). Presampling
  uses a fixed seed and is done once; the same point set is reused every inference tick.

- **`tools/`** — one-off inspection/verification scripts, not part of the runtime loop:
  - `inspect_so101_urdf.py` — dumps every visual mesh's link parent and the joint order
    `RobotSampler` actually uses for FK (from `pytorch_kinematics`'s URDF tree walk, which is not
    necessarily the URDF's declaration order). Run this first when bringing up any new robot
    asset — don't hardcode joint/link names from reading URDF XML alone.
  - `check_mesh_link_matching.py` — compares full-4x4-Frobenius vs. translation-only mesh↔link
    matching across all three robot URDFs in the repo, to confirm the SO101 fix doesn't change
    Franka/R1Pro attachment.
  - `make_so101_norm_stats.py` — one-off script that produced `stats/so101/norm_stats.json` by
    copying `stats/droid/norm_stats.json` and renaming the domain key. Re-run only if you want to
    regenerate that file from scratch.
  - `verify_dummy_forward_pass.py` — loads a checkpoint and runs `BaseModel.forward` on an
    all-synthetic, correctly-shaped `so101`-domain sample. Pure shape/contract check; no camera,
    robot, or `sample_builder` involved.
  - `verify_sample_builder.py` — runs `build_live_sample()` end-to-end on a synthetic RGBD frame
    + a real SO101 joint trajectory, then feeds the result into the model. Closest thing to an
    integration test for everything except the actual hardware reads and the live loop/viewer.
  - `verify_so101_gripper_viser.py` — standalone viser sanity check: sweeps the gripper joint
    through its URDF limits, dumps the FK'd point cloud with a frame slider at
    `http://localhost:<port>`. No model/checkpoint needed — pure `RobotSampler` visual check.

## Design decisions (locked — see rationale before changing)

- **Action window = delayed buffer, not extrapolation.** `BaseModel`/`DynamicsPredictor` needs a
  *future* 10-step robot action trajectory (`robot_flows`/`joint_positions` for all T=11 steps),
  but a teleop leader arm only ever reports the *current* pose. Rather than guessing t=1..10,
  `RollingBuffer` looks backwards: it treats a frame from N steps ago as t=0 and uses the real,
  now-known leader states since then as t=1..10. This costs a small fixed display latency but
  keeps action-conditioning real — predictions can be checked directly against what the camera
  actually saw happen (the "actual future" overlay in `live_loop.py`).

- **`stats/so101/norm_stats.json` is `stats/droid`'s stats with the domain key renamed.** SO101
  was never part of PointWorld's training data, so no real normalization statistics exist for it.
  Aliasing to `droid` (the closest existing domain: real, single-arm, non-simulated) is an
  explicit, labeled approximation, not a silent hack — see `tools/make_so101_norm_stats.py`.

- **Normals are estimated *after* downsampling, not before.** `sample_builder.estimate_normals`
  is deliberately called on the already voxel-downsampled (`args.max_scene_points`, e.g. ~12k
  points at grid_size) point cloud, not the raw backprojected cloud (tens of thousands of
  points). Estimating normals before downsampling was the dominant per-update cost and made each
  live update take multiple seconds — enough that the `viz_hz` throttle in `live_loop.py`
  couldn't help, since a single update already exceeded the throttle period.

- **Hardware is assumed ready.** `pyrealsense2`, the LeRobot SO101 leader+follower driver, and
  the camera-to-robot-base extrinsic calibration are assumed to already work on your setup — this
  package only covers the model/data-contract glue between a live sensor stream and
  `BaseModel.forward`, not driver bring-up or calibration.

## Open calibration items

These depend on facts baked into the offline `data`-branch dataset conversion that aren't present
in this `main` checkout, so they can't be derived from code here — resolve empirically once
hardware is connected:

- **`N_STEPS_PER_MODEL_STEP`** (`rolling_buffer.py`) — the real-world time interval between
  consecutive predicted steps. Currently a placeholder (`1`). Calibrate by comparing the live
  loop's predicted-vs-actual overlay and adjusting until they line up in time, or by checking the
  paper (arXiv:2601.03782) for a stated capture/prediction frequency.
- **`right_gripper_open` value convention/range** — same story; confirm your LeRobot driver's
  gripper reading convention matches what the model expects (droid-derived, via the aliased norm
  stats).

## Running it

Dry run (synthetic camera + leader data, exercises the full loop with no hardware):

```bash
python online_eval/live_loop.py --dry_run --model_path pretrained_checkpoints/small-droid/model-best.pt
```

Opens a viser viewer at `http://localhost:8080` (`--viewer_port` to change). Useful flags:
`--tick_hz` (buffer-feed rate), `--viz_hz` (inference/redraw rate, keep well below `tick_hz`),
`--max_iterations` (for finite test runs), `--n_steps_per_model_step` (see calibration item
above).

Running without `--dry_run` currently raises immediately — real-hardware mode isn't wired up
until you fill in the two TODOs (see below).

Standalone verification scripts (no live loop / viewer needed for the first three):

```bash
python online_eval/tools/verify_dummy_forward_pass.py    # model load + forward shape/contract check
python online_eval/tools/verify_sample_builder.py        # sample_builder + model, synthetic frame
python online_eval/tools/verify_so101_gripper_viser.py   # RobotSampler FK visual sanity check (viser)
python online_eval/tools/inspect_so101_urdf.py           # dump URDF mesh/link/joint-order info
```

## Plugging in real hardware

1. In `live_loop.py`, implement `read_camera_frame_TODO` using `pyrealsense2`: grab a color +
   depth frame, convert depth to meters, and return a `RawCameraFrame` (rgb, depth, intrinsic —
   from the RealSense stream profile, extrinsic — your camera-to-robot-base calibration, loaded
   once at startup, not per frame). See the function's docstring for a sketch.
2. Implement `read_leader_state_TODO` using your LeRobot SO101 leader driver: read joint
   positions **in the same order as `sampler.joint_names`** (confirm this order with
   `tools/inspect_so101_urdf.py` — it is FK traversal order via `pytorch_kinematics`, not
   necessarily URDF declaration order) and convert your driver's joint convention/units to the
   URDF's radians if needed. Return `(joint_positions, gripper_position)`.
3. Remove or update the `if not cli_args.dry_run: raise SystemExit(...)` guard at the bottom of
   `live_loop.py` once both are implemented, and swap the `pipeline=None` / `so101_leader=None`
   placeholders in `run_live_loop` for real handles (opened once, before the loop starts).
4. Run without `--dry_run` and watch the "actual future" overlay against predictions to calibrate
   `N_STEPS_PER_MODEL_STEP` (see Open calibration items).

Nothing else in this package should need to change — `RollingBuffer`, `build_live_sample`, model
loading, and visualization only depend on the `RawCameraFrame` / `(joint_positions,
gripper_position)` shapes these two functions produce, not on how they're produced.
