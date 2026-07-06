# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rolling buffer implementing the delayed-buffer / mini-replay action-window construction.

The model needs a *future* 10-step action trajectory (see pointworld/base.py::DynamicsPredictor
-- robot_flows/robot_features for ALL T=11 steps are fed in as action-conditioning input, not
just t=0), but a teleop leader arm only ever reports the *current* commanded pose in real time.

Rather than extrapolating a guess for the future, this buffer looks BACKWARDS: it treats a
frame from N steps ago as the model's t=0, and uses the REAL, now-known leader states between
then and now as the real t=1..10 action window. This costs a small fixed display latency
(N steps of real time) but the action-conditioning is always real, not guessed -- predictions
can be directly checked against what the camera actually saw happen (see live_loop.py's
"actual future" overlay).

OPEN CALIBRATION ITEM (see CLAUDE.md / plan notes): the model's real-world time interval per
predicted step is baked into the offline `data`-branch dataset conversion, not present in this
`main` checkout. N_STEPS_PER_MODEL_STEP below is a placeholder default -- calibrate it by
comparing the live loop's predicted-vs-actual overlay and adjusting until they line up in time,
or by checking the paper (arXiv:2601.03782) for a stated capture/prediction frequency.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from online_eval.sample_builder import RawCameraFrame

PRED_HORIZON = 10  # see pointworld/base.py: CONTEXT_HORIZON(1) + PRED_HORIZON(10) = T(11)
N_STEPS_PER_MODEL_STEP = 1  # PLACEHOLDER -- calibrate empirically, see module docstring.


@dataclass
class BufferedTick:
    frame: RawCameraFrame
    joint_positions: np.ndarray  # (n_joints,)
    gripper_position: float
    timestamp: float


@dataclass
class BufferedWindow:
    """One model-ready window: t=0 (from N steps ago) + real t=1..10 trajectory."""

    now_frame: RawCameraFrame  # the most-recent ("now") camera frame, for the actual-future overlay
    t0_frame: RawCameraFrame  # the delayed frame used as the model's t=0
    joint_trajectory: np.ndarray  # (11, n_joints), t=0..10, real (not extrapolated)
    gripper_trajectory: np.ndarray  # (11,)
    t0_timestamp: float
    now_timestamp: float


class RollingBuffer:
    """Ingests live (camera frame, joint state) ticks and yields delayed 11-step windows."""

    def __init__(self, n_steps_per_model_step: int = N_STEPS_PER_MODEL_STEP, maxlen: Optional[int] = None):
        assert n_steps_per_model_step >= 1
        self.n_steps_per_model_step = n_steps_per_model_step
        required_span = PRED_HORIZON * n_steps_per_model_step + 1
        self._buffer: deque[BufferedTick] = deque(maxlen=maxlen or (required_span + 8))
        self._required_span = required_span

    def push(
        self,
        frame: RawCameraFrame,
        joint_positions: np.ndarray,
        gripper_position: float,
        timestamp: float,
    ) -> None:
        self._buffer.append(
            BufferedTick(
                frame=frame,
                joint_positions=np.asarray(joint_positions, dtype=np.float32),
                gripper_position=float(gripper_position),
                timestamp=timestamp,
            )
        )

    def __len__(self) -> int:
        return len(self._buffer)

    def try_get_window(self) -> Optional[BufferedWindow]:
        """Return a BufferedWindow once enough history exists, else None.

        t=0 is the tick from `PRED_HORIZON * n_steps_per_model_step` ticks before "now";
        t=1..10 are the real ticks every `n_steps_per_model_step` steps after that, up to "now".
        """
        if len(self._buffer) < self._required_span:
            return None

        buf = list(self._buffer)
        now_idx = len(buf) - 1
        t0_idx = now_idx - self._required_span + 1

        indices = [t0_idx + k * self.n_steps_per_model_step for k in range(PRED_HORIZON + 1)]
        assert indices[-1] <= now_idx, (indices, now_idx)

        ticks = [buf[i] for i in indices]
        joint_trajectory = np.stack([t.joint_positions for t in ticks], axis=0)  # (11, n_joints)
        gripper_trajectory = np.array([t.gripper_position for t in ticks], dtype=np.float32)  # (11,)

        return BufferedWindow(
            now_frame=buf[now_idx].frame,
            t0_frame=ticks[0].frame,
            joint_trajectory=joint_trajectory,
            gripper_trajectory=gripper_trajectory,
            t0_timestamp=ticks[0].timestamp,
            now_timestamp=buf[now_idx].timestamp,
        )
