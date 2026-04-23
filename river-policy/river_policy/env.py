"""
Gymnasium environment wrapping AIC Gazebo callbacks for a single episode.

Each insert_cable() call = one episode. The env does not reset Gazebo state;
board randomization happens between calls by the eval system.

Observation (21D):
    [0:3]   tcp_pose.position  (x, y, z)  in base_link
    [3:7]   tcp_pose.orientation  (x, y, z, w)
    [7:10]  tcp_velocity.linear  (vx, vy, vz)
    [10:13] tcp_velocity.angular  (wx, wy, wz)
    [13:16] wrist_force  (fx, fy, fz)  [N]
    [16:19] wrist_torque  (tx, ty, tz)  [Nm]
    [19:21] plug_type one-hot  [sfp=1 0, sc=0 1]

Action (6D):  delta TCP pose in base_link, values in [-1, 1],
              scaled to MAX_DELTA_POS / MAX_DELTA_ROT before commanding.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from std_msgs.msg import Header

OBS_DIM = 21
ACT_DIM = 6

# Plug-type → one-hot index.  Add entries here as new plug types appear.
PLUG_TYPE_MAP: dict[str, int] = {
    "sfp_module": 0,
    "sc_plug": 1,
}

MAX_DELTA_POS = 0.005   # 5 mm per 50 ms step  (~10 cm/s max)
MAX_DELTA_ROT = 0.03    # ~1.7 deg per step

# Force/torque thresholds for reward shaping.
LATERAL_FORCE_SCALE = 0.02   # penalty weight for (fx² + fy²)
INSERTION_FZ_MIN = 5.0       # N downward force that signals contact/insertion
FORCE_LIMIT = 25.0           # N  – hard excess penalty beyond this
TORQUE_LIMIT = 2.5           # Nm

MAX_STEPS = 300              # ~15 s at 20 Hz


def _default_stiffness() -> list[float]:
    return np.diag([90.0, 90.0, 90.0, 50.0, 50.0, 50.0]).flatten().tolist()


def _default_damping() -> list[float]:
    return np.diag([50.0, 50.0, 50.0, 20.0, 20.0, 20.0]).flatten().tolist()


class CableInsertionEnv(gym.Env):
    """Single-episode Gymnasium env wrapping live AIC Gazebo callbacks."""

    metadata = {"render_modes": []}

    def __init__(self, task, get_observation, move_robot, sleep_fn, logger):
        super().__init__()

        self._task = task
        self._get_obs = get_observation
        self._move_robot = move_robot
        self._sleep = sleep_fn   # Policy.sleep_for(seconds)
        self._log = logger

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
        )

        plug_idx = PLUG_TYPE_MAP.get(task.plug_name, 0)
        self._plug_one_hot = np.zeros(2, dtype=np.float32)
        self._plug_one_hot[plug_idx] = 1.0

        self._step_count = 0
        self._inserted = False

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._inserted = False

        obs_msg = self._wait_for_observation()
        return self._extract_obs(obs_msg), {}

    def step(self, action: np.ndarray):
        obs_msg = self._wait_for_observation()

        target_pose = self._apply_delta(obs_msg.controller_state.tcp_pose, action)
        self._command_pose(target_pose)
        self._sleep(0.05)  # 20 Hz

        next_obs_msg = self._wait_for_observation()
        next_obs = self._extract_obs(next_obs_msg)
        reward, success = self._compute_reward(next_obs_msg)

        self._step_count += 1
        self._inserted = success
        terminated = success
        truncated = self._step_count >= MAX_STEPS

        return next_obs, reward, terminated, truncated, {"success": success}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _wait_for_observation(self):
        obs = self._get_obs()
        while obs is None:
            self._sleep(0.02)
            obs = self._get_obs()
        return obs

    def _extract_obs(self, obs_msg) -> np.ndarray:
        cs = obs_msg.controller_state
        pos = cs.tcp_pose.position
        ori = cs.tcp_pose.orientation
        vel = cs.tcp_velocity
        wr = obs_msg.wrist_wrench.wrench

        return np.array(
            [
                pos.x, pos.y, pos.z,
                ori.x, ori.y, ori.z, ori.w,
                vel.linear.x, vel.linear.y, vel.linear.z,
                vel.angular.x, vel.angular.y, vel.angular.z,
                wr.force.x, wr.force.y, wr.force.z,
                wr.torque.x, wr.torque.y, wr.torque.z,
                *self._plug_one_hot,
            ],
            dtype=np.float32,
        )

    def _apply_delta(self, current_pose: Pose, action: np.ndarray) -> Pose:
        """Return a new Pose = current_pose ⊕ scaled action delta."""
        d_pos = action[:3] * MAX_DELTA_POS
        d_rot = action[3:] * MAX_DELTA_ROT

        new_pos = Point(
            x=current_pose.position.x + d_pos[0],
            y=current_pose.position.y + d_pos[1],
            z=current_pose.position.z + d_pos[2],
        )

        # transforms3d quaternion convention: (w, x, y, z)
        q_cur = np.array([
            current_pose.orientation.w,
            current_pose.orientation.x,
            current_pose.orientation.y,
            current_pose.orientation.z,
        ])
        # Small rotation expressed as euler (sxyz) → quaternion
        q_delta = euler2quat(float(d_rot[0]), float(d_rot[1]), float(d_rot[2]), axes="sxyz")
        q_new = qmult(q_delta, q_cur)  # (w, x, y, z)

        new_ori = Quaternion(
            w=float(q_new[0]),
            x=float(q_new[1]),
            y=float(q_new[2]),
            z=float(q_new[3]),
        )
        return Pose(position=new_pos, orientation=new_ori)

    def _command_pose(self, pose: Pose) -> None:
        motion_update = MotionUpdate(
            header=Header(frame_id="base_link"),
            pose=pose,
            target_stiffness=_default_stiffness(),
            target_damping=_default_damping(),
            feedforward_wrench_at_tip=Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0),
                torque=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            wrench_feedback_gains_at_tip=[0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION
            ),
        )
        try:
            self._move_robot(motion_update=motion_update)
        except Exception as exc:
            self._log.warn(f"move_robot error: {exc}")

    def _compute_reward(self, obs_msg) -> tuple[float, bool]:
        wr = obs_msg.wrist_wrench.wrench
        fx, fy, fz = wr.force.x, wr.force.y, wr.force.z
        torque = np.array([wr.torque.x, wr.torque.y, wr.torque.z])

        # Penalise lateral forces (misalignment during contact)
        r_align = -LATERAL_FORCE_SCALE * (fx**2 + fy**2)

        # Reward sustained downward contact force (insertion progress)
        r_progress = 0.05 * max(0.0, -fz)

        # Hard penalty for exceeding force/torque limits
        force_excess = max(0.0, np.sqrt(fx**2 + fy**2 + fz**2) - FORCE_LIMIT)
        torque_excess = max(0.0, float(np.linalg.norm(torque)) - TORQUE_LIMIT)
        r_limit = -5.0 * (force_excess**2 + torque_excess**2)

        # Small per-step time penalty to encourage speed
        r_step = -0.05

        reward = r_align + r_progress + r_limit + r_step

        # Heuristic success: downward force sustained for at least a few steps,
        # lateral force small → plug locked into port.
        lateral = np.sqrt(fx**2 + fy**2)
        success = (
            self._step_count >= 5
            and (-fz) > INSERTION_FZ_MIN
            and lateral < 3.0
        )
        if success:
            reward += 100.0

        return float(reward), success
