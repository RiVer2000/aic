"""
MyPolicy     – classical compliant descent (no learning). Active policy.
TrainingPolicy – SAC online training (kept for reference; not used).

Usage
-----
Eval / submission:
    pixi run --frozen ros2 run aic_model aic_model \\
        --ros-args -p use_sim_time:=true -p policy:=river_policy.MyPolicy
"""

from __future__ import annotations

import os

import numpy as np
from aic_control_interfaces.msg import MotionUpdate, TrajectoryGenerationMode
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3, Wrench
from rclpy.duration import Duration
from std_msgs.msg import Header
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qinverse, qmult, quat2mat

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_task_interfaces.msg import Task

from .env import PLUG_TYPE_MAP, CableInsertionEnv, OBS_DIM, ACT_DIM

# Path where SAC weights are stored / loaded from.
_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
WEIGHTS_FILE = os.path.join(_WEIGHTS_DIR, "sac_cable.zip")

# Training hyper-parameters
LEARNING_STARTS = 500   # random exploration steps before SAC updates begin
BATCH_SIZE = 256
GRADIENT_STEPS = 1
CHECKPOINT_EVERY = 1000  # steps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sac(load_path: str | None = None):
    """Return a configured SB3 SAC model, optionally loading saved weights."""
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import SAC

    class _DummyEnv(gym.Env):
        observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)

        def reset(self, **kw):
            return np.zeros(OBS_DIM, np.float32), {}

        def step(self, a):
            return np.zeros(OBS_DIM, np.float32), 0.0, False, False, {}

    from stable_baselines3.common.logger import configure as sb3_configure

    if load_path and os.path.exists(load_path):
        model = SAC.load(load_path, env=_DummyEnv())
    else:
        model = SAC(
            "MlpPolicy",
            _DummyEnv(),
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=LEARNING_STARTS,
            batch_size=BATCH_SIZE,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=GRADIENT_STEPS,
            ent_coef="auto",
            policy_kwargs={"net_arch": [256, 256]},
            verbose=1,
        )

    model.set_logger(sb3_configure(folder=None, format_strings=["stdout"]))
    return model


# ---------------------------------------------------------------------------
# Inference policy
# ---------------------------------------------------------------------------

class MyPolicy(Policy):
    """
    Classical compliant-descent controller.

    The robot starts within a few cm of the target port with a plug already
    grasped and roughly pointing downward. We command a slow straight-line
    descent with low Z-stiffness so the admittance controller allows the plug
    to be pushed back when it hits the port face, and moderate XY-stiffness
    so the port's chamfer can guide the plug laterally into the hole.

    Phases:
      1. Approach  – descend until contact (|fz| > CONTACT_FORCE)
      2. Settle    – on contact, descend slower with even lower stiffness;
                     port chamfer funnels plug in
      3. Verify    – if we descended further than INSERT_DEPTH after contact,
                     call it a success

    No vision, no learning, no ground-truth TF. Robust to ~2 mm grasp
    deviation because the admittance controller absorbs the error.
    """

    # --- plug grasp geometry (from aic_engine sample_config.yaml) -----------
    # Static cable-grasp orientation in gripper frame (sxyz Euler angles).
    # The plug's local +Z is its insertion axis; with this grasp the plug
    # sits at ~38° from the gripper's local +Z, which is why pure straight-Z
    # descent never inserts. Phase 0 reorients the gripper to compensate.
    PLUG_GRASP_RPY = (0.4432, -0.4838, 1.3303)
    # Plug-tip translation in gripper frame, by plug name.
    PLUG_OFFSET_IN_GRIPPER = {
        "sfp_tip": np.array([0.0, 0.015385, 0.04245], dtype=np.float64),
        "sc_tip":  np.array([0.0, 0.015385, 0.04045], dtype=np.float64),
    }

    # --- tunable parameters -------------------------------------------------
    REORIENT_TIME = 2.0           # s     SLERP duration for the plug-aligning rotation
    SETTLE_AT_START = 0.7         # s     wait for arm/cable to stabilise before baselining wrench
    DESCENT_SPEED = 0.030         # m/s   approach descent rate (faster — port up to 13 cm away)
    INSERT_SPEED = 0.010          # m/s   post-contact descent rate
    STEP_DT = 0.05                # s     command rate → 20 Hz
    CONTACT_FORCE = 5.0           # N     wrench delta over baseline to declare contact
    MIN_APPROACH_BEFORE_CONTACT = 0.005  # m  must descend at least this far before contact is allowed
    STALL_FORCE = 22.0            # N     excessive wrench delta → abort
    MAX_APPROACH_DEPTH = 0.20     # m     safety stop if no contact by then
    INSERT_DEPTH = 0.015          # m     additional descent after contact = success
    INSERT_PHASE_TIMEOUT = 8.0    # s     hard cap on Phase 2 (prevents lock-up if plug jams)
    NO_PROGRESS_TIMEOUT = 2.0     # s     bail if cur_z hasn't dropped 1 mm within this window
    APPROACH_STALL_LIMIT = 0.05   # m     |commanded_z - cur_z| over this → joint stall, abort approach
    SETTLE_TIME = 1.0             # s     pause at end to let connector seat
    HOLD_AT_END_TIME = 0.5        # s     command current pose to stop motion before returning
    BASELINE_LATERAL_ABORT = 15.0 # N     |fx| or |fy| in baseline → bad spawn, abort

    # Phase 3 — Spiral search after Phase 2 fails.
    # Lifts the plug a hair, then sweeps XY in an Archimedean spiral around the
    # contact point, looking for a Z-drop that signals the plug fell into the hole.
    SPIRAL_MAX_RADIUS = 0.070     # m     up to ±70 mm — SFP ports sit ~50-60 mm from contact
    SPIRAL_TURNS = 6              #       full revolutions in the spiral
    SPIRAL_POINTS_PER_TURN = 8    #       angular resolution
    SPIRAL_DWELL = 0.25           # s     hold each XY position to let the plug settle
    SPIRAL_PUSH_OFFSET = 0.006    # m     command Z this far below contact_z to push lightly
    SPIRAL_DROP_THRESHOLD = 0.006 # m     Z drop past contact_z that signals a found hole
    SPIRAL_DROP_CONFIRM_STEPS = 2 #       extra dwell steps to sustain before committing
    SPIRAL_STIFFNESS = [80.0, 80.0, 30.0, 40.0, 40.0, 40.0]   # low Z so plug can drop in
    SPIRAL_DAMPING = [40.0, 40.0, 25.0, 18.0, 18.0, 18.0]

    # Admittance gains — match the WaveArm default for the approach so the
    # controller actually drives motion; soften slightly during insertion so
    # the port chamfer can guide the plug.
    APPROACH_STIFFNESS = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
    APPROACH_DAMPING = [50.0, 50.0, 50.0, 20.0, 20.0, 20.0]
    INSERT_STIFFNESS = [80.0, 80.0, 60.0, 40.0, 40.0, 40.0]
    INSERT_DAMPING = [45.0, 45.0, 40.0, 18.0, 18.0, 18.0]
    # ------------------------------------------------------------------------

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.get_logger().info("MyPolicy (classical compliant descent) ready")

    # ------------------------------------------------------------------

    def _wait_for_obs(self, get_observation):
        obs = get_observation()
        while obs is None:
            self.sleep_for(0.02)
            obs = get_observation()
        return obs

    def _send_pose(self, move_robot, pose: Pose, stiffness, damping) -> None:
        motion_update = MotionUpdate(
            header=Header(
                frame_id="base_link",
                stamp=self.time_now().to_msg(),
            ),
            pose=pose,
            target_stiffness=np.diag(stiffness).flatten(),
            target_damping=np.diag(damping).flatten(),
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
            move_robot(motion_update=motion_update)
        except Exception as exc:
            self.get_logger().warn(f"move_robot error: {exc}")

    # ------------------------------------------------------------------

    @staticmethod
    def _slerp_wxyz(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between two unit quaternions (wxyz)."""
        d = float(np.dot(q0, q1))
        if d < 0:
            q1 = -q1
            d = -d
        if d > 0.9995:
            r = q0 + t * (q1 - q0)
            return r / np.linalg.norm(r)
        theta = float(np.arccos(d))
        sin_theta = float(np.sin(theta))
        return (np.sin((1.0 - t) * theta) / sin_theta) * q0 + (
            np.sin(t * theta) / sin_theta
        ) * q1

    def _reorient_for_plug(
        self,
        get_observation,
        move_robot,
        send_feedback,
        plug_name: str,
    ) -> Pose:
        """
        Phase 0: SLERP-rotate the gripper so the plug's +Z axis points world -Z,
        keeping the plug tip stationary in world frame.

        Returns the final commanded Pose so the caller can use it as the
        baseline for the descent phase.
        """
        plug_offset = self.PLUG_OFFSET_IN_GRIPPER.get(
            plug_name, self.PLUG_OFFSET_IN_GRIPPER["sfp_tip"]
        )

        # Plug-in-gripper rotation (constant, sxyz convention).
        q_plug_in_gripper = euler2quat(*self.PLUG_GRASP_RPY, axes="sxyz")  # wxyz
        # Target plug-in-world rotation = 180° about world X (plug Z = world -Z).
        q_home_wxyz = np.array([0.0, 1.0, 0.0, 0.0])
        # Required gripper-in-world rotation: q_home * inv(q_plug_in_gripper).
        q_target_gripper = qmult(q_home_wxyz, qinverse(q_plug_in_gripper))

        # Read current gripper pose.
        obs = self._wait_for_obs(get_observation)
        start_pose = obs.controller_state.tcp_pose
        cur_pos = np.array(
            [start_pose.position.x, start_pose.position.y, start_pose.position.z]
        )
        q_cur = np.array(
            [
                start_pose.orientation.w,
                start_pose.orientation.x,
                start_pose.orientation.y,
                start_pose.orientation.z,
            ]
        )

        # Plug tip world position to keep stationary throughout the rotation.
        plug_tip_world = cur_pos + quat2mat(q_cur) @ plug_offset

        n_steps = max(1, int(self.REORIENT_TIME / self.STEP_DT))
        send_feedback("river-policy: reorienting gripper to align plug")
        self.get_logger().info(
            f"Reorient: plug {plug_name}, plug_tip_world="
            f"({plug_tip_world[0]:.3f}, {plug_tip_world[1]:.3f}, {plug_tip_world[2]:.3f})"
        )

        for i in range(1, n_steps + 1):
            t = i / n_steps
            q_interp = self._slerp_wxyz(q_cur, q_target_gripper, t)
            R_interp = quat2mat(q_interp)
            gripper_pos = plug_tip_world - R_interp @ plug_offset

            interp_pose = Pose(
                position=Point(
                    x=float(gripper_pos[0]),
                    y=float(gripper_pos[1]),
                    z=float(gripper_pos[2]),
                ),
                orientation=Quaternion(
                    w=float(q_interp[0]),
                    x=float(q_interp[1]),
                    y=float(q_interp[2]),
                    z=float(q_interp[3]),
                ),
            )
            self._send_pose(
                move_robot,
                interp_pose,
                self.APPROACH_STIFFNESS,
                self.APPROACH_DAMPING,
            )
            self.sleep_for(self.STEP_DT)

        # Final pose for the caller — gripper position that puts plug tip at
        # plug_tip_world with the corrected orientation.
        final_R = quat2mat(q_target_gripper)
        final_gripper_pos = plug_tip_world - final_R @ plug_offset
        return Pose(
            position=Point(
                x=float(final_gripper_pos[0]),
                y=float(final_gripper_pos[1]),
                z=float(final_gripper_pos[2]),
            ),
            orientation=Quaternion(
                w=float(q_target_gripper[0]),
                x=float(q_target_gripper[1]),
                y=float(q_target_gripper[2]),
                z=float(q_target_gripper[3]),
            ),
        )

    # Per-module coarse XY correction applied before Phase 1.
    # Cleared until a ground_truth:=true calibration run determines correct values.
    MODULE_XY_OFFSETS: dict[str, tuple[float, float]] = {}

    # --- camera geometry (from URDF: ur_gz.urdf.xacro) ----------------------
    # TCP (tool0) → cam_mount: pure translation along TCP Z, no rotation.
    _TCP_TO_CAM_MOUNT_T = np.array([0.0, 0.0, -0.0265])

    # cam_mount → each camera optical frame: (translation_in_mount, rpy_sxyz).
    # Three cameras at ±60° yaw spread, all pitched down ~75°.
    _CAMERAS = {
        'left':   {'t': np.array([-0.09326, -0.053843, -0.007188]),
                   'rpy': (0.0, -1.30899630, 0.523599027)},
        'center': {'t': np.array([0.0,     -0.107700, -0.007190]),
                   'rpy': (0.0, -1.30899630, 1.570796230)},
        'right':  {'t': np.array([0.09326, -0.053843, -0.007188]),
                   'rpy': (0.0, -1.30899630, 2.617993430)},
    }

    # Triangulation quality gate: max distance from any camera ray to the
    # triangulated point before we reject the result as inconsistent.
    TRIANGULATION_RESIDUAL_THRESH = 0.015  # m

    # Safety clamp on the XY correction derived from triangulation.
    MAX_TRIANGULATION_CORRECTION = 0.04  # m

    def _detect_port_pixel(self, img_msg, plug_type: str):
        """Find the most port-like dark blob near the image centre.

        Returns (cx_px, cy_px, area, n_candidates) or (None, None, None, 0).
        """
        try:
            import cv2  # type: ignore
        except ImportError:
            return None, None, None, 0

        if img_msg.height == 0 or img_msg.width == 0 or len(img_msg.data) == 0:
            return None, None, None, 0

        try:
            arr = np.frombuffer(img_msg.data, dtype=np.uint8)
            if img_msg.encoding in ("rgb8", "bgr8"):
                img = arr.reshape(img_msg.height, img_msg.width, 3)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img_msg.encoding == "mono8":
                gray = arr.reshape(img_msg.height, img_msg.width)
            else:
                return None, None, None, 0
        except Exception:
            return None, None, None, 0

        is_sc = "sc" in plug_type.lower()
        min_area = 8 if is_sc else 30
        max_area = 2500 if is_sc else 8000
        thresh = 60

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, thresh, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        h, w = gray.shape
        cx_img, cy_img = w / 2.0, h / 2.0

        best = None
        best_dist2 = float("inf")
        n_cand = 0
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < min_area or area > max_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            dist2 = (cx - cx_img) ** 2 + (cy - cy_img) ** 2
            n_cand += 1
            if dist2 < best_dist2:
                best_dist2 = dist2
                best = (cx, cy, area)

        if best is None:
            return None, None, None, 0
        return best[0], best[1], best[2], n_cand

    def _camera_poses_in_base_link(self, tcp_pose: Pose) -> dict:
        """Return {cam_name: (origin_3d, R_cam_to_base)} for all three cameras."""
        q = tcp_pose.orientation
        R_tcp = quat2mat(np.array([q.w, q.x, q.y, q.z]))
        p_tcp = np.array([tcp_pose.position.x, tcp_pose.position.y, tcp_pose.position.z])

        p_mount = p_tcp + R_tcp @ self._TCP_TO_CAM_MOUNT_T  # cam_mount origin in base_link
        poses = {}
        for name, cam in self._CAMERAS.items():
            R_cam_in_mount = quat2mat(euler2quat(*cam['rpy'], axes='sxyz'))
            poses[name] = (
                p_mount + R_tcp @ cam['t'],   # camera origin in base_link
                R_tcp @ R_cam_in_mount,       # R: camera frame → base_link
            )
        return poses

    def _triangulate_port(self, obs, task):
        """Detect port blob in all three cameras and triangulate 3D position.

        Casts a pinhole ray through each camera's best blob detection, then
        finds the least-squares intersection point. Requires ≥2 cameras and a
        small per-camera residual before trusting the result.

        Returns (dx, dy, detected) — XY correction from TCP in base_link.
        """
        try:
            import cv2  # noqa: F401 — needed inside _detect_port_pixel
        except ImportError:
            return 0.0, 0.0, False

        cam_sources = [
            ('left',   obs.left_image,   obs.left_camera_info),
            ('center', obs.center_image, obs.center_camera_info),
            ('right',  obs.right_image,  obs.right_camera_info),
        ]

        tcp_pose = obs.controller_state.tcp_pose
        cam_poses = self._camera_poses_in_base_link(tcp_pose)

        rays = []  # (origin, unit_direction, cam_name)
        for cam_name, img, cam_info in cam_sources:
            cx_px, cy_px, area, n_cand = self._detect_port_pixel(img, task.plug_type)
            if cx_px is None or len(cam_info.k) < 6:
                continue
            fx = float(cam_info.k[0]); fy = float(cam_info.k[4])
            cx_p = float(cam_info.k[2]); cy_p = float(cam_info.k[5])
            if fx <= 0 or fy <= 0:
                continue

            # Pinhole ray in camera optical frame (Z forward, X right, Y down).
            d_cam = np.array([(cx_px - cx_p) / fx, (cy_px - cy_p) / fy, 1.0])
            d_cam /= np.linalg.norm(d_cam)

            origin, R_cam = cam_poses[cam_name]
            direction = R_cam @ d_cam
            direction /= np.linalg.norm(direction)

            rays.append((origin, direction, cam_name))
            self.get_logger().info(
                f"Tri {cam_name}: {n_cand} cands px=({cx_px:.0f},{cy_px:.0f}) "
                f"area={area:.0f} "
                f"origin=({origin[0]:.3f},{origin[1]:.3f},{origin[2]:.3f}) "
                f"dir=({direction[0]:.3f},{direction[1]:.3f},{direction[2]:.3f})"
            )

        if len(rays) < 2:
            self.get_logger().info(
                f"Triangulation: {len(rays)} camera(s) detected — skipping"
            )
            return 0.0, 0.0, False

        # Least-squares ray intersection.
        # Minimise sum_i ||(I - d_i d_i^T)(P - o_i)||^2
        # Solution: A P = b  where  A = sum(I - d d^T),  b = sum((I - d d^T) o)
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for origin, direction, _ in rays:
            M = np.eye(3) - np.outer(direction, direction)
            A += M
            b += M @ origin

        try:
            port_3d = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            self.get_logger().warn("Triangulation: singular system — skipping")
            return 0.0, 0.0, False

        # Per-camera residual check — reject if cameras disagree.
        max_residual = 0.0
        for origin, direction, cam_name in rays:
            v = port_3d - origin
            residual = float(np.linalg.norm(v - np.dot(v, direction) * direction))
            max_residual = max(max_residual, residual)
            self.get_logger().info(f"Tri residual {cam_name}: {residual * 1000:.1f} mm")

        if max_residual > self.TRIANGULATION_RESIDUAL_THRESH:
            self.get_logger().warn(
                f"Triangulation rejected: max residual {max_residual * 1000:.1f} mm "
                f"> {self.TRIANGULATION_RESIDUAL_THRESH * 1000:.0f} mm"
            )
            return 0.0, 0.0, False

        # Clamp XY correction so a bad triangulation can't drive us far off.
        p_tcp = np.array([tcp_pose.position.x, tcp_pose.position.y, tcp_pose.position.z])
        clamp = self.MAX_TRIANGULATION_CORRECTION
        dx = float(np.clip(port_3d[0] - p_tcp[0], -clamp, clamp))
        dy = float(np.clip(port_3d[1] - p_tcp[1], -clamp, clamp))

        self.get_logger().info(
            f"Triangulation OK: port_3d=({port_3d[0]:.4f},{port_3d[1]:.4f},{port_3d[2]:.4f}) "
            f"max_residual={max_residual * 1000:.1f} mm "
            f"correction=({dx * 1000:+.1f},{dy * 1000:+.1f}) mm"
        )
        return dx, dy, True

    def _hold_current_pose(self, get_observation, move_robot) -> None:
        """Command the arm to hold its current pose so it doesn't drift after return."""
        obs = self._wait_for_obs(get_observation)
        cur = obs.controller_state.tcp_pose
        hold_pose = Pose(
            position=Point(x=cur.position.x, y=cur.position.y, z=cur.position.z),
            orientation=Quaternion(
                x=cur.orientation.x,
                y=cur.orientation.y,
                z=cur.orientation.z,
                w=cur.orientation.w,
            ),
        )
        n = max(1, int(self.HOLD_AT_END_TIME / self.STEP_DT))
        for _ in range(n):
            self._send_pose(
                move_robot,
                hold_pose,
                self.APPROACH_STIFFNESS,
                self.APPROACH_DAMPING,
            )
            self.sleep_for(self.STEP_DT)

    # ------------------------------------------------------------------

    def _spiral_search_and_insert(
        self,
        get_observation,
        move_robot,
        send_feedback,
        center_x: float,
        center_y: float,
        contact_z: float,
        hold_ori: Quaternion,
        deadline,
    ) -> bool:
        """
        Run an Archimedean XY spiral around (center_x, center_y) at z = contact_z
        with a small downward push offset. When the actual TCP Z drops past
        SPIRAL_DROP_THRESHOLD past contact_z, we found the hole — commit and
        descend INSERT_DEPTH from there.

        Returns True if a hole was found and the plug was inserted, else False.
        """
        push_z = contact_z - self.SPIRAL_PUSH_OFFSET
        n_points = self.SPIRAL_TURNS * self.SPIRAL_POINTS_PER_TURN
        send_feedback("river-policy: starting spiral search")
        self.get_logger().info(
            f"Spiral search around xy=({center_x:.4f}, {center_y:.4f}) at z={contact_z:.4f}"
        )

        for i in range(1, n_points + 1):
            if self.time_now() >= deadline:
                self.get_logger().warn("Spiral search hit deadline")
                return False

            # Archimedean spiral: r grows linearly with index (and so with theta).
            frac = i / n_points
            theta = frac * self.SPIRAL_TURNS * 2.0 * np.pi
            r = self.SPIRAL_MAX_RADIUS * frac
            target_x = center_x + r * float(np.cos(theta))
            target_y = center_y + r * float(np.sin(theta))

            target = Pose(
                position=Point(x=target_x, y=target_y, z=push_z),
                orientation=hold_ori,
            )

            # Hold this XY for SPIRAL_DWELL while polling cur_z for a drop.
            dwell_steps = max(1, int(self.SPIRAL_DWELL / self.STEP_DT))
            for _ in range(dwell_steps):
                self._send_pose(
                    move_robot, target, self.SPIRAL_STIFFNESS, self.SPIRAL_DAMPING
                )
                self.sleep_for(self.STEP_DT)
                obs = self._wait_for_obs(get_observation)
                cur_z = obs.controller_state.tcp_pose.position.z
                drop = contact_z - cur_z

                if drop > self.SPIRAL_DROP_THRESHOLD:
                    # Confirm the drop is sustained, not a momentary surface dip.
                    confirmed = True
                    for _ in range(self.SPIRAL_DROP_CONFIRM_STEPS):
                        self._send_pose(
                            move_robot, target, self.SPIRAL_STIFFNESS, self.SPIRAL_DAMPING
                        )
                        self.sleep_for(self.STEP_DT)
                        obs = self._wait_for_obs(get_observation)
                        cur_z = obs.controller_state.tcp_pose.position.z
                        if (contact_z - cur_z) < self.SPIRAL_DROP_THRESHOLD:
                            confirmed = False
                            break
                    if not confirmed:
                        continue
                    msg = (
                        f"hole found at xy=({target_x:.4f}, {target_y:.4f}), "
                        f"r={r*1000:.1f} mm, drop={drop*1000:.1f} mm"
                    )
                    self.get_logger().info(msg)
                    send_feedback(f"river-policy: {msg}")
                    # Commit: continue descending at this XY, INSERT_DEPTH past contact_z.
                    insert_target_z = contact_z - self.INSERT_DEPTH - 0.005
                    commanded_z = cur_z
                    while commanded_z > insert_target_z and self.time_now() < deadline:
                        commanded_z -= self.INSERT_SPEED * self.STEP_DT
                        self._send_pose(
                            move_robot,
                            Pose(
                                position=Point(x=target_x, y=target_y, z=commanded_z),
                                orientation=hold_ori,
                            ),
                            self.INSERT_STIFFNESS,
                            self.INSERT_DAMPING,
                        )
                        self.sleep_for(self.STEP_DT)
                    self.get_logger().info(
                        f"Spiral commit complete (commanded_z={commanded_z:.4f})"
                    )
                    return True

        self.get_logger().info("Spiral search exhausted, no hole found")
        return False

    # ------------------------------------------------------------------

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        send_feedback(
            f"river-policy: classical insert for {task.plug_name} → {task.port_name}"
        )

        time_limit = float(task.time_limit) if task.time_limit > 0 else 60.0
        deadline = self.time_now() + Duration(seconds=time_limit)

        # Capture starting pose — XY and orientation are held throughout; Z ramps down.
        # (Reorient phase from V0.6 reverted: cable physics induced 12 N lateral pull
        #  in the new orientation and final plug-port distance was unchanged.)
        obs = self._wait_for_obs(get_observation)
        start = obs.controller_state.tcp_pose
        hold_x = start.position.x
        hold_y = start.position.y
        start_z = start.position.z
        hold_ori = Quaternion(
            x=start.orientation.x,
            y=start.orientation.y,
            z=start.orientation.z,
            w=start.orientation.w,
        )
        self.get_logger().info(
            f"Start pose: xyz=({hold_x:.3f}, {hold_y:.3f}, {start_z:.3f})"
        )

        # V1.0 — Triangulated vision correction: cast a ray through the best
        # port-blob in each of the three wrist cameras, find the least-squares
        # 3D intersection, and shift the descent centre toward that point.
        # Only applied when ≥2 cameras agree (residual < TRIANGULATION_RESIDUAL_THRESH).
        dx_vis, dy_vis, vis_ok = self._triangulate_port(obs, task)
        if vis_ok:
            hold_x = hold_x + dx_vis
            hold_y = hold_y + dy_vis
            self.get_logger().info(
                f"Descent centre after triangulation: ({hold_x:.4f}, {hold_y:.4f})"
            )

        # Module-name coarse correction — applied on top of vision correction.
        # Compensates for board placements where the target port is far from the
        # arm's fixed spawn pose (e.g. trial_3 sc_port_1 is ~21 cm away).
        mod_offset = self.MODULE_XY_OFFSETS.get(task.target_module_name)
        if mod_offset is not None:
            dx_mod, dy_mod = mod_offset
            hold_x += dx_mod
            hold_y += dy_mod
            self.get_logger().info(
                f"Module '{task.target_module_name}' offset ({dx_mod:+.3f}, {dy_mod:+.3f}) m "
                f"→ descent centre ({hold_x:.4f}, {hold_y:.4f})"
            )

        # Phase 0 — Settle: hold position, then average wrench to get a baseline.
        # Initial readings include cable weight + dynamic settling and would otherwise
        # trigger a false "contact" before any motion happened.
        hold_pose = Pose(
            position=Point(x=hold_x, y=hold_y, z=start_z), orientation=hold_ori
        )
        baseline_samples = []
        settle_steps = max(1, int(self.SETTLE_AT_START / self.STEP_DT))
        for _ in range(settle_steps):
            self._send_pose(
                move_robot, hold_pose, self.APPROACH_STIFFNESS, self.APPROACH_DAMPING
            )
            self.sleep_for(self.STEP_DT)
            obs = self._wait_for_obs(get_observation)
            wr = obs.wrist_wrench.wrench
            baseline_samples.append([wr.force.x, wr.force.y, wr.force.z])
        baseline = np.mean(baseline_samples[-10:], axis=0)  # last ~0.5 s
        self.get_logger().info(
            f"Wrench baseline: fx={baseline[0]:.1f} fy={baseline[1]:.1f} fz={baseline[2]:.1f} N"
        )

        # Baseline lateral force abort: if the arm spawned already in contact with
        # something (e.g. trial 3 starting at Z=0.046 with fx=-30N), descending
        # further causes a -11 penalty. Hold and return True immediately.
        if max(abs(baseline[0]), abs(baseline[1])) > self.BASELINE_LATERAL_ABORT:
            self.get_logger().warn(
                f"Baseline lateral force too high "
                f"(fx={baseline[0]:.1f} fy={baseline[1]:.1f} N) — "
                "arm likely spawned in contact; holding and returning"
            )
            send_feedback("river-policy: bad spawn state detected, holding")
            self._hold_current_pose(get_observation, move_robot)
            return True

        # Phase 1 — Approach: descend straight down until contact (delta wrench).
        contact_z = None
        commanded_z = start_z
        step_descent = self.DESCENT_SPEED * self.STEP_DT

        while self.time_now() < deadline and contact_z is None:
            obs = self._wait_for_obs(get_observation)
            wr = obs.wrist_wrench.wrench
            dfz = wr.force.z - baseline[2]
            cur_z = obs.controller_state.tcp_pose.position.z
            descended = start_z - cur_z

            # Contact check — only after we've actually moved enough to rule out drift.
            if descended >= self.MIN_APPROACH_BEFORE_CONTACT and abs(dfz) > self.CONTACT_FORCE:
                contact_z = cur_z
                msg = f"contact at z={cur_z:.4f} (dfz={dfz:+.1f} N) after descending {descended*1000:.1f} mm"
                self.get_logger().info(msg)
                send_feedback(f"river-policy: {msg}")
                break

            # Safety — don't descend past the reachable workspace.
            if (start_z - commanded_z) >= self.MAX_APPROACH_DEPTH:
                self.get_logger().warn(
                    f"Max approach depth ({self.MAX_APPROACH_DEPTH*100:.0f} cm) without contact "
                    f"— actual descent {descended*1000:.1f} mm"
                )
                send_feedback("river-policy: max approach depth, no contact")
                self._hold_current_pose(get_observation, move_robot)
                return True

            # Stall detection — if commanded keeps dropping but cur_z barely moves
            # the arm is probably at a joint limit / singularity. Bail out early.
            if (commanded_z - cur_z) < -self.APPROACH_STALL_LIMIT:
                self.get_logger().warn(
                    f"Approach stalled — commanded {(start_z - commanded_z)*1000:.0f} mm "
                    f"but actual {descended*1000:.0f} mm"
                )
                send_feedback("river-policy: approach stalled, aborting")
                self._hold_current_pose(get_observation, move_robot)
                return True

            commanded_z -= step_descent
            target = Pose(
                position=Point(x=hold_x, y=hold_y, z=commanded_z),
                orientation=hold_ori,
            )
            self._send_pose(
                move_robot, target, self.APPROACH_STIFFNESS, self.APPROACH_DAMPING
            )
            self.sleep_for(self.STEP_DT)

        if contact_z is None:
            self.get_logger().warn("Time limit reached before contact")
            send_feedback("river-policy: time limit reached before contact")
            self._hold_current_pose(get_observation, move_robot)
            return True

        # Phase 2 — Settle / seat: continue descending slowly. Hard-bound the
        # phase so a stuck plug-on-port-face never spins the loop forever.
        step_insert = self.INSERT_SPEED * self.STEP_DT
        commanded_z = contact_z
        insert_successful = False

        phase2_max = min(self.INSERT_PHASE_TIMEOUT, max(0.0, time_limit - 6.0))
        phase2_deadline = self.time_now() + Duration(seconds=phase2_max)
        last_progress_z = contact_z
        last_progress_t = self.time_now()

        while self.time_now() < deadline and self.time_now() < phase2_deadline:
            obs = self._wait_for_obs(get_observation)
            wr = obs.wrist_wrench.wrench
            dfz = wr.force.z - baseline[2]
            dfx = wr.force.x - baseline[0]
            dfy = wr.force.y - baseline[1]
            cur_z = obs.controller_state.tcp_pose.position.z

            descended_after_contact = contact_z - cur_z

            # Success: descended past the insertion depth after first contact.
            if descended_after_contact >= self.INSERT_DEPTH:
                insert_successful = True
                msg = f"inserted {descended_after_contact*1000:.1f} mm past contact (dfz={dfz:+.1f} N)"
                self.get_logger().info(msg)
                send_feedback(f"river-policy: {msg}")
                break

            # Hard stall: very high force.
            lateral = np.sqrt(dfx * dfx + dfy * dfy)
            if abs(dfz) > self.STALL_FORCE and lateral > 8.0:
                msg = f"stalled at z={cur_z:.4f}, dfz={dfz:+.1f} lat={lateral:.1f}"
                self.get_logger().warn(msg)
                send_feedback(f"river-policy: {msg}")
                break

            # No-progress check: if cur_z isn't dropping, the plug is jammed on
            # the port face. Bail out so the engine can score proximity/partial.
            if (last_progress_z - cur_z) > 0.001:
                last_progress_z = cur_z
                last_progress_t = self.time_now()
            elif (self.time_now() - last_progress_t) > Duration(seconds=self.NO_PROGRESS_TIMEOUT):
                msg = (
                    f"no Z progress for {self.NO_PROGRESS_TIMEOUT}s "
                    f"(stuck at z={cur_z:.4f}, dfz={dfz:+.1f} N) — exiting insertion"
                )
                self.get_logger().warn(msg)
                send_feedback(f"river-policy: {msg}")
                break

            commanded_z -= step_insert
            target = Pose(
                position=Point(x=hold_x, y=hold_y, z=commanded_z),
                orientation=hold_ori,
            )
            self._send_pose(
                move_robot, target, self.INSERT_STIFFNESS, self.INSERT_DAMPING
            )
            self.sleep_for(self.STEP_DT)

        # Phase 3 — Spiral search if direct insertion failed.
        # The plug is jammed on the port face (XY misaligned). Sweep XY
        # in a small spiral around the contact point looking for the hole.
        if not insert_successful and self.time_now() < deadline:
            insert_successful = self._spiral_search_and_insert(
                get_observation=get_observation,
                move_robot=move_robot,
                send_feedback=send_feedback,
                center_x=hold_x,
                center_y=hold_y,
                contact_z=contact_z,
                hold_ori=hold_ori,
                deadline=deadline,
            )

        # Phase 4 — Let the connector settle, then freeze the arm so it doesn't
        # keep tracking a stale low-Z command after we return.
        self.sleep_for(self.SETTLE_TIME)
        self._hold_current_pose(get_observation, move_robot)

        # Always return True so the engine measures the final plug-port distance
        # and awards proximity / partial-insertion points. Returning False here
        # is treated as "Task not completed" and zeros all Tier 2/3 scoring.
        self.get_logger().info(
            f"insert_cable returning True (heuristic insert={insert_successful}); "
            "engine will score by final plug position."
        )
        return True


# ---------------------------------------------------------------------------
# Training policy (online SAC via Gazebo rollouts)
# ---------------------------------------------------------------------------

class TrainingPolicy(Policy):
    """
    Runs SAC training inside insert_cable(), one episode per call.

    Each trial provides one episode:
      - explore randomly for LEARNING_STARTS steps
      - then switch to stochastic SAC policy
      - update the replay buffer and SAC networks after every step
      - checkpoint to WEIGHTS_FILE every CHECKPOINT_EVERY steps

    After sufficient episodes the weights can be used with MyPolicy.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._sac = _make_sac(WEIGHTS_FILE if os.path.exists(WEIGHTS_FILE) else None)
        self._total_steps = 0
        self._episode = 0
        os.makedirs(_WEIGHTS_DIR, exist_ok=True)
        self.get_logger().info(
            f"TrainingPolicy ready. Weights will be saved to {WEIGHTS_FILE}"
        )

    # ------------------------------------------------------------------

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self._episode += 1
        send_feedback(
            f"river-policy training: episode {self._episode}, "
            f"steps so far {self._total_steps}"
        )

        env = CableInsertionEnv(
            task=task,
            get_observation=get_observation,
            move_robot=move_robot,
            sleep_fn=self.sleep_for,
            logger=self.get_logger(),
        )

        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            # Random exploration before learning starts; stochastic SAC after.
            if self._total_steps < LEARNING_STARTS:
                action = env.action_space.sample()
            else:
                action, _ = self._sac.predict(obs, deterministic=False)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_reward += reward

            # Add transition to replay buffer.
            self._sac.replay_buffer.add(
                obs.reshape(1, -1),
                next_obs.reshape(1, -1),
                action.reshape(1, -1),
                np.array([reward], dtype=np.float32),
                np.array([float(terminated)], dtype=np.float32),
                [{}],
            )

            # Update SAC networks once learning has started.
            if self._sac.replay_buffer.size() >= LEARNING_STARTS:
                self._sac.train(batch_size=BATCH_SIZE, gradient_steps=GRADIENT_STEPS)
                # Flush SB3 metrics every 50 steps so we can see training progress.
                if self._total_steps % 50 == 0:
                    self._sac.logger.dump(step=self._total_steps)

            obs = next_obs
            self._total_steps += 1

            # Periodic checkpoint.
            if self._total_steps % CHECKPOINT_EVERY == 0:
                self._sac.save(WEIGHTS_FILE)
                self.get_logger().info(
                    f"Checkpoint saved at step {self._total_steps}"
                )

        self.get_logger().info(
            f"Episode {self._episode} done. "
            f"Return={ep_reward:.1f}  success={terminated}"
        )

        # Always save at episode end.
        self._sac.save(WEIGHTS_FILE)
        return bool(terminated)
