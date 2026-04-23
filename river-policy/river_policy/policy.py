"""
MyPolicy  – inference-only: loads a trained SAC checkpoint and runs it.
TrainingPolicy – online RL training: collects rollouts in Gazebo and updates SAC.

Usage
-----
Inference (eval submission):
    pixi run ros2 run aic_model aic_model \\
        --ros-args -p use_sim_time:=true -p policy:=river_policy.MyPolicy

Training (run against a live Gazebo env with multiple trials):
    pixi run ros2 run aic_model aic_model \\
        --ros-args -p use_sim_time:=true -p policy:=river_policy.TrainingPolicy
"""

from __future__ import annotations

import os

import numpy as np
from rclpy.duration import Duration

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

    if load_path and os.path.exists(load_path):
        return SAC.load(load_path, env=_DummyEnv())

    return SAC(
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


# ---------------------------------------------------------------------------
# Inference policy
# ---------------------------------------------------------------------------

class MyPolicy(Policy):
    """
    Loads a pre-trained SAC checkpoint and runs deterministic inference.

    The same weights handle both SFP_MODULE and SC_PLUG because the plug
    type is encoded as a 2D one-hot appended to the observation vector.
    """

    def __init__(self, parent_node):
        super().__init__(parent_node)
        self._model = None
        if os.path.exists(WEIGHTS_FILE):
            try:
                self._model = _make_sac(WEIGHTS_FILE)
                self.get_logger().info(f"Loaded SAC weights from {WEIGHTS_FILE}")
            except Exception as exc:
                self.get_logger().error(f"Failed to load weights: {exc}")
        else:
            self.get_logger().warn(
                f"No weights found at {WEIGHTS_FILE}. "
                "Run with TrainingPolicy first, or copy a checkpoint there."
            )

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        if self._model is None:
            self.get_logger().error("No model loaded – cannot run inference.")
            return False

        env = CableInsertionEnv(
            task=task,
            get_observation=get_observation,
            move_robot=move_robot,
            sleep_fn=self.sleep_for,
            logger=self.get_logger(),
        )

        obs, _ = env.reset()
        time_limit = float(task.time_limit) if task.time_limit > 0 else 60.0
        deadline = self.time_now() + Duration(seconds=time_limit)

        send_feedback(
            f"river-policy inference: {task.plug_name} → {task.port_name}"
        )

        while self.time_now() < deadline:
            action, _ = self._model.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(action)

            if terminated:
                send_feedback("river-policy: insertion succeeded")
                self.sleep_for(2.0)
                return True

            if truncated:
                send_feedback("river-policy: max steps reached without success")
                return False

        send_feedback("river-policy: time limit reached")
        return False


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
