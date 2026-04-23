"""
Training entry point for the river-policy SAC agent.

The AIC framework does not expose a standalone env reset API, so training
runs *inside* the policy node itself: each `insert_cable()` call provides
one episode, and the eval system randomises the board between calls.

To train:
    1.  Start the distrobox eval container with Gazebo running.
    2.  In a second terminal, launch the model node using TrainingPolicy:

        $ pixi reinstall ros-kilted-river-policy
        $ pixi run ros2 run aic_model aic_model \\
              --ros-args -p use_sim_time:=true \\
                         -p policy:=river_policy.TrainingPolicy

    3.  Trigger repeated insert_cable tasks (e.g. via the scoring runner or
        a manual action client) until enough episodes have accumulated.
        Weights are auto-saved to river_policy/weights/sac_cable.zip after
        every episode and every CHECKPOINT_EVERY steps.

    4.  Switch to inference once training converges:

        $ pixi run ros2 run aic_model aic_model \\
              --ros-args -p use_sim_time:=true \\
                         -p policy:=river_policy.MyPolicy

Design notes
------------
Algorithm: SAC (Soft Actor-Critic)
  - Off-policy → sample-efficient; each Gazebo step is expensive.
  - Maximum-entropy framework → systematic exploration of contact-rich spaces.
  - Handles continuous 6-DoF delta-pose action space naturally.
  - Outperforms PPO on contact-rich low-DoF manipulation in practice.

Observation (21D) – sensor-only, no ground-truth TF:
  tcp_pose.position (3)  + tcp_pose.orientation (4)  # absolute TCP state
  tcp_velocity.linear (3) + tcp_velocity.angular (3) # velocity feedback
  wrist_force (3) + wrist_torque (3)                  # contact sensing
  plug_type one-hot (2)                               # task conditioning

Action (6D): delta TCP pose [dx, dy, dz, drx, dry, drz]
  Normalised to [-1, 1], scaled to ±5 mm / ±30 mrad per 50 ms step.
  This gives ~10 cm/s and ~0.6 rad/s maximum speed — safe for contact.

Reward:
  r_align    = -0.02 * (fx² + fy²)      # penalise lateral contact forces
  r_progress = +0.05 * max(0, −fz)      # reward downward insertion force
  r_limit    = −5 * (excess_force²      # hard penalty above FORCE_LIMIT
                     + excess_torque²)
  r_step     = −0.05                    # time penalty
  r_success  = +100                     # sparse bonus on insertion

Plug-type generalisation:
  A 2D one-hot vector [sfp=1 0, sc=0 1] is appended to every observation.
  The policy learns separate insertion dynamics for each type while sharing
  all network weights. Extend PLUG_TYPE_MAP in env.py for new connectors.

Robustness to pose deviations (~2 mm / ~0.04 rad):
  - SAC entropy bonus drives exploration that covers this neighbourhood.
  - Wrench feedback lets the policy detect and correct misalignment at
    contact rather than relying on open-loop accuracy.
  - Board randomisation across episodes prevents pose overfitting.
"""
