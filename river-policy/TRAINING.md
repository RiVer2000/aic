# River Policy — Training Guide

## Overview

Training runs **online** against the live Gazebo simulation. Each `insert_cable`
action call is one episode. The SAC agent explores randomly for the first 500
steps (~2 episodes), then starts updating its networks after every step.

Weights are saved automatically to:
```
river-policy/river_policy/weights/sac_cable.zip
```

---

## Prerequisites

- Gazebo evaluation container available as `aic_eval` in distrobox
- NVIDIA GPU (the policy node will use CUDA automatically)
- All packages installed via `uv` as described in the repo README

---

## Step-by-Step Launch

Open **three terminals**, all from `/home/rishabh/ws_aic/src/aic`.

---

### Terminal 1 — Gazebo + RViz2

```bash
distrobox enter -r aic_eval -- bash -c \
  "__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
   /entrypoint.sh ground_truth:=false start_aic_engine:=true"
```

Wait until you see Gazebo and RViz2 fully loaded before continuing.

---

### Terminal 2 — Policy Node

```bash
pixi run --frozen ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true -p policy:=river_policy.TrainingPolicy
```

Expected output:
```
[INFO] Loading policy module: river_policy.TrainingPolicy
[INFO] Using policy: TrainingPolicy
```

Wait here — the node is loaded but not yet active.

---

### Terminal 3 — Activate + Send Goals

**Step 1: Activate the lifecycle node** (once per session)

```bash
pixi run --frozen ros2 lifecycle set /aic_model configure
pixi run --frozen ros2 lifecycle set /aic_model activate
```

Expected output in Terminal 2:
```
[INFO] on_configure(...)
[INFO] Instantiating policy...
Using cuda device
[INFO] TrainingPolicy ready. Weights will be saved to .../sac_cable.zip
[INFO] on_activate()
```

**Step 2: Send training goals** (see scenarios below)

---

## Training Scenarios

Run these one at a time in Terminal 3. Each command blocks until the episode
finishes, then you run the next one. Cycle through all three trials to cover
both plug types and board positions.

After each goal you will see in Terminal 2:
```
[INFO] Episode N done. Return=XXX  success=True/False
```

---

### Smoke Test — Single Episode Per Trial

Run one episode of each trial type to confirm the full pipeline works before
starting a long training run.

**Trial 1 — SFP, NIC card rail 0**
```bash
pixi run --frozen ros2 action send_goal /insert_cable \
  aic_task_interfaces/action/InsertCable \
  "{task: {id: 'trial_1', cable_type: 'sfp_sc', cable_name: 'cable_0', \
    plug_type: 'sfp', plug_name: 'sfp_tip', port_type: 'sfp', \
    port_name: 'sfp_port_0', target_module_name: 'nic_card_mount_0', \
    time_limit: 180}}"
```

**Trial 2 — SFP, NIC card rail 1**
```bash
pixi run --frozen ros2 action send_goal /insert_cable \
  aic_task_interfaces/action/InsertCable \
  "{task: {id: 'trial_2', cable_type: 'sfp_sc', cable_name: 'cable_0', \
    plug_type: 'sfp', plug_name: 'sfp_tip', port_type: 'sfp', \
    port_name: 'sfp_port_0', target_module_name: 'nic_card_mount_1', \
    time_limit: 180}}"
```

**Trial 3 — SC plug**
```bash
pixi run --frozen ros2 action send_goal /insert_cable \
  aic_task_interfaces/action/InsertCable \
  "{task: {id: 'trial_3', cable_type: 'sfp_sc', cable_name: 'cable_1', \
    plug_type: 'sc', plug_name: 'sc_tip', port_type: 'sc', \
    port_name: 'sc_port_base', target_module_name: 'sc_port_1', \
    time_limit: 180}}"
```

**Smoke test pass criteria:**
- Terminal 2 shows `Episode N done. Return=... success=False` (False is expected
  during random exploration — the episode completing without a crash is the pass)
- No Python tracebacks in Terminal 2
- The robot arm visibly moves in Gazebo during each episode

---

## Main Training Run

Once smoke tests pass, cycle through trials repeatedly. After ~500 total steps
(end of episode 2) you will see SAC loss logs in Terminal 2:
```
train/actor_loss   -0.23
train/critic_loss   0.45
train/ent_coef      0.42
```

Recommended cycle — run each command after the previous one returns:

```bash
# Round 1
pixi run --frozen ros2 action send_goal /insert_cable aic_task_interfaces/action/InsertCable "{task: {id: 'trial_1', cable_type: 'sfp_sc', cable_name: 'cable_0', plug_type: 'sfp', plug_name: 'sfp_tip', port_type: 'sfp', port_name: 'sfp_port_0', target_module_name: 'nic_card_mount_0', time_limit: 180}}"
pixi run --frozen ros2 action send_goal /insert_cable aic_task_interfaces/action/InsertCable "{task: {id: 'trial_2', cable_type: 'sfp_sc', cable_name: 'cable_0', plug_type: 'sfp', plug_name: 'sfp_tip', port_type: 'sfp', port_name: 'sfp_port_0', target_module_name: 'nic_card_mount_1', time_limit: 180}}"
pixi run --frozen ros2 action send_goal /insert_cable aic_task_interfaces/action/InsertCable "{task: {id: 'trial_3', cable_type: 'sfp_sc', cable_name: 'cable_1', plug_type: 'sc', plug_name: 'sc_tip', port_type: 'sc', port_name: 'sc_port_base', target_module_name: 'sc_port_1', time_limit: 180}}"

# Repeat rounds until success rate improves
```

**Training milestones to watch for:**

| Episodes | Expected behaviour |
|----------|--------------------|
| 1–2      | Random actions, `Return` around -300 to -400 |
| 3–10     | SAC loss logs appear, `Return` slowly improves |
| 20–50    | Return starts trending above -100 |
| 50+      | Occasional `success=True` episodes |
| 100+     | Consistent successes on familiar trials |

---

## Switching to Inference

Once `success=True` appears consistently, stop the TrainingPolicy node
(Ctrl+C in Terminal 2) and run the inference policy instead:

```bash
pixi run --frozen ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true -p policy:=river_policy.MyPolicy
```

Then activate and send goals exactly as above. The policy will run
deterministically using the saved weights.

---

## Troubleshooting

**`pixi run` fails with "failed to solve requirements"**
Use `--frozen` flag on all `pixi run` commands (lockfile is stale upstream).

**`/aic_model` not found by lifecycle commands**
The policy node is not running. Check Terminal 2.

**`Goal rejected`**
The node is not in the active lifecycle state. Run configure + activate again.

**Episode returns -400 after many rounds (no improvement)**
The reward shaping may need tuning for your specific board configuration.
Check the wrench values being observed:
```bash
pixi run --frozen ros2 topic echo /fts_broadcaster/wrench --once
```
If forces are near zero during episodes the robot is not making contact —
check that the robot arm is actually near the port in Gazebo.

**`success=True` but aic_engine does not score it**
The wrench-based success heuristic in `env.py:_compute_reward` fired but the
actual connector was not fully seated. Tune `INSERTION_FZ_MIN` (currently 5 N)
downward in `river_policy/env.py` if the connector seats with less force.
