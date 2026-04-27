# River Policy — Run Guide

## Overview

**Active policy: `MyPolicy` — classical compliant-descent controller (no learning).**

The SAC training code (`TrainingPolicy`, `env.py`) is kept in the repo as
reference but is **not used for evaluation**. Blind RL cannot learn target
localization without camera or TF signals, so we switched to a classical
two-phase compliant descent that relies on the admittance controller and
the port chamfer to guide the plug in.

Algorithm (see `policy.py:MyPolicy`):
1. **Approach** — slow descent (15 mm/s) with low Z-stiffness until wrist
   force indicates contact with the port face.
2. **Settle** — slower descent (8 mm/s) with even lower stiffness so the
   chamfer can funnel the plug into the hole.
3. **Verify** — if we descended > 15 mm past the contact point, it's seated.

No vision, no learning, no ground-truth TF needed.

---

## Prerequisites

- Gazebo evaluation container available as `aic_eval` in distrobox
- NVIDIA GPU (the policy node will use CUDA automatically)
- All packages installed via `uv` as described in the repo README

---

## Step-by-Step Launch

Open **two terminals**, both from `/home/rishabh/ws_aic/src/aic`.

> `aic_engine` (launched with Gazebo) automatically sends tasks to the policy
> node once it activates. You do **not** need to send goals manually.

---

### Terminal 1 — Gazebo + RViz2

```bash
distrobox enter -r aic_eval -- bash -c \ 
    "__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
     /entrypoint.sh ground_truth:=false start_aic_engine:=true" 2>&1 | tee river-policy/gazebo.log
```

Wait until you see Gazebo and RViz2 fully loaded before continuing.

---

### Terminal 2 — Policy Node

```bash
pixi run --frozen ros2 run aic_model aic_model \
    --ros-args -p use_sim_time:=true -p policy:=river_policy.MyPolicy 2>&1 | tee river-policy/run.log
```

`aic_engine` will configure and activate the node automatically. Once you see:
```
[INFO] on_activate()
[INFO] Goal accepted
```
each trial runs back-to-back without any manual intervention.

> **Note:** The `[ERROR] aic_model lifecycle is not in the active state` message
> that appears briefly during startup is a harmless race condition — the engine
> retries and succeeds immediately after.

---

## What to Expect

`aic_engine` cycles through its configured trials automatically. Per trial
you will see the policy log:
```
[INFO] MyPolicy (classical compliant descent) ready
[INFO] Start pose: xyz=(0.xxx, 0.yyy, 1.zzz)
[INFO] river-policy: contact at z=1.zzz (fz=X.X N)
[INFO] river-policy: inserted N.N mm
```
followed by the `aic_engine` score summary.

### Smoke test pass criteria
- Policy logs `Start pose` within 1 s of goal acceptance
- Policy logs `contact at z=…` during descent (arm hits port face)
- No Python tracebacks
- `aic_engine` reports a Tier 3 score > 0 (proximity or partial insertion)

---

## Main Training Run

`aic_engine` drives all episodes automatically. Just leave both terminals
running. After ~500 total steps (end of episode 2) SAC loss logs will appear
in Terminal 2:
```
train/actor_loss   -0.23
train/critic_loss   0.45
train/ent_coef      0.42
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

Then activate the node if needed (see lifecycle check above). `aic_engine`
will send goals automatically and the policy runs deterministically.

---

## Troubleshooting

**`pixi run` fails with "failed to solve requirements"**
Use `--frozen` flag on all `pixi run` commands (lockfile is stale upstream).

**`/aic_model` not found by lifecycle commands**
The policy node is not running. Check Terminal 2.

**`Goal rejected` or node stuck in inactive state**
`aic_engine` did not auto-activate the node. Run manually:
```bash
pixi run --frozen ros2 lifecycle set /aic_model configure
pixi run --frozen ros2 lifecycle set /aic_model activate
```

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
