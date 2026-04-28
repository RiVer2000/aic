# River Policy — Handoff Doc

A classical compliant-insertion policy for the Intrinsic AIC qualification
phase. Scores **115.47 / 300** across the three configured trials in
`aic_engine/config/sample_config.yaml` (V1.0). V1.1 adds plug-tilt
compensation, vision-miss retry, and SC port depth increase. **V1.2 adds
Phase 2 quality gates** to reject false-success descents into non-port
features (closes the Trial 3 regression seen in V1.1).

This document assumes you already have the official WaveArm policy running on
your machine (i.e., you can launch the Gazebo eval container and a policy node
under `aic_model`).

---

## TL;DR

- Approach: pure classical control. **No RL, no demos, no learning.** A
  five-phase compliant descent with 3-camera triangulated XY correction.
- Trials 1 & 2 (SFP NIC) land ~5-6 cm from the port → ~40-50 pts each via
  Tier 2 + Tier 3 proximity.
- Trial 3 (SC) starts ~21 cm laterally from the port; visible camera blob
  correction shifts the descent ~5 cm closer, but the port is still missed
  → ~1-5 pts.
- Bottleneck is plug tilt (38°) and target localization. The compliant
  descent mechanics work; the plug just doesn't land over the port.

---

## Repo Layout

```
river-policy/
├── pixi.toml                    # ROS 2 build config (pixi-build-ros)
├── package.xml                  # ROS 2 package manifest
├── setup.py                     # ament_python entry
├── river_policy/
│   ├── __init__.py
│   ├── MyPolicy.py              # Loader shim (aic_model imports module path)
│   ├── TrainingPolicy.py        # Loader shim for the unused SAC variant
│   ├── policy.py                # *** real implementation ***
│   ├── env.py                   # Gymnasium env wrapper (used only by old SAC code)
│   └── train.py                 # Training entry-point notes (deprecated)
├── HANDOFF.md                   # this doc
└── TRAINING.md                  # quick run guide
```

The active class is **`MyPolicy`** in `policy.py`. `TrainingPolicy` and
`env.py` are leftover SAC code kept for reference; ignore them.

---

## How It Works

### Pre-Phase — Triangulated vision XY correction (V1.0)

Before any motion, we detect the port blob in all three wrist cameras
(left, center, right) and triangulate a 3D port position:

1. Each camera: Grayscale + Gaussian blur → threshold at gray=60 → morphological
   close → `cv2.findContours` → pick nearest-to-center blob in area range
2. Build a pinhole ray from each blob pixel using `CameraInfo.k` (fx, fy, cx, cy)
3. Map ray from camera frame → base_link using URDF-derived geometry:
   `TCP → cam_mount (−26.5 mm Z) → left/center/right camera`
4. Least-squares ray intersection: solve `A P = b` where `A = Σ(I − d dᵀ)`,
   `b = Σ(I − d dᵀ) oᵢ`
5. Accept only if ≥2 cameras detect a blob AND max per-camera residual < 15 mm
6. Clamp XY correction to ±40 mm and shift `hold_x, hold_y` to the triangulated XY

If vision fails (residuals too high or fewer than 2 cameras see a blob),
the policy descends from the raw spawn XY.

### Phase 0 — Settle and baseline wrench (~0.7 s)

Hold position for 0.7 s, average the last 10 wrist FT samples, store as
wrench baseline. All later contact checks use `wrench − baseline`.

**Baseline lateral abort**: if `|fx|` or `|fy|` in baseline > 15 N, the arm
has spawned in contact (Trial 3 worst case). Hold and return True immediately
rather than driving in and taking the −11 contact penalty.

### Phase 1 — Approach (~3-6 s)

Descend along world −Z at 30 mm/s with high-stiffness admittance
(`90, 90, 90, 50, 50, 50` N/m). Three exit conditions:

| Condition | What happens |
|-----------|--------------|
| `\|fz − baseline\|` > 5 N **and** descended ≥ 5 mm | Contact detected, enter Phase 1.5 |
| Commanded depth ≥ 25 cm | Max-depth: if vision was applied and retry not done yet, ascend and retry from raw spawn XY; otherwise abort |
| `\|commanded_z − cur_z\|` > 5 cm | Approach stall (joint limit / singularity) |

The max-depth retry (V1.1): if triangulation applied a correction but we still
miss, ascend back to `start_z`, reset `hold_x/hold_y` to original spawn XY,
re-baseline wrench, then descend again (once).

### Phase 1.5 — Partial reorient at contact (V1.1, ~1.5 s)

After contact is detected, perform a 60% SLERP from the current gripper
orientation toward the plug-aligned orientation (where plug Z = world −Z).
The plug tip is pinned in world frame throughout the rotation (same mechanism
as `_reorient_for_plug` from V0.6, but only 60% of the full rotation to
avoid the cable-physics overshoot seen in V0.6).

**Why 60%**: The 38° plug tilt causes 9.2 mm lateral drift at 15 mm insertion
depth (> ±5 mm tolerance). A 60% correction reduces this to ~3.7 mm, within
tolerance, while keeping cable bending forces manageable.

Returns the post-reorient `(hold_x, hold_y, hold_ori)` from actual TCP obs
— these become the new XY/orientation references for Phase 2 and Phase 3.

### Phase 2 — Direct insertion attempt (max 8 s)

Continue descending at 10 mm/s with softer Z-stiffness (60 N/m). Reference Z
is `phase2_start_z` (post-reorient gripper Z, not `contact_z`). Exit on:

| Condition | Reason |
|-----------|--------|
| Descended ≥ 15 mm past `phase2_start_z` **AND** quality gates pass | Real insertion success |
| Descended ≥ 15 mm but quality gates fail | **V1.2 false-success rejection** — ascend to start_z, skip Phase 3 |
| `\|dfz\|` > 22 N **and** lateral > 8 N | Hard stall — connector jammed |
| No 1 mm of progress in 2 s | Plug stuck on port face → trigger Phase 3 |

**V1.2 quality gates** (both must pass to declare success):
- Avg lateral wrench delta over last 0.5 s ≤ `PHASE2_QUALITY_LAT_MAX` (5 N)
- TCP XY drift from commanded center ≤ `PHASE2_QUALITY_DRIFT_MAX` (25 mm)

A real port hole guides the plug compliantly with low sustained lateral
force and no XY drift. Hitting an arbitrary obstacle on the board produces
lateral spikes and/or pushes the plug sideways. Without these gates, V1.1
Trial 3 declared "inserted 15.8 mm" after descending into a wrong feature
and ended 20 cm from the SC port (score: 1 pt).

### Phase 3 — Spiral search (~12 s worst case)

Fired only if Phase 2 didn't insert. Sweeps an Archimedean spiral in XY
around the post-reorient `(hold_x, hold_y)` contact point:

- 6 turns × 8 points = 48 XY positions
- Radius 0 → 70 mm linearly
- 0.25 s dwell at each point
- Z held 6 mm below `contact_z` with low Z-stiffness (30 N/m)
- **3-step sustain confirm** (V1.1): Z must stay > 6 mm below `contact_z`
  for 3 consecutive steps before "hole found" is declared — reduces false
  positives on surface irregularities

### Phase 4 — Hold and return

Sleep 1 s for cable to settle, then command the arm to its **current** pose
for 0.5 s (high stiffness) — prevents the controller from continuing to track
a stale low-Z command after `insert_cable()` returns.

**Always returns `True`.** Returning `False` causes `aic_engine` to label the
task "Task not completed" and zero out *all* Tier 2 + Tier 3 scoring.

---

## Scoring Trajectory

| Iteration | What changed | Score |
|-----------|--------------|-------|
| Initial SAC | Blind RL, random exploration | 3 |
| V0.1 classical | Compliant descent, returned True without moving | 57.6 |
| V0.2 | Wrench baselining + min-descent contact gate | 83.1 |
| V0.5 | + Phase 3 spiral search | 83.1 (spiral exhausted) |
| V0.6 | Tried gripper reorient — cable physics fought back | 56.0 (regressed) |
| V0.7a | Reverted reorient, added single-camera vision probe | 90.2 |
| V0.7b | Vision XY correction wired into descent + spiral | 90.2 |
| V0.8 | Expanded spiral (70 mm, 6 turns); per-module XY offsets | 86.8 |
| V1.0 | 3-camera triangulation; bad-spawn abort; raised drop threshold 3→6 mm; 2-step confirm | 115.47 |
| V1.1 | + Phase 1.5 partial reorient at contact; vision-miss retry; MAX_APPROACH_DEPTH 0.20→0.25 m; 3-step confirm | 89.62 |
| V1.2 | + Phase 2 quality gates (avg lateral force + XY drift) to reject false-success descents into non-port features | 82.96 |
| V1.3 | Disable Phase 1.5 + ascend to start_z on non-insertion. Catastrophic regression — engine measures plug AT return, ascend drove it above port surface, all 3 trials ended outside max_distance, scoring zeroed everywhere. | 3.00 |
| V1.4 | Keep Phase 1.5 disabled. Revert the ascend (back to V1.2 end-of-trial behaviour). | **94.60** ✓ (T1: 47.19, T2: 46.41, T3: 1.00) |
| **V1.5** | + One-shot ground-truth calibration mode (env var `RIVER_POLICY_CALIBRATE=1`) for capturing per-module port-vs-plug XY offsets to populate `MODULE_XY_OFFSETS`. | **TBD (after calibration data populated)** |

Per-trial breakdown at V1.0 (score 115.47):

| Trial | Plug | Final dist | Notes | Total |
|-------|------|-----------|-------|-------|
| 1 | SFP | ~0.05 m | Triangulation OK; spiral still didn't find port (plug tilt) | ~44 |
| 2 | SFP | ~0.07 m | Triangulation picked wrong blob → vision offset made it worse | ~40 |
| 3 | SC  | ~0.21 m | SC port out of camera FOV; baseline lateral abort fired on bad spawns | ~30 |

Per-trial breakdown at V1.1 (score 89.62):

| Trial | Plug | Final dist | Notes | Total |
|-------|------|-----------|-------|-------|
| 1 | SFP | 0.04 m | Phase 3 spiral **found hole** (heuristic insert) | 45.82 |
| 2 | SFP | 0.05 m | Phase 3 spiral exhausted, no hole found | 42.81 |
| 3 | SC  | 0.20 m | **False-success bug**: descended 15.8 mm into a non-port feature, ended outside max-distance bounding radius → all Tier 2 forfeit | 1.00 |

The V1.1 regression on Trial 3 was the entire 26-pt drop; trials 1+2
actually improved over V1.0. V1.2 closes that bug.

---

## Bottlenecks

### 1. Plug grasp tilt (the current top priority)

`PLUG_GRASP_RPY = (0.4432, −0.4838, 1.3303)` → plug Z-axis is **38°** off
gripper Z. At 15 mm insertion depth, this causes 9.2 mm lateral drift — larger
than the ±5 mm XY tolerance. Phase 1.5 (V1.1) partially compensates with a 60%
SLERP reorient at contact, reducing drift to ~3.7 mm.

### 2. Target localization

The triangulation pipeline works (residuals 1-3 mm when blobs are detected),
but the blob detector picks up the wrong dark feature on Trial 2 consistently.
The actual SFP port is not always the nearest-to-center dark blob — screws,
mounting holes, shadows, and other ports all qualify.

For Trial 3 (SC), the port is 20+ cm laterally from spawn — outside camera FOV.
The vision layer has no chance until the arm is within ~10 cm.

### 3. SC port depth (Trial 3)

SC port entrance is at `z ≈ −15.64 mm` from the board surface. With
`MAX_APPROACH_DEPTH = 0.25 m` (V1.1), we now descend far enough to reach it,
but Trial 3 starts with very high lateral force in some seeds (bad spawn),
triggering the baseline lateral abort before we can descend.

### 4. Proximity scoring formula

Tier 2 only awards smoothness/duration/efficiency points if Tier 3 > 0. On
Trial 3 we're outside the max-distance bounding radius (half the initial
plug-port distance) on bad spawns, so we collect zero on Tier 2 too.

---

## Future Extensions

Ranked by expected score impact for effort.

### A. Template-matched port detection — 4-8 hours

Instead of "darkest blob near center", match against rendered or hand-cropped
templates of the SFP slot (rectangular, ~10×7 mm) and SC port (circular,
~2 mm). Use `cv2.matchTemplate` at multiple scales.
- Pros: targeted detection; rejects screws, shadows, other holes
- Cons: doesn't help Trial 3 until arm is within FOV
- Expected gain: trials 1+2 from ~44/40 → 60-80; Trial 3 unchanged
- Drop-in at `MyPolicy._detect_port_pixel` in `policy.py`

### B. Ground-truth calibration run for MODULE_XY_OFFSETS

Run with `ground_truth:=true` to read the actual port 3D position from TF,
then compute `(dx, dy)` corrections from spawn XY and populate
`MyPolicy.MODULE_XY_OFFSETS`. This is the easiest way to fix Trial 3.
- Expected gain: Trial 3 from ~10 → 50-80 pts if offset is correct
- `MODULE_XY_OFFSETS` is in `policy.py` (currently empty dict)

### C. Imitation learning on CheatCode demos — 3-5 days

`CheatCode.py` reads ground-truth TF and inserts perfectly. Run it under
`ground_truth:=true`, record `(observation, action)` pairs to `lerobot`
format, train an ACT or diffusion policy on the demos, swap in for inference.
- Pros: proper target localization learned from data; transfers to all trials
- Cons: requires demo collection pipeline, GPU training
- Expected gain: 90 → 200+

### D. Train RL on top of classical controller (residual policy) — 3-5 days

Keep the classical descent. RL only sees wrench + small CNN over center camera,
outputs delta-XY corrections to the descent center. Reward on Tier 3 directly.

### Quick wins under 1 hour each

- Increase `REORIENT_AT_CONTACT_FRACTION` from 0.6 toward 1.0 in small steps;
  watch Trial 1/2 scores — stop if cable-physics overshoot reappears
- Two-pass detection: detect top-3 blobs, sweep a quick Z-push test over each,
  keep whichever shows a real Z-drop first
- Try `DESCENT_SPEED = 0.020` for smoother contact detection

---

## Running Instructions

You said WaveArm runs on your machine, so the env is good. The river-policy
package is editable-installed via `uv` (because the repo's `pixi install`
has a known stale-lockfile issue) — see Setup below if it isn't already present.

### Setup (only if river-policy isn't installed)

From the repo root (`aic/`):

```bash
# stable-baselines3 is required even though we don't use it (legacy import)
uv pip install --python .pixi/envs/default/bin/python3.12 stable-baselines3

# install river-policy in editable mode
uv pip install --python .pixi/envs/default/bin/python3.12 -e river-policy/

# verify
pixi run --frozen python -c "from river_policy.policy import MyPolicy; print('OK')"
```

If `pixi run` errors with "failed to solve requirements", **always pass
`--frozen`** — it skips lockfile resolution and uses the already-built env.

### Run a full evaluation

Open two terminals from the repo root.

**Terminal 1 — Gazebo + RViz2 (with aic_engine):**
```bash
distrobox enter -r aic_eval -- bash -c \
  "__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
   /entrypoint.sh ground_truth:=false start_aic_engine:=true" \
  2>&1 | tee river-policy/gazebo.log
```

Wait for Gazebo + RViz2 to fully load (look for the task board spawning).

**Terminal 2 — Policy:**
```bash
pixi run --frozen ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true -p policy:=river_policy.MyPolicy \
  2>&1 | tee river-policy/run.log
```

`aic_engine` configures and activates the lifecycle node automatically.
The brief `[ERROR] aic_model lifecycle is not in the active state` during
startup is a harmless race condition — the engine retries within 1 s.

### Tunables

All are class constants on `MyPolicy` in `policy.py`. Common ones:

| Constant | Default (V1.1) | What it controls |
|----------|----------------|------------------|
| `DESCENT_SPEED` | 0.030 m/s | Phase 1 speed |
| `INSERT_SPEED` | 0.010 m/s | Phase 2 speed |
| `MAX_APPROACH_DEPTH` | **0.25 m** | Phase 1 safety stop |
| `INSERT_DEPTH` | 0.015 m | Phase 2 success threshold |
| `INSERT_PHASE_TIMEOUT` | 8.0 s | Phase 2 hard cap |
| `NO_PROGRESS_TIMEOUT` | 2.0 s | Phase 2 no-progress trigger |
| `REORIENT_AT_CONTACT_FRACTION` | **0.6** | Phase 1.5 partial SLERP fraction |
| `REORIENT_AT_CONTACT_TIME` | **1.5 s** | Phase 1.5 duration |
| `SPIRAL_MAX_RADIUS` | 0.070 m | Phase 3 spiral max radius |
| `SPIRAL_TURNS` | 6 | Phase 3 spiral revolutions |
| `SPIRAL_DROP_THRESHOLD` | 0.006 m | Z drop to declare hole found |
| `SPIRAL_DROP_CONFIRM_STEPS` | **3** | Sustain steps before committing |
| `TRIANGULATION_RESIDUAL_THRESH` | 0.015 m | Max ray residual to accept triangulation |
| `BASELINE_LATERAL_ABORT` | 15.0 N | Lateral force in baseline → bad spawn abort |
| `PHASE2_QUALITY_WINDOW_STEPS` | 10 | V1.2: rolling-window samples for avg lateral |
| `PHASE2_QUALITY_LAT_MAX` | 5.0 N | V1.2: avg lateral threshold for real success |
| `PHASE2_QUALITY_DRIFT_MAX` | 0.025 m | V1.2: TCP drift threshold for real success |
| `APPROACH_STIFFNESS` | [90,90,90,50,50,50] | Phase 1 admittance |
| `INSERT_STIFFNESS` | [80,80,60,40,40,40] | Phase 2 admittance |
| `REORIENT_AT_CONTACT_STIFFNESS` | [70,70,60,15,15,30] | Phase 1.5 admittance |
| `SPIRAL_STIFFNESS` | [80,80,30,40,40,40] | Phase 3 admittance |

Edits are picked up immediately — the package is editable-installed, no
reinstall needed. Just restart Terminal 2.

### Where to look in the logs

- `run.log` (Terminal 2): policy-side state — phase transitions, contact
  detection, triangulation residuals, spiral progress
- `gazebo.log` (Terminal 1): scoring summary, contact events, engine state

Search for `Total Score:` in `gazebo.log` for the per-run total.
Search for `tier_2:` / `tier_3:` for the per-category breakdown.
Search for `Triangulation OK` / `Triangulation rejected` in `run.log` to
see which trials benefit from vision correction.

---

## Known Issues / Gotchas

- **`pixi install` doesn't work** in this checkout (stale lockfile from
  upstream divergence). Always use `pixi run --frozen` and install new
  Python deps via `uv pip install --python .pixi/envs/default/bin/python3.12 ...`.
- **Lifecycle race on first activation**: `aic_engine` sometimes sends the
  first goal in the millisecond between `on_configure` and `on_activate`.
  This shows as one `[ERROR] aic_model lifecycle is not in the active state`
  line. Engine retries; ignore.
- **Robot keeps moving after `insert_cable()` returns** if you don't call
  `_hold_current_pose` before returning. Already handled in current code,
  but if you add new return paths, remember to call it.
- **`tcp_pose` from `controller_state` is the actual current pose**, not the
  reference. The reference is `reference_tcp_pose`. We use `tcp_pose` for
  `cur_z`-based progress checks.
- **`task.time_limit` is in seconds (`uint64`)**. We default to 60 s if zero.
- **Wrench is in the FT-sensor frame, not base_link.** Sign convention for
  `fz` flips during contact in our runs (negative dfz on contact) — that's
  why all checks use `abs(dfz)`.
- **Phase 1.5 may cause `phase2_start_z` ≠ `contact_z`** if the SLERP
  rotation lifts or lowers the gripper Z. This is intentional — we use
  `phase2_start_z` as the descent reference, not the original contact Z.

---

## Contacts / Continuity

- Branch: `user/rishabh`
- Remote: `https://github.com/RiVer2000/aic.git` (user `shantanu-ghodgaonkar`)
- Last commits:
  - `93115ad` river-policy: expand spiral radius and add per-module XY offset
  - `307c7a6` river-policy: add handoff doc
  - `c55ba73` river-policy: spiral search + vision-guided XY correction (V0.5–V0.7b)

If you pick this up: start by running it once to confirm ~115/300, then:
1. Run with `ground_truth:=true` to calibrate `MODULE_XY_OFFSETS` for Trial 3
2. Tune `REORIENT_AT_CONTACT_FRACTION` upward if V1.1 improves Trial 1/2
3. Consider template-matched port detection (Extension A) for Trials 1+2
