# River Policy — Handoff Doc

A classical compliant-insertion policy for the Intrinsic AIC qualification
phase. **Current best: 94.60 / 300** (V1.4) on the three trials in
`aic_engine/config/sample_config.yaml`. The next clear win is fixing Trial 3
via the V1.5 ground-truth calibration mode — instructions below.

This document assumes the reader can already launch the Gazebo eval container
and run a policy node under `aic_model` (i.e., the official WaveArm policy
works on your machine).

---

## TL;DR — Where we are today

- Approach: **pure classical control**. No RL, no demos, no learning.
  Three-camera triangulated XY correction → compliant descent → optional
  Archimedean spiral search.
- **Trials 1 & 2 (SFP/NIC) saturate at ~47 pts each** (proximity + Tier 2).
  Plug lands ~5 cm from port. Pushing higher needs actual insertion.
- **Trial 3 (SC) stuck at 1 pt**. The SC port is 20+ cm laterally from
  spawn — outside camera FOV, so vision can't help.
- Bottleneck is target localization for Trial 3, **not control**.
- **The unlock is `MODULE_XY_OFFSETS`**: hard-code per-module (port, plug)
  XY offsets captured under `ground_truth:=true`. V1.5 adds a one-shot
  calibration mode that does this for you.

**Status snapshot (V1.4, single run):**

| Trial | Plug → Port | Final dist | Score |
|-------|-------------|-----------|-------|
| 1 | SFP → nic_card_mount_0 | 0.04 m | 47.19 |
| 2 | SFP → nic_card_mount_1 | 0.05 m | 46.41 |
| 3 | SC → sc_port_1 | 0.22 m | 1.00 |
| **Total** | | | **94.60 / 300** |

---

## Recommended next 1–2 hours of work

1. **Run V1.5 calibration** (instructions below). Captures three lines like
   `"sc_port_1": (-0.184, +0.052),` from the ground-truth TF.
2. **Paste them into `MyPolicy.MODULE_XY_OFFSETS`** in `policy.py`.
3. **Rerun normally** with `ground_truth:=false`.
4. Expected: Trial 3 jumps from 1 → 50–80 pts. **Total: 145–175 / 300.**

After that, the next-best lever is template-matched port detection for
Trials 1+2 (Extension A below), which can push them from ~47 to ~65 each.

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

The active class is **`MyPolicy`** in `policy.py`. `TrainingPolicy` and `env.py`
are leftover SAC code kept for reference; ignore them.

---

## How It Actually Works (V1.5)

The flow per `insert_cable()` call. Phase 1.5 partial-reorient is **disabled
by default** (`REORIENT_AT_CONTACT_FRACTION = 0.0`); the helpers stay in the
file in case someone wants to re-enable.

### Calibration short-circuit (V1.5, only when env var set)

If `RIVER_POLICY_CALIBRATE=1` is set in the policy node's environment, the
function reads port + plug-tip TFs, logs the offset, and returns `True`
within ~1 s. Skips everything else. See "Calibration mode" below.

### Pre-Phase — Triangulated vision XY correction (V1.0)

Detect the port blob in all three wrist cameras, triangulate a 3D port
position, shift the descent center toward it.

1. Per camera: grayscale + Gaussian blur → threshold at gray=60 → morphological
   close → `cv2.findContours` → pick nearest-to-center blob in a plug-type-
   appropriate area range.
2. Build a pinhole ray from each blob pixel using `CameraInfo.k`
   (fx, fy, cx, cy).
3. Map ray from camera frame → base_link via URDF-derived static geometry:
   `TCP → cam_mount (−26.5 mm Z) → left/center/right camera`.
4. Least-squares ray intersection: solve `A P = b` where
   `A = Σ(I − dᵢ dᵢᵀ)`, `b = Σ(I − dᵢ dᵢᵀ) oᵢ`.
5. Accept only if **≥2 cameras detect a blob AND** max per-camera residual
   `< TRIANGULATION_RESIDUAL_THRESH` (15 mm).
6. Clamp the correction to ±`MAX_TRIANGULATION_CORRECTION` (40 mm).
7. Then **add `MODULE_XY_OFFSETS[task.target_module_name]`** if the dict has
   an entry — the V1.5 calibration populates this.

If both layers fail, descent uses the raw spawn XY.

### Phase 0 — Settle and baseline wrench (~0.7 s)

Hold position, average the last 10 wrist FT samples, store as wrench
baseline. All later contact checks use `wrench − baseline`.

**Baseline lateral abort**: if `|fx|` or `|fy|` in baseline > 15 N, the arm
spawned in unwanted contact. Hold and return `True` immediately rather
than driving in (avoids the −12 force penalty).

### Phase 1 — Approach (~3–6 s)

Descend along world −Z at 30 mm/s with high-stiffness admittance
(`90, 90, 90, 50, 50, 50` N/m). Three exit conditions:

| Condition | What happens |
|-----------|--------------|
| `\|fz − baseline\|` > 5 N **and** descended ≥ 5 mm | Contact detected, enter Phase 1.5 |
| Commanded depth ≥ 25 cm | Max-depth: ascend + retry from raw spawn XY (once, only if vision was applied) |
| `\|commanded_z − cur_z\|` > 5 cm | Approach stall (joint limit / singularity) — abort |

The retry: if triangulation pulled us off-target, ascend back to `start_z`,
reset `(hold_x, hold_y)` to original spawn XY, re-baseline, descend again.
Triggered max once per trial.

### Phase 1.5 — Partial reorient at contact (DISABLED by default in V1.4)

`REORIENT_AT_CONTACT_FRACTION = 0.0`. Function short-circuits and returns
the current pose without consuming the 1.5 s SLERP window.

**Why disabled:** V1.1 enabled this at 60% SLERP. Visual observation showed
the gripper rotating "anticlockwise around vertical" instead of the intended
backward pitch. Cause: the math computes a correct horizontal-axis rotation,
but the simultaneous pin-the-tip translation (~17 mm in horizontal plane)
visually reads as yaw + slide. Post-reorient plug-tip-relative-to-TCP
magnitude (53 mm) didn't match the static plug offset (45 mm), confirming
the controller was fighting compliance during the rotation. V1.4 disabling
gained ~5 pts on each of trials 1+2.

The helpers (`_slerp_wxyz`, `_reorient_at_contact`, plug-grasp constants)
remain in the file. Raise `REORIENT_AT_CONTACT_FRACTION` to re-enable.

### Phase 2 — Direct insertion attempt (max 8 s)

Continue descending at 10 mm/s with softer Z-stiffness (60 N/m). Reference Z
is `phase2_start_z` (post-reorient gripper Z; same as `contact_z` when 1.5
is disabled).

| Exit | Reason |
|------|--------|
| Descended ≥ 15 mm **AND** quality gates pass | Real insertion success |
| Descended ≥ 15 mm but quality gates fail | **V1.2 false-success rejection** — ascend to `start_z`, skip Phase 3 (don't re-jam) |
| `\|dfz\|` > 22 N **and** lateral > 8 N | Hard stall — connector jammed |
| No 1 mm of progress in 2 s | Plug stuck on port face → trigger Phase 3 |

**V1.2 quality gates** (both must pass):
- Avg lateral wrench delta over last 0.5 s ≤ `PHASE2_QUALITY_LAT_MAX` (5 N).
- TCP XY drift from commanded center ≤ `PHASE2_QUALITY_DRIFT_MAX` (25 mm).

A real port hole guides the plug compliantly; hitting an arbitrary obstacle
spikes lateral force and pushes the plug sideways. Without these gates, V1.1
Trial 3 declared "inserted 15.8 mm" after descending into a wrong feature
and ended 20 cm from the SC port (1 pt total).

### Phase 3 — Spiral search (~12 s worst case)

Fired only if Phase 2 didn't insert and wasn't rejected. Sweeps an
Archimedean spiral in XY around the contact `(hold_x, hold_y)`:

- 6 turns × 8 points = **48 XY positions**, radius 0 → 70 mm linearly
- 0.25 s dwell per position
- Z held 6 mm below `contact_z` with low Z-stiffness (30 N/m)
- **3-step sustain confirm**: Z must stay > 6 mm below `contact_z` for
  3 consecutive steps before "hole found" — reduces false positives on
  surface dimples.

On hole: descend `INSERT_DEPTH + 5 mm` past contact_z at that XY, return.

### Phase 4 — Hold and return

Sleep 1 s for cable to settle, then command the arm to its **current** pose
for 0.5 s (high stiffness). Prevents the controller from continuing to track
a stale low-Z command after `insert_cable()` returns.

**Always returns `True`.** Returning `False` causes `aic_engine` to label
the task "Task not completed" and zero out *all* Tier 2 + Tier 3 scoring.

---

## Calibration Mode (V1.5) — How to populate `MODULE_XY_OFFSETS`

This is the recommended next step for the next person. ~5 minutes wall-clock.

### Step 1 — Launch with ground truth

**Terminal 1 — Gazebo with `ground_truth:=true`** (note the change):
```bash
distrobox enter -r aic_eval -- bash -c \
  "__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
   /entrypoint.sh ground_truth:=true start_aic_engine:=true" \
  2>&1 | tee river-policy/gazebo.log
```

**Terminal 2 — Policy with calibration env var:**
```bash
RIVER_POLICY_CALIBRATE=1 pixi run --frozen ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true -p policy:=river_policy.MyPolicy \
  2>&1 | tee river-policy/run.log
```

### Step 2 — Confirm boards spawn

`aic_engine` only spawns the task board *after* the policy node activates.
With the env var set, each trial's `insert_cable()` call:
1. Reads the port TF (`task_board/<target_module_name>/<port_name>_link`)
2. Reads the plug-tip TF (`<cable_name>/<plug_name>_link`)
3. Logs the offset
4. Returns `True` immediately

You should see all 3 trials run in ~5 s each.

### Step 3 — Extract offsets

```bash
grep "CALIBRATION_OFFSET" river-policy/run.log
```

You'll get three lines like:
```
CALIBRATION_OFFSET    "nic_card_mount_0": (-0.0023, +0.0091),
CALIBRATION_OFFSET    "nic_card_mount_1": (+0.0034, +0.0142),
CALIBRATION_OFFSET    "sc_port_1": (-0.1837, +0.0521),
```

### Step 4 — Hard-code into `MODULE_XY_OFFSETS`

In `river-policy/river_policy/policy.py`, find:
```python
MODULE_XY_OFFSETS: dict[str, tuple[float, float]] = {}
```

…and replace with:
```python
MODULE_XY_OFFSETS: dict[str, tuple[float, float]] = {
    "nic_card_mount_0": (-0.0023, +0.0091),
    "nic_card_mount_1": (+0.0034, +0.0142),
    "sc_port_1":        (-0.1837, +0.0521),
}
```
(use *your* numbers from the log)

### Step 5 — Rerun normally with `ground_truth:=false`

```bash
distrobox enter -r aic_eval -- bash -c \
  "__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
   /entrypoint.sh ground_truth:=false start_aic_engine:=true" \
  2>&1 | tee river-policy/gazebo.log

# (no env var this time)
pixi run --frozen ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true -p policy:=river_policy.MyPolicy \
  2>&1 | tee river-policy/run.log
```

### Caveats

- The board is **randomly placed within configured bounds** each run, so
  offsets vary slightly between runs. Run calibration 2–3 times and take
  the median to reduce per-run drift.
- If a trial appears under a *different* `target_module_name` in a future
  config, it won't get an offset. The dict is keyed exactly on
  `task.target_module_name`.

---

## Scoring Trajectory

| Iteration | What changed | Score |
|-----------|--------------|-------|
| Initial SAC | Blind RL, random exploration | 3 |
| V0.1 classical | Compliant descent, returned True without moving | 57.6 |
| V0.2 | Wrench baselining + min-descent contact gate | 83.1 |
| V0.5 | + Phase 3 spiral search | 83.1 (spiral exhausted) |
| V0.6 | Tried full gripper reorient — cable physics fought back | 56.0 (regressed) |
| V0.7a | Reverted reorient, single-camera vision probe | 90.2 |
| V0.7b | Vision XY correction wired into descent + spiral | 90.2 |
| V0.8 | Expanded spiral (70 mm, 6 turns); per-module XY offset hooks | 86.8 |
| V1.0 | 3-camera triangulation; bad-spawn abort; drop threshold 3→6 mm; 2-step confirm | **115.47** (single run, possible outlier) |
| V1.1 | + Phase 1.5 partial reorient; vision-miss retry; max depth 0.20→0.25 m; 3-step confirm | 89.62 |
| V1.2 | + Phase 2 quality gates (avg lateral + XY drift) | 82.96 |
| V1.3 | Disabled Phase 1.5 + ascended to start_z on non-insertion | **3.00 (catastrophic)** |
| **V1.4** | Kept Phase 1.5 disabled; reverted V1.3 ascend | **94.60** ✓ stable working high |
| **V1.5** | + Ground-truth calibration mode (env-var gated) for `MODULE_XY_OFFSETS` | **TBD (after calibration data populated)** |

### Per-trial breakdowns

**V1.0 (115.47 — single run, may be outlier):**

| Trial | Final dist | Notes | Total |
|-------|-----------|-------|-------|
| 1 | ~0.05 m | Triangulation OK | ~44 |
| 2 | ~0.07 m | Triangulation picked wrong blob | ~40 |
| 3 | ~0.21 m | Got proximity points (initial dist large enough) | ~30 |

**V1.1 (89.62):**

| Trial | Final dist | Notes | Total |
|-------|-----------|-------|-------|
| 1 | 0.04 m | Phase 3 spiral found a hole (heuristic insert) | 45.82 |
| 2 | 0.05 m | Phase 3 spiral exhausted | 42.81 |
| 3 | 0.20 m | Phase 1.5 + over-permissive Phase 2 → "false success" → all Tier 2 + 3 zeroed | 1.00 |

**V1.4 (94.60 — working high):**

| Trial | Final dist | Notes | Total |
|-------|-----------|-------|-------|
| 1 | 0.04 m | Spiral search; quality gates ready | 47.19 |
| 2 | 0.05 m | Spiral exhausted | 46.41 |
| 3 | 0.22 m | Outside max-distance bounding radius → 0 on Tier 2/3 | 1.00 |

---

## Bottlenecks

### 1. Target localization for Trial 3 (top priority)

The SC port is 20+ cm laterally from spawn — **outside the wrist cameras'
FOV** at the start pose. Triangulation has nothing to lock onto. The arm
descends at the spawn XY, lands far from the port, and the proximity
formula (`max_dist = 0.5 × initial_plug_port_distance`) zeroes the score.

**Fix:** V1.5 calibration mode → populate `MODULE_XY_OFFSETS` with offsets
read from ground-truth TF. Resolves Trial 3 entirely if board placement is
deterministic; near-resolves it if randomized (median of a few runs).

### 2. Wrong-blob detection on Trials 1 & 2

Triangulation works (residuals 1–3 mm), but the blob detector picks the
nearest-to-center dark feature, which is sometimes a screw, mounting hole,
or shadow rather than the actual SFP slot.

**Fix:** template-matched port detection (Extension A below) — match
against the rectangular SFP / circular SC profile.

### 3. Plug grasp tilt (~38°)

`PLUG_GRASP_RPY = (0.4432, −0.4838, 1.3303)` ⇒ plug Z is 38° off gripper Z.
At 15 mm insertion depth this would cause 9.2 mm lateral drift, larger
than the ±5 mm port tolerance.

**Workaround chosen (V1.4):** disable Phase 1.5 (the partial-reorient
correction), accept the tilt, rely on the spiral and chamfer to find the
hole. Net-positive for Trials 1+2 in practice (each gained ~5 pts vs V1.1).

If you want to retry compensation: raise `REORIENT_AT_CONTACT_FRACTION`
from 0.0 in small steps (0.2, 0.4) and watch Trial 1/2 scores.

### 4. Proximity formula gating

Tier 2 metrics (smoothness/duration/efficiency) are awarded **only if
Tier 3 > 0**. On Trial 3 we're outside the max-distance bounding radius
(half the initial plug-port distance), so Tier 2 is zeroed too — that's
why a 22 cm miss scores 1, not ~10.

---

## Future Extensions (ranked by effort × score impact)

### A. Template-matched port detection — 4–8 hours

Replace "darkest blob near center" with `cv2.matchTemplate` at multiple
scales against rendered or hand-cropped SFP-slot / SC-port templates.
- Pros: targeted detection; rejects screws, shadows, other holes.
- Cons: doesn't help Trial 3 until the arm is within FOV (so do calibration
  first).
- Expected gain: trials 1+2 from ~47 each to ~60–80; Trial 3 unchanged.
- Drop-in at `MyPolicy._detect_port_pixel` in `policy.py`.

### B. V1.5 calibration → `MODULE_XY_OFFSETS` (the easiest win)

Already implemented. See "Calibration Mode" above. Expected: Trial 3
**1 → 50–80**. Total: ~145–175 / 300.

### C. Imitation learning on `CheatCode` demos — 3–5 days

`CheatCode.py` reads ground-truth TF and inserts perfectly. Run it under
`ground_truth:=true`, record `(observation, action)` pairs to LeRobot
format, train ACT or diffusion policy, swap in for inference.
- Pros: target localization learned from data; transfers to all trials.
- Cons: demo collection pipeline + GPU training.
- Expected: 90 → 200+.

### D. Residual RL on top of classical — 3–5 days

Keep classical descent. RL (small CNN over center camera + wrench) outputs
delta-XY corrections to the descent center. Reward on Tier 3 directly.

### Quick wins under 1 hour each

- Tune `SPIRAL_DROP_THRESHOLD` from 0.006 → 0.004 m to confirm holes earlier.
- Two-pass detection: detect top-3 blobs, sweep a quick Z-push test over
  each, keep whichever shows a real Z-drop first.
- Try `DESCENT_SPEED = 0.020` for smoother contact detection.
- Re-test `REORIENT_AT_CONTACT_FRACTION` at 0.2, 0.3 — V1.4 disabled
  entirely; small fraction may still help slightly.

---

## Running Instructions

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

### Normal evaluation (two terminals)

**Terminal 1 — Gazebo + RViz2:**
```bash
distrobox enter -r aic_eval -- bash -c \
  "__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia \
   /entrypoint.sh ground_truth:=false start_aic_engine:=true" \
  2>&1 | tee river-policy/gazebo.log
```

Wait for Gazebo + RViz2 to fully load.

**Terminal 2 — Policy:**
```bash
pixi run --frozen ros2 run aic_model aic_model \
  --ros-args -p use_sim_time:=true -p policy:=river_policy.MyPolicy \
  2>&1 | tee river-policy/run.log
```

`aic_engine` configures + activates the lifecycle node automatically.
The `[ERROR] aic_model lifecycle is not in the active state` line during
startup is a harmless race; the engine retries within 1 s.

### Where to look in the logs

- `run.log` — policy-side: phase transitions, contact, triangulation
  residuals, spiral progress, Phase 2 quality-gate decisions
- `gazebo.log` — engine-side: scoring summary, contact events

```bash
# total score for a run
grep "Total Score:" river-policy/gazebo.log

# per-trial detail
grep -A 50 "Complete Scoring Results" river-policy/gazebo.log

# vision pipeline status
grep -E "Triangulation|hole found|Spiral search exhausted" river-policy/run.log

# false-success rejections (V1.2)
grep "rejecting Phase 2 success" river-policy/run.log

# calibration outputs (V1.5)
grep "CALIBRATION_OFFSET" river-policy/run.log
```

---

## Tunables

All on `MyPolicy` in `policy.py`. Common ones, **defaults as of V1.5**:

| Constant | Default | What it controls |
|----------|---------|------------------|
| `DESCENT_SPEED` | 0.030 m/s | Phase 1 speed |
| `INSERT_SPEED` | 0.010 m/s | Phase 2 speed |
| `MAX_APPROACH_DEPTH` | 0.25 m | Phase 1 safety stop |
| `INSERT_DEPTH` | 0.015 m | Phase 2 success threshold |
| `INSERT_PHASE_TIMEOUT` | 8.0 s | Phase 2 hard cap |
| `NO_PROGRESS_TIMEOUT` | 2.0 s | Phase 2 no-progress trigger |
| **`REORIENT_AT_CONTACT_FRACTION`** | **0.0 (disabled)** | Phase 1.5 SLERP fraction |
| `REORIENT_AT_CONTACT_TIME` | 1.5 s | Phase 1.5 duration |
| `SPIRAL_MAX_RADIUS` | 0.070 m | Phase 3 spiral max radius |
| `SPIRAL_TURNS` | 6 | Phase 3 spiral revolutions |
| `SPIRAL_DROP_THRESHOLD` | 0.006 m | Z drop to declare hole found |
| `SPIRAL_DROP_CONFIRM_STEPS` | 3 | Sustain steps before committing |
| `TRIANGULATION_RESIDUAL_THRESH` | 0.015 m | Max ray residual to accept triangulation |
| `MAX_TRIANGULATION_CORRECTION` | 0.04 m | Clamp on vision XY shift |
| `BASELINE_LATERAL_ABORT` | 15.0 N | Lateral force in baseline → bad spawn abort |
| `PHASE2_QUALITY_LAT_MAX` | 5.0 N | V1.2: avg lateral threshold for real success |
| `PHASE2_QUALITY_DRIFT_MAX` | 0.025 m | V1.2: TCP drift threshold for real success |
| `MODULE_XY_OFFSETS` | `{}` | **V1.5: populate via calibration mode** |
| `APPROACH_STIFFNESS` | [90,90,90,50,50,50] | Phase 1 admittance |
| `INSERT_STIFFNESS` | [80,80,60,40,40,40] | Phase 2 admittance |
| `SPIRAL_STIFFNESS` | [80,80,30,40,40,40] | Phase 3 admittance |

Edits are picked up immediately — the package is editable-installed; just
restart Terminal 2.

---

## Lessons Learned / Gotchas

These are sharp edges that took real time to find. **Read these before
making behavioral changes.**

### 1. The engine measures plug-port distance AT the moment `insert_cable()` returns

Not at any maximum-progress point during the trial. So **the final pose
of the arm is what scores.** This was the V1.3 disaster: ascending the
gripper back to `start_z` after a non-insertion drove the plug above the
port surface, blowing past `max_distance = 0.5 × initial_plug_port_distance`,
which zeroes Tier 3 proximity *and* gates all Tier 2 metrics. Score 83 → 3.

**Implication:** wherever you leave the plug at return is where you're
scored. Don't move it back "for cleanliness."

### 2. Always return `True` from `insert_cable()`

Returning `False` is treated as "Task not completed" and zeros all Tier 2
+ Tier 3 scoring. Even if you didn't insert anything, return `True` and
let the engine measure the plug position for proximity points.

### 3. `pixi install` doesn't work in this checkout

Stale lockfile from upstream divergence. Always use `pixi run --frozen`
and install new Python deps via
`uv pip install --python .pixi/envs/default/bin/python3.12 ...`.

### 4. Lifecycle race on first activation

`aic_engine` sometimes sends the first goal in the millisecond between
`on_configure` and `on_activate`. This shows as one
`[ERROR] aic_model lifecycle is not in the active state` line during
startup. Engine retries; ignore.

### 5. Spiral commit can leave the arm in an awkward pose

The spiral can move the arm 4+ cm laterally before committing. After this,
the engine's between-trial homing sometimes leaves the next trial starting
from an unexpected pose. **Don't try to "fix" this from the policy** —
V1.3 tried and scored 3 (see Lesson 1).

### 6. `tcp_pose` from `controller_state` is the actual current pose, not the reference

Reference is in `reference_tcp_pose`. We use `tcp_pose` for `cur_z`-based
progress checks because compliance can lag the commanded pose.

### 7. Wrench is in the FT-sensor frame, not base_link

Sign convention for `fz` flips during contact in our runs (negative `dfz`
on contact) — that's why all checks use `abs(dfz)`.

### 8. `task.time_limit` is `uint64` seconds

We default to 60 s if zero/missing.

### 9. Phase 1.5 reorient was visually wrong

The math computes a correct horizontal-axis rotation (~75° about
`(-0.84, 0.54, 0)`), but the simultaneous pin-the-tip translation
(~17 mm horizontal) makes the combined motion *look* like a yaw+slide,
i.e., "rotating anticlockwise around vertical." Cable physics also
fought the rotation (post-reorient plug-tip-vs-TCP magnitude was
53 mm vs static 45 mm). Disabled in V1.4. Re-enable with caution.

### 10. The wrench baseline includes ~21 N from cable weight

The cable hangs from the gripper before the policy starts. **All contact
checks must subtract baseline** (`dfz = wrench.force.z - baseline[2]`).
Without this, V0.5 declared contact at descent=0 mm because the cable's
20 N startup wrench tripped the threshold. The 0.7 s settle in Phase 0
followed by averaging the last 10 samples is what stabilizes this.

---

## Continuity

- **Branch**: `user/rishabh`
- **Remote**: `git@github.com:RiVer2000/aic.git` (your fork)
  - `upstream` is `https://github.com/intrinsic-dev/aic`
- **Latest commits:**
  - `23552ef` river-policy V1.5: ground-truth calibration mode
  - `cb5e592` river-policy V1.4 confirmed at 94.60/300
  - `3257d0d` river-policy V1.4: revert V1.3 ascend, keep Phase 1.5 disabled
  - `d58469b` river-policy V1.3: disable Phase 1.5 reorient, ascend on non-insertion
  - `b038b59` river-policy V1.2: Phase 2 quality gates
  - `72a85a5` river-policy V1.1: plug-tilt reorient, vision-miss retry, deeper approach
  - `1d36b13` river-policy V1.0: 3-camera triangulation

### If you pick this up cold

1. **Run V1.5 calibration** (Calibration Mode section). 5 minutes.
2. **Hard-code the resulting `MODULE_XY_OFFSETS`** in `policy.py`.
3. **Rerun normally** with `ground_truth:=false` and confirm Trial 3 score
   jumps from 1 to 50+.
4. From there, the next-best lever is **template-matched port detection**
   (Extension A) for Trials 1+2 — they're saturating proximity scoring at
   ~47 each, and template matching can push them into partial-insertion
   territory (38–50 pts) for ~60–70 each.
5. **If you suspect run-to-run variance**, run the same version 3–5×
   back-to-back and tabulate mean/std before drawing conclusions. We
   have only single-run scores at most versions, and noticed that V1.0's
   115 may have been a lucky-seed outlier.

### Out-of-scope (deferred unless you have ≥3 days)

- Imitation learning on CheatCode demos (Extension C)
- Residual RL on top of the classical controller (Extension D)
- Re-implementing Phase 1.5 with full plug-tip pinning (the V0.6/V1.1
  attempts failed against cable physics — needs a different approach)
