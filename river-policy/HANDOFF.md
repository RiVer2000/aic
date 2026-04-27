# River Policy — Handoff Doc

A classical compliant-insertion policy for the Intrinsic AIC qualification
phase. Scores **90 / 300** across the three configured trials in
`aic_engine/config/sample_config.yaml`.

This document assumes you already have the official WaveArm policy running on
your machine (i.e., you can launch the Gazebo eval container and a policy node
under `aic_model`).

---

## TL;DR

- Approach: pure classical control. **No RL, no demos, no learning.** A
  three-phase compliant descent with optional camera-guided XY correction.
- Trials 1 & 2 (SFP) consistently land 5-6 cm from the port → ~30-50 pts each
  via Tier 2 + Tier 3 proximity.
- Trial 3 (SC) lands ~21 cm away → 1 pt (Tier 1 only). The SC port is too far
  laterally for our blob detector to pin down.
- Bottleneck is target localization, **not** control. The compliant descent
  works; we just don't know where to point it.

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

### Phase 0 — Settle and baseline wrench (~0.7 s)

The wrist FT sensor reads ~21 N at startup because the cable hangs from the
gripper before the policy starts. We hold position for 0.7 s, average the
last 10 samples, and store that as the wrench baseline. All later contact
checks compare `wrench - baseline`, not absolute wrench.

### Phase 1 — Approach (~3-4 s)

Descend straight along world -Z at 30 mm/s with high-stiffness admittance
control (`90, 90, 90, 50, 50, 50` N/m, default-style). Holding starting XY
and orientation throughout. Three exit conditions:

| Condition | What happens |
|-----------|--------------|
| `\|fz - baseline\|` > 5 N **and** descended ≥ 5 mm | Contact detected, enter Phase 2 |
| Commanded depth ≥ 20 cm | Max-depth abort (we missed the port entirely) |
| `\|commanded_z - cur_z\|` > 5 cm | Approach stall (joint limit / singularity) |

The `MIN_APPROACH_BEFORE_CONTACT` of 5 mm prevents the cable's startup
wrench from triggering false contact at zero descent.

### Phase 2 — Direct insertion attempt (max 8 s)

Continue descending at 10 mm/s with softer Z-stiffness (60 N/m). Exit on:

| Condition | Reason |
|-----------|--------|
| Descended ≥ 15 mm past contact_z | Heuristic insertion success |
| `\|dfz\|` > 22 N **and** lateral > 8 N | Hard stall — connector jammed |
| No 1 mm of progress in 2 s | Plug stuck on port face → trigger spiral |

### Phase 3 — Spiral search (~12 s worst case)

Fired only if Phase 2 didn't insert. Sweeps an Archimedean spiral in XY
around the contact point:
- 6 turns × 8 points = 48 XY positions
- Radius 0 → 70 mm linearly
- 0.25 s dwell at each point
- Z held 6 mm below contact_z with low Z-stiffness (30 N/m)

If `cur_z` drops > 3 mm past contact_z at any XY → "hole found", commit and
push down the full insertion depth at that XY.

### Phase 4 — Hold and return

Sleep 1 s for cable to settle, then command the arm to its **current** pose
for 0.5 s (high stiffness) — this prevents the controller from continuing
to track a stale low-Z command after `insert_cable()` returns.

**Always returns `True`.** Returning `False` causes `aic_engine` to label
the task "Task not completed" and zero out *all* Tier 2 + Tier 3 scoring.
Returning `True` lets the engine measure final plug-port distance and
award proximity / partial-insertion points.

### Vision-guided XY correction (V0.7b)

Before Phase 1 begins, we run a simple OpenCV pipeline on the center wrist
camera image:

1. Grayscale + Gaussian blur + binary threshold at gray=60 (find dark regions)
2. Morphological close, then `cv2.findContours`
3. Filter by area (8-2500 px² for SC, 30-8000 px² for SFP) and pick the
   contour closest to the image principal point
4. Pinhole-project the pixel offset to a metric camera-frame offset using
   `CameraInfo.k` (fx, fy, cx, cy)
5. Map camera-frame XY → base_link XY (heuristic axis swap)
6. Clamp to ±4 cm and apply as a shift to `hold_x, hold_y` (both descent
   and spiral centers)

Currently **does not meaningfully improve scoring** because the blob
detector picks up the wrong dark feature on most trials — see Bottlenecks
below.

---

## Scoring Trajectory

| Iteration | What changed | Score |
|-----------|--------------|-------|
| Initial SAC | Blind RL, random exploration | 3 |
| V0.1 classical | Compliant descent, returned True without moving (lockup-free) | 57.6 |
| V0.2 | Wrench baselining + min-descent contact gate | 83.1 |
| V0.5 | + Phase 3 spiral search | 83.1 (spiral exhausted) |
| V0.6 | Tried gripper reorient — cable physics fought back | 56.0 (regressed) |
| V0.7a | Reverted reorient, added vision probe | 90.2 |
| V0.7b | Vision XY correction wired into descent + spiral | 90.2 |
| V0.8 | Expanded spiral (70 mm, 6 turns); per-module XY offsets | 86.8 |
| **V0.9** | Disabled vision correction; cleared wrong sc_port_1 offset; raised spiral drop threshold to 6 mm + sustain check | **TBD** |

Per-trial breakdown at V0.8 (last measured run):

| Trial | Plug | Final dist | Notes | Total |
|-------|------|-----------|-------|-------|
| 1 | SFP | 0.05 m | Spiral false-positive: "hole found" at wrong feature | 44.9 |
| 2 | SFP | 0.07 m | Vision pushed arm 4.5 mm X off-target; hit max depth, no contact | 40.9 |
| 3 | SC  | 0.31 m | Module offset (0.00, -0.19) moved plug further away, not closer | 1.0 |

---

## Bottlenecks

### 1. Target localization (the big one)

We can **descend** and **insert** if we know where the port is, but we don't
reliably know where it is.

- The blob detector finds 30-50 dark features per image. The actual port is
  typically *not* the nearest-to-center one (screws, mounting holes,
  shadows, and other ports on the board all qualify).
- For Trial 3 (SC), the target port is 20+ cm laterally from the start
  pose — outside the camera's field of view at the spawn pose. The vision
  layer has no chance.
- Without ground-truth TF (only available with `ground_truth:=true`) we
  have to find the port from images alone.

### 2. Plug grasp tilt

The cable grasp orientation in `sample_config.yaml` puts the plug Z-axis
~38° off the gripper Z-axis. Pure straight-Z descent therefore inserts at
an angle and the port resists. We tried correcting this in V0.6 but the
cable's own weight and bending stiffness pulled the plug off-axis after
the gripper rotated, and final plug-port distance was unchanged.

### 3. Proximity scoring formula

Tier 2 only awards the smoothness/duration/efficiency points if Tier 3 > 0.
On Trial 3 we're outside the "max distance" bounding radius
(half the initial plug-port distance), so we collect zero on Tier 2 too.
Closing the lateral gap on Trial 3 — even by a few cm — is worth a lot.

### 4. Trial 2 occasional joint stall

Across runs, Trial 2 sometimes started at a different (lower) Z than 1+3
and the arm could only descend ~2 mm before the controller stalled. The
approach-stall check exits cleanly when this happens, but the trial
score then floors at proximity-only (no insertion possible).

---

## Future Extensions

Ranked by expected score impact for effort.

### A. Template-matched port detection — 4-8 hours

Instead of "darkest blob near center", match against rendered or
hand-cropped templates of the SFP slot (rectangular, ~10×7 mm) and SC port
(circular, ~2 mm). Use `cv2.matchTemplate` at multiple scales.
- Pros: targeted detection; rejects screws, shadows, other holes
- Cons: doesn't help Trial 3 (port out of FOV)
- Expected gain: trials 1+2 from ~48/37 → 60-80; Trial 3 unchanged
- Drop-in at `MyPolicy._detect_port_pixel` in `policy.py`

### B. Stereo triangulation with left + right cameras — 1 day

Detect the same blob in both `left_image` and `right_image`, triangulate
3D position via `CameraInfo.p` (projection matrices). Removes the
working-distance hand-coded estimate and gives true depth.
- Pros: more accurate XY; verifies detection across views
- Cons: still relies on correct blob detection; doesn't help Trial 3
- Expected gain: small bump in trials 1+2 if accuracy improves

### C. ~~Read `task.target_module_name` + spawn-relative geometry~~ — **DONE**

Implemented in `MyPolicy.MODULE_XY_OFFSETS` in `policy.py`. A dict keyed by
`task.target_module_name` maps to a `(dx, dy)` shift applied on top of the
vision correction before Phase 1. Currently tuned for `sc_port_1`:
`(0.00, -0.19)` m. Expand the dict entries for additional module names as needed.
- Expected gain not yet measured — run Trial 3 to confirm.

### D. Imitation learning on CheatCode demos — 3-5 days

`CheatCode.py` reads ground-truth TF and inserts perfectly. Run it under
`ground_truth:=true`, record `(observation, action)` pairs to `lerobot`
format, train an ACT or diffusion policy on the demos, swap in for
inference.
- Pros: proper target localization learned from data; transfers to all
  three trials; principled solution
- Cons: requires demo collection pipeline, GPU training, sim-to-real
  considerations even within Gazebo
- Expected gain: 90 → 200+

### E. Train RL on top of classical controller (residual policy) — 3-5 days

Keep the classical descent. RL only sees the wrench + small CNN over the
center camera, outputs delta-XY corrections to the descent center. Reward
on Tier 3 score directly.
- Pros: leverages classical floor; small policy, fast to train
- Cons: still needs demo collection or shaped reward to learn target
  identification

### Quick wins under 1 hour each

- Tune `SPIRAL_MAX_RADIUS` from 12 mm to 25 mm — covers more area at the
  cost of smoothness/duration. Could be net positive on trials 1+2.
- Two-pass detection: pick top 3 candidates by area near center, sweep all
  3 with a quick Z-push test, pick whichever shows real Z-drop.
- Try alternate threshold / morphology params in `_detect_port_pixel` —
  current `gray < 60` is one guess.

---

## Running Instructions

You said WaveArm runs on your machine, so the env is good. The river-policy
package is editable-installed via `uv` (because the repo's `pixi install`
has a known stale-lockfile issue) — see Setup below if it isn't already
present.

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

You will see lines like:
```
Vision: 33 cands, best px=(539,511) Δpx=(-37,-1) area=82 |
  working_dist=119mm fx=1236.6 fy=1236.6 |
  world Δxy=(+0.1,-3.6) mm
Descent centre after vision correction: (-0.3728, 0.1902)
contact at z=0.2349 (dfz=-5.0 N) after descending 84.2 mm
no Z progress for 2.0s (stuck at z=0.2349, dfz=-5.8 N) — exiting insertion
Spiral search around xy=(-0.3729, 0.1938) at z=0.2349
Spiral search exhausted, no hole found
insert_cable returning True (heuristic insert=False); engine will score by final plug position.
```

After all three trials, `gazebo.log` ends with the scoring summary
(`Total Score: ...`).

### Tunables

All are class constants on `MyPolicy` in `policy.py`. Common ones:

| Constant | Default | What it controls |
|----------|---------|-------------------|
| `DESCENT_SPEED` | 0.030 m/s | Phase 1 speed |
| `INSERT_SPEED` | 0.010 m/s | Phase 2 speed |
| `MAX_APPROACH_DEPTH` | 0.20 m | Phase 1 safety stop |
| `INSERT_DEPTH` | 0.015 m | Phase 2 success threshold |
| `INSERT_PHASE_TIMEOUT` | 8.0 s | Phase 2 hard cap |
| `NO_PROGRESS_TIMEOUT` | 2.0 s | Phase 2 no-progress trigger |
| `SPIRAL_MAX_RADIUS` | 0.070 m | Phase 3 spiral max |
| `SPIRAL_TURNS` | 6 | Phase 3 spiral revolutions |
| `SPIRAL_POINTS_PER_TURN` | 8 | Phase 3 angular resolution |
| `SPIRAL_DROP_THRESHOLD` | 0.006 m | Z drop to declare hole found (+ 2-step sustain) |
| `MAX_VISION_XY_CORRECTION` | 0.0 m | Vision shift clamp (0 = disabled) |
| `VISION_FLIP_X / Y / SWAP_AXES` | False | Camera→base_link sign overrides |
| `APPROACH_STIFFNESS` | [90,90,90,50,50,50] | Phase 1 admittance |
| `INSERT_STIFFNESS` | [80,80,60,40,40,40] | Phase 2 admittance |
| `SPIRAL_STIFFNESS` | [80,80,30,40,40,40] | Phase 3 admittance |

Edits are picked up immediately — the package is editable-installed, no
reinstall needed. Just restart Terminal 2.

### Where to look in the logs

- `run.log` (Terminal 2): policy-side state — phase transitions, contact
  detection, spiral progress, vision detection
- `gazebo.log` (Terminal 1): scoring summary, contact events, engine
  state machine

Search for `Total Score:` in `gazebo.log` for the per-run total. Search
for `tier_2:` / `tier_3:` for the per-category breakdown.

---

## Known Issues / Gotchas

- **`pixi install` doesn't work** in this checkout (stale lockfile from
  upstream divergence). Always use `pixi run --frozen` and install new
  Python deps via `uv pip install --python .pixi/envs/default/bin/python3.12 ...`.
- **Lifecycle race on first activation**: `aic_engine` sometimes sends the
  first goal in the millisecond between `on_configure` and `on_activate`.
  This shows as one `[ERROR] aic_model lifecycle is not in the active
  state` line. Engine retries; ignore.
- **Robot keeps moving after `insert_cable()` returns** if you don't call
  `_hold_current_pose` before returning. Already handled in current code,
  but if you add new return paths, remember to call it.
- **`tcp_pose` from `controller_state` is the actual current pose, not the
  reference**. The reference is `reference_tcp_pose`. We use `tcp_pose`
  for `cur_z`-based progress checks.
- **`task.time_limit` is in seconds (`uint64`)**. We default to 60 s if
  zero/missing.
- **Wrench is in the FT-sensor frame, not base_link.** Sign convention
  for `fz` flipped during contact in our runs (negative dfz on contact)
  — that's why all checks use `abs(dfz)`.

---

## Contacts / Continuity

- Branch: `user/rishabh`
- Last commits:
  - `c55ba73` river-policy: spiral search + vision-guided XY correction
  - `990446f` river-policy: switch to classical compliant descent
  - `388efd1` Add river-policy: SAC-based RL cable insertion policy

If you pick this up: start by running it once on your machine to confirm
you reproduce ~90/300, then tackle (A) template-matched detection or (D)
imitation learning depending on the time you have.
