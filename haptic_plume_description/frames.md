# Frames, units, and conventions — HapticPlume

**This file is the authority.** 
 If code disagrees with what is written
here, the code is wrong — fix the code, or change this file first and deliberately.

Every number below is traceable to a line of Phase A source. Citations are given so the
claim can be re-checked rather than trusted.

:author: premmm
:date: July 31, 2026

---

## 1. TF tree

```
map
 └── odom                        (static identity)
      └── base_link
           ├── gas_sensor_link
           └── camera_link
                └── camera_optical_frame
```

| Frame | Meaning | Published by |
|---|---|---|
| `map` | Scenario/world frame. Fixed. All Phase A math lives here. | static transform in the launch file |
| `odom` | Odometry origin. Identity to `map` — a kinematic drone has zero drift. | static transform in the launch file |
| `base_link` | Drone body origin, at the geometric centre of the airframe. | `drone_kinematics_node` (B3) |
| `gas_sensor_link` | Point at which concentration is sampled. | `robot_state_publisher` (B2) |
| `camera_link` | FPV camera body frame. | `robot_state_publisher` (B2) |
| `camera_optical_frame` | Optical frame for the image. | `robot_state_publisher` (B2) |


**Orientation conventions (REP-103):**
- `base_link`, `gas_sensor_link`, `camera_link`: **x forward, y left, z up**.
- `camera_optical_frame`: **z forward, x right, y down**. This is the optical convention
  and it differs from its own parent by a fixed rotation.

Angles are radians everywhere in code. One documented exception, a paper deviation already
recorded in `CLAUDE.md`: Table 2's `sigma_theta = 0.33` is read as **degrees**, matching the
±30° domain printed on the line above it. As radians it would be 31% of the whole domain
per step, and no wind-direction information could survive.

---

## 2. World extent

| Quantity | Value | Source |
|---|---|---|
| Search area | **20 m × 20 m**, `x ∈ [0, 20]`, `y ∈ [0, 20]` | `SEARCH_AREA`, `scenario.py:33` |
| Source placement margin | 1.0 m — sources land inside a centred 18 × 18 m box | `SOURCE_MARGIN`, `scenario.py:34` |
| Minimum source separation | 2.0 m | `MIN_SOURCE_SEPARATION`, `scenario.py:35` |
| Source height `z_s` | 1.0 m | `SOURCE_HEIGHT`, `scenario.py:36` |
| Ground plane | `z = 0` | `config/obstacles.yaml` |

---

## 3. Units

| Quantity | Unit | Notes |
|---|---|---|
| Position, distance | m | |
| Time | s | Wall clock (D3 — `use_sim_time:=false`) |
| Angle | rad | |
| Velocity | m/s | |
| Concentration | kg/m³ | The whole chain: truth, measurement, compensated |
| Leak rate `Q` | kg/s | Range `[2.0e-2, 3.0e-2]`, `scenario.py:38` |
| Wind speed `v` | m/s | Range `[0.2, 0.8]`, `scenario.py:39` |
| Diffusion `d_y`, `d_z` | m²/s | `[2.0e-2, 5.0e-2]` and `[3.0e-4, 7.5e-4]`, `scenario.py:40-41` |
| Force | N | Budget: PRF ≤ 5, well ≤ 2, heaviness ≤ 2, total ≤ 8 (R6) |

**Concentration magnitudes are small — this is the R7 trap.** With `Q ~ 2.5e-2 kg/s` the
readings of interest are order `1e-2 … 1e-1 kg/m³`, sensor noise is `sigma_noise = 2.0e-3`,
and the particle-filter likelihood scale is `PROJECT_Q = 0.1 kg/m³`
(`particle_filter.py:85`). Anything that renders or thresholds concentration must use the
D5 log scaling `k·log(1 + c/c₀)` with globally fixed constants. Per-scenario normalisation
is an experimental confound and is forbidden.

---

## 4. Wind direction — sign convention

**`theta` is the direction the wind blows TOWARD** — the direction the gas travels.

Authority: `plume_model.world_to_plume` (`plume_model.py:90`) computes

```
x_p = cos(theta) * dx + sin(theta) * dy
```

so the downwind unit vector in the `map` frame is `(cos theta, sin theta, 0)`.
`concentration_plume_frame` returns exactly zero for `x_p <= 0`, so there is never any gas
upwind of a source. With `THETA_RANGE = (-pi/6, +pi/6)` (`scenario.py:37`), every plume
trails roughly toward **+x**.

### Acceptance check (R4)

Run this in RViz after B5 and after any change to the plume math or the marker publisher:

1. Publish the wind arrow along `(cos theta, sin theta, 0)` from the source position.
2. Publish the gas field markers.
3. **The markers must extend from the source in the same direction the arrow points**,
   and there must be zero concentration on the opposite side.

If the plume trails *into* the arrow, a sign is flipped. Do not proceed past this check.

---

## 5. Operating altitude

**Fixed altitude = 1.0 m.**

Justification: `random_scenario` sets `start_position[2] = SOURCE_HEIGHT = 1.0`
(`scenario.py:184`), and the Phase A trial loop's `_step_towards` clips only x and y
(`eval_estimator.py:281-282`) — z is never modified. The Phase A agent therefore flew at
exactly source height for the entire 240 s of every trial in the validated suite.

> **Changing this value silently invalidates the Phase A results as a baseline.** At
> `z ≠ z_s` the vertical Gaussian term is no longer at its maximum, every concentration
> reading drops, and the 73%-correct-count / 2–3% localization-error numbers no longer
> describe the system. If the altitude must change, the Phase A suite has to be re-run at
> the new altitude before any comparison is made.

`drone_kinematics_node` keeps its integrator fully 3D internally (locked decision: "all
math 3D internally, fixed-alt first") and enforces the altitude with a `fixed_altitude`
parameter that ignores commanded z velocity. This is a policy at the top of the node, not
an assumption baked into the math.

**Consequence for world geometry:** any obstacle intended to be collidable must intersect
`z = 1.0`. A pipe whose top surface sits below 1.0 m is invisible to a fixed-altitude
drone. See `config/obstacles.yaml`.

---

## 6. Fixed-altitude force allocation

World semantics, per the locked decision in `CLAUDE.md`:

| Channel | Rendered in |
|---|---|
| Planar velocity command, gravity wells, PRF repulsion | horizontal (world x, y) |
| Heaviness ∝ concentration | vertical |

---

## 7. Novint Falcon device axes

Carried forward from `CLAUDE.md` so the frame authority holds it, not just the plan:

- The Falcon is a 3-DOF delta with roughly a 10 cm cubic workspace.
- **Device-y is vertical. Device-z points at the operator.** This does not match the world
  frame and must never be assumed to.
- Heaviness (vertical in world semantics, §6) therefore maps to **device-y** — and it does
  so through the axis-map YAML in `haptic_plume_teleop` / `haptic_plume_haptics`.
- **Never hardcode an axis index or sign in node source.** The axis map is config. This is
  the whole reason `teleop_mapper_node` exists rather than using stock `teleop_twist_joy`.

Because the device workspace is ~10 cm and the world is ~20 m, wells render as
**bearing-biasing forces** (Crossan navigation scheme), not positional wells. Device
position is a rate command, not a position command.

---

## 8. Known gaps

- **Sources are not attached to the pipeline.** `random_scenario` (`scenario.py:172`)
  places leaks by rejection sampling anywhere in the 18 × 18 m inner box, with no
  knowledge of where the pipe is. Physically a leak belongs *on* a pipeline. Deferred to
  Phase E, where hand-authored scenario YAMLs replace random placement; the Phase A
  statistics are all against random placement and stay valid on their own terms.
- **`obstacles.yaml` and `pipeline.world` are two hand-maintained copies of one geometry.**
  B6 adds a test asserting they agree. Until that test exists, edit both together.
