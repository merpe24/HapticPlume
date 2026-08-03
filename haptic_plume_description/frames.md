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
           ├── lidar_link
           ├── prop_front_left_link
           ├── prop_front_right_link
           ├── prop_back_left_link
           ├── prop_back_right_link
           └── camera_link
                └── camera_optical_frame
```

`/tf`, `/tf_static`, and `/robot_description` are **global**, not under `/hp`. The `/hp`
namespace covers this project's data topics; TF is global by ROS convention, and
namespacing it would force a `/tf` remap in every consumer, RViz and `tf2_tools` included.

| Frame | Meaning | Published by |
|---|---|---|
| `map` | Scenario/world frame. Fixed. All Phase A math lives here. | `static_transform_publisher` in `haptic_plume_drone/launch/drone.launch.py` |
| `odom` | Odometry origin. Identity to `map` — a kinematic drone has zero drift, and `drone_kinematics_node` *is* ground truth, so there is nothing to correct. | same launch file (identity, no arguments) |
| `base_link` | Drone body origin, at the geometric centre of the airframe. Yaw is cosmetic — see §6.3. | `drone_kinematics_node` (B3) |
| `gas_sensor_link` | Point at which concentration is sampled. Coincident with `base_link` — see §5. | `robot_state_publisher` (B2) |
| `lidar_link` | 3D lidar (Unitree). **Description-only** — nothing subscribes; it exists for airframe fidelity and the Phase F hardware path. Its gz sensor is off by default (`enable_gz_sensor:=false`) to protect the RTF budget. Origin sits on the spin axis with the mounting foot at its own `z = 0`; the vendor CAD datum is not centred on the body, so `mesh_xyz_offset` corrects for it. | `robot_state_publisher` (B2) |
| `prop_*_link` | Four propellers, visual only, fixed joints (no `/joint_states`). | `robot_state_publisher` (B2) |
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
| Airframe | Holybro X500 V2, 500 mm wheelbase | product spec |
| Drone collision radius | 0.377 m (250 mm motor + 127 mm prop) | derived, `drone_prop.urdf.xacro` |

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

Since the setting is indoor (decision, 2026-08-03), `theta` is better described as the
**prevailing airflow direction** — HVAC- and doorway-driven, not outdoor wind. The maths and
the sign convention below are unchanged; only the physical story is. That the free-field
Gaussian plume is retained indoors is a disclosed limitation, listed in `CLAUDE.md`.

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

### 6.1 `/hp/cmd_vel` is a WORLD-frame velocity (decided B3, 2026-08-04)

`geometry_msgs/TwistStamped` conventionally carries a **body-frame** velocity. **Ours does
not.** `/hp/cmd_vel` is interpreted in the `odom` frame, and publishers should stamp it
`odom`.

Reason: the force allocation above is defined in world semantics, and the Falcon is a
3-DOF device with **no yaw axis** — the operator cannot rotate the body, so a body-frame
command would be unusable. `drone_kinematics_node` logs a one-shot `RCLCPP_WARN` if it
receives a twist stamped `base_link`, so the convention is self-enforcing rather than
merely documented.

### 6.2 `/hp/odom` twist is also WORLD-frame — a deliberate deviation

`nav_msgs/Odometry` documents its `twist` as being expressed in `child_frame_id`, i.e.
the body frame. `drone_kinematics_node` publishes it in the **world** frame instead.

Reason: the drone's yaw is *synthesised* from the velocity direction (§6.3), so rotating
the velocity into the body frame would collapse it to approximately `(|v|, 0, 0)` — the
same scalar twice, expressed relative to a heading that exists only to aim a camera. The
world-frame twist carries strictly more information and matches how every consumer in
this system thinks.

> **Consumers must NOT rotate `/hp/odom.twist` into the world frame — it is already
> there.** In particular Phase C's PRF look-ahead `d_ahead = |v|·t_ahead` uses it as-is.

Disclose this in the paper alongside the other deviations listed in `CLAUDE.md`.

### 6.3 Yaw is cosmetic

`base_link`'s yaw follows the horizontal velocity direction (held below
`yaw_speed_threshold` so hover does not spin, slew-limited by `yaw_rate_limit`). Roll and
pitch are always zero — the kinematic model has no attitude dynamics.

It exists **only so the FPV camera points where the drone is flying**, which matters
because the camera is the visual channel of the experiment and is never condition-gated
(D6). Nothing in the gas, estimation, or plume chain reads orientation: `gas_sensor_link`
is coincident with `base_link` and the Gaussian plume is isotropic about its axis. Do not
build anything load-bearing on this heading.

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
