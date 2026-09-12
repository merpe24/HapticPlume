# HapticPlume

Haptic-guided gas leak localization for a teleoperated inspection drone — ROS 2 Jazzy, Gazebo Harmonic.

An operator flies an inspection drone through an industrial hall looking for a gas leak, but gas is invisible and the onboard sensor is slow and noisy. HapticPlume estimates where the leak is, and how certain that estimate is, with a Bayesian particle filter over a Gaussian plume model, and is built to render that belief back to the operator's hand as force through a Novint Falcon controller. The research question is whether *confidence*, rendered as force, helps a human find a leak faster than seeing the same information on a screen; the system is designed so that visual and haptic guidance can be switched per trial for a user study.

The simulation is deliberately split so that each part can be tested on its own. A C++ kinematic node at 100 Hz owns the drone's motion, integrating world-frame velocity commands with acceleration limits and sliding the drone along the pipes instead of stopping dead at them; Gazebo is only a renderer, driven by a small world plugin that mirrors that pose and produces the FPV camera image. A Python gas server holds the ground-truth plume and models the sensor on top of it: first-order lag with a slow asymmetric recovery, electronic noise and turbulent fluctuation. Everything above the ROS layer — the plume maths, the scenario generator, the filter — is plain, seeded, ROS-free Python, so a trial replays bit-exactly from a recorded bag and hundreds of trials can be run headless.

The estimator compensates the sensor lag at the sensor's own 20 Hz rate, decimates to 5 Hz, and updates 1500 particles over the eight plume parameters (leak position, release rate, airflow direction and speed, dispersion). It publishes the belief mean, its spread, the effective sample size and a status, and declares the leak once the spread falls below 0.5 m. For the haptic target it uses the densest cluster of particles rather than their mean, which outliers drag around by tens of centimetres — enough to point a force cue at empty floor.

**Status.** The simulation, teleoperation and estimator all run, and the estimator is the part being worked on now. In headless evaluation against the live sensor rates it declares the leak in every 240-second trial, within 1.5 m of truth 80% of the time, with a median error of 0.71 m and a median time of 55 seconds; the first closed-loop run in ROS declared a leak at 30 seconds, 0.43 m from truth. Those figures are for scripted flight paths, and how the filter behaves when a *human* chooses the path is the open question: flying straight down the plume makes it converge confidently in the wrong place, and its own spread does not reveal that. The haptic cue library and the Falcon driver come after that gate is settled. One limitation is stated up front: the gas model is a steady free-field plume while the setting is indoors, so a CFD-based simulator is the planned upgrade.

<!-- Demo: a GIF of the drone, plume and particle cloud in RViz belongs here -->

```
gamepad ──► teleop_mapper (C++, 100 Hz, deadman)
                   │ /hp/cmd_vel
                   ▼
            drone_kinematics (C++, 100 Hz) ──► Gazebo (pose mirror, FPV camera)
                   │ /hp/odom
                   ▼
            gas_field_server (20 Hz, lag + noise)
                   │ /hp/gas/measurement
                   ▼
            plume_estimator ── lag compensation (20 Hz) ── particle filter (5 Hz)
                   │ /hp/belief/*
                   ▼
            RViz   ·   force composer and Falcon driver (planned)
```

## Running it

Requires Ubuntu 24.04, ROS 2 Jazzy and Gazebo Harmonic (`ros-jazzy-ros-gz`). Build from the workspace root, never from inside a package.

```bash
mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
git clone https://github.com/merpe24/HapticPlume.git haptic_plume
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-up-to haptic_plume
source install/setup.bash
```

```bash
# Simulation: world, drone, gas field, RViz, gamepad teleop
# (hold button 0 as deadman and fly with the right stick)
ros2 launch haptic_plume_bringup sim_teleop.launch.py

# Estimator, in a second terminal
ros2 launch haptic_plume_estimation estimator.launch.py
```

Without a gamepad, `teleop_twist_keyboard` works in holonomic mode:
`ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p stamped:=true -p frame_id:=map -r /cmd_vel:=/hp/cmd_vel`.

For a repeatable run without a human, launch with `use_teleop:=false` and fly the scripted check flight, which pins a scenario, resets the belief, flies up the plume and reports the belief against ground truth every five seconds:

```bash
python3 src/haptic_plume/haptic_plume_bringup/scripts/fly_plume.py 3 120   # seed 3, 120 s
```

Headless studies and tests:

```bash
python3 -m haptic_plume_experiment.eval_estimator --study suite --trials 10 --sources 1
python3 -m haptic_plume_experiment.eval_estimator --study non-informative  # effect of the flight path
colcon test --packages-select-regex haptic_plume && colcon test-result --verbose
```

## Layout

```
haptic_plume_core          plume model, seeded scenario generator (no ROS)
haptic_plume_gas_sim       ground-truth gas field, sensor lag and noise
haptic_plume_estimation    lag compensation, particle filter, consume, clustering, information gain
haptic_plume_drone         100 Hz kinematic drone, capsule obstacle library (C++)
haptic_plume_teleop        gamepad to world-frame velocity, YAML axis map (C++)
haptic_plume_description   airframe, sensors, and frames.md — the authority on frames and units
haptic_plume_gazebo        pipeline world, pose-mirror plugin, bridges
haptic_plume_interfaces    messages and services
haptic_plume_experiment    headless evaluation harness
haptic_plume_bringup       launch files, scripted check flight
haptic_plume_system_tests  headless smoke test of the full bring-up
haptic_plume_haptics       cue library, force composer, Falcon driver (planned)
```

Tests: 127 pytest cases, 16 gtest cases covering the collision geometry, and a launch test that boots the whole simulation headless and checks rates, transforms and clean shutdown.

## Notes

- **A bias that only appeared at the live sample rate.** The lag compensator picks between the sensor's rise and recovery constants from the sign of the last sample step. On a noisy signal noise makes that choice, and the asymmetric gains rectify it into a negative offset that grows with the sample rate. Every offline study had run at 5 Hz; at the live 20 Hz the offset was larger than the plume signal nine metres downwind, and the filter was effectively blind. Correcting it moved detection from 6% to 100% of trials and the median error from 7.65 m to 0.71 m.
- **An alternative filter was implemented and rejected on data.** A competing particle-filter method was run as a passive observer on the same 30 seeded worlds and the same flights: 40% success at 3.01 m against 73% at 0.63 m. What ruled it out was calibration — its belief spread barely correlated with its actual error, and this project renders confidence as force, so a filter whose confidence means nothing is unusable.
- **The obvious haptic cue was measured before it was built.** A pull toward the belief, switched on whenever the filter reports tracking, would be on from the first update and point more than 10° off for most of the time it was rendered. That result set the design: the cue starts wide and narrows as confidence grows, and it needs a real gate on when the belief is trustworthy.
- **Multi-leak trials were dropped for a measured reason.** Eight configurations failed, and the cause was that the particle spread was shrinking through resampling depletion rather than evidence, which meant the earlier multi-leak results had been resting on an artifact. Trials now use one leak, which also makes them short enough to be practical for a user study.

## References

- Goodell et al., *Mobile-robotic sensor for estimation and localization of multiple chemical-gas leaks: a find-and-consume infotaxis approach*, Robotics and Autonomous Systems.
- Lam et al., *Artificial force field for haptic feedback in UAV teleoperation*, IEEE Transactions on Systems, Man, and Cybernetics.
- Menelas et al., *Survey on haptic rendering of data sets: exploration of scalar and vector fields*.

Apache License 2.0.
