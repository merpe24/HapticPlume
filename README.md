# HapticPlume

Haptic-guided gas leak localization for a teleoperated drone · ROS 2 Jazzy · Gazebo Harmonic

An inspection drone searches an industrial hall for a gas leak. A particle filter estimates where the leak is, and how confident that estimate is, from a slow and noisy gas sensor. The goal is to render that belief to the operator as force feedback through a Novint Falcon haptic controller, so the operator can feel which way to fly.

- **Simulation:** C++ kinematic drone, simulated gas plume and sensor, Gazebo rendering, gamepad teleop
- **Estimation:** sensor-lag compensation and a particle filter over a Gaussian plume model, based on MultiPLE (Goodell et al.)
- **Evaluation:** seeded, reproducible headless trials and unit tests for all algorithm code

**Status:** in active development. The simulation, teleoperation and estimator run in ROS 2, and the localization algorithm is being refined; in simulated trials it currently locates a single leak with a median error of about 0.7 m. Haptic rendering and the Falcon driver are next.

## Quick start

```bash
cd ~/ros2_ws/src && git clone https://github.com/merpe24/HapticPlume.git haptic_plume
cd ~/ros2_ws && rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-up-to haptic_plume && source install/setup.bash

ros2 launch haptic_plume_bringup sim_teleop.launch.py     # simulation + gamepad teleop
ros2 launch haptic_plume_estimation estimator.launch.py   # gas source estimator
```

Apache-2.0.
