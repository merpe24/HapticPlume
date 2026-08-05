#!/bin/bash
# Launch the full Phase B simulation: world, drone, gas field, gamepad teleop.

cleanup() {
    echo "Shutting down HapticPlume simulation..."
    pkill -f "ros2 launch haptic_plume_bringup"
    sleep 3
    
    pkill -f "gz sim"
    pkill -f "ros_gz_bridge"
    pkill -f "parameter_bridge"
    pkill -f "image_bridge"
    pkill -f "rviz2"
    sleep 2
    echo "Done."
}

trap 'cleanup' SIGINT SIGTERM

# Every argument is spelled out even at its default: this script doubles as the
# record of a known-good configuration.
#
# The x/y/z spawn pose must match start_position in
# haptic_plume_drone/config/drone_params.yaml, and obstacles_file must describe the
# same pipes that world_file draws.
ros2 launch haptic_plume_bringup sim_teleop.launch.py \
    headless:=false \
    obstacles_file:="$(ros2 pkg prefix haptic_plume_description)/share/haptic_plume_description/config/obstacles.yaml" \
    prefix:="" \
    use_rviz:=true \
    use_teleop:=true \
    world_file:=pipeline.world \
    x:=1.0 \
    y:=1.0 \
    z:=1.0 &

wait
