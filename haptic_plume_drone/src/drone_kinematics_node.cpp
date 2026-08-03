/**
 * @file drone_kinematics_node.cpp
 * @brief Motion authority for the simulated drone (decision D1).
 *
 * Integrates velocity commands at a fixed rate, clamps the resulting position out of
 * the analytic world geometry with sliding, and publishes the only pose the rest of
 * the system believes. Gazebo renders the result; it never moves the drone. Because
 * this node IS ground truth, map -> odom is identity and no localization error can
 * contaminate the particle filter.
 *
 * /hp/cmd_vel is interpreted in the WORLD frame (odom), NOT the body frame. The
 * fixed-altitude force allocation is defined in world semantics and the Falcon has no
 * yaw axis, so body-frame commands would be unusable. See frames.md.
 *
 * Subscription Topics:
 *     /hp/cmd_vel (geometry_msgs/TwistStamped): World-frame velocity command [m/s]
 *
 * Publishing Topics:
 *     /hp/odom (nav_msgs/Odometry): Pose and twist, odom -> base_link
 *     /tf (tf2_msgs/TFMessage): odom -> base_link, republished every tick
 *     /hp/events/collision (haptic_plume_interfaces/TaskEvent): Rising edge of contact
 *
 * Services:
 *     /hp/reset_drone (std_srvs/Trigger): Return to start_position, zero velocity
 *
 * Parameters:
 *     update_rate (double): Tick and publish rate [Hz]
 *     max_speed (double): Speed cap applied to the command [m/s]
 *     max_accel (double): Acceleration limit [m/s^2]
 *     cmd_timeout (double): Command staleness before commanding zero [s]
 *     fixed_altitude_enabled (bool): Ignore commanded z velocity and hold altitude
 *     fixed_altitude (double): Held altitude when enabled [m]
 *     drone_radius (double): Collision sphere radius [m]
 *     collision_margin (double): Extra standoff beyond bare contact [m]
 *     min_altitude (double): Floor clearance above ground_z [m]
 *     search_area_min/max (double[2]): Room bounds in x and y [m]
 *     start_position (double[3]): Spawn and reset position [m]
 *     yaw_follows_velocity (bool): Point the body along the velocity (camera only)
 *     yaw_speed_threshold (double): Below this speed the yaw is held [m/s]
 *     yaw_rate_limit (double): Yaw slew limit [rad/s]
 *     obstacles_file (string): Absolute path to obstacles.yaml
 *     odom_frame, base_frame (string): TF frame names
 *
 * @author premmm
 * @date August 4, 2026
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "geometry_msgs/msg/twist_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "tf2/LinearMath/Quaternion.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/transform_broadcaster.hpp"

#include "haptic_plume_drone/obstacle_field.hpp"
#include "haptic_plume_interfaces/msg/task_event.hpp"

using TaskEvent = haptic_plume_interfaces::msg::TaskEvent;
using TwistStamped = geometry_msgs::msg::TwistStamped;
using Trigger = std_srvs::srv::Trigger;

/**
 * @brief Kinematic drone: integrates velocity, clamps to free space, owns the pose.
 */
class DroneKinematics : public rclcpp::Node
{
public:
  /**
   * @brief Constructor: reads parameters, loads the world, wires up the interface.
   */
  DroneKinematics()
  : Node("drone_kinematics")
  {
    // Motion limits and timing
    update_rate_ = this->declare_parameter<double>("update_rate", 100.0);
    max_speed_ = this->declare_parameter<double>("max_speed", 1.5);
    max_accel_ = this->declare_parameter<double>("max_accel", 3.0);
    cmd_timeout_ = this->declare_parameter<double>("cmd_timeout", 0.3);

    // Fixed-altitude policy. The integrator below stays fully 3D; this is a policy
    // applied on top of it, so the 3D path stays exercised (locked decision).
    fixed_altitude_enabled_ = this->declare_parameter<bool>("fixed_altitude_enabled", true);
    fixed_altitude_ = this->declare_parameter<double>("fixed_altitude", 1.0);

    // Yaw is cosmetic: it exists so the FPV camera points where the drone is flying.
    // Nothing in the gas or estimation chain reads orientation.
    yaw_follows_velocity_ = this->declare_parameter<bool>("yaw_follows_velocity", true);
    yaw_speed_threshold_ = this->declare_parameter<double>("yaw_speed_threshold", 0.05);
    yaw_rate_limit_ = this->declare_parameter<double>("yaw_rate_limit", 2.0);

    // Frame names, parameterised so a prefix can be applied for multi-robot later.
    odom_frame_ = this->declare_parameter<std::string>("odom_frame", "odom");
    base_frame_ = this->declare_parameter<std::string>("base_frame", "base_link");

    // Collision geometry. drone_radius is the Holybro X500 V2 footprint:
    // 250 mm motor arm + 127 mm propeller.
    clamp_cfg_.drone_radius = this->declare_parameter<double>("drone_radius", 0.377);
    clamp_cfg_.margin = this->declare_parameter<double>("collision_margin", 0.05);
    clamp_cfg_.min_altitude = this->declare_parameter<double>("min_altitude", 0.15);
    clamp_cfg_.max_passes = this->declare_parameter<int>("clamp_max_passes", 4);

    const std::vector<double> area_min =
      this->declare_parameter<std::vector<double>>("search_area_min", {0.0, 0.0});
    const std::vector<double> area_max =
      this->declare_parameter<std::vector<double>>("search_area_max", {20.0, 20.0});
    const std::vector<double> start =
      this->declare_parameter<std::vector<double>>("start_position", {1.0, 1.0, 1.0});

    if (area_min.size() != 2 || area_max.size() != 2 || start.size() != 3) {
      throw std::runtime_error(
              "search_area_min/max must have 2 elements and start_position 3");
    }
    clamp_cfg_.area_min = Eigen::Vector2d(area_min[0], area_min[1]);
    clamp_cfg_.area_max = Eigen::Vector2d(area_max[0], area_max[1]);
    start_position_ = Eigen::Vector3d(start[0], start[1], start[2]);

    // Load the world. The same YAML feeds Phase C's risk field (D1), so a missing or
    // malformed file is fatal rather than something to limp along without.
    const std::string obstacles_file =
      this->declare_parameter<std::string>("obstacles_file", "");
    if (obstacles_file.empty()) {
      throw std::runtime_error("obstacles_file parameter is required");
    }
    field_ = haptic_plume_drone::ObstacleField::from_yaml(obstacles_file);
    RCLCPP_INFO(
      this->get_logger(), "Loaded %zu obstacle segments from %s",
      field_.segments.size(), obstacles_file.c_str());

    // Start at rest, at the configured spawn point.
    reset_state();

    // Best-effort keep-last-1 on the command: latest sample wins, and a best-effort
    // subscriber also accepts a reliable publisher, so no teleop source can create a
    // silent QoS mismatch.
    rclcpp::QoS cmd_qos(1);
    cmd_qos.best_effort();
    subscription_cmd_vel_ = this->create_subscription<TwistStamped>(
      "/hp/cmd_vel", cmd_qos,
      std::bind(&DroneKinematics::cmd_vel_callback, this, std::placeholders::_1));

    // Odometry stays reliable: the gas server samples concentration at this pose, so
    // a dropped pose is a wrong measurement.
    publisher_odom_ = this->create_publisher<nav_msgs::msg::Odometry>("/hp/odom", 10);
    publisher_events_ = this->create_publisher<TaskEvent>("/hp/events/collision", 10);

    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    service_reset_ = this->create_service<Trigger>(
      "/hp/reset_drone",
      std::bind(
        &DroneKinematics::reset_callback, this, std::placeholders::_1,
        std::placeholders::_2));

    // Fixed-rate tick. Everything integrates against this dt, never against wall-clock
    // deltas, so the motion is reproducible (D3: wall clock, no /clock bridge).
    dt_ = 1.0 / update_rate_;
    timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::duration<double>(dt_)),
      std::bind(&DroneKinematics::tick, this));

    RCLCPP_INFO(this->get_logger(), "drone_kinematics up at %.1f Hz", update_rate_);
  }

private:
  // --- methods defined in the next part -------------------------------------
  void cmd_vel_callback(const TwistStamped::SharedPtr msg);
  void tick();
  void publish_state();
  void emit_collision_events(const std::vector<std::string> & contacts);
  void reset_state();
  void reset_callback(
    const std::shared_ptr<Trigger::Request> request,
    std::shared_ptr<Trigger::Response> response);

  // --- world ----------------------------------------------------------------
  haptic_plume_drone::ObstacleField field_;
  haptic_plume_drone::ClampConfig clamp_cfg_;

  // --- state ----------------------------------------------------------------
  Eigen::Vector3d position_;
  Eigen::Vector3d velocity_;
  Eigen::Vector3d start_position_;
  double yaw_{0.0};

  // --- latest command -------------------------------------------------------
  Eigen::Vector3d cmd_velocity_{Eigen::Vector3d::Zero()};
  rclcpp::Time last_cmd_time_;
  bool cmd_received_{false};
  bool warned_body_frame_{false};

  // --- contact bookkeeping for rising-edge events ---------------------------
  std::vector<std::string> prev_contacts_;

  // --- parameters -----------------------------------------------------------
  double update_rate_;
  double dt_;
  double max_speed_;
  double max_accel_;
  double cmd_timeout_;
  bool fixed_altitude_enabled_;
  double fixed_altitude_;
  bool yaw_follows_velocity_;
  double yaw_speed_threshold_;
  double yaw_rate_limit_;
  std::string odom_frame_;
  std::string base_frame_;

  // --- ROS handles ----------------------------------------------------------
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Subscription<TwistStamped>::SharedPtr subscription_cmd_vel_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr publisher_odom_;
  rclcpp::Publisher<TaskEvent>::SharedPtr publisher_events_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Service<Trigger>::SharedPtr service_reset_;
};

void DroneKinematics::cmd_vel_callback(const TwistStamped::SharedPtr msg)
{
  // /hp/cmd_vel is world-frame by project decision. A body-frame stamp means the
  // publisher misunderstood the contract; say so once rather than silently obeying.
  if (!warned_body_frame_ && msg->header.frame_id == base_frame_) {
    RCLCPP_WARN(
      this->get_logger(),
      "cmd_vel stamped '%s': /hp/cmd_vel is interpreted in the WORLD frame ('%s'). "
      "See frames.md.", base_frame_.c_str(), odom_frame_.c_str());
    warned_body_frame_ = true;
  }

  cmd_velocity_ = Eigen::Vector3d(
    msg->twist.linear.x, msg->twist.linear.y, msg->twist.linear.z);

  // Arrival time, NOT header.stamp. The question this watchdog answers is "is the
  // publisher still alive", and arrival time answers it without trusting a clock we
  // do not own.
  last_cmd_time_ = this->now();
  cmd_received_ = true;
}

void DroneKinematics::tick()
{
  // --- 1. Staleness gate ----------------------------------------------------
  // A teleop node that dies must not leave the drone flying. Note the short-circuit:
  // last_cmd_time_ is only read once cmd_received_ is true, so we never subtract an
  // uninitialised Time (which would throw over mismatched clock sources).
  Eigen::Vector3d target = cmd_velocity_;
  if (!cmd_received_ || (this->now() - last_cmd_time_).seconds() > cmd_timeout_) {
    target.setZero();
  }

  // --- 2. Limits ------------------------------------------------------------
  // Cap the command first, then rate-limit toward it, so max_accel shapes the
  // response and max_speed bounds it, instead of the two fighting each other.
  const double speed = target.norm();
  if (speed > max_speed_) {
    target *= max_speed_ / speed;
  }

  Eigen::Vector3d dv = target - velocity_;
  const double dv_max = max_accel_ * dt_;
  if (dv.norm() > dv_max) {
    dv *= dv_max / dv.norm();
  }
  velocity_ += dv;

  // --- 3. Fixed-altitude policy ---------------------------------------------
  // The integrator below is fully 3D; this is a policy applied on top of it, so the
  // 3D path stays exercised for the eventual full-3D remap.
  if (fixed_altitude_enabled_) {
    velocity_.z() = 0.0;
  }

  // --- 4. Integrate ---------------------------------------------------------
  Eigen::Vector3d candidate = position_ + velocity_ * dt_;
  if (fixed_altitude_enabled_) {
    candidate.z() = fixed_altitude_;
  }

  // --- 5. Clamp out of solid geometry, preserving tangential motion ---------
  std::vector<std::string> contacts;
  const Eigen::Vector3d clamped =
    haptic_plume_drone::clamp_to_free_space(field_, clamp_cfg_, candidate, &contacts);

  // Remove only the velocity component driven into the surface. What survives is the
  // sliding component. Without this the published twist would claim motion into a
  // pipe while the pose never moves, and Phase C's risk field reads that velocity.
  const Eigen::Vector3d correction = clamped - candidate;
  if (correction.squaredNorm() > 0.0) {
    const Eigen::Vector3d n = correction.normalized();
    const double into_surface = velocity_.dot(n);
    if (into_surface < 0.0) {
      velocity_ -= into_surface * n;
    }
  }
  position_ = clamped;

  // --- 6. Yaw, publish, events ----------------------------------------------
  // Yaw is COSMETIC: it exists so the FPV camera looks where the drone is flying.
  // Nothing in the gas or estimation chain reads orientation.
  if (yaw_follows_velocity_) {
    const Eigen::Vector2d horizontal(velocity_.x(), velocity_.y());
    if (horizontal.norm() > yaw_speed_threshold_) {
      const double desired = std::atan2(horizontal.y(), horizontal.x());
      // std::remainder wraps to [-pi, pi], giving the shortest way round.
      double diff = std::remainder(desired - yaw_, 2.0 * M_PI);
      const double max_step = yaw_rate_limit_ * dt_;
      diff = std::clamp(diff, -max_step, max_step);
      yaw_ = std::remainder(yaw_ + diff, 2.0 * M_PI);
    }
  }

  publish_state();
  emit_collision_events(contacts);
}

void DroneKinematics::publish_state()
{
  const rclcpp::Time stamp = this->now();

  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, yaw_);

  auto odom = std::make_unique<nav_msgs::msg::Odometry>();
  odom->header.stamp = stamp;
  odom->header.frame_id = odom_frame_;
  odom->child_frame_id = base_frame_;
  odom->pose.pose.position.x = position_.x();
  odom->pose.pose.position.y = position_.y();
  odom->pose.pose.position.z = position_.z();
  odom->pose.pose.orientation = tf2::toMsg(q);
  // DEVIATION: nav_msgs/Odometry documents twist in child_frame_id (body frame); we
  // publish it in the world frame. Rationale in frames.md -- the whole control stack
  // is world-frame and yaw is derived from the velocity itself, so rotating into body
  // frame would yield a near-constant (|v|, 0, 0) built on a cosmetic heading.
  odom->twist.twist.linear.x = velocity_.x();
  odom->twist.twist.linear.y = velocity_.y();
  odom->twist.twist.linear.z = velocity_.z();
  publisher_odom_->publish(std::move(odom));

  geometry_msgs::msg::TransformStamped tf;
  tf.header.stamp = stamp;
  tf.header.frame_id = odom_frame_;
  tf.child_frame_id = base_frame_;
  tf.transform.translation.x = position_.x();
  tf.transform.translation.y = position_.y();
  tf.transform.translation.z = position_.z();
  tf.transform.rotation = tf2::toMsg(q);
  tf_broadcaster_->sendTransform(tf);
}

void DroneKinematics::emit_collision_events(const std::vector<std::string> & contacts)
{
  for (const std::string & name : contacts) {
    // Rising edge only: sliding along a pipe is one event, not 100 per second.
    if (std::find(prev_contacts_.begin(), prev_contacts_.end(), name) !=
      prev_contacts_.end())
    {
      continue;
    }

    auto event = std::make_unique<TaskEvent>();
    event->header.stamp = this->now();
    event->header.frame_id = odom_frame_;
    event->event_type = TaskEvent::COLLISION;
    event->position.x = position_.x();
    event->position.y = position_.y();
    event->position.z = position_.z();
    event->detail = name;
    publisher_events_->publish(std::move(event));

    RCLCPP_INFO(this->get_logger(), "Collision with '%s'", name.c_str());
  }
  prev_contacts_ = contacts;
}

void DroneKinematics::reset_state()
{
  position_ = start_position_;
  if (fixed_altitude_enabled_) {
    position_.z() = fixed_altitude_;
  }
  velocity_.setZero();
  cmd_velocity_.setZero();
  yaw_ = 0.0;
  cmd_received_ = false;
  prev_contacts_.clear();
}

void DroneKinematics::reset_callback(
  const std::shared_ptr<Trigger::Request> request,
  std::shared_ptr<Trigger::Response> response)
{
  (void)request;   // Trigger has no fields; Phase E replaces this with a posed reset
  reset_state();
  response->success = true;
  response->message = "drone reset to start position";
  RCLCPP_INFO(this->get_logger(), "Reset to start position");
}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<DroneKinematics>());
  } catch (const std::exception & e) {
    // The constructor throws on a missing or malformed obstacles.yaml. Catching it
    // here turns "terminate called after throwing an instance of..." into a sentence
    // that says what is wrong.
    RCLCPP_FATAL(rclcpp::get_logger("drone_kinematics"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
