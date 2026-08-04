/**
 * @file teleop_mapper_node.cpp
 * @brief Maps gamepad axes to world-frame velocity commands for the drone.
 *
 * Reads sensor_msgs/Joy and emits geometry_msgs/TwistStamped on /hp/cmd_vel at a
 * fixed rate. Every axis and button is looked up through the parameter file: not one
 * index is hardcoded. That is the whole reason this node exists rather than
 * teleop_twist_joy -- the Falcon in Phase D has device-y vertical and device-z
 * pointing at the operator, and the same YAML schema has to describe it.
 *
 * The command is published from a TIMER, not from the joy callback. An input device
 * delivers events at whatever rate it feels like (a keyboard, at the terminal's
 * 500 ms / 33 Hz repeat), which would make /hp/cmd_vel bursty and leave
 * drone_kinematics_node's 0.3 s staleness gate reasoning about the input device
 * rather than about whether teleop is alive. At a steady 100 Hz the gate means
 * exactly one thing: this node died.
 *
 * /hp/cmd_vel is WORLD-frame (frames.md 6.1), so there is no body-frame rotation
 * here and no yaw command at all -- the drone derives yaw from its own velocity.
 *
 * Subscription Topics:
 *     /joy (sensor_msgs/Joy): Raw device state from joy_node
 *
 * Publishing Topics:
 *     /hp/cmd_vel (geometry_msgs/TwistStamped): World-frame velocity command [m/s]
 *
 * Parameters:
 *     publish_rate (double): Fixed publish rate [Hz]
 *     deadman_button (int): Button index that must be HELD to command motion
 *     joy_timeout (double): No /joy for this long -> command zero [s]
 *     frame_id (string): Frame stamped on the command; world frame per frames.md
 *     axes.<name>.index (int): Joy axis index feeding this command
 *     axes.<name>.scale (double): Command at full deflection [m/s]
 *     axes.<name>.deadband (double): Ignore |axis| below this [-]
 *     axes.<name>.invert (bool): Negate the axis before scaling
 *
 * @author premmm
 * @date August 4, 2026
 */

#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include "geometry_msgs/msg/twist_stamped.hpp"
#include "geometry_msgs/msg/vector3.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joy.hpp"

/**
 * @brief One configured mapping from a joy axis to a velocity component.
 */
struct AxisSpec
{
  int index = -1;           ///< Joy axis index; negative disables the axis
  double scale = 1.0;       ///< Command at full deflection [m/s]
  double deadband = 0.05;   ///< Ignore |axis| below this [-]
  bool invert = false;      ///< Negate the axis before scaling
};

/**
 * @brief Deadman-gated mapper from device axes to /hp/cmd_vel.
 */
class TeleopMapper : public rclcpp::Node
{
public:
  /**
   * @brief Constructor: read the axis map, then start publishing on a fixed timer.
   */
  TeleopMapper()
  : Node("teleop_mapper")
  {
    // Timing and safety parameters
    publish_rate_ = this->declare_parameter<double>("publish_rate", 100.0);
    deadman_button_ = this->declare_parameter<int>("deadman_button", 0);
    joy_timeout_ = this->declare_parameter<double>("joy_timeout", 0.5);
    frame_id_ = this->declare_parameter<std::string>("frame_id", "map");

    if (publish_rate_ <= 0.0) {
      throw std::invalid_argument("publish_rate must be positive");
    }

    // The axis map. These defaults describe the gamepad measured in B4, but the
    // YAML is the authority: a different device changes the YAML and nothing else.
    linear_x_ = declare_axis("linear_x", 0, 1.0);
    linear_y_ = declare_axis("linear_y", 1, 1.0);
    linear_z_ = declare_axis("linear_z", 2, 0.5);

    // Create the publisher for world-frame velocity commands
    publisher_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
      "/hp/cmd_vel", 10);

    // Subscribe to the raw device state
    subscription_joy_ = this->create_subscription<sensor_msgs::msg::Joy>(
      "/joy", 10,
      std::bind(&TeleopMapper::joy_callback, this, std::placeholders::_1));

    // Publish on a fixed schedule, independent of the device's event rate
    const auto period = std::chrono::duration<double>(1.0 / publish_rate_);
    timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&TeleopMapper::timer_callback, this));

    RCLCPP_INFO(
      this->get_logger(), "teleop_mapper up at %.1f Hz, deadman button %d",
      publish_rate_, deadman_button_);
  }

  /**
   * @brief Command zero on the way out, so Ctrl-C never leaves the drone flying.
   */
  ~TeleopMapper()
  {
    command_ = geometry_msgs::msg::Vector3();
    publish_command();
  }

private:
  /**
   * @brief Declare one axis's four parameters and return the resulting spec.
   *
   * @param name Axis name as it appears under axes: in the YAML
   * @param default_index Joy axis index used when the YAML is silent
   * @param default_scale Command at full deflection [m/s]
   * @return The populated spec
   */
  AxisSpec declare_axis(const std::string & name, int default_index, double default_scale)
  {
    AxisSpec spec;
    spec.index = this->declare_parameter<int>("axes." + name + ".index", default_index);
    spec.scale = this->declare_parameter<double>("axes." + name + ".scale", default_scale);
    spec.deadband = this->declare_parameter<double>("axes." + name + ".deadband", 0.05);
    spec.invert = this->declare_parameter<bool>("axes." + name + ".invert", false);
    return spec;
  }

  /**
   * @brief Convert one joy axis into a velocity component [m/s].
   *
   * @param spec The configured mapping for this component
   * @param joy The message being read
   * @return The commanded velocity component [m/s], zero inside the deadband
   */
  double apply_axis(const AxisSpec & spec, const sensor_msgs::msg::Joy & joy)
  {
    if (spec.index < 0 || static_cast<size_t>(spec.index) >= joy.axes.size()) {
      return 0.0;
    }

    double value = static_cast<double>(joy.axes[spec.index]);
    if (std::abs(value) < spec.deadband) {
      return 0.0;
    }

    // Re-span the live range onto [0, 1] so the command leaves the deadband from
    // zero, instead of jumping to deadband * scale the instant the stick moves.
    value = (value - std::copysign(spec.deadband, value)) / (1.0 - spec.deadband);

    if (spec.invert) {
      value = -value;
    }
    return value * spec.scale;
  }

  /**
   * @brief Latch the newest device state, gated by the deadman button.
   *
   * @param msg Shared pointer to the incoming joy message
   */
  void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg)
  {
    joy_received_ = true;
    last_joy_time_ = this->now();

    // A device with fewer buttons than the map expects is almost always the wrong
    // controller plugged in. Silent zeros would be baffling to debug, so say so.
    if (deadman_button_ < 0 ||
      static_cast<size_t>(deadman_button_) >= msg->buttons.size())
    {
      RCLCPP_WARN_ONCE(
        this->get_logger(),
        "deadman button %d out of range for a device with %zu buttons -- refusing to move",
        deadman_button_, msg->buttons.size());
      command_ = geometry_msgs::msg::Vector3();
      return;
    }

    // Deadman: motion only while it is HELD. Releasing zeroes the command right
    // here, and the next timer tick puts that zero on the wire within 1/rate.
    if (msg->buttons[deadman_button_] == 0) {
      command_ = geometry_msgs::msg::Vector3();
      return;
    }

    command_.x = apply_axis(linear_x_, *msg);
    command_.y = apply_axis(linear_y_, *msg);
    command_.z = apply_axis(linear_z_, *msg);
  }

  /**
   * @brief Publish the current command at the fixed rate, checking device liveness.
   */
  void timer_callback()
  {
    // Losing the device (unplugged, joy_node died) must stop the drone, exactly
    // as releasing the deadman does.
    if (!joy_received_ || (this->now() - last_joy_time_).seconds() > joy_timeout_) {
      command_ = geometry_msgs::msg::Vector3();
    }
    publish_command();
  }

  /**
   * @brief Stamp and send the current command on /hp/cmd_vel.
   */
  void publish_command()
  {
    if (!rclcpp::ok()) {
      return;
    }
    auto msg = std::make_unique<geometry_msgs::msg::TwistStamped>();
    msg->header.stamp = this->now();
    msg->header.frame_id = frame_id_;
    msg->twist.linear = command_;
    publisher_->publish(std::move(msg));
  }

  // Class member variables
  double publish_rate_;
  int deadman_button_;
  double joy_timeout_;
  std::string frame_id_;

  AxisSpec linear_x_;
  AxisSpec linear_y_;
  AxisSpec linear_z_;

  bool joy_received_ = false;
  rclcpp::Time last_joy_time_;
  geometry_msgs::msg::Vector3 command_;

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr publisher_;
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr subscription_joy_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TeleopMapper>());
  rclcpp::shutdown();
  return 0;
}
