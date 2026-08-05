/**
 * @file pose_mirror.hpp
 * @brief gz-sim 8 system plugin that drives a model's world pose from a gz topic.
 *
 * Decision D1 makes drone_kinematics_node the motion authority: it integrates
 * /hp/cmd_vel, clamps against the analytic obstacles in obstacles.yaml, and owns
 * odom -> base_link. Gazebo is a renderer and a camera host, never a physics
 * authority. This plugin is the seam that keeps that true — on every simulation step
 * it teleports the model onto the latest pose received from ROS, so Gazebo's own
 * integrator can never disagree with the node about where the drone is.
 *
 * The pose arrives as a gz.msgs.Odometry, bridged ROS_TO_GZ from /hp/odom by
 * ros_gz_bridge (see config/ros_gz_bridge.yaml). There is deliberately no new ROS
 * node and no new message type on this path.
 * 
 * Requires a world with gravity disabled. SetWorldPoseCmd teleports the body but does
 * not zero its velocity, so under gravity the teleport is undone by one step of
 * integration at an ever-growing speed. pipeline.world sets <gravity>0 0 0</gravity>
 *
 * SDF parameters:
 *     <topic> (string): gz topic carrying gz.msgs.Odometry. Default "/hp/odom"
 *
 * @author premmm
 * @date August 5, 2026
 */

#ifndef HAPTIC_PLUME_GAZEBO__POSE_MIRROR_HPP_
#define HAPTIC_PLUME_GAZEBO__POSE_MIRROR_HPP_

#include <memory>                       // std headers first
#include <mutex>
#include <string>

#include <gz/math/Pose3.hh>             // then gz headers
#include <gz/msgs/odometry.pb.h>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/transport/Node.hh>
#include <sdf/Element.hh>

namespace haptic_plume
{

/**
 * @brief Mirrors a ROS-authored pose onto a Gazebo model every simulation step.
 */
class PoseMirror
  : public gz::sim::System,
    public gz::sim::ISystemConfigure,
    public gz::sim::ISystemPreUpdate
{
public:
  PoseMirror() = default;
  ~PoseMirror() override = default;

  /**
   * @brief Bind to the model entity and subscribe to the pose topic.
   * @param _entity   Entity this plugin is attached to (the drone model)
   * @param _sdf      The plugin's own SDF element, source of <topic>
   * @param _ecm      Entity-component manager
   * @param _eventMgr Event manager (unused)
   */
  void Configure(
    const gz::sim::Entity & _entity,
    const std::shared_ptr<const sdf::Element> & _sdf,
    gz::sim::EntityComponentManager & _ecm,
    gz::sim::EventManager & _eventMgr) override;

  /**
   * @brief Teleport the model onto the most recently received pose.
   * @param _info Simulation timing and pause state
   * @param _ecm  Entity-component manager
   */
  void PreUpdate(
    const gz::sim::UpdateInfo & _info,
    gz::sim::EntityComponentManager & _ecm) override;

private:
  /**
   * @brief Store the latest pose. Runs on a gz-transport thread, not the sim thread.
   * @param _msg Odometry message whose pose field is the commanded world pose
   */
  void OnOdometry(const gz::msgs::Odometry & _msg);

  // Class member variables
  gz::sim::Model model_{gz::sim::kNullEntity};
  gz::transport::Node node_;
  std::string topic_{"/hp/odom"};       // overridable from SDF

  // Single-slot latest-message buffer. Written by the transport thread, read by the
  // sim thread — the same discipline falcon_node will use for its 1 kHz servo loop.
  std::mutex pose_mutex_;
  gz::math::Pose3d latest_pose_;
  bool have_pose_{false};
};

}  // namespace haptic_plume

#endif  // HAPTIC_PLUME_GAZEBO__POSE_MIRROR_HPP_
