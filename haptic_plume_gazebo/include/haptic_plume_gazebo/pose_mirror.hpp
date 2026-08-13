/**
 * @file pose_mirror.hpp
 * @brief gz-sim 8 system plugin that drives a model's world pose from a gz topic.
 *
 * D1 makes drone_kinematics_node the motion authority and Gazebo a renderer only. This
 * plugin is the seam that keeps that true: every step it teleports the model onto the
 * latest pose bridged from /hp/odom (gz.msgs.Odometry, ROS_TO_GZ, ros_gz_bridge.yaml),
 * so Gazebo's integrator can never disagree about where the drone is. Requires gravity
 * disabled -- SetWorldPoseCmd moves the body without zeroing its velocity, so one step
 * of integration would undo each teleport at an ever-growing speed.
 *
 * Declared as a WORLD plugin in pipeline.world, not on the drone's URDF: it then loads
 * deterministically at world load, binds explicitly and logged, and rebinds for free
 * when a Phase E trial reset respawns the drone. The cost is a lookup by name that must
 * tolerate the model not existing yet, because the world loads before the spawn.
 *
 * SDF parameters:
 *     <topic> (string): gz topic carrying gz.msgs.Odometry. Default "/hp/odom"
 *     <model_name> (string): model to drive. Default "drone"; must match the -name
 *         passed to `create` in haptic_plume.gazebo.launch.py
 *
 * @author premmm
 * @date August 5, 2026
 */

#ifndef HAPTIC_PLUME_GAZEBO__POSE_MIRROR_HPP_
#define HAPTIC_PLUME_GAZEBO__POSE_MIRROR_HPP_

#include <memory>
#include <mutex>
#include <string>

#include <gz/math/Pose3.hh>
#include <gz/msgs/odometry.pb.h>
#include <gz/sim/Model.hh>
#include <gz/sim/System.hh>
#include <gz/transport/Node.hh>
#include <sdf/Element.hh>

namespace haptic_plume
{

/// Mirrors a ROS-authored pose onto a Gazebo model every simulation step.
class PoseMirror
  : public gz::sim::System,
  public gz::sim::ISystemConfigure,
  public gz::sim::ISystemPreUpdate
{
public:
  PoseMirror() = default;
  ~PoseMirror() override = default;

  /// Read <topic>/<model_name> from SDF and subscribe.
  /// _entity is the WORLD, not the drone.
  void Configure(
    const gz::sim::Entity & _entity,
    const std::shared_ptr<const sdf::Element> & _sdf,
    gz::sim::EntityComponentManager & _ecm,
    gz::sim::EventManager & _eventMgr) override;

  /// Teleport the model onto the latest received pose, binding the model lazily.
  void PreUpdate(
    const gz::sim::UpdateInfo & _info,
    gz::sim::EntityComponentManager & _ecm) override;

private:
  /// Store the latest pose. Runs on a gz-transport thread, not the sim thread.
  void OnOdometry(const gz::msgs::Odometry & _msg);

  // Class member variables
  gz::sim::Model model_{gz::sim::kNullEntity};
  gz::transport::Node node_;
  std::string topic_{"/hp/odom"};       // overridable from SDF
  std::string model_name_{"drone"};     // overridable from SDF

  // Single-slot latest-message buffer. Written by the transport thread, read by the
  // sim thread — the same discipline falcon_node will use for its 1 kHz servo loop.
  std::mutex pose_mutex_;
  gz::math::Pose3d latest_pose_;
  bool have_pose_{false};
};

}  // namespace haptic_plume

#endif  // HAPTIC_PLUME_GAZEBO__POSE_MIRROR_HPP_
