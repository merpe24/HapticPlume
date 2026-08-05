/**
 * @file pose_mirror.cpp
 * @brief Implementation of the PoseMirror gz-sim 8 system plugin.
 *
 * See pose_mirror.hpp for the interface and the D1 rationale.
 *
 * @author premmm
 * @date August 5, 2026
 */

#include "haptic_plume_gazebo/pose_mirror.hpp"

#include <gz/common/Console.hh>
#include <gz/msgs/Utility.hh>
#include <gz/plugin/Register.hh>

namespace haptic_plume
{

void PoseMirror::Configure(
  const gz::sim::Entity & _entity,
  const std::shared_ptr<const sdf::Element> & _sdf,
  gz::sim::EntityComponentManager & _ecm,
  gz::sim::EventManager & /*_eventMgr*/)
{
  // Bind to the entity the plugin was attached to. Because the plugin is declared in
  // the drone's URDF it always lands on the drone model, so there is no lookup by
  // name and no "model has not spawned yet" race to handle.
  this->model_ = gz::sim::Model(_entity);

  if (!this->model_.Valid(_ecm)) {
    gzerr << "PoseMirror must be attached to a model entity. Plugin disabled.\n";
    return;
  }

  // Read the gz topic carrying the commanded pose
  if (_sdf->HasElement("topic")) {
    this->topic_ = _sdf->Get<std::string>("topic");
  }

  // Subscribe. The callback runs on a gz-transport thread, never the sim thread.
  if (!this->node_.Subscribe(this->topic_, &PoseMirror::OnOdometry, this)) {
    gzerr << "PoseMirror failed to subscribe to [" << this->topic_ << "]\n";
    return;
  }

  gzmsg << "PoseMirror mirroring [" << this->topic_ << "] onto model ["
        << this->model_.Name(_ecm) << "]\n";
}

void PoseMirror::PreUpdate(
  const gz::sim::UpdateInfo & _info,
  gz::sim::EntityComponentManager & _ecm)
{
  // A paused simulation must not be teleported: the operator may be dragging the
  // model around in the GUI, and fighting that makes the world feel broken.
  if (_info.paused) {
    return;
  }

  if (!this->model_.Valid(_ecm)) {
    return;
  }

  gz::math::Pose3d pose;
  {
    std::lock_guard<std::mutex> lock(this->pose_mutex_);

    // Before the first message arrives, leave the spawn pose alone. Teleporting to
    // the identity pose here would park the drone at the world origin for however
    // long the bridge takes to connect, which looks exactly like a broken spawn.
    if (!this->have_pose_) {
      return;
    }
    pose = this->latest_pose_;
  }

  // Teleport. Gazebo never integrates the drone (D1).
  this->model_.SetWorldPoseCmd(_ecm, pose);
}

void PoseMirror::OnOdometry(const gz::msgs::Odometry & _msg)
{
  std::lock_guard<std::mutex> lock(this->pose_mutex_);
  this->latest_pose_ = gz::msgs::Convert(_msg.pose());
  this->have_pose_ = true;
}

}  // namespace haptic_plume

// Register the plugin with the gz-plugin loader
GZ_ADD_PLUGIN(
  haptic_plume::PoseMirror,
  gz::sim::System,
  haptic_plume::PoseMirror::ISystemConfigure,
  haptic_plume::PoseMirror::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(haptic_plume::PoseMirror, "haptic_plume::PoseMirror")
