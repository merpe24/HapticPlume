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
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>

namespace haptic_plume
{

void PoseMirror::Configure(
  const gz::sim::Entity & /*_entity*/,
  const std::shared_ptr<const sdf::Element> & _sdf,
  gz::sim::EntityComponentManager & /*_ecm*/,
  gz::sim::EventManager & /*_eventMgr*/)
{
  // _entity is the world. The drone does not exist yet, so binding happens lazily in
  // PreUpdate — see the header for why this is a world plugin.
  if (_sdf->HasElement("topic")) {
    this->topic_ = _sdf->Get<std::string>("topic");
  }

  if (_sdf->HasElement("model_name")) {
    this->model_name_ = _sdf->Get<std::string>("model_name");
  }

  // Subscribe. The callback runs on a gz-transport thread, never the sim thread.
  if (!this->node_.Subscribe(this->topic_, &PoseMirror::OnOdometry, this)) {
    gzerr << "PoseMirror failed to subscribe to [" << this->topic_ << "]\n";
    return;
  }

  gzmsg << "PoseMirror mirroring [" << this->topic_ << "] onto model ["
        << this->model_name_ << "]\n";
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
    // The world loads before `create` spawns the drone, so this lookup cannot happen
    // in Configure and must be retried each step. Re-binding also means a Phase E
    // trial reset that respawns the model is handled for free.
    const gz::sim::Entity entity = _ecm.EntityByComponents(
      gz::sim::components::Name(this->model_name_),
      gz::sim::components::Model());

    if (entity == gz::sim::kNullEntity) {
      return;
    }

    this->model_ = gz::sim::Model(entity);
    gzmsg << "PoseMirror bound to model [" << this->model_name_
          << "] entity [" << entity << "]\n";
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
