/**
 * @file well.cpp
 * @brief Implementation of the confidence-scaled belief well. See well.hpp for the contract.
 *
 * @author premmm
 * @date October 9, 2026
 */

#include "haptic_plume_haptics/well.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace haptic_plume_haptics
{

namespace
{
// Same reason as the drone's clamp_max_passes: main_run and riser share the endpoint
// (10, 5, 1), so one pass can push a target out of one capsule and into the other.
constexpr int kTargetClampPasses = 4;
}  // namespace

void WellConfig::validate() const
{
  if (width_min <= 0.0 || width_max <= 0.0) {
    throw std::invalid_argument("WellConfig: widths must be positive");
  }
  if (width_min > width_max) {
    throw std::invalid_argument("WellConfig: width_min must not exceed width_max, or "
                                "confidence would widen the well instead of narrowing it");
  }
  if (depth_min < 0.0) {
    throw std::invalid_argument("WellConfig: depth_min must not be negative");
  }
  if (depth_min > depth_max) {
    throw std::invalid_argument("WellConfig: depth_min must not exceed depth_max");
  }
  if (target_margin < 0.0) {
    throw std::invalid_argument("WellConfig: target_margin must not be negative");
  }
  if (speed_threshold < 0.0) {
    throw std::invalid_argument("WellConfig: speed_threshold must not be negative");
  }
}

double well_width(const WellConfig & cfg, double confidence)
{
  const double k = std::clamp(confidence, 0.0, 1.0);
  return cfg.width_max + k * (cfg.width_min - cfg.width_max);
}

double well_depth(const WellConfig & cfg, double confidence)
{
  const double k = std::clamp(confidence, 0.0, 1.0);
  // Exactly 0 is reserved for "render nothing" (DIVERGED, or GuidanceTarget.valid false);
  // the continuous band is (0, 1]. The step from 0 to depth_min is deliberate and is NOT
  // smoothed here -- the composer owns per-cue fade envelopes (D2, risk R5), and a cue
  // library that fades on its own would fade twice.
  if (k <= 0.0) {
    return 0.0;
  }
  return cfg.depth_min + k * (cfg.depth_max - cfg.depth_min);
}

Eigen::Vector3d clamp_target_outward(
  const haptic_plume_drone::ObstacleField & field, double standoff,
  const Eigen::Vector3d & target)
{
  Eigen::Vector3d out = target;

  for (int pass = 0; pass < kTargetClampPasses; ++pass) {
    bool moved = false;
    for (const haptic_plume_drone::Capsule & c : field.segments) {
      const haptic_plume_drone::Clearance cl = haptic_plume_drone::clearance(c, out);
      if (cl.distance < standoff) {
        // Along the outward normal only, by exactly the shortfall -- so a leak on the
        // near face of a pipe clamps to the near face, not around to the far side.
        out += cl.normal * (standoff - cl.distance);
        moved = true;
      }
    }
    if (!moved) {
      break;
    }
  }
  return out;
}

Eigen::Vector3d well_force(
  const haptic_plume_drone::ObstacleField & field, const PrfConfig & prf_cfg,
  const WellConfig & cfg, const Eigen::Vector3d & p, const Eigen::Vector3d & v,
  const Eigen::Vector3d & target, double confidence, double risk)
{
  const double depth = well_depth(cfg, confidence);
  if (depth <= 0.0) {
    return Eigen::Vector3d::Zero();
  }

  const Eigen::Vector3d safe =
    clamp_target_outward(field, prf_cfg.r_pz() + cfg.target_margin, target);
  const Eigen::Vector3d offset = safe - p;
  const double range = offset.norm();
  if (range <= 0.0) {
    return Eigen::Vector3d::Zero();
  }
  const Eigen::Vector3d bearing = offset / range;

  // Heading error. Undefined while parked, and a parked operator should still be able to
  // feel which way the belief lies, so the full depth is rendered along the bearing.
  double psi = 0.0;
  const double speed = v.norm();
  if (speed > cfg.speed_threshold) {
    psi = std::acos(std::clamp(v.dot(bearing) / speed, -1.0, 1.0));
  }

  const double width = well_width(cfg, confidence);
  const double shape = std::exp(-(psi * psi) / (2.0 * width * width));
  const double yield_to_prf = std::clamp(1.0 - risk, 0.0, 1.0);

  return bearing * (depth * shape * yield_to_prf);
}

}  // namespace haptic_plume_haptics
