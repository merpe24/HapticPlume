/**
 * @file prf.cpp
 * @brief Implementation of Lam's parametric risk field. See prf.hpp for the contract.
 *
 * @author premmm
 * @date October 9, 2026
 */

#include "haptic_plume_haptics/prf.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace haptic_plume_haptics
{

namespace
{
constexpr double kPi = 3.14159265358979323846;
}  // namespace

double PrfConfig::r_pz() const
{
  return clamp.drone_radius + clamp.margin;
}

void PrfConfig::validate() const
{
  // d_min is the whole reason the field does not vanish at hover: with the velocity zero
  // both the braking and look-ahead terms go to zero, and d_min is all that is left
  // between the critical region and zero risk (risk R10).
  if (d_min <= 0.0) {
    throw std::invalid_argument("PrfConfig: d_min must be positive, else the field "
                                "collapses onto the clamp boundary at hover");
  }
  if (t_ahead < 0.0) {
    throw std::invalid_argument("PrfConfig: t_ahead must not be negative");
  }
  if (a_max <= 0.0) {
    throw std::invalid_argument("PrfConfig: a_max must be positive");
  }
  if (max_force < 0.0) {
    throw std::invalid_argument("PrfConfig: max_force must not be negative");
  }
  if (clamp.drone_radius < 0.0 || clamp.margin < 0.0) {
    throw std::invalid_argument("PrfConfig: clamp radii must not be negative");
  }
}

double p_cos2(double x)
{
  if (x <= 0.0) {
    return 1.0;
  }
  if (x >= 1.0) {
    return 0.0;
  }
  return 0.5 * (std::cos(kPi * x) + 1.0);
}

double braking_distance(double speed, double a_max)
{
  return speed * speed / (2.0 * a_max);
}

double look_ahead(double speed, double t_ahead)
{
  return speed * t_ahead;
}

Risk risk_at(
  const haptic_plume_drone::ObstacleField & field, const PrfConfig & cfg,
  const Eigen::Vector3d & p, const Eigen::Vector3d & v)
{
  const double speed = v.norm();
  const Eigen::Vector3d heading = speed > 0.0 ? Eigen::Vector3d(v / speed) :
    Eigen::Vector3d::Zero();
  const double d_brake = braking_distance(speed, cfg.a_max);
  const double d_look = look_ahead(speed, cfg.t_ahead);

  Risk out{0.0, Eigen::Vector3d::Zero(), std::string()};

  // One surface's contribution. `slack` is the distance left before the kinematic clamp
  // would start pushing, so slack <= 0 is risk 1 by construction and the D1 agreement is
  // structural rather than tuned. `normal` points away from the surface.
  // Ties keep the earlier surface, so the result is deterministic in segment order.
  const auto score =
    [&out, &heading, d_brake, d_look, &cfg](
    double slack, const Eigen::Vector3d & normal, const std::string & name) {
      // How much of the motion heads into this surface, in [0, 1]. Lam extends the
      // critical region and the outer boundary along the velocity; projecting onto the
      // surface direction is the same statement without segment-to-segment geometry.
      const double toward = std::max(0.0, -heading.dot(normal));
      const double d = slack - d_brake * toward;
      const double d_outer = cfg.d_min + d_look * toward;

      double value;
      if (d <= 0.0) {
        value = 1.0;
      } else if (d >= d_outer) {
        value = 0.0;
      } else {
        value = p_cos2(d / d_outer);
      }

      if (value > out.value) {
        out.value = value;
        out.direction = normal;
        out.surface = name;
      }
    };

  // 1. Pipes. The clamp requires drone_radius + margin from a capsule surface.
  for (const haptic_plume_drone::Capsule & c : field.segments) {
    const haptic_plume_drone::Clearance cl = haptic_plume_drone::clearance(c, p);
    score(cl.distance - cfg.r_pz(), cl.normal, c.name);
  }

  // 2. Floor, only when asked. At the fixed 1.0 m altitude it sits 0.85 m away, inside a
  //    d_min of 1.50 m, so scoring it would render ~2 N upward everywhere all trial from a
  //    surface the drone cannot descend toward. See the header for the full reasoning.
  if (cfg.include_floor) {
    score(
      p.z() - (field.ground_z + cfg.clamp.min_altitude), Eigen::Vector3d::UnitZ(), "ground");
  }

  // 3. Room walls. NOTE the asymmetry with the pipes: clamp_to_free_space insets the bounds
  //    by drone_radius alone, with no margin, so the risk must saturate there too or the
  //    field and the clamp would disagree about the wall (D1).
  for (int axis = 0; axis < 2; ++axis) {
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    normal[axis] = 1.0;
    score(p[axis] - (cfg.clamp.area_min[axis] + cfg.clamp.drone_radius), normal, "wall");
    score(
      (cfg.clamp.area_max[axis] - cfg.clamp.drone_radius) - p[axis], -normal, "wall");
  }

  return out;
}

Eigen::Vector3d prf_force(
  const haptic_plume_drone::ObstacleField & field, const PrfConfig & cfg,
  const Eigen::Vector3d & p, const Eigen::Vector3d & v)
{
  const Risk risk = risk_at(field, cfg, p, v);
  return risk.direction * (risk.value * cfg.max_force);
}

}  // namespace haptic_plume_haptics
