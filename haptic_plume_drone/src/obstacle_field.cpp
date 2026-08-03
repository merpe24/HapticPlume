/**
 * @file obstacle_field.cpp
 * @brief Capsule/ground/wall geometry and the sliding position clamp.
 *
 * @author premmm
 * @date August 3, 2026
 */

#include "haptic_plume_drone/obstacle_field.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <yaml-cpp/yaml.h>

namespace haptic_plume_drone
{

namespace
{

/// Lengths below this are treated as zero. Guards the divisions below.
constexpr double kEps = 1e-9;

/// Read a 3-element YAML sequence, failing with a message that names the field.
Eigen::Vector3d parse_vec3(const YAML::Node & node, const std::string & what)
{
  if (!node || !node.IsSequence() || node.size() != 3) {
    throw std::runtime_error("obstacles.yaml: " + what + " must be a 3-element sequence");
  }
  return Eigen::Vector3d(node[0].as<double>(), node[1].as<double>(), node[2].as<double>());
}

/// Any unit vector perpendicular to u, for when there is no natural push direction.
Eigen::Vector3d any_perpendicular(const Eigen::Vector3d & u)
{
  // Cross with whichever world axis is least aligned with u, so the product is
  // never itself near-zero.
  const Eigen::Vector3d axis =
    (std::abs(u.x()) < 0.9 * u.norm()) ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
  const Eigen::Vector3d perp = u.cross(axis);
  const double n = perp.norm();
  return (n < kEps) ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d(perp / n);
}

}  // namespace

Eigen::Vector3d closest_point_on_segment(
  const Eigen::Vector3d & p, const Eigen::Vector3d & p0, const Eigen::Vector3d & p1)
{
  const Eigen::Vector3d u = p1 - p0;
  const double uu = u.squaredNorm();

  // Degenerate segment: the two endpoints coincide, so the capsule is a sphere.
  if (uu < kEps) {
    return p0;
  }

  // Project p onto the infinite line through p0 and p1, then clamp the parameter
  // to [0, 1] so the result stays on the finite segment.
  const double t = std::clamp((p - p0).dot(u) / uu, 0.0, 1.0);
  return p0 + t * u;
}

Clearance clearance(const Capsule & c, const Eigen::Vector3d & p)
{
  const Eigen::Vector3d q = closest_point_on_segment(p, c.p0, c.p1);
  const Eigen::Vector3d d = p - q;
  const double dist = d.norm();

  Clearance out;
  if (dist < kEps) {
    // p sits exactly on the axis, so d/dist would be NaN. A NaN here propagates
    // into the published pose and TF, and RViz dies with no usable error. Pick an
    // arbitrary outward direction instead; the drone is maximally inside, so the
    // distance is minus the full radius.
    out.normal = any_perpendicular(c.p1 - c.p0);
    out.distance = -c.radius;
  } else {
    out.normal = d / dist;
    out.distance = dist - c.radius;
  }
  return out;
}

ObstacleField ObstacleField::from_yaml(const std::string & path)
{
  YAML::Node root;
  try {
    root = YAML::LoadFile(path);
  } catch (const YAML::Exception & e) {
    throw std::runtime_error("obstacles.yaml: cannot load '" + path + "': " + e.what());
  }

  ObstacleField field;
  field.ground_z = root["ground_z"] ? root["ground_z"].as<double>() : 0.0;

  const YAML::Node segments = root["segments"];
  if (!segments || !segments.IsSequence()) {
    throw std::runtime_error("obstacles.yaml: missing 'segments' sequence");
  }

  for (const YAML::Node & s : segments) {
    Capsule c;
    c.name = s["name"] ? s["name"].as<std::string>() : "unnamed";
    c.p0 = parse_vec3(s["p0"], c.name + ".p0");
    c.p1 = parse_vec3(s["p1"], c.name + ".p1");

    if (!s["radius"]) {
      throw std::runtime_error("obstacles.yaml: " + c.name + " has no radius");
    }
    c.radius = s["radius"].as<double>();
    if (c.radius <= 0.0) {
      throw std::runtime_error("obstacles.yaml: " + c.name + " radius must be positive");
    }
    field.segments.push_back(c);
  }
  return field;
}

Eigen::Vector3d clamp_to_free_space(
  const ObstacleField & field, const ClampConfig & cfg, const Eigen::Vector3d & p,
  std::vector<std::string> * contacts)
{
  Eigen::Vector3d out = p;
  if (contacts) {
    contacts->clear();
  }

  // Record a surface once, however many passes touch it.
  auto note = [contacts](const std::string & name) {
      if (!contacts) {
        return;
      }
      if (std::find(contacts->begin(), contacts->end(), name) == contacts->end()) {
        contacts->push_back(name);
      }
    };

  const int passes = std::max(1, cfg.max_passes);
  for (int pass = 0; pass < passes; ++pass) {
    bool moved = false;

    // 1. Pipes. Move along the surface normal ONLY, by exactly the shortfall. The
    //    component of motion along the pipe is untouched, which is what makes the
    //    drone slide instead of stopping dead.
    for (const Capsule & c : field.segments) {
      const Clearance cl = clearance(c, out);
      const double required = cfg.drone_radius + cfg.margin;
      if (cl.distance < required) {
        out += cl.normal * (required - cl.distance);
        note(c.name);
        moved = true;
      }
    }

    // 2. Floor.
    const double floor_z = field.ground_z + cfg.min_altitude;
    if (out.z() < floor_z) {
      out.z() = floor_z;
      note("ground");
      moved = true;
    }

    // 3. Room walls. The 20 x 20 m search area IS the hall (Phase B decision), so
    //    the bounds clamp gives us walls for free. Inset by the drone radius so the
    //    body stays inside, not just the centre point.
    for (int a = 0; a < 2; ++a) {
      const double lo = cfg.area_min[a] + cfg.drone_radius;
      const double hi = cfg.area_max[a] - cfg.drone_radius;
      if (out[a] < lo) {
        out[a] = lo;
        note("wall");
        moved = true;
      } else if (out[a] > hi) {
        out[a] = hi;
        note("wall");
        moved = true;
      }
    }

    if (!moved) {
      break;
    }
  }
  return out;
}

}  // namespace haptic_plume_drone
