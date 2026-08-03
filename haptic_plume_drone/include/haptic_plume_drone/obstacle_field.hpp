/**
 * @file obstacle_field.hpp
 * @brief Analytic world geometry: capsule pipes, the ground, and the room walls.
 *
 * This is the geometric authority for the simulated world. The pipeline is modelled
 * as a set of capsules (a cylinder with hemispherical caps) loaded from
 * haptic_plume_description/config/obstacles.yaml.
 *
 * Two consumers link this library and they MUST agree:
 *   - drone_kinematics_node clamps the integrated position out of solid geometry (D1)
 *   - Phase C's Parametric Risk Field renders repulsion from the same surfaces
 * If each computed distance-to-pipe its own way, the drone would slide off a pipe
 * while the haptics pushed from somewhere slightly different.
 *
 * Deliberately free of ROS: no rclcpp here, so the geometry is unit-testable with
 * plain gtest and reusable offline.
 *
 * @author premmm
 * @date August 3, 2026
 */

#ifndef HAPTIC_PLUME_DRONE__OBSTACLE_FIELD_HPP_
#define HAPTIC_PLUME_DRONE__OBSTACLE_FIELD_HPP_

#include <string>
#include <vector>

#include <Eigen/Dense>

namespace haptic_plume_drone
{

/// One pipeline segment: a cylinder of `radius` about p0->p1 with hemispherical caps.
struct Capsule
{
  std::string name;
  Eigen::Vector3d p0;     // [m] segment start, map frame
  Eigen::Vector3d p1;     // [m] segment end, map frame
  double radius;          // [m] pipe outer radius
};

/// Result of querying a point against one capsule.
struct Clearance
{
  double distance;        // [m] surface-to-point; negative means inside
  Eigen::Vector3d normal; // unit, points from the surface out toward the query point
};

/// Tunables the clamp needs. Mirrors the node's ROS parameters.
struct ClampConfig
{
  double drone_radius;        // [m] the drone is treated as a sphere of this radius
  double margin;              // [m] extra standoff kept beyond bare contact
  double min_altitude;        // [m] lowest allowed height above ground_z
  Eigen::Vector2d area_min;   // [m] search-area lower corner (x, y)
  Eigen::Vector2d area_max;   // [m] search-area upper corner (x, y)
  int max_passes;             // clamp iterations; capsules share endpoints
};

/**
 * @brief Closest point on the segment p0->p1 to an arbitrary point.
 * @param p  Query point [m]
 * @param p0 Segment start [m]
 * @param p1 Segment end [m]
 * @return The closest point lying on the segment [m]
 */
Eigen::Vector3d closest_point_on_segment(
  const Eigen::Vector3d & p, const Eigen::Vector3d & p0, const Eigen::Vector3d & p1);

/**
 * @brief Distance and outward normal from a capsule surface to a point.
 * @param c Capsule to query against
 * @param p Query point [m]
 * @return Clearance; distance is negative when p is inside the capsule
 */
Clearance clearance(const Capsule & c, const Eigen::Vector3d & p);

/// The whole analytic world, as loaded from obstacles.yaml.
struct ObstacleField
{
  double ground_z;                  // [m] floor plane in the map frame
  std::vector<Capsule> segments;

  /**
   * @brief Parse an obstacles.yaml into an ObstacleField.
   * @param path Absolute path to the YAML file
   * @return The populated field
   * @throw std::runtime_error if the file is missing or a key is malformed
   */
  static ObstacleField from_yaml(const std::string & path);
};

/**
 * @brief Push a position out of all solid geometry, preserving tangential motion.
 *
 * For each violated capsule the point is moved along the surface normal ONLY, so the
 * component of motion along the pipe survives. That projection is what makes the drone
 * slide along an obstacle instead of stopping dead against it.
 *
 * @param field    The world geometry
 * @param cfg      Radii, margins and bounds
 * @param p        Candidate position after integration [m]
 * @param contacts If non-null, receives the name of every surface actually touched
 * @return A position guaranteed to be outside every obstacle [m]
 */
Eigen::Vector3d clamp_to_free_space(
  const ObstacleField & field, const ClampConfig & cfg, const Eigen::Vector3d & p,
  std::vector<std::string> * contacts = nullptr);

}  // namespace haptic_plume_drone

#endif  // HAPTIC_PLUME_DRONE__OBSTACLE_FIELD_HPP_
