/**
 * @file test_obstacle_field.cpp
 * @brief Unit tests for the capsule geometry and the sliding position clamp.
 *
 * These run with plain gtest, no ROS and no launching, because obstacle_field
 * deliberately has no rclcpp dependency.
 *
 * @author premmm
 * @date August 3, 2026
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "haptic_plume_drone/obstacle_field.hpp"

using haptic_plume_drone::Capsule;
using haptic_plume_drone::ClampConfig;
using haptic_plume_drone::Clearance;
using haptic_plume_drone::ObstacleField;
using haptic_plume_drone::clamp_to_free_space;
using haptic_plume_drone::clearance;
using haptic_plume_drone::closest_point_on_segment;
using Vec3 = Eigen::Vector3d;

namespace
{

/// The real world, mirroring haptic_plume_description/config/obstacles.yaml.
ObstacleField test_world()
{
  ObstacleField f;
  f.ground_z = 0.0;
  f.segments.push_back({"main_run", Vec3(10.0, 5.0, 1.0), Vec3(10.0, 15.0, 1.0), 0.30});
  f.segments.push_back({"riser", Vec3(10.0, 5.0, 0.0), Vec3(10.0, 5.0, 1.0), 0.30});
  return f;
}

/// Matches the node's defaults: Holybro X500 V2 in the 20 x 20 m hall.
ClampConfig test_config()
{
  ClampConfig c;
  c.drone_radius = 0.377;
  c.margin = 0.05;
  c.min_altitude = 0.15;
  c.area_min = Eigen::Vector2d(0.0, 0.0);
  c.area_max = Eigen::Vector2d(20.0, 20.0);
  c.max_passes = 4;
  return c;
}

}  // namespace

// ---------------------------------------------------------------- projection

TEST(ClosestPointOnSegment, HitsEndpointsAndMidpoint)
{
  const Vec3 a(0.0, 0.0, 0.0), b(10.0, 0.0, 0.0);
  EXPECT_NEAR((closest_point_on_segment(Vec3(0.0, 5.0, 0.0), a, b) - a).norm(), 0.0, 1e-9);
  EXPECT_NEAR((closest_point_on_segment(Vec3(10.0, 5.0, 0.0), a, b) - b).norm(), 0.0, 1e-9);
  EXPECT_NEAR(
    (closest_point_on_segment(Vec3(5.0, 5.0, 0.0), a, b) - Vec3(5.0, 0.0, 0.0)).norm(), 0.0, 1e-9);
}

TEST(ClosestPointOnSegment, ClampsBeyondEachEnd)
{
  const Vec3 a(0.0, 0.0, 0.0), b(10.0, 0.0, 0.0);
  // Raw projections land at t = -1 and t = +2; both must clamp back on to the segment.
  EXPECT_NEAR((closest_point_on_segment(Vec3(-10.0, 3.0, 0.0), a, b) - a).norm(), 0.0, 1e-9);
  EXPECT_NEAR((closest_point_on_segment(Vec3(20.0, 3.0, 0.0), a, b) - b).norm(), 0.0, 1e-9);
}

TEST(ClosestPointOnSegment, DegenerateSegmentIsAPoint)
{
  const Vec3 a(1.0, 2.0, 3.0);
  EXPECT_NEAR((closest_point_on_segment(Vec3(9.0, 9.0, 9.0), a, a) - a).norm(), 0.0, 1e-9);
}

// ----------------------------------------------------------------- clearance

TEST(ClearanceTest, OutsideIsPositiveAndNormalPointsOutward)
{
  const Capsule c{"main_run", Vec3(10.0, 5.0, 1.0), Vec3(10.0, 15.0, 1.0), 0.30};
  const Clearance cl = clearance(c, Vec3(8.0, 10.0, 1.0));   // 2 m to the -x side
  EXPECT_NEAR(cl.distance, 2.0 - 0.30, 1e-9);
  EXPECT_NEAR((cl.normal - Vec3(-1.0, 0.0, 0.0)).norm(), 0.0, 1e-9);
  EXPECT_NEAR(cl.normal.norm(), 1.0, 1e-9);
}

TEST(ClearanceTest, InsideIsNegative)
{
  const Capsule c{"main_run", Vec3(10.0, 5.0, 1.0), Vec3(10.0, 15.0, 1.0), 0.30};
  const Clearance cl = clearance(c, Vec3(10.1, 10.0, 1.0));
  EXPECT_LT(cl.distance, 0.0);
  EXPECT_NEAR(cl.distance, 0.1 - 0.30, 1e-9);
}

TEST(ClearanceTest, CapRegionMeasuresToTheEndpoint)
{
  const Capsule c{"main_run", Vec3(10.0, 5.0, 1.0), Vec3(10.0, 15.0, 1.0), 0.30};
  // 1 m past the p1 end, on the axis line: this is the hemispherical cap.
  const Clearance cl = clearance(c, Vec3(10.0, 16.0, 1.0));
  EXPECT_NEAR(cl.distance, 1.0 - 0.30, 1e-9);
  EXPECT_NEAR((cl.normal - Vec3(0.0, 1.0, 0.0)).norm(), 0.0, 1e-9);
}

TEST(ClearanceTest, OnAxisDoesNotProduceNaN)
{
  const Capsule c{"riser", Vec3(10.0, 5.0, 0.0), Vec3(10.0, 5.0, 1.0), 0.30};
  const Clearance cl = clearance(c, Vec3(10.0, 5.0, 0.5));   // exactly on the axis
  EXPECT_FALSE(std::isnan(cl.distance));
  EXPECT_FALSE(cl.normal.hasNaN());
  EXPECT_NEAR(cl.normal.norm(), 1.0, 1e-9);
  EXPECT_NEAR(cl.distance, -0.30, 1e-9);
}

// --------------------------------------------------------------------- clamp

TEST(ClampToFreeSpace, LeavesFreeSpaceUntouched)
{
  const Vec3 p(2.0, 2.0, 1.0);
  EXPECT_NEAR((clamp_to_free_space(test_world(), test_config(), p) - p).norm(), 0.0, 1e-12);
}

TEST(ClampToFreeSpace, SlidingPreservesTangentialMotion)
{
  const ObstacleField f = test_world();
  const ClampConfig cfg = test_config();
  const double standoff = 0.30 + cfg.drone_radius + cfg.margin;   // 0.727 m from the axis

  // Start west of main_run and fly mostly straight into it, with a small
  // component along the pipe.
  Vec3 p(9.0, 8.0, 1.0);
  const Vec3 v(1.0, 0.3, 0.0);
  const double dt = 0.01;
  const double y0 = p.y();

  for (int i = 0; i < 200; ++i) {
    p = clamp_to_free_space(f, cfg, p + v * dt);
  }

  // Pinned at the standoff surface on the -x side ...
  EXPECT_NEAR(p.x(), 10.0 - standoff, 1e-6);
  // ... but still travelling along the pipe. This is the "collisions slide" DoD item.
  EXPECT_GT(p.y(), y0 + 0.5);
  EXPECT_NEAR(p.z(), 1.0, 1e-9);
}

TEST(ClampToFreeSpace, NeverLeavesThePointInsideAnything)
{
  const ObstacleField f = test_world();
  const ClampConfig cfg = test_config();
  const double required = cfg.drone_radius + cfg.margin;

  // Sweep a grid straddling both capsules and the corner they share at (10, 5, 1).
  for (double x = 9.0; x <= 11.0; x += 0.05) {
    for (double y = 4.0; y <= 6.0; y += 0.05) {
      for (double z = 0.2; z <= 1.4; z += 0.2) {
        const Vec3 out = clamp_to_free_space(f, cfg, Vec3(x, y, z));
        ASSERT_FALSE(out.hasNaN()) << "NaN from start " << x << ", " << y << ", " << z;
        for (const Capsule & c : f.segments) {
          ASSERT_GE(clearance(c, out).distance, required - 1e-6)
            << c.name << " violated from start " << x << ", " << y << ", " << z;
        }
      }
    }
  }
}

TEST(ClampToFreeSpace, PushesOutFromRestInsideAnObstacle)
{
  const ObstacleField f = test_world();
  const ClampConfig cfg = test_config();
  // Sitting exactly on the axis with zero velocity: nothing is "moving into" the
  // pipe, so a velocity-based check would never fire. The clamp is positional.
  const Vec3 out = clamp_to_free_space(f, cfg, Vec3(10.0, 10.0, 1.0));
  EXPECT_FALSE(out.hasNaN());
  EXPECT_GE(clearance(f.segments[0], out).distance, cfg.drone_radius + cfg.margin - 1e-6);
}

TEST(ClampToFreeSpace, ReportsContactsByName)
{
  const ObstacleField f = test_world();
  const ClampConfig cfg = test_config();
  std::vector<std::string> contacts;

  clamp_to_free_space(f, cfg, Vec3(9.5, 10.0, 1.0), &contacts);
  ASSERT_EQ(contacts.size(), 1u);
  EXPECT_EQ(contacts[0], "main_run");

  clamp_to_free_space(f, cfg, Vec3(2.0, 2.0, 1.0), &contacts);
  EXPECT_TRUE(contacts.empty());
}

TEST(ClampToFreeSpace, HoldsTheDroneAboveTheFloor)
{
  const ObstacleField f = test_world();
  const ClampConfig cfg = test_config();
  const Vec3 out = clamp_to_free_space(f, cfg, Vec3(2.0, 2.0, -5.0));
  EXPECT_NEAR(out.z(), f.ground_z + cfg.min_altitude, 1e-9);
}

TEST(ClampToFreeSpace, KeepsTheBodyInsideTheRoom)
{
  const ObstacleField f = test_world();
  const ClampConfig cfg = test_config();
  const Vec3 out = clamp_to_free_space(f, cfg, Vec3(-3.0, 25.0, 1.0));
  EXPECT_NEAR(out.x(), cfg.drone_radius, 1e-9);
  EXPECT_NEAR(out.y(), 20.0 - cfg.drone_radius, 1e-9);
}

// ---------------------------------------------------------------- yaml loader

TEST(FromYaml, ParsesSegments)
{
  const std::string path = "/tmp/haptic_plume_test_obstacles.yaml";
  {
    std::ofstream out(path);
    out << "ground_z: 0.0\n"
      "segments:\n"
      "  - name: main_run\n"
      "    p0: [10.0, 5.0, 1.0]\n"
      "    p1: [10.0, 15.0, 1.0]\n"
      "    radius: 0.30\n";
  }

  const ObstacleField f = ObstacleField::from_yaml(path);
  EXPECT_NEAR(f.ground_z, 0.0, 1e-12);
  ASSERT_EQ(f.segments.size(), 1u);
  EXPECT_EQ(f.segments[0].name, "main_run");
  EXPECT_NEAR(f.segments[0].radius, 0.30, 1e-12);
  EXPECT_NEAR((f.segments[0].p1 - Vec3(10.0, 15.0, 1.0)).norm(), 0.0, 1e-12);

  std::remove(path.c_str());
}

TEST(FromYaml, ThrowsOnMissingFile)
{
  EXPECT_THROW(ObstacleField::from_yaml("/tmp/definitely_not_here.yaml"), std::runtime_error);
}
