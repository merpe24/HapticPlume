/**
 * @file test_prf.cpp
 * @brief Unit tests for the parametric risk field (Lam et al. 2009).
 *
 * Plain gtest, no ROS: prf.hpp pulls in haptic_plume_drone::obstacle_field but never
 * rclcpp. The D1 invariant test is the one that matters most here -- it is what proves
 * the risk field and drone_kinematics_node's clamp cannot disagree about where a pipe is.
 *
 * @author premmm
 * @date October 10, 2026
 */

#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "haptic_plume_haptics/prf.hpp"

using haptic_plume_drone::Capsule;
using haptic_plume_drone::ClampConfig;
using haptic_plume_drone::ObstacleField;
using haptic_plume_drone::clamp_to_free_space;
using haptic_plume_haptics::PrfConfig;
using haptic_plume_haptics::Risk;
using haptic_plume_haptics::braking_distance;
using haptic_plume_haptics::look_ahead;
using haptic_plume_haptics::p_cos2;
using haptic_plume_haptics::prf_force;
using haptic_plume_haptics::risk_at;
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
ClampConfig test_clamp_config()
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

/// config/haptics_params.yaml's prf block.
PrfConfig test_prf_config()
{
  PrfConfig cfg;
  cfg.clamp = test_clamp_config();
  cfg.d_min = 1.50;
  cfg.t_ahead = 2.0;
  cfg.a_max = 3.0;
  cfg.max_force = 5.0;
  cfg.include_floor = false;   // as shipped: fixed-altitude flight, see prf.hpp
  return cfg;
}

/// True iff clamp_to_free_space would actually move p -- an exact comparison, since
/// the clamp leaves "out" bit-identical to p when nothing touches it.
bool clamp_would_move(const ObstacleField & f, const ClampConfig & c, const Vec3 & p)
{
  const Vec3 out = clamp_to_free_space(f, c, p);
  return out.x() != p.x() || out.y() != p.y() || out.z() != p.z();
}

}  // namespace

// ------------------------------------------------------------------- p_cos2

TEST(PCos2, HitsTheThreeNamedPoints)
{
  EXPECT_DOUBLE_EQ(p_cos2(0.0), 1.0);
  EXPECT_DOUBLE_EQ(p_cos2(1.0), 0.0);
  EXPECT_DOUBLE_EQ(p_cos2(0.5), 0.5);
}

TEST(PCos2, ClampsOutsideTheUnitInterval)
{
  EXPECT_DOUBLE_EQ(p_cos2(-2.0), 1.0);
  EXPECT_DOUBLE_EQ(p_cos2(1.5), 0.0);
}

TEST(PCos2, GradientIsZeroAtBothEnds)
{
  // Central difference near each end. Lam picked this function over the linear and
  // plain-cosine forms exactly so the force has no corner at either boundary.
  const double h = 1e-3;
  const double slope_at_zero = (p_cos2(h) - p_cos2(0.0)) / h;
  const double slope_at_one = (p_cos2(1.0) - p_cos2(1.0 - h)) / h;
  EXPECT_NEAR(slope_at_zero, 0.0, 0.01);
  EXPECT_NEAR(slope_at_one, 0.0, 0.01);
}

// ------------------------------------------------------------ braking / look-ahead

TEST(BrakingDistance, MatchesEq10OnHandValues)
{
  EXPECT_DOUBLE_EQ(braking_distance(3.0, 3.0), 1.5);   // 3^2 / (2*3)
  EXPECT_DOUBLE_EQ(braking_distance(0.0, 3.0), 0.0);
}

TEST(LookAhead, MatchesEq11OnHandValues)
{
  EXPECT_DOUBLE_EQ(look_ahead(3.0, 2.0), 6.0);
  EXPECT_DOUBLE_EQ(look_ahead(0.0, 2.0), 0.0);
}

// -------------------------------------------------------------------- validate()

TEST(PrfConfigValidate, ThrowsOnNonPositiveDMin)
{
  PrfConfig cfg = test_prf_config();
  cfg.d_min = 0.0;
  EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(PrfConfigValidate, ThrowsOnNonPositiveAMax)
{
  PrfConfig cfg = test_prf_config();
  cfg.a_max = 0.0;
  EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(PrfConfigValidate, ThrowsOnNegativeTAhead)
{
  PrfConfig cfg = test_prf_config();
  cfg.t_ahead = -0.1;
  EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(PrfConfigValidate, ThrowsOnNegativeMaxForce)
{
  PrfConfig cfg = test_prf_config();
  cfg.max_force = -1.0;
  EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

// -------------------------------------------------------------------- R10 hover

TEST(Prf, DoesNotCollapseAtHoverJustOutsideTheStandoff)
{
  const ObstacleField f = test_world();
  const PrfConfig cfg = test_prf_config();
  // r_pz() == 0.427. Clearance here is 0.437 -- 1 cm outside the standoff, so the
  // clamp would NOT move this point, yet d_min must still give the hover field a
  // nonzero reading this close in (risk R10).
  const Vec3 p(10.0 + 0.737, 10.0, 1.0);
  ASSERT_FALSE(clamp_would_move(f, cfg.clamp, p));

  const Risk risk = risk_at(f, cfg, p, Vec3::Zero());
  EXPECT_GT(risk.value, 0.0);
  EXPECT_LT(risk.value, 1.0);
  EXPECT_EQ(risk.surface, "main_run");  // confirms this is the pipe field, not the floor
}

TEST(Prf, RiskSaturatesEarlierThanTheClampWhenMovingFast)
{
  const ObstacleField f = test_world();
  const PrfConfig cfg = test_prf_config();
  // 1 m of clearance beyond the standoff -- the positional clamp leaves this point
  // completely alone.
  const Vec3 p(9.0, 10.0, 1.0);
  ASSERT_FALSE(clamp_would_move(f, cfg.clamp, p));

  // Flying straight at the pipe at 5 m/s: braking_distance(5, 3) = 4.17 m, which alone
  // exceeds the 0.273 m of slack between this point and the standoff.
  const Vec3 v(5.0, 0.0, 0.0);
  const Risk risk = risk_at(f, cfg, p, v);
  EXPECT_DOUBLE_EQ(risk.value, 1.0);
  EXPECT_EQ(risk.surface, "main_run");
}

// --------------------------------------------------------- D1 invariant (the big one)

TEST(Prf, RiskSaturatesExactlyWhereTheClampWouldMove)
{
  const ObstacleField f = test_world();
  const PrfConfig cfg = test_prf_config();
  const Vec3 v = Vec3::Zero();   // D1 is a claim about hover; speed only ever widens risk's
                                 // lead over the clamp (previous test), never narrows it.
  int checked = 0;

  // Dense sweep straddling both capsules and the (10, 5, 1) corner they share.
  for (double x = 8.5; x <= 11.5 + 1e-9; x += 0.1) {
    for (double y = 4.0; y <= 16.0 + 1e-9; y += 0.25) {
      const Vec3 p(x, y, 1.0);
      const bool moved = clamp_would_move(f, cfg.clamp, p);
      const Risk risk = risk_at(f, cfg, p, v);
      if (risk.value >= 1.0) {
        EXPECT_TRUE(moved)  << "risk saturated but clamp left p alone at (" << x << ", " << y
                            << ", 1.0)";
      } else {
        EXPECT_FALSE(moved)  << "clamp moved p but risk " << risk.value << " < 1 at (" << x
                             << ", " << y << ", 1.0)";
      }
      ++checked;
    }
  }
  EXPECT_GT(checked, 1000);

  // Near the room walls. Deliberately straddling the inset boundary (drone_radius alone,
  // no margin -- the asymmetry D1 calls out) without landing exactly on it.
  const double wall_xs[] = {-0.5, -0.1, 0.0, 0.1, 0.3, 0.3769, 0.3771, 0.4, 0.5, 1.0,
    19.0, 19.5, 19.6, 19.6229, 19.6231, 19.7, 19.9, 20.0, 20.1, 20.5};
  for (const double x : wall_xs) {
    for (const double y : {1.0, 10.0, 19.0}) {
      const Vec3 p(x, y, 1.0);
      const bool moved = clamp_would_move(f, cfg.clamp, p);
      const Risk risk = risk_at(f, cfg, p, v);
      if (risk.value >= 1.0) {
        EXPECT_TRUE(moved)  << "risk saturated but clamp left p alone at (" << x << ", " << y
                            << ", 1.0)";
      } else {
        EXPECT_FALSE(moved)  << "clamp moved p but risk " << risk.value << " < 1 at (" << x
                             << ", " << y << ", 1.0)";
      }
    }
  }

  // Near the floor, far from every pipe and wall so only the floor term is in play. The
  // clamp ALWAYS enforces the floor, so the invariant can only be claimed here with the
  // floor scored -- which is exactly what include_floor turns on for Phase F. With it off
  // (as shipped) the invariant is scoped to the pipes and walls swept above, and the
  // following test pins down what the floor does instead.
  PrfConfig floor_cfg = cfg;
  floor_cfg.include_floor = true;
  for (const double z : {-0.5, -0.1, 0.0, 0.1, 0.1499, 0.1501, 0.2, 0.5, 1.0}) {
    const Vec3 p(5.0, 10.0, z);
    const bool moved = clamp_would_move(f, floor_cfg.clamp, p);
    const Risk risk = risk_at(f, floor_cfg, p, v);
    if (risk.value >= 1.0) {
      EXPECT_TRUE(moved) << "risk saturated but clamp left p alone at z = " << z;
    } else {
      EXPECT_FALSE(moved) << "clamp moved p but risk " << risk.value << " < 1 at z = " << z;
    }
  }
}

// ----------------------------------------------------------------- monotonicity

TEST(Prf, RiskIsNonIncreasingMovingRadiallyAwayFromMainRun)
{
  const ObstacleField f = test_world();
  const PrfConfig cfg = test_prf_config();
  const Vec3 v = Vec3::Zero();

  double prev = risk_at(f, cfg, Vec3(10.0, 10.0, 1.0), v).value;   // on the axis: risk 1
  for (double r = 0.1; r <= 5.0; r += 0.1) {
    const double value = risk_at(f, cfg, Vec3(10.0 - r, 10.0, 1.0), v).value;
    EXPECT_LE(value, prev + 1e-9) << "risk increased moving outward at r = " << r;
    prev = value;
  }
}

// --------------------------------------------------------------------- direction

TEST(Prf, PushesInMinusXWhenApproachingMainRunFromTheWest)
{
  const ObstacleField f = test_world();
  const PrfConfig cfg = test_prf_config();
  const Vec3 p(9.0, 10.0, 1.0);

  const Risk risk = risk_at(f, cfg, p, Vec3::Zero());
  EXPECT_NEAR((risk.direction - Vec3(-1.0, 0.0, 0.0)).norm(), 0.0, 1e-9);

  const Vec3 force = prf_force(f, cfg, p, Vec3::Zero());
  EXPECT_LT(force.x(), 0.0);
  EXPECT_NEAR(force.y(), 0.0, 1e-9);
  EXPECT_NEAR(force.z(), 0.0, 1e-9);
}

// ------------------------------------------------------------------ max-combine

TEST(Prf, MaxCombinesToExactlyOneAtTheSharedEndpoint)
{
  const ObstacleField f = test_world();
  const PrfConfig cfg = test_prf_config();
  // (10, 5, 1) is on the axis of BOTH main_run and riser.
  const Vec3 p(10.0, 5.0, 1.0);

  const Risk risk = risk_at(f, cfg, p, Vec3::Zero());
  EXPECT_DOUBLE_EQ(risk.value, 1.0);
  EXPECT_EQ(risk.surface, "main_run");   // ties keep the earlier segment, deterministically

  const Vec3 force = prf_force(f, cfg, p, Vec3::Zero());
  EXPECT_NEAR(force.norm(), cfg.max_force, 1e-9);
}

TEST(Prf, ForceMagnitudeNeverExceedsMaxForce)
{
  const ObstacleField f = test_world();
  const PrfConfig cfg = test_prf_config();
  // High speed, straight at the pipe from several headings, including the shared corner.
  const Vec3 v(10.0, 10.0, 0.0);
  const Vec3 probes[] = {Vec3(9.5, 10.0, 1.0), Vec3(10.5, 10.0, 1.0), Vec3(10.0, 5.0, 1.0),
    Vec3(10.0, 5.0, 0.5), Vec3(0.1, 0.1, 1.0), Vec3(19.9, 19.9, 1.0)};
  for (const Vec3 & p : probes) {
    const Risk risk = risk_at(f, cfg, p, v);
    EXPECT_LE(risk.value, 1.0);
    const Vec3 force = prf_force(f, cfg, p, v);
    EXPECT_LE(force.norm(), cfg.max_force + 1e-9);
  }
}

// --------------------------------------------------------------------- far field

TEST(Prf, RiskIsZeroAndDirectionlessFarFromEverything)
{
  const ObstacleField f = test_world();
  const PrfConfig cfg = test_prf_config();
  const Risk risk = risk_at(f, cfg, Vec3(5.0, 10.0, 1.0), Vec3::Zero());
  EXPECT_DOUBLE_EQ(risk.value, 0.0);
  EXPECT_NEAR(risk.direction.norm(), 0.0, 1e-12);
  EXPECT_TRUE(risk.surface.empty());
}

// This is the regression test for the bug the test above found on 2026-10-10: with the
// floor scored, the middle of an empty room at flight altitude is NOT risk-free, because
// the floor is 0.85 m below and d_min is 1.50 m. That is about 2 N straight up, constantly,
// everywhere, all trial -- 40% of the PRF budget spent on a surface the drone cannot
// descend toward. Both settings are pinned so neither can drift silently.
TEST(Prf, TheFloorIsInsideTheFieldAtFlightAltitudeWhichIsWhyItIsOffByDefault)
{
  const ObstacleField f = test_world();
  const PrfConfig shipped = test_prf_config();
  ASSERT_FALSE(shipped.include_floor);

  PrfConfig scored = shipped;
  scored.include_floor = true;

  const Vec3 middle_of_the_room(5.0, 10.0, 1.0);
  const Vec3 hover = Vec3::Zero();

  // The floor really is inside the field: 1.00 - min_altitude = 0.85 against d_min 1.50.
  const double slack = 1.0 - (f.ground_z + shipped.clamp.min_altitude);
  ASSERT_LT(slack, shipped.d_min) << "this test is meaningless unless the floor is inside "
                                     "the field; retune d_min or min_altitude and revisit";

  const Risk with_floor = risk_at(f, scored, middle_of_the_room, hover);
  EXPECT_NEAR(with_floor.value, p_cos2(slack / scored.d_min), 1e-12);
  EXPECT_NEAR(with_floor.value, 0.396, 0.001);
  EXPECT_EQ(with_floor.surface, "ground");
  EXPECT_NEAR(with_floor.direction.z(), 1.0, 1e-12);
  EXPECT_GT(prf_force(f, scored, middle_of_the_room, hover).norm(), 1.9);

  // As shipped, the same point is silent.
  EXPECT_DOUBLE_EQ(risk_at(f, shipped, middle_of_the_room, hover).value, 0.0);
  EXPECT_DOUBLE_EQ(prf_force(f, shipped, middle_of_the_room, hover).norm(), 0.0);

  // Turning the floor off must not touch anything else. Close to main_run, both agree.
  const Vec3 near_pipe(9.0, 10.0, 1.0);
  EXPECT_DOUBLE_EQ(
    risk_at(f, shipped, near_pipe, hover).value, risk_at(f, scored, near_pipe, hover).value);
}
