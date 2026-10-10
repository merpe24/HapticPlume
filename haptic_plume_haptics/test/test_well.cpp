/**
 * @file test_well.cpp
 * @brief Unit tests for the confidence-scaled belief well.
 *
 * Plain gtest, no ROS. The headline test is ClampTargetOutward -- commit 1b7557b places
 * leaks on the pipe surface, so every real trial starts with a believed source inside the
 * standoff, and this clamp is what keeps the well from fighting the PRF there (D4, risk R2).
 *
 * @author premmm
 * @date October 10, 2026
 */

#include <gtest/gtest.h>

#include <cmath>

#include "haptic_plume_haptics/prf.hpp"
#include "haptic_plume_haptics/well.hpp"

using haptic_plume_drone::Capsule;
using haptic_plume_drone::ClampConfig;
using haptic_plume_drone::ObstacleField;
using haptic_plume_drone::clearance;
using haptic_plume_haptics::PrfConfig;
using haptic_plume_haptics::Risk;
using haptic_plume_haptics::WellConfig;
using haptic_plume_haptics::clamp_target_outward;
using haptic_plume_haptics::prf_force;
using haptic_plume_haptics::risk_at;
using haptic_plume_haptics::well_depth;
using haptic_plume_haptics::well_force;
using haptic_plume_haptics::well_width;
using Vec3 = Eigen::Vector3d;

namespace
{

constexpr double kPi = 3.14159265358979323846;

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

/// config/haptics_params.yaml's prf block -- needed for r_pz() in the well's own clamp.
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

/// config/haptics_params.yaml's well block.
WellConfig test_well_config()
{
  WellConfig cfg;
  cfg.width_max = 0.785;
  cfg.width_min = 0.140;
  cfg.depth_min = 0.30;
  cfg.depth_max = 2.0;
  cfg.target_margin = 0.15;
  cfg.speed_threshold = 0.05;
  return cfg;
}

}  // namespace

// ----------------------------------------------------------------------- well_width

TEST(WellWidth, FallsMonotonicallyFromMaxToMin)
{
  const WellConfig cfg = test_well_config();
  EXPECT_DOUBLE_EQ(well_width(cfg, 0.0), cfg.width_max);
  EXPECT_DOUBLE_EQ(well_width(cfg, 1.0), cfg.width_min);

  double prev = well_width(cfg, 0.0);
  for (double k = 0.05; k <= 1.0001; k += 0.05) {
    const double w = well_width(cfg, k);
    EXPECT_LE(w, prev + 1e-12) << "width rose at confidence " << k;
    prev = w;
  }
}

TEST(WellWidth, ClampsConfidenceOutsideTheUnitInterval)
{
  const WellConfig cfg = test_well_config();
  EXPECT_DOUBLE_EQ(well_width(cfg, -0.5), cfg.width_max);
  EXPECT_DOUBLE_EQ(well_width(cfg, 1.5), cfg.width_min);
}

// ----------------------------------------------------------------------- well_depth

TEST(WellDepth, RisesFromDepthMinToDepthMaxOverTheOpenBand)
{
  const WellConfig cfg = test_well_config();
  double prev = well_depth(cfg, 0.01);
  EXPECT_GE(prev, cfg.depth_min);
  for (double k = 0.05; k <= 1.0001; k += 0.05) {
    const double d = well_depth(cfg, k);
    EXPECT_GE(d, prev - 1e-12) << "depth fell at confidence " << k;
    prev = d;
  }
  EXPECT_DOUBLE_EQ(well_depth(cfg, 1.0), cfg.depth_max);
}

TEST(WellDepth, IsExactlyZeroAtTheReservedConfidenceZero)
{
  const WellConfig cfg = test_well_config();
  // Exactly 0.0 means "render nothing" (DIVERGED / GuidanceTarget.valid false) -- must
  // not be smoothed into depth_min.
  EXPECT_EQ(well_depth(cfg, 0.0), 0.0);
}

TEST(WellDepth, ClampsConfidenceOutsideTheUnitInterval)
{
  const WellConfig cfg = test_well_config();
  EXPECT_EQ(well_depth(cfg, -0.5), 0.0);         // clamps to 0 -> the reserved value
  EXPECT_DOUBLE_EQ(well_depth(cfg, 1.5), cfg.depth_max);
}

// -------------------------------------------------------------------------- validate()

TEST(WellConfigValidate, ThrowsWhenWidthMinExceedsWidthMax)
{
  WellConfig cfg = test_well_config();
  cfg.width_min = 0.9;   // > width_max = 0.785
  EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(WellConfigValidate, ThrowsWhenDepthMinExceedsDepthMax)
{
  WellConfig cfg = test_well_config();
  cfg.depth_min = 3.0;   // > depth_max = 2.0
  EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(WellConfigValidate, ThrowsOnNonPositiveWidths)
{
  WellConfig cfg = test_well_config();
  cfg.width_min = 0.0;
  EXPECT_THROW(cfg.validate(), std::invalid_argument);

  cfg = test_well_config();
  cfg.width_max = 0.0;
  EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(WellConfigValidate, ThrowsOnNegativeTargetMargin)
{
  WellConfig cfg = test_well_config();
  cfg.target_margin = -0.01;
  EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

// -------------------------------------------------------------- clamp_target_outward

TEST(ClampTargetOutward, NeverLeavesClearanceBelowStandoff)
{
  const ObstacleField f = test_world();
  const PrfConfig prf = test_prf_config();
  const WellConfig well = test_well_config();
  const double standoff = prf.r_pz() + well.target_margin;   // 0.577

  // Dense sweep over and around both capsules, including inside them, at the caps,
  // and at the shared (10, 5, 1) corner.
  for (double x = 9.0; x <= 11.0 + 1e-9; x += 0.05) {
    for (double y = 4.0; y <= 16.0 + 1e-9; y += 0.2) {
      for (const double z : {0.0, 0.5, 1.0}) {
        const Vec3 target(x, y, z);
        const Vec3 out = clamp_target_outward(f, standoff, target);
        ASSERT_FALSE(out.hasNaN()) << "NaN clamping target " << x << ", " << y << ", " << z;
        for (const Capsule & c : f.segments) {
          ASSERT_GE(clearance(c, out).distance, standoff - 1e-6)
            << c.name << " violated clamping target " << x << ", " << y << ", " << z;
        }
      }
    }
  }
}

TEST(ClampTargetOutward, ClampsRealLeaksToTheirOwnSideOfThePipe)
{
  const ObstacleField f = test_world();
  const PrfConfig prf = test_prf_config();
  const WellConfig well = test_well_config();
  const double standoff = prf.r_pz() + well.target_margin;   // 0.577
  const double expected_offset = 0.30 + prf.r_pz() + well.target_margin;   // 0.877

  // Leak on the near face of main_run (commit 1b7557b's pipe-surface placement).
  const Vec3 leak1(9.70, 7.37, 1.0);
  const Vec3 out1 = clamp_target_outward(f, standoff, leak1);
  EXPECT_LT(out1.x(), 9.70);
  EXPECT_NEAR(out1.x(), 10.0 - expected_offset, 1e-9);
  EXPECT_NEAR(out1.y(), 7.37, 1e-9);
  EXPECT_NEAR(out1.z(), 1.0, 1e-9);
  EXPECT_LT(risk_at(f, prf, out1, Vec3::Zero()).value, 1.0);

  // Leak on the far face.
  const Vec3 leak2(10.30, 6.49, 1.0);
  const Vec3 out2 = clamp_target_outward(f, standoff, leak2);
  EXPECT_GT(out2.x(), 10.30);
  EXPECT_NEAR(out2.x(), 10.0 + expected_offset, 1e-9);
  EXPECT_NEAR(out2.y(), 6.49, 1e-9);
  EXPECT_LT(risk_at(f, prf, out2, Vec3::Zero()).value, 1.0);
}

// -------------------------------------------------------------------------- well_force

TEST(WellForce, PointsAtTheClampedTargetNotTheRawOne)
{
  const ObstacleField f = test_world();
  const PrfConfig prf = test_prf_config();
  const WellConfig well = test_well_config();
  const double standoff = prf.r_pz() + well.target_margin;

  // Positioned off-axis so the raw leak and its clamped target give measurably
  // different bearings -- a straight-ahead approach would not distinguish them.
  const Vec3 p(5.0, 11.37, 1.0);
  const Vec3 raw_leak(9.70, 7.37, 1.0);

  const Vec3 force = well_force(f, prf, well, p, Vec3::Zero(), raw_leak, 1.0, 0.0);
  const Vec3 clamped = clamp_target_outward(f, standoff, raw_leak);
  const Vec3 expected_bearing = (clamped - p).normalized();
  const Vec3 wrong_bearing = (raw_leak - p).normalized();

  ASSERT_GT((expected_bearing - wrong_bearing).norm(), 1e-3);   // the two really do differ
  EXPECT_NEAR((force.normalized() - expected_bearing).norm(), 0.0, 1e-9);
}

TEST(WellForce, MagnitudeNeverExceedsDepthMax)
{
  const ObstacleField f = test_world();
  const PrfConfig prf = test_prf_config();
  const WellConfig well = test_well_config();
  const Vec3 leak(9.70, 7.37, 1.0);
  const Vec3 headings[] = {Vec3(0.0, 0.0, 0.0), Vec3(1.0, 0.0, 0.0), Vec3(0.0, -1.0, 0.0),
    Vec3(0.7, 0.7, 0.0)};

  for (double confidence = 0.0; confidence <= 1.0001; confidence += 0.1) {
    for (double risk = 0.0; risk <= 1.0001; risk += 0.25) {
      for (const Vec3 & v : headings) {
        const Vec3 force = well_force(f, prf, well, Vec3(5.0, 7.37, 1.0), v, leak, confidence,
            risk);
        EXPECT_LE(force.norm(), well.depth_max + 1e-9);
      }
    }
  }
}

TEST(WellForce, RiskOneZerosTheForceAndRiskZeroLeavesItUnscaled)
{
  const ObstacleField f = test_world();
  const PrfConfig prf = test_prf_config();
  const WellConfig well = test_well_config();
  const double standoff = prf.r_pz() + well.target_margin;
  const Vec3 p(5.0, 7.37, 1.0);
  const Vec3 leak(9.70, 7.37, 1.0);

  const Vec3 f_risk_one = well_force(f, prf, well, p, Vec3::Zero(), leak, 1.0, 1.0);
  EXPECT_NEAR(f_risk_one.norm(), 0.0, 1e-12);

  const Vec3 f_risk_zero = well_force(f, prf, well, p, Vec3::Zero(), leak, 1.0, 0.0);
  const Vec3 clamped = clamp_target_outward(f, standoff, leak);
  const Vec3 expected = (clamped - p).normalized() * well.depth_max;   // shape = 1, yield = 1
  EXPECT_NEAR((f_risk_zero - expected).norm(), 0.0, 1e-9);
}

TEST(WellForce, FallsOffWithHeadingErrorAndAWiderWellForgivesItMore)
{
  const ObstacleField f = test_world();
  const PrfConfig prf = test_prf_config();
  const WellConfig well = test_well_config();
  const Vec3 p(5.0, 7.37, 1.0);
  const Vec3 leak(9.70, 7.37, 1.0);   // same y, z as p -> bearing is pure +x

  const double speed = 1.0;   // above speed_threshold
  const Vec3 bearing(1.0, 0.0, 0.0);
  const Vec3 v_straight = bearing * speed;
  const Vec3 v_45 =
    (bearing * std::cos(kPi / 4.0) + Vec3(0.0, 1.0, 0.0) * std::sin(kPi / 4.0)) * speed;

  const double narrow = 1.0;    // confidence 1.0 -> width_min, the sharp detent
  const double wide = 0.001;    // confidence near 0 (not exactly 0 -- that renders nothing)

  const Vec3 f_straight_narrow = well_force(f, prf, well, p, v_straight, leak, narrow, 0.0);
  const Vec3 f_45_narrow = well_force(f, prf, well, p, v_45, leak, narrow, 0.0);
  const Vec3 f_45_wide = well_force(f, prf, well, p, v_45, leak, wide, 0.0);

  EXPECT_LT(f_45_narrow.norm(), f_straight_narrow.norm())
    << "force did not fall off with heading error";
  EXPECT_GT(f_45_wide.norm(), f_45_narrow.norm())
    << "a wide (low-confidence) well did not forgive 45 deg of heading error more than a "
       "narrow one";
}

TEST(WellForce, RendersFullDepthAlongTheBearingBelowSpeedThreshold)
{
  const ObstacleField f = test_world();
  const PrfConfig prf = test_prf_config();
  const WellConfig well = test_well_config();
  const double standoff = prf.r_pz() + well.target_margin;
  const Vec3 p(5.0, 7.37, 1.0);
  const Vec3 leak(9.70, 7.37, 1.0);
  const Vec3 clamped = clamp_target_outward(f, standoff, leak);
  const Vec3 expected_bearing = (clamped - p).normalized();

  for (const Vec3 & v : {Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.03, 0.0), Vec3(-0.02, 0.0, 0.04)}) {
    ASSERT_LT(v.norm(), well.speed_threshold);
    const Vec3 force = well_force(f, prf, well, p, v, leak, 1.0, 0.0);
    const Vec3 expected = expected_bearing * well.depth_max;   // shape = 1 when parked
    EXPECT_NEAR((force - expected).norm(), 0.0, 1e-9);
  }
}

TEST(WellForce, ApproachToALeakOnAPipeDoesNotLimitCycle)
{
  const ObstacleField f = test_world();
  const PrfConfig prf = test_prf_config();
  const WellConfig well = test_well_config();
  const Vec3 leak(9.70, 7.37, 1.0);
  const Vec3 v(0.5, 0.0, 0.0);   // flying straight toward the leak

  int sign_changes = 0;
  int prev_sign = 0;
  for (double x = 6.0; x <= 9.70 + 1e-9; x += 0.01) {
    const Vec3 p(x, 7.37, 1.0);
    const Risk risk = risk_at(f, prf, p, v);
    const Vec3 combined = prf_force(f, prf, p, v) + well_force(f, prf, well, p, v, leak, 1.0,
        risk.value);
    const int sign = (combined.x() > 1e-9) - (combined.x() < -1e-9);
    if (sign != 0 && prev_sign != 0 && sign != prev_sign) {
      ++sign_changes;
    }
    if (sign != 0) {
      prev_sign = sign;
    }
  }

  // One crossing is the standoff being reached; repeated crossings would be R2's
  // well/PRF fight. If this fails, that fight is real -- do not retune constants to
  // force it to pass.
  EXPECT_LE(sign_changes, 1);
}
