/**
 * @file test_heaviness.cpp
 * @brief Unit tests for the pseudo-gravity heaviness cue.
 *
 * Plain gtest, no ROS, no geometry -- this cue reads only a scalar concentration.
 *
 * @author premmm
 * @date October 10, 2026
 */

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "haptic_plume_haptics/heaviness.hpp"

using haptic_plume_haptics::HeavinessConfig;
using haptic_plume_haptics::heaviness_force;

namespace
{

/// config/haptics_params.yaml's heaviness block.
HeavinessConfig test_config()
{
  HeavinessConfig cfg;
  cfg.gain = 1.15;
  cfg.c_ref = 0.05;
  cfg.c_floor = 0.03;
  cfg.max_force = 2.0;
  return cfg;
}

}  // namespace

// -------------------------------------------------------------------------- validate()

TEST(HeavinessConfigValidate, ThrowsOnNonPositiveCRef)
{
  HeavinessConfig cfg = test_config();
  cfg.c_ref = 0.0;
  EXPECT_THROW(cfg.validate(), std::invalid_argument);

  cfg.c_ref = -0.01;
  EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(HeavinessConfigValidate, ThrowsOnNegativeCFloor)
{
  HeavinessConfig cfg = test_config();
  cfg.c_floor = -0.001;
  EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

TEST(HeavinessConfigValidate, ThrowsOnNegativeForces)
{
  HeavinessConfig cfg = test_config();
  cfg.gain = -0.1;
  EXPECT_THROW(cfg.validate(), std::invalid_argument);

  cfg = test_config();
  cfg.max_force = -0.1;
  EXPECT_THROW(cfg.validate(), std::invalid_argument);
}

// ------------------------------------------------------------------------- monotonicity

TEST(HeavinessForce, IsMonotoneNonDecreasingInConcentration)
{
  const HeavinessConfig cfg = test_config();
  double prev = heaviness_force(cfg, -1.0);
  for (double c = -0.9; c <= 5.0; c += 0.01) {
    const double value = heaviness_force(cfg, c);
    EXPECT_GE(value, prev - 1e-12) << "force fell at concentration " << c;
    prev = value;
  }
}

// ------------------------------------------------------------------------------- floor

TEST(HeavinessForce, IsExactlyZeroAtAndBelowTheFloor)
{
  const HeavinessConfig cfg = test_config();
  EXPECT_EQ(heaviness_force(cfg, cfg.c_floor), 0.0);
  EXPECT_EQ(heaviness_force(cfg, cfg.c_floor - 0.01), 0.0);
  EXPECT_EQ(heaviness_force(cfg, 0.0), 0.0);
  EXPECT_EQ(heaviness_force(cfg, -5.0), 0.0);
}

TEST(HeavinessForce, NanInputReturnsZeroNotNan)
{
  const HeavinessConfig cfg = test_config();
  const double value = heaviness_force(cfg, std::numeric_limits<double>::quiet_NaN());
  EXPECT_FALSE(std::isnan(value));
  EXPECT_EQ(value, 0.0);
}

// --------------------------------------------------------------------- hand-checked values

TEST(HeavinessForce, MatchesHandComputedValues)
{
  const HeavinessConfig cfg = test_config();
  // gain * log1p((c - c_floor) / c_ref)
  EXPECT_NEAR(heaviness_force(cfg, 0.05), 0.39, 0.02);
  EXPECT_NEAR(heaviness_force(cfg, 0.116), 1.15, 0.02);
  EXPECT_NEAR(heaviness_force(cfg, 0.26), 2.0, 0.02);   // at the cap
}

// ----------------------------------------------------------------------------------- cap

TEST(HeavinessForce, NeverExceedsMaxForce)
{
  const HeavinessConfig cfg = test_config();
  for (const double c : {0.3, 1.0, 100.0, 1.0e9, std::numeric_limits<double>::max() / 2.0}) {
    EXPECT_LE(heaviness_force(cfg, c), cfg.max_force + 1e-9);
  }
}
