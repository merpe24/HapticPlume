/**
 * @file heaviness.cpp
 * @brief Implementation of the pseudo-gravity cue. See heaviness.hpp for the contract.
 *
 * @author premmm
 * @date October 9, 2026
 */

#include "haptic_plume_haptics/heaviness.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace haptic_plume_haptics
{

void HeavinessConfig::validate() const
{
  if (c_ref <= 0.0) {
    throw std::invalid_argument("HeavinessConfig: c_ref must be positive");
  }
  if (c_floor < 0.0) {
    throw std::invalid_argument("HeavinessConfig: c_floor must not be negative");
  }
  if (gain < 0.0 || max_force < 0.0) {
    throw std::invalid_argument("HeavinessConfig: forces must not be negative");
  }
}

double heaviness_force(const HeavinessConfig & cfg, double concentration)
{
  // The compensator undershoots into negative values on clean air; a signed force there
  // would read as a pull up out of nothing, so the floor is zero, not a reflection.
  // Written as a negated comparison so a NaN measurement also lands on zero instead of
  // propagating into the wrench and out to the device.
  const double excess = concentration - cfg.c_floor;
  if (!(excess > 0.0)) {
    return 0.0;
  }
  const double magnitude = cfg.gain * std::log1p(excess / cfg.c_ref);
  return std::min(magnitude, cfg.max_force);
}

}  // namespace haptic_plume_haptics
