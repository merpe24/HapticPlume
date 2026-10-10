/**
 * @file heaviness.hpp
 * @brief Pseudo-gravity: the drone feels heavier where the gas is thicker.
 *
 * Menelas et al. (2009) pseudo-gravity, rendered on the LAG-COMPENSATED concentration (D5).
 * The raw sensor trails truth by ~2.2 s and peaks at ~65% of it, so rendering the raw signal
 * would put the heaviest point downwind of the leak -- the same failure mode that residual
 * lag causes in the estimator. Log scaling handles the dynamic range (risk R7): the plume is
 * ~0.26 kg/m^3 at the source and ~0.05 nine metres downwind, against a compensated clean-air
 * noise band of +/-0.0149.
 *
 * THE CONSTANTS ARE GLOBAL AND MUST STAY GLOBAL. Normalizing per scenario would make the
 * same concentration feel different in different trials, which is an uncontrolled confound in
 * an experiment whose dependent variable is teleoperation modality. D5 forbids it.
 *
 * This cue is the one that is deliberately NOT belief-dependent (risk R1): it reads the
 * measurement, so a starved or diverged filter cannot corrupt it. Together with the PRF,
 * which reads only geometry, it is what the operator still has when the well goes quiet.
 *
 * Parameters (mirrors config/haptics_params.yaml):
 *     gain [N]: scale k of the logarithm; the force at c - c_floor = c_ref * (e - 1)
 *     c_ref [kg/m^3]: concentration scale c_0 inside the logarithm
 *     c_floor [kg/m^3]: subtracted before the logarithm, so clean air renders exactly zero
 *     max_force [N]: cap; risk R6 budgets heaviness 2 N
 *
 * @author premmm
 * @date October 9, 2026
 */

#ifndef HAPTIC_PLUME_HAPTICS__HEAVINESS_HPP_
#define HAPTIC_PLUME_HAPTICS__HEAVINESS_HPP_

namespace haptic_plume_haptics
{

/// Tunables of the heaviness cue. Mirrors config/haptics_params.yaml.
struct HeavinessConfig
{
  double gain;       // [N] scale of the logarithm
  double c_ref;      // [kg/m^3] concentration scale inside the logarithm
  double c_floor;    // [kg/m^3] noise floor subtracted before the logarithm
  double max_force;  // [N] cap

  /**
   * @brief Reject a configuration that would make the mapping undefined.
   * @throw std::invalid_argument if c_ref is non-positive, or c_floor or any force is
   *     negative
   */
  void validate() const;
};

/**
 * @brief Heaviness magnitude for a compensated concentration -- D5.
 *
 * k * log1p((c - c_floor) / c_ref), capped, and zero at or below the floor.
 *
 * A negative input is clamped to zero rather than reflected: the inverse feedforward
 * compensator undershoots on clean air (measured +0.0008 +/- 0.0149), and a signed force
 * there would read as a pull upward out of nothing.
 *
 * WHY c_floor EXISTS. Without it the logarithm's compression works against the cue: at
 * c_ref = 0.015 a one-sigma noise sample renders 0.48 N, a quarter of the budget, so clean
 * air would hum instead of going quiet. Subtracting roughly two sigma first makes silence
 * mean "outside the plume", which is itself information the operator can use.
 *
 * @param cfg Heaviness tunables; must have passed validate()
 * @param concentration Lag-compensated concentration [kg/m^3]; may be negative or NaN
 * @return Magnitude [N] in [0, max_force], monotone non-decreasing in concentration
 */
double heaviness_force(const HeavinessConfig & cfg, double concentration);

}  // namespace haptic_plume_haptics

#endif  // HAPTIC_PLUME_HAPTICS__HEAVINESS_HPP_
