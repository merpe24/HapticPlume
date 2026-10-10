/**
 * @file well.hpp
 * @brief Belief well: a bearing bias whose width and depth carry the filter's confidence.
 *
 * This is the project's haptic novelty -- rendering a Bayesian belief's CONFIDENCE as force,
 * after Menelas et al. (2009) gravity wells, under the rate-control scheme that a 10 cm
 * device and a 20 m world force (Crossan navigation, not a positional spring). A wide well
 * is a weak pull felt across almost every heading, saying "roughly that way"; a narrow one
 * is a sharp detent on a precise bearing. Both ends are continuous in the confidence, so
 * there is no switch-on transition to fade and risk R5 has no pop to render.
 *
 * WHY THERE IS NO ON/OFF GATE (risk R1's third gate, closed by decision 2026-10-09). Study
 * C0 measured the only gate that existed, status != DIVERGED, as on from t = 0 and wrong for
 * 66-87% of rendered time. Three attempts to find a better one failed. The way out was in
 * C0's own numbers: scored at 10 deg the pull is wrong 66.6% of the time and never becomes
 * usable on 40% of hunts, but at 30 deg it is wrong 47.4% and never-usable on 0.0%. Direction
 * survives where position does not -- even wrong declarations point only 9.8 deg off, because
 * the error runs along the plume axis. So width is the quantity that must scale, not a gate.
 *
 * CONFIDENCE 0 IS RESERVED. The continuous band is (0, 1]; exactly 0.0 means render nothing,
 * which is how plume_estimator_node's DIVERGED case and GuidanceTarget.valid keep a hard
 * off-switch without reintroducing a threshold. Do not map 0.0 into depth_min.
 *
 * The force stays 3D here and the composer allocates it to the device's axes (C5); at the
 * fixed 1.0 m altitude the bearing comes out horizontal on its own.
 *
 * Parameters (mirrors config/haptics_params.yaml):
 *     width_min [rad]: angular std at confidence 1.0, the narrow detent
 *     width_max [rad]: angular std as confidence approaches 0, the wide "roughly that way"
 *     depth_min [N]: force at the diffuse end of the band
 *     depth_max [N]: force at confidence 1.0; risk R6 budgets the well 2 N
 *     target_margin [m]: standoff kept beyond r_pz so the clamped target is not itself at risk 1
 *     speed_threshold [m/s]: below this the heading is undefined and full depth is rendered
 *
 * @author premmm
 * @date October 9, 2026
 */

#ifndef HAPTIC_PLUME_HAPTICS__WELL_HPP_
#define HAPTIC_PLUME_HAPTICS__WELL_HPP_

#include <Eigen/Dense>

#include "haptic_plume_drone/obstacle_field.hpp"
#include "haptic_plume_haptics/prf.hpp"

namespace haptic_plume_haptics
{

/// Tunables of the belief well. Mirrors config/haptics_params.yaml.
struct WellConfig
{
  double width_min;        // [rad] angular std at confidence 1.0
  double width_max;        // [rad] angular std as confidence -> 0
  double depth_min;        // [N] force at the diffuse end
  double depth_max;        // [N] force at confidence 1.0
  double target_margin;    // [m] extra standoff for the clamped target
  double speed_threshold;  // [m/s] below this the heading is undefined

  /**
   * @brief Reject a configuration that would inverse the confidence mapping.
   * @throw std::invalid_argument if width_min > width_max, depth_min > depth_max,
   *     any width is non-positive, or target_margin is negative
   */
  void validate() const;
};

/**
 * @brief Angular half-width of the well at a given confidence.
 * @param cfg Well tunables
 * @param confidence BeliefSummary.well_confidence, in [0, 1]
 * @return Angular standard deviation [rad], falling from width_max to width_min
 */
double well_width(const WellConfig & cfg, double confidence);

/**
 * @brief Peak force of the well at a given confidence.
 * @param cfg Well tunables
 * @param confidence BeliefSummary.well_confidence, in [0, 1]
 * @return Force [N], rising from depth_min to depth_max; exactly 0 when confidence is 0
 */
double well_depth(const WellConfig & cfg, double confidence);

/**
 * @brief Push a well target out of every obstacle's standoff surface -- D4.
 *
 * NOT an edge case: commit 1b7557b places leaks on the pipe SURFACE, so the believed source
 * is inside the protection zone on every trial and this clamp fires every time. A leak at
 * x = 9.7 on the main_run axis at x = 10.0 (pipe radius 0.30, r_pz 0.427) clamps to
 * x = 10.0 - (0.30 + 0.427 + target_margin), i.e. onto the near face of the pipe, which is
 * the closest place the operator can actually be flown to. Risk R2 is defused here rather
 * than damped later: an unclamped target plus the (1 - risk) scaling is exactly the standing
 * fight between the well and the PRF that would show up as a limit cycle.
 *
 * Mirrors clamp_to_free_space's multi-pass structure because main_run and riser share an
 * endpoint, so one pass can push out of one capsule into the other.
 *
 * @param field    World geometry, as loaded from obstacles.yaml
 * @param standoff Required clearance from every capsule surface [m]
 * @param target   Believed source position [m], map frame
 * @return A target at least standoff clear of every capsule [m]
 */
Eigen::Vector3d clamp_target_outward(
  const haptic_plume_drone::ObstacleField & field, double standoff,
  const Eigen::Vector3d & target);

/**
 * @brief The well force: a confidence-scaled bias toward the believed source.
 *
 * Magnitude is depth(confidence) * exp(-psi^2 / 2 width(confidence)^2) * (1 - risk), where
 * psi is the angle between the commanded direction and the bearing to the clamped target.
 * The (1 - risk) factor is D4's requirement that the PRF always wins near a surface.
 *
 * @param field      World geometry, as loaded from obstacles.yaml
 * @param prf_cfg    Risk-field tunables, for r_pz
 * @param cfg        Well tunables; must have passed validate()
 * @param p          Drone position [m], map frame
 * @param v          Commanded or current velocity [m/s], map frame
 * @param target     BeliefSummary.haptic_target, the MNE position [m], map frame
 * @param confidence BeliefSummary.well_confidence, in [0, 1]; exactly 0 renders nothing
 * @param risk       Risk from risk_at(), in [0, 1]
 * @return Force [N], pointing at the clamped target
 */
Eigen::Vector3d well_force(
  const haptic_plume_drone::ObstacleField & field, const PrfConfig & prf_cfg,
  const WellConfig & cfg, const Eigen::Vector3d & p, const Eigen::Vector3d & v,
  const Eigen::Vector3d & target, double confidence, double risk);

}  // namespace haptic_plume_haptics

#endif  // HAPTIC_PLUME_HAPTICS__WELL_HPP_
