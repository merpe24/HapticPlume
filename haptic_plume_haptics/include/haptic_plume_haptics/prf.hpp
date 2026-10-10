/**
 * @file prf.hpp
 * @brief Parametric Risk Field: obstacle repulsion that grows with speed.
 *
 * Lam et al. (IEEE SMC-A 2009) Section III-C, Eqs. (10)-(15). Risk is a scalar in [0, 1]
 * over the space around each obstacle: 1 inside a critical region the vehicle must never
 * enter, 0 outside the field, and a shifted cosine in between. The field extends forward
 * along the velocity by a braking distance and a look-ahead distance, so flying fast at a
 * pipe is felt sooner than drifting past it. Risk is max-combined over obstacles and the
 * force points radially away from whichever surface governs -- Lam's winning variant.
 *
 * DEVIATION, deliberate: Eqs. (10) and (11) are applied as a PROJECTION onto the direction
 * of the obstacle rather than by sweeping the protection zone along the velocity. Sweeping
 * needs segment-to-segment distance, which obstacle_field does not provide and which this
 * library must not reinvent; projecting reproduces Fig. 7's intent (the full braking
 * distance counts when heading straight at a surface, none when heading away) with one
 * cosine and no new geometry.
 *
 * D1 -- the reason PrfConfig embeds a ClampConfig instead of restating its fields: the
 * standoff the risk saturates at is read from the clamp's own configuration, so the
 * haptics physically cannot disagree with drone_kinematics_node about where a pipe is.
 * Note the clamp is not uniform -- pipes require drone_radius + margin, walls only
 * drone_radius -- so each surface carries its own required standoff.
 *
 * THE FLOOR IS OFF BY DEFAULT, and the reason is measured, not stylistic. At the fixed 1.0 m
 * altitude the floor is only 1.00 - min_altitude = 0.85 m away, which is INSIDE a d_min of
 * 1.50 m, so scoring it renders p_cos2(0.85/1.50) = 0.396 -> about 2 N straight up,
 * constantly, everywhere in the room, for a whole trial. The drone cannot descend anyway
 * (drone_kinematics_node zeroes the z command and holds altitude), so that is a force the
 * operator can never act on: it only eats 40% of the PRF budget and masks the cues that
 * matter (risk R6). include_floor exists so Phase F's full-3D flight can switch it back on
 * without the geometry having to be rewritten. A gtest covers both settings.
 *
 * Parameters (mirrors config/haptics_params.yaml):
 *     clamp: the drone's ClampConfig; supplies drone_radius, margin, min_altitude and bounds
 *     d_min [m]: field extent beyond the critical region, in every direction
 *     t_ahead [s]: look-ahead time; extends the field along the velocity
 *     a_max [m/s^2]: maximum deceleration; sets the braking distance
 *     max_force [N]: force rendered at risk 1.0
 *     include_floor: score the floor; false while flight is fixed-altitude
 *
 * @author premmm
 * @date October 9, 2026
 */

#ifndef HAPTIC_PLUME_HAPTICS__PRF_HPP_
#define HAPTIC_PLUME_HAPTICS__PRF_HPP_

#include <string>

#include <Eigen/Dense>

#include "haptic_plume_drone/obstacle_field.hpp"

namespace haptic_plume_haptics
{

/// Tunables of the risk field. Embeds the drone's clamp configuration by value (D1).
struct PrfConfig
{
  haptic_plume_drone::ClampConfig clamp;
  double d_min;        // [m] field extent beyond the critical region
  double t_ahead;      // [s] look-ahead time
  double a_max;        // [m/s^2] maximum deceleration
  double max_force;    // [N] force at risk 1.0
  bool include_floor;  // score the floor at all; false while flight is fixed-altitude

  /**
   * @brief Required standoff from a pipe, the radius at which risk saturates.
   * @return drone_radius + margin [m], exactly what clamp_to_free_space demands
   */
  double r_pz() const;

  /**
   * @brief Reject a configuration that would make the field degenerate.
   *
   * d_min must be positive or the field collapses to the critical region at hover and the
   * operator feels nothing until the clamp is already sliding (risk R10).
   *
   * @throw std::invalid_argument if any parameter is non-positive
   */
  void validate() const;
};

/// Risk at one point and the direction it pushes.
struct Risk
{
  double value;               // [0, 1] max-combined over every surface
  Eigen::Vector3d direction;  // unit, away from the governing surface; zero when value is 0
  std::string surface;        // governing surface's name; empty outside the field. Short
                              // enough for the small-string optimization, so no allocation
};

/**
 * @brief Shifted-cosine risk function -- Lam Eq. (15).
 *
 * Chosen over the linear and plain-cosine forms of Eqs. (13)-(14) because its gradient is
 * zero at both ends, so the force has no corner where the field begins or saturates.
 *
 * @param x Normalized distance d/d0, clamped to [0, 1] by the caller
 * @return 1 at x = 0 falling smoothly to 0 at x = 1
 */
double p_cos2(double x);

/**
 * @brief Distance needed to stop from a given speed -- Lam Eq. (10).
 * @param speed Current speed [m/s]
 * @param a_max Maximum deceleration [m/s^2]
 * @return Braking distance [m]
 */
double braking_distance(double speed, double a_max);

/**
 * @brief Distance covered during the operator's reaction window -- Lam Eq. (11).
 * @param speed Current speed [m/s]
 * @param t_ahead Look-ahead time [s]
 * @return Look-ahead distance [m]
 */
double look_ahead(double speed, double t_ahead);

/**
 * @brief Max-combined risk at a point, given the velocity it is moving with -- Eq. (12).
 *
 * Every pipe, the floor and the four room walls are scored; the largest wins and lends its
 * outward normal as the push direction. Max-combination rather than summation is what keeps
 * the force bounded by max_force where surfaces meet, such as the shared main_run/riser
 * endpoint.
 *
 * @param field  World geometry, as loaded from obstacles.yaml
 * @param cfg    Field tunables; must have passed validate()
 * @param p      Query position [m], map frame
 * @param v      Velocity [m/s], map frame; the zero vector is valid and gives the hover field
 * @return Risk in [0, 1] with its direction and governing surface
 */
Risk risk_at(
  const haptic_plume_drone::ObstacleField & field, const PrfConfig & cfg,
  const Eigen::Vector3d & p, const Eigen::Vector3d & v);

/**
 * @brief Repulsive force from the risk field.
 * @param field World geometry, as loaded from obstacles.yaml
 * @param cfg   Field tunables; must have passed validate()
 * @param p     Query position [m], map frame
 * @param v     Velocity [m/s], map frame
 * @return Force [N], magnitude risk * max_force, pointing away from the governing surface
 */
Eigen::Vector3d prf_force(
  const haptic_plume_drone::ObstacleField & field, const PrfConfig & cfg,
  const Eigen::Vector3d & p, const Eigen::Vector3d & v);

}  // namespace haptic_plume_haptics

#endif  // HAPTIC_PLUME_HAPTICS__PRF_HPP_
