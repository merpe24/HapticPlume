/**
 * @file composer_core.hpp
 * @brief Arbitration of the three haptic cues into one bounded, pop-free force.
 *
 * The cue library (prf, well, heaviness) is pure: each call depends only on its arguments.
 * This is the stateful half. It gates each cue on the trial's ConditionConfig (D6), fades
 * the well in and out instead of switching it (risk R5), enforces the 8 N budget with the
 * PRF keeping priority (risk R6), and caps how fast the total may change.
 *
 * Zero rclcpp, so it is testable with plain gtest; force_composer_node is a thin shell that
 * fills ComposerInputs from topics and copies ComposerOutput into a HapticCue.
 *
 * Parameters (mirrors config/haptics_params.yaml):
 *     total_max [N]: budget for the summed force; the PRF is never scaled to meet it
 *     slew_max [N/s]: largest change of the total force per second
 *     well_fade_out [s]: time for the well envelope to fall from 1 to 0
 *     well_fade_in [s]: time for the well envelope to rise from 0 to 1
 *     odom_timeout [s]: older pose than this zeroes every cue
 *     belief_timeout [s]: older belief than this fades the well out
 *     measurement_timeout [s]: older concentration than this zeroes heaviness
 *
 * @author premmm
 * @date October 14, 2026
 */

#ifndef HAPTIC_PLUME_HAPTICS__COMPOSER_CORE_HPP_
#define HAPTIC_PLUME_HAPTICS__COMPOSER_CORE_HPP_

#include <cstdint>
 
#include <Eigen/Dense>

#include "haptic_plume_drone/obstacle_field.hpp"
#include "haptic_plume_haptics/heaviness.hpp"
#include "haptic_plume_haptics/prf.hpp"
#include "haptic_plume_haptics/well.hpp"

namespace haptic_plume_haptics
{

// BeliefSummary.status values. Restated because this library must not depend on the
// message package; MUST match haptic_plume_interfaces/msg/BeliefSummary.msg.
constexpr std::uint8_t kStatusReset = 3;

// HapticCue.active_mask bits; MUST match HapticCue.msg.
constexpr std::uint8_t kCuePrf = 1;
constexpr std::uint8_t kCueWell = 2;
constexpr std::uint8_t kCueHeaviness = 4;

/// Every tunable of the composer, with the three cue configs embedded by value.
struct ComposerConfig
{
  PrfConfig prf;
  WellConfig well;
  HeavinessConfig heaviness;
  double total_max;            // [N] summed-force budget
  double slew_max;             // [N/s] largest change of the total per second
  double well_fade_out;        // [s] envelope 1 -> 0
  double well_fade_in;         // [s] envelope 0 -> 1
  double odom_timeout;         // [s]
  double belief_timeout;       // [s]
  double measurement_timeout;  // [s]

  /**
   * @brief Validate the embedded cue configs and the composer's own values.
   * @throw std::invalid_argument if any time or force limit is non-positive
   */
  void validate() const;
};

/// One tick's worth of input, filled by the node from the latest message of each topic.
struct ComposerInputs
{
  // /hp/odom -- the twist is already world frame (frames.md 6.2), do not rotate it
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Vector3d velocity{Eigen::Vector3d::Zero()};
  double odom_age{1e9};  // [s] since arrival; huge = never received

  // /hp/gas/measurement_compensated
  double concentration{0.0};
  double measurement_age{1e9};

  // /hp/belief/summary
  Eigen::Vector3d haptic_target{Eigen::Vector3d::Zero()};
  double well_confidence{0.0};
  std::uint8_t status{0};
  std::uint16_t n_consumed{0};
  double belief_age{1e9};

  // /hp/condition -- until one arrives every cue counts as disabled
  bool have_condition{false};
  bool prf_enabled{false};
  double prf_gain{1.0};
  bool well_enabled{false};
  double well_gain{1.0};
  bool heaviness_enabled{false};
  double heaviness_gain{1.0};
};

/// One tick's output; field for field what HapticCue.msg carries.
struct ComposerOutput
{
  Eigen::Vector3d total{Eigen::Vector3d::Zero()};        // after budget and slew cap
  Eigen::Vector3d f_prf{Eigen::Vector3d::Zero()};        // before rescale
  Eigen::Vector3d f_well{Eigen::Vector3d::Zero()};       // before rescale
  Eigen::Vector3d f_heaviness{Eigen::Vector3d::Zero()};  // before rescale
  double risk{0.0};
  std::uint8_t active_mask{0};
  bool saturated{false};
};

/**
 * @brief The stateful arbiter. Holds the well envelope and the last total, nothing else.
 */
class Composer
{
public:
  /**
   * @brief Bind the composer to the world geometry and its tunables.
   * @param field World geometry; must outlive the composer
   * @param cfg   Tunables; validated here
   * @throw std::invalid_argument if cfg fails validate()
   */
  Composer(const haptic_plume_drone::ObstacleField & field, const ComposerConfig & cfg);

  /**
   * @brief Advance one tick.
   * @param in Latest inputs
   * @param dt Nominal tick period [s], never a measured wall-clock delta (D7)
   * @return The composed force and its per-cue breakdown
   */
  ComposerOutput tick(const ComposerInputs & in, double dt);

  /// Zero the envelope and the slew state; called between trials (Phase E).
  void reset();

private:
  const haptic_plume_drone::ObstacleField & field_;
  ComposerConfig cfg_;
  double envelope_{0.0};                                // well gain in [0, 1]
  Eigen::Vector3d held_well_{Eigen::Vector3d::Zero()};  // last well force, kept during fade-out
  Eigen::Vector3d last_total_{Eigen::Vector3d::Zero()};  // for the slew cap
};

}  // namespace haptic_plume_haptics

#endif  // HAPTIC_PLUME_HAPTICS__COMPOSER_CORE_HPP_