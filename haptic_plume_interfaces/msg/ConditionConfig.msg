# The per-trial experiment condition. Reliable transient_local; exactly two consumers (D6)

# condition values
uint8 VISUAL_ONLY=0
uint8 HAPTIC_ONLY=1
uint8 VISUAL_HAPTIC=2

uint8 condition            # one of the constants above
bool prf_enabled           # false: f_prf is bitwise zero, not merely small (T3 asserts this)
float64 prf_gain           # scale applied to the PRF cue, nominal 1.0
bool well_enabled          # false: f_well is bitwise zero
float64 well_gain          # scale applied to the belief well, nominal 1.0
bool heaviness_enabled     # false: f_heaviness is bitwise zero
float64 heaviness_gain     # scale applied to the heaviness cue, nominal 1.0
bool show_belief_markers   # viz_gate: publish the belief mean and covariance to /hp/display/*
bool show_particles        # viz_gate: publish the ParticleSet cloud
bool show_gauge            # viz_gate: publish the concentration gauge
bool show_wind_arrow       # viz_gate: publish the airflow direction arrow
string trial_id            # opaque label recorded in the bag metadata, e.g. "P07_C2_S3"
int32 scenario_seed        # the seed handed to SetScenario, for exact world replay
int32 pf_seed              # the particle filter's Generator seed, for bit-exact belief replay (D7)
