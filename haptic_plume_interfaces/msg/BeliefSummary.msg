# Summary of the particle filter's belief over one plume, published at ~5 Hz

# status values
uint8 INIT=0
uint8 TRACKING=1
uint8 CONVERGING=2
uint8 RESET=3
uint8 DIVERGED=4

std_msgs/Header header              # stamp: the measurement that produced this belief. frame_id: "map"
float64[8] mean                     # mmse alpha, in order: x_s, y_s, z_s [m], theta [rad], Q [kg/s], v [m/s], d_y, d_z [m^2/s]
float64[8] sigma                    # per-parameter weighted standard deviation, same order and units as mean
float64 sigma_position_norm         # the consume test (< sigma_c) and well's width driver
float64 n_eff                       # effective sample size  1 / sum(w^2)
uint8 status                        # one of the constants above
uint16 n_consumed                   # len(consumed) AFTER merge, matching SourceEstimateArray;
                                    # can DECREASE when a merge collapses two - not monotone
geometry_msgs/Point haptic_target   # mne_position [m], map frame: what the well pulls toward
float64 well_confidence             # [0, 1] well depth AND width driver; 0.0 = render nothing.
                                    # PROVISIONAL: R1's third gate replaces the mapping, not the
                                    # field. C0 measured the != DIVERGED gate as on from t=0 and
                                    # wrong 66-87% of rendered time
