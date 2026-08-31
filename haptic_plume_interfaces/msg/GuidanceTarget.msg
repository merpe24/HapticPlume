# Where the estimator suggests flying next, published at 2 Hz

# source values, deciding which target D4 is rendering
uint8 MI_WAYPOINT=0
uint8 STANDOFF_BELIEF=1

std_msgs/Header header      # stamp: the belief this target was derived from. frame id: "map"
geometry_msgs/Point target  # suggested position [m], map frame
float64 info_gain           # mutual information of the chosen waypoint [nats]; 0.0 when source is STANDOFF_BELIEF
uint8 source                # one of the constants above: search phase vs converged
bool valid                  # false: the belief is unusable, render no well; the composer must not re-derive the gate
