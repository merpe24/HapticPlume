# One composed force command and its per-cue breakdown, published at 200 Hz

# active_mask bits, OR-combined
uint8 CUE_PRF=1
uint8 CUE_WELL=2
uint8 CUE_HEAVINESS=4

std_msgs/Header header             # stamp: the composer tick. frame_id: "map"
geometry_msgs/Vector3 total        # commanded force [N] after arbitration, the 8 N budget and the slew cap
geometry_msgs/Vector3 f_prf        # obstacle-repulsion contribution [N], before PRF-priority rescale
geometry_msgs/Vector3 f_well       # belief-well contribution [N], before rescale
geometry_msgs/Vector3 f_heaviness  # concentration-heaviness contribution [N], before rescale
float64 risk                       # max-combined PRF risk P, in [0, 1]
uint8 active_mask                  # OR of the CUE_* bits contributing on this tick
bool saturated                     # true when the budget rescaled any cue this tick (R6 logging)
