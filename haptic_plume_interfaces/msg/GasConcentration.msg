# One chemical sensor sample, as the estimator will consume it

std_msgs/Header header          # stamp: when the sample was taken. frame id: "map"
float64 concentration           # measured gas concentration [kg/m^3], >= 0
geometry_msgs/Point position    # where the sample was taken, map frame [m]