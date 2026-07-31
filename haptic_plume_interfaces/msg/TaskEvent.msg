# A discrete, timestamped event for logging or reacting to

# Event type constants
string COLLISION="COLLISION"        # drone collided on obstacles (rising edge only)
string TRIAL_START="TRIAL_START"    # Phase E
string TRIAL_END="TRIAL_END"        # Phase E

std_msgs/Header header          # stamp: when the event occurred
string event_type               # one of the constants above
geometry_msgs/Point position    # where it occurred, map frame [m]
string detail                   # free text, e.g. the obstacle segment name