# Every source consumed so far this trial, republished on each consume

std_msgs/Header header      # stamp: the most recent consume. frame id: "map"
SourceEstimate[] sources    # index == id within THIS message only; a merge renumbers
                            # survivors, so ids are NOT a stable handle across messages
