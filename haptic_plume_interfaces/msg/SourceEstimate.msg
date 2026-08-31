# One consumed plume, its eight-parameter alpha unpacked into named fields

geometry_msgs/Point position    # source location x_s, y_s, z_s [m]. map frame
float64 emission_rate           # Q [kg/s]
float64 wind_direction          # theta [rad], the direction the wind blows toward
float64 wind_speed              # v [m/s]
float64 d_y                     # cross-wind diffusion constant [m^2/s]
float64 d_z                     # vertical diffusion constant [m^2/s]
float64 sigma_position_norm     # ||sigma_pos|| [m] at the moment of consume: how tight the
                                # belief was when should_consume() accepted it
uint16 id                       # index within the parent SourceEstimateArray, 0-based;
                                # a merge renumbers survivors, so this is not a stable handle
