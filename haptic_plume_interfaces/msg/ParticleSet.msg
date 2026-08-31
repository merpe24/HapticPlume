# The raw particle cloud, a visualization channel only, published at 1 Hz

std_msgs/Header header      # stamp: the belief update that produced this cloud. frame id: "map"
uint16 n_particles          # number of rows in particles; also the length of the weights
uint8 n_params              # number of columns in particles; 8 for Gaussian plume
float32[] particles         # row-major n_particles x n_params, alpha order; float32 halves the
                            # validated 1500x8 doubles from 96 kB, and RViz cannot see digit 8
float32[] weights           # normalized importance weights, summing up to 1
