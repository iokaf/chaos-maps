"""This module contains an example for the study of the logistic map"""

import numpy as np
import matplotlib.pyplot as plt

from chaotic_maps import ChaoticMap
from chaotic_maps.plotting import ChaoticMapPlot

# Define the iteration step for the logistic map

def logistic_step(x, r):
    x, = x  # Unpack the variable values
    r, = r  # Upack the parameters
    new_point = r * x * (1 - x)  # calculate the next step
    return (new_point, )  # Be sure to return a tuple

# Define the logistic map and intial conditions
logistic_map = ChaoticMap(logistic_step)

x0 = (0.4, )  # Tuple
r = (3.99, )  # Tuple

# Create a trajectory and prints its 5 first points
trajectory = logistic_map.trajectory(point=x0, pams=r, num_point=500)

for _ in range(5):
    print(next(trajectory))


# Estimate the Lyapunov exponent
lyapunov_exponent = logistic_map.approximate_lyapunov_exponents(point=x0, pams=r)
print(lyapunov_exponent)


# Initialize the plotter
logistic_map_plotter = ChaoticMapPlot(logistic_map)

# Plot a trajectory
fig = plt.figure()
fig = logistic_map_plotter.plot_trajectory(x0, r, fig=fig)


# Select a parameter range for the bifurcation diagram and plot it
parameter_range = (np.arange(0, 3.99, 0.01), )  # This still has to be a tuple

fig = logistic_map_plotter.bifurcation_diagram(x0, parameter_range, s=0.05, c='k')


# Select a parameter range for the Lyapunov exponent diagram and plot it
parameter_range = (np.arange(0.02, 3.99, 0.01), )  # This still has to be a tuple

fig = logistic_map_plotter.lyapunov_exponent_plot(x0, parameter_range)

# Create the return map diagram
r = (3.99, )
fig = logistic_map_plotter.return_map(x0, r, num_points=1500, c='k', s=0.5)

# Create the Cobweb diagram
r = (3.99, )
fig = logistic_map_plotter.cobweb_diagram(x0, r, num_points=100, c='b', linewidth=1)

plt.show()
