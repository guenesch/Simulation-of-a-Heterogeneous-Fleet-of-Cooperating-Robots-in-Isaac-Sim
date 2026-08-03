"""
Power-law path generator.

Given a start point and an end point, this builds a smooth curve between them
using a "power-law easing" rule:

    - x is advanced LINEARLY along the path
    - y is advanced by a POWER of the progress, y proportional to t**p

Because x is linear in the progress parameter t, eliminating t shows that y is a
power function of x (y - y_start) proportional to (x - x_start)**p. With p = 2
that curve is a parabola; p = 1 gives a straight line; higher p bends it more.

NOTE: the curve is defined ONLY by the first and last points. The intermediate
points of `original_points` are not used to shape the curve — they are just the
raw data the endpoints are taken from.
"""

import numpy as np
import matplotlib.pyplot as plt


def calculate_power_law_trajectory(x_start, y_start, x_end, y_end, n_points, p_exponent):
    """
    Generate a power-law easing curve between two points.

    The path is described parametrically by a normalized progress value t that
    runs from 0 (start) to 1 (end):

        x(t) = x_start + delta_x * t            # linear in t
        y(t) = y_start + delta_y * t**p         # power law in t

    Parameters
    ----------
    x_start, y_start : float
        Coordinates of the first point (reached at t = 0).
    x_end, y_end : float
        Coordinates of the last point (reached at t = 1).
    n_points : int
        How many points to sample along the curve (>= 2).
    p_exponent : float
        The power p. p = 1 -> straight line, p = 2 -> parabola (ease-in),
        larger p -> curve starts flatter and steepens more sharply.

    Returns
    -------
    x_out, y_out : np.ndarray
        Arrays of length n_points giving the sampled curve.
    """
    # Total change to cover from start to end, along each axis.
    delta_x = x_end - x_start
    delta_y = y_end - y_start

    # Pre-allocate the output arrays.
    x_out = np.zeros(n_points)
    y_out = np.zeros(n_points)

    for i in range(n_points):
        # Normalized progress: 0.0 at the first sample, 1.0 at the last.
        t = i / (n_points - 1)

        # x moves at a constant rate (equal steps in t give equal steps in x).
        x_out[i] = x_start + (delta_x * t)

        # y moves according to the power law: slow at first, then accelerating
        # (for p > 1). This is what bends the path into a curve.
        y_out[i] = y_start + (delta_y * (t ** p_exponent))

    return x_out, y_out


# --- Define and transform the raw points -----------------------------------

# Raw (x, y) data points. Stored as float so the in-place math below is valid.
original_points = np.array([
    (0.00, 0.00), (5.00, 0.00), (10.00, 0.25), (14.83, 1.24), (19.14, 3.36),
    (22.29, 6.72), (24.23, 11.13), (25.54, 15.87), (27.58, 20.44),
    (30.25, 24.58), (33.36, 28.50)
])



original_points *= 1

# Scale every coordinate down by 15 and round to 2 decimals, so the numbers sit
# in a smaller, tidier range for plotting.
scaled_points = np.round(original_points / 15, 2)

# --- Build the smooth curve -------------------------------------------------

# Take only the first and last scaled points as the curve's endpoints.
x_start, y_start = scaled_points[0]
x_end, y_end = scaled_points[-1]

# Sample 100 points along a parabolic (p = 2) power-law path between them.
x_smooth, y_smooth = calculate_power_law_trajectory(
    x_start, y_start,
    x_end, y_end,
    n_points=100,
    p_exponent=2.0
)

# --- Report -----------------------------------------------------------------

print("Scaled input points:")
print(scaled_points)

print("\nSmoothed trajectory (first 10 of 100):")
for x, y in zip(x_smooth[:10], y_smooth[:10]):
    print(f"  ({x:.4f}, {y:.4f})")