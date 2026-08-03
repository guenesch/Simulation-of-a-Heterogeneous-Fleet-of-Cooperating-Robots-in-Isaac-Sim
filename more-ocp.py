"""
Cubic-spline path smoothing.

Unlike the power-law version (which used only the two endpoints), this fits a
smooth curve that passes through ALL of the points. Two ideas make it work:

1. PARAMETRIC form. Instead of fitting y as a function of x, both x and y are
   fitted as functions of a shared progress parameter t. This matters because
   a plain y = f(x) spline needs x to be strictly increasing -- but after the
   sign flip below, x runs from 0 down to -2.22 (decreasing), and paths can
   even curl back on themselves. Parametrizing against a t that always
   increases sidesteps that restriction entirely.

2. CHORD-LENGTH parameter. t is built from the cumulative straight-line
   distance between consecutive points, then normalized to [0, 1]. So points
   that are far apart get more "room" in t than points that are close together,
   which keeps the curve from bulging where the data is unevenly spaced.

The result is a piecewise-cubic curve with continuous first and second
derivatives at every knot -- that continuity is what makes it look smooth.
"""

import numpy as np
from scipy.interpolate import CubicSpline


def generate_spline_trajectory(x_orig, y_orig):
    """
    Fit a chord-length parametric cubic spline through the given points and
    resample it at 21 evenly spaced parameter values.

    Parameters
    ----------
    x_orig, y_orig : np.ndarray
        Coordinates of the points to interpolate. The curve passes through
        every one of them, in order.

    Returns
    -------
    x_out, y_out : np.ndarray
        21 points sampled along the smooth curve.
    """
    # 1. CHORD-LENGTH PARAMETER -------------------------------------------
    # Walk along the points and accumulate the straight-line (Euclidean)
    # distance from the start. cumulative_distance[i] = path length up to
    # point i.
    n = len(x_orig)
    cumulative_distance = np.zeros(n)
    for i in range(n - 1):
        dx = x_orig[i+1] - x_orig[i]
        dy = y_orig[i+1] - y_orig[i]
        segment_dist = np.sqrt(dx**2 + dy**2)          # length of this segment
        cumulative_distance[i+1] = cumulative_distance[i] + segment_dist

    # Rescale so the parameter runs 0.0 at the first point to 1.0 at the last.
    total_length = cumulative_distance[-1]
    t_norm = cumulative_distance / total_length

    # 2. TWO 1-D CUBIC SPLINES --------------------------------------------
    # One spline maps t -> x, another maps t -> y. Each is a set of cubic
    # polynomials joined so that value, slope, and curvature are continuous
    # across the joins (knots). Both pass exactly through the input points.
    spline_X = CubicSpline(t_norm, x_orig)
    spline_Y = CubicSpline(t_norm, y_orig)

    # 3. RESAMPLE ----------------------------------------------------------
    # Evaluate the curve at 21 equally spaced parameter values (t = 0, 0.05,
    # ..., 1.0), giving 20 segments. NOTE: equal steps in t are NOT exactly
    # equal steps in distance along the curve -- they are close, because t is
    # chord-length based, but not perfectly uniform in arc length.
    x_out = np.zeros(21)
    y_out = np.zeros(21)
    for j in range(21):
        t_target = j / 20
        x_out[j] = spline_X(t_target)
        y_out[j] = spline_Y(t_target)

    return x_out, y_out


# --- Data preparation -------------------------------------------------------

# Raw (x, y) points. Kept as float so the in-place sign flip below is valid.
original_points = np.array([
    (0.00, 0.00), (5.00, 0.00), (10.00, 0.25), (14.83, 1.24), (19.14, 3.36),
    (22.29, 6.72), (24.23, 11.13), (25.54, 15.87), (27.58, 20.44),
    (30.25, 24.58), (33.36, 28.50)
])

original_points *= -1                               # mirror through the origin (flip x and y)
scaled_points = np.round(original_points / 15, 2)   # shrink coordinates and round

# Split into separate x and y arrays for the spline routine.
x_orig = scaled_points[:, 0]
y_orig = scaled_points[:, 1]

# --- Run the spline ---------------------------------------------------------

x_out, y_out = generate_spline_trajectory(x_orig, y_orig)

# --- Report -----------------------------------------------------------------

print("Scaled input points (11):")
for pt in scaled_points:
    print(f"  ({pt[0]:.2f}, {pt[1]:.2f})")

print("\nSpline output points (21):")
for x, y in zip(x_out, y_out):
    print(f"  ({x:.4f}, {y:.4f})")