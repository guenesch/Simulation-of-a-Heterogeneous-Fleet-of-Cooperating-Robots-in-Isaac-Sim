"""
Turn a trajectory into differential-drive wheel speeds via the OCP.

This extends the OCP calculation with a final step for a two-wheel
(differential-drive) robot: it converts the motion of each path segment into a
left-wheel and right-wheel speed.

The OCP describe motion with dimensionless numbers:
    kn  -- nominal curvature, on a scale of [-2, 2]. |kn| < 1 is a "soft" (wide)
           curve; 1 <= |kn| <= 2 is a "sharp" curve, up to spinning on the spot
           at |kn| = 2.
    vn  -- nominal velocity in [-1, 1]; a speed scale, sign = direction.

Two vehicle constants:
    kg    -- boundary curvature: the value of |k| where the soft/sharp formulas
             switch. Raising kg (here 17.0) widens the "soft" band, so all but
             extremely tight turns are treated as soft.
    v_max -- maximum wheel speed. vs = vn * v_max is the effective service speed.

Differential-drive idea: a two-wheel robot turns by driving its wheels at
different speeds. Straight ahead means both wheels equal; a curve means one
wheel faster than the other. Steps 1-3 estimate how curved each segment is
(kn); step 4 splits the forward speed into the two wheels using that curvature.
"""

import numpy as np
import pandas as pd


def calculate_robot_kinematics(x_coords, y_coords, dt=5.0, kg=17.0, v_max=0.3):
    """
    Calculate OCP and differential-drive wheel velocities along a path.

    Parameters
    ----------
    x_coords, y_coords : sequence of float
        The sampled path, assumed evenly spaced in time.
    dt : float
        Time between consecutive samples.
    kg : float
        Boundary curvature (soft/sharp switch happens at |k| = kg).
    v_max : float
        Maximum wheel speed.

    Returns
    -------
    pandas.DataFrame
        One row per segment: curvature, normalized curvature, service speed,
        the two wheel speeds, and which regime applied.
    """
    # Stack the coordinate lists into an (N, 2) array of points.
    pts = np.column_stack((x_coords, y_coords))
    results = []

    # Loop over segments: point i to point i+1.
    for i in range(len(pts) - 1):
        # 1. LINEAR VELOCITY ----------------------------------------------
        # Finite difference between successive points; magnitude = actual speed.
        vx = (pts[i + 1, 0] - pts[i, 0]) / dt
        vy = (pts[i + 1, 1] - pts[i, 1]) / dt
        v_actual = np.sqrt(vx ** 2 + vy ** 2)

        # 2. SIGNED MENGER CURVATURE --------------------------------------
        # Curvature at an interior vertex from the circle through three
        # consecutive points: k = 4*Area / (a*b*c), where a, b, c are the
        # triangle's side lengths (this equals 1 / circumradius). Endpoints
        # have no neighbour on one side, so they are treated as straight (k=0).
        if i > 0 and i < len(pts) - 1:
            p0, p1, p2 = pts[i - 1], pts[i], pts[i + 1]
            a = np.linalg.norm(p1 - p0)
            b = np.linalg.norm(p2 - p1)
            c = np.linalg.norm(p2 - p0)
            # Triangle area via the shoelace formula.
            area = 0.5 * abs(p0[0] * (p1[1] - p2[1]) + p1[0] * (p2[1] - p0[1]) + p2[0] * (p0[1] - p1[1]))
            # Guard against a zero denominator (collinear / coincident points).
            k = (4 * area) / (a * b * c) if (a * b * c) > 1e-9 else 0.0
            # Sign the curvature: left turn (counter-clockwise) is positive.
            cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
            k *= np.sign(cross)
        else:
            k = 0.0

        # 3. REGIME SWITCHING + NORMALIZED CURVATURE ----------------------
        if abs(k) < kg:
            # Soft curve: kn is just k scaled by the boundary curvature.
            kn = k / kg
            regime = "Soft"
            v_profile = v_actual              # actual speed IS the service speed
        else:
            # Sharp curve: map the measured curvature to |kn| in [1, 2].
            sign_k = np.sign(k) if k != 0 else 1
            kn = sign_k * (2 - kg / abs(k))
            regime = "Sharp"
            # In sharp curves translation is scaled by (2 - |kn|); recover it.
            v_profile = v_actual / (2 - abs(kn)) if (2 - abs(kn)) > 1e-6 else v_actual

        # 4. WHEEL SPEEDS -------------------------------------------------
        # vn is the dimensionless speed; vs = vn * v_max the service speed.
        # (This round-trip returns vs == v_profile; vn is NOT clamped to [-1,1].)
        vn = v_profile / v_max
        vs = vn * v_max
        # Differential drive: subtract curvature from one wheel, add to the
        # other. Equal wheels when kn = 0 (straight); the gap grows with kn.
        u1 = vs * (1 - kn)   # left wheel
        u2 = vs * (1 + kn)   # right wheel

        results.append({
            'Step': i + 1,
            'Curvature_k': round(k, 4),
            'Norm_kn': round(kn, 4),
            'Linear_vs': round(vs, 4),
            'Left_u1': round(u1, 4),
            'Right_u2': round(u2, 4),
            'Regime': regime
        })

    return pd.DataFrame(results)


# --- Execution --------------------------------------------------------------

# 1/5 scaled path coordinates.
x_scaled = [0.00, 1.00, 2.00, 2.97, 3.83, 4.46, 4.85, 5.11, 5.52, 6.05, 6.67]
y_scaled = [0.00, 0.00, 0.05, 0.25, 0.67, 1.34, 2.23, 3.17, 4.09, 4.92, 5.70]

df_results = calculate_robot_kinematics(x_scaled, y_scaled, kg=17.0, v_max=0.3)

print("Differential Robot Velocity Sets (kg=17.0, vmax=0.3):")
print("-" * 70)
print(df_results.to_string(index=False))