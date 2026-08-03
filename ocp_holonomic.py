"""
Recover Omni-Curve-Parameters (OCP) from a trajectory.

The OCP describe a vehicle's motion with three dimensionless numbers:

    kn  -- nominal curvature, on a scale of [-2, 2]. It captures the ratio of
           rotation to translation. |kn| < 1 is a "soft" (wide) curve;
           1 <= |kn| <= 2 is a "sharp" curve, up to spinning on the spot at
           |kn| = 2.
    beta -- heading: the direction the vehicle is travelling, measured from the
           x-axis. (Vehicle and world frame coincide here, so this is the slip
           angle directly.)
    vn  -- nominal velocity in [-1, 1]; a pure speed scale, sign = direction.

Two vehicle constants tie these to physical units:
    kg    -- boundary curvature: the curvature at |kn| = 1, where the soft and
             sharp formulas switch. A fixed, vehicle-specific value.
    v_max -- maximum roll speed. vs = vn * v_max is the effective service speed
             used in the motion equations.

WHAT THIS SCRIPT DOES: for each segment of a sampled (x, y) path it estimates
the OCP (kn, beta, vn) and then expands them  motion
vector [vx, vy, omega].
    points -> estimated OCP -> reconstructed motion.
"""

import numpy as np
import pandas as pd


def calculate_full_ocp_state(x_coords, y_coords, dt=5.0, kg=1.0, v_max=1.5):
    """
    Walk along a path and, for each step, estimate the OCP and rebuild the
    corresponding [vx, vy, omega] motion vector.

    Parameters
    ----------
    x_coords, y_coords : sequence of float
        The sampled path, assumed evenly spaced in time.
    dt : float
        Time between consecutive samples (turns position differences into
        velocities).
    kg : float
        Boundary curvature -- the soft/sharp switch happens at |k| = kg.
    v_max : float
        Vehicle maximum speed, used to normalize / de-normalize the speed.

    Returns
    -------
    pandas.DataFrame
        One row per segment with the estimated OCP and the rebuilt motion.
    """
    # Stack the two coordinate lists into an (N, 2) array of points.
    pts = np.column_stack((x_coords, y_coords))
    results = []

    # Loop over segments: point i to point i+1.
    for i in range(len(pts) - 1):
        # 1. WORLD-FRAME VELOCITY -----------------------------------------
        # Finite difference between successive points gives this segment's
        # velocity; its magnitude is the actual speed.
        vx_raw = (pts[i + 1, 0] - pts[i, 0]) / dt
        vy_raw = (pts[i + 1, 1] - pts[i, 1]) / dt
        v_actual = np.sqrt(vx_raw ** 2 + vy_raw ** 2)

        # 2. SIGNED MENGER CURVATURE --------------------------------------
        # Curvature at an interior vertex from the circle through three
        # consecutive points: k = 4*Area / (a*b*c), where a, b, c are the
        # triangle's side lengths (this equals 1 / circumradius). Endpoints
        # have no neighbour on one side, so they are treated as straight (k=0).
        if i > 0 and i < len(pts) - 1:
            p_prev, p_curr, p_next = pts[i - 1], pts[i], pts[i + 1]
            a = np.linalg.norm(p_curr - p_prev)
            b = np.linalg.norm(p_next - p_curr)
            c = np.linalg.norm(p_next - p_prev)
            # Triangle area via the shoelace formula.
            area = 0.5 * abs(p_prev[0] * (p_curr[1] - p_next[1]) +
                             p_curr[0] * (p_next[1] - p_prev[1]) +
                             p_next[0] * (p_prev[1] - p_curr[1]))
            # Guard against a zero denominator (collinear / coincident points).
            k = (4 * area) / (a * b * c) if (a * b * c) > 1e-9 else 0.0
            # Sign the curvature: left turn (counter-clockwise) is positive.
            # The 2D cross product of the two step vectors gives turn direction.
            cross = (p_curr[0] - p_prev[0]) * (p_next[1] - p_curr[1]) - \
                    (p_curr[1] - p_prev[1]) * (p_next[0] - p_curr[0])
            k *= np.sign(cross)
        else:
            k = 0.0

        # 3. HEADING (beta) -----------------------------------------------
        # Direction of travel of this segment. With the vehicle and world frame
        # aligned, this is the slip angle.
        beta = np.arctan2(vy_raw, vx_raw)

        # 4. NORMALIZED CURVATURE + BASE SPEED ----------------------------
        if abs(k) < kg:
            # Soft curve: kn is just k scaled by the boundary curvature.
            kn = k / kg
            vs_temp = v_actual              # here actual speed IS the service speed
            regime = "Soft"
        else:
            # Sharp curve: map the measured curvature to |kn| in [1, 2] via the
            # inverse of the radius relation R = (2*sgn(kn) - kn)/kg.
            sign_k = np.sign(k) if k != 0 else 1
            kn = sign_k * (2 - kg / abs(k))
            # In sharp curves the translation obeys v = vs*(2 - |kn|); recover vs.
            vs_temp = v_actual / (2 - abs(kn)) if (2 - abs(kn)) > 1e-6 else v_actual
            regime = "Sharp"

        # 5. NOMINAL vs SERVICE SPEED -------------------------------------
        # vn is the dimensionless speed; vs = vn * v_max is the service speed.
        # (This round-trip returns vs == vs_temp; vn is NOT clamped to [-1, 1].)
        vn = vs_temp / v_max
        vs = vn * v_max

        # 6. REBUILD THE MOTION VECTOR [vx, vy, omega] --------------------
        # Choose the branch by regime. omega is the yaw (rotation) rate.
        if abs(kn) < 1:
            # Soft regime.
            vx = vs * np.cos(beta)
            vy = vs * np.sin(beta)
            omega = vs * kg * kn
        else:
            # Sharp regime: translation is scaled by (2 - |kn|); rotation is
            # fixed in magnitude and only its sign follows kn.
            multiplier = (2 - abs(kn))
            vx = vs * multiplier * np.cos(beta)
            vy = vs * multiplier * np.sin(beta)
            omega = vs * kg * np.sign(kn)

        results.append({
            'Step': i + 1,
            'kn': round(kn, 4),
            'beta_deg': round(np.degrees(beta), 2),
            'vn': round(vn, 4),
            'vs': round(vs, 4),
            'vx': round(vx, 4),
            'vy': round(vy, 4),
            'omega': round(omega, 4),
            'Regime': regime
        })

    return pd.DataFrame(results)


# --- Data execution ---------------------------------------------------------

# Scaled path coordinates to analyze.
x_scaled = [0.00, 1.00, 2.00, 2.97, 3.83, 4.46, 4.85, 5.11, 5.52, 6.05, 6.67]
y_scaled = [0.00, 0.00, 0.05, 0.25, 0.67, 1.34, 2.23, 3.17, 4.09, 4.92, 5.70]

# Vehicle / sampling parameters.
KG_VAL = 1.0      # boundary curvature
VMAX_VAL = 1.5    # maximum speed
DT_VAL = 5.0      # time step between samples

df_final = calculate_full_ocp_state(x_scaled, y_scaled, dt=DT_VAL, kg=KG_VAL, v_max=VMAX_VAL)

print("OCP Final State Vectors (vx, vy, omega):")
print("-" * 80)
print(df_final[['Step', 'kn', 'beta_deg', 'vs', 'vx', 'vy', 'omega', 'Regime']].to_string(index=False))