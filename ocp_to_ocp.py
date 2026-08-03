"""
OCP-to-OCP formation conversion: turn a LEADER robot's OCP into a FOLLOWER's OCP.

Several robots hold a fixed formation (e.g. a leader with followers offset by
various (dx, dy) distances). Because the formation is rigid, every robot turns about
the SAME center at the SAME rate, but a robot offset in 2D space sits at a
different radius from that center, so it must drive at a different speed and
curvature. This script computes those follower commands step-by-step.

The OCP describe motion with three dimensionless numbers:
    kn   -- nominal curvature in [-2, 2]. |kn| < 1 = gentle curve; 1..2 = sharp,
            up to spinning in place at |kn| = 2.
    beta -- drift / heading angle (degrees at the interface, radians internally).
    vn   -- normalized speed in [-1, 1].

Vehicle constants: kappa_g (boundary curvature, the gentle/sharp switch) and
v_max (max speed); the service speed is v_s = vn * v_max.

The conversion routes through velocities as a bridge, because the rigid-body
link between leader and follower is simplest in plain velocity terms:
    leader OCP -> leader velocity -> follower velocity -> follower OCP
"""

import numpy as np


def convert_S_curve_step(kn_L, beta_L, vn_L, dx, dy, kappa_g=1.0, v_max=1.5):
    """
    Converts a Leader's OCP state (kn, beta, vn) into a Follower's OCP state
    for a single time step, given 2D spatial offsets (dx, dy) in the Leader frame.

    Parameters:
        kn_L    : Leader nominal curvature [-2.0, 2.0]
        beta_L  : Leader drift angle (degrees)
        vn_L    : Leader normalized speed ratio [-1.0, 1.0]
        dx      : Forward/longitudinal offset of follower (meters)
        dy      : Lateral offset of follower (meters, >0 for Left, <0 for Right)
        kappa_g : Boundary curvature parameter (default 1.0)
        v_max   : Maximum attainable linear velocity (default 1.5 m/s)

    Returns:
        (kn_F, beta_deg_F, vn_F) : Tuple of follower OCP values rounded for control output.
    """

    # =========================================================================
    # 1. DECODE LEADER OCP -> BODY-FRAME VELOCITY (vx_L, vy_L, omega_L)
    # =========================================================================
    # Convert normalized speed to true physical service speed v_s (m/s)
    v_s_L = vn_L * v_max
    beta_rad_L = np.radians(beta_L)

    if abs(kn_L) < 1.0:
        # --- Gentle Curve Regime (|kn| < 1) ---
        # Rotational speed scales linearly with curvature kn.
        omega_L = kn_L * v_s_L * kappa_g
        vx_L = v_s_L * np.cos(beta_rad_L)
        vy_L = v_s_L * np.sin(beta_rad_L)
    else:
        # --- Sharp Curve Regime (1 <= |kn| <= 2) ---
        # Yaw rate saturates; translation speed drops linearly with (2 - |kn|).
        v_actual_L = v_s_L * (2.0 - abs(kn_L))
        omega_L = np.sign(kn_L) * v_s_L * kappa_g
        vx_L = v_actual_L * np.cos(beta_rad_L)
        vy_L = v_actual_L * np.sin(beta_rad_L)

    # =========================================================================
    # 2. DYNAMIC RIGID BODY BRIDGE (Leader Velocity -> Follower Velocity)
    # =========================================================================
    # Full 2D kinematic transformation using planar rigid body cross-product:
    # v_F = v_L + (omega x r_F/L)
    #
    #  vx_F = vx_L - (omega_L * dy)   <- Lateral offset (dy) creates longitudinal velocity
    #  vy_F = vy_L + (omega_L * dx)   <- Longitudinal offset (dx) creates lateral velocity
    #  omega_F = omega_L              <- Entire rigid formation shares identical yaw rate
    vx_F = vx_L - (omega_L * dy)
    vy_F = vy_L + (omega_L * dx)
    omega_F = omega_L

    # =========================================================================
    # 3. ENCODE FOLLOWER VELOCITY -> FOLLOWER OCP (ROBUST & NUMERICALLY SAFE)
    # =========================================================================
    # Compute resulting linear speed and drift angle
    v_tran_F = np.sqrt(vx_F ** 2 + vy_F ** 2)
    beta_rad_F = np.arctan2(vy_F, vx_F)

    # Guard 1: Complete stop edge case (no translation and no rotation)
    if v_tran_F < 1e-6 and abs(omega_F) < 1e-6:
        kn_F = 0.0
        v_s_F = 0.0
    else:
        # Estimate nominal curvature; handle zero-translation (pure point-rotation)
        if v_tran_F > 1e-6:
            kn_estimate = omega_F / (v_tran_F * kappa_g)
        else:
            kn_estimate = 2.0 * np.sign(omega_F) if abs(omega_F) > 1e-6 else 0.0

        # Re-branch follower state into Gentle or Sharp curve regimes
        if abs(kn_estimate) <= 1.0:
            # --- Follower stays in Gentle Regime ---
            kn_F = kn_estimate
            v_s_F = v_tran_F
        else:
            # --- Follower enters Sharp Regime ---
            # Guard 2: Safe service speed recovery from yaw rate
            v_s_F = abs(omega_F) / kappa_g if abs(omega_F) > 1e-6 else v_tran_F / 2.0

            if v_s_F > 1e-6:
                # Calculate curvature ratio and clip strictly within physical limits [1.0, 2.0]
                kn_abs = np.clip(2.0 - (v_tran_F / v_s_F), 1.0, 2.0)
                kn_F = np.sign(omega_F) * kn_abs if abs(omega_F) > 1e-6 else kn_abs
            else:
                kn_F = np.sign(omega_F) * 2.0 if abs(omega_F) > 1e-6 else 0.0

    # Guard 3: Normalize speed ratio and clip to physical bounds [-1.0, 1.0]
    vn_F = np.clip(v_s_F / v_max, -1.0, 1.0) if v_max > 1e-6 else 0.0

    return (
        round(kn_F, 4),
        round(np.degrees(beta_rad_F), 2),
        round(vn_F, 4),
    )


def print_follower_trajectories(leader_data, offsets, kappa_g=1.0, v_max=1.5):
    """
    Computes and neatly prints follower trajectories step-by-step for multiple
    robots across a given leader trajectory.
    """
    print("=" * 80)
    print(" OCP-TO-OCP FORMATION CONVERSION (FULL 2D RIGID BODY KINEMATICS)")
    print("=" * 80)
    print(
        f"{'Step':<6} | {'Robot':<6} | {'kn_F (Curvature)':<18} | {'beta_F (deg)':<15} | {'vn_F (Vel Ratio)':<15}"
    )
    print("=" * 80)

    # Process each trajectory step for all configured followers
    for step_data in leader_data:
        step = step_data[0]
        kn_L, beta_L, vn_L = step_data[1], step_data[2], step_data[3]

        for name, (dx, dy) in offsets.items():
            kn_F, beta_deg_F, vn_F = convert_S_curve_step(
                kn_L, beta_L, vn_L, dx, dy, kappa_g, v_max
            )
            print(
                f"{step:<6} | {name:<6} | {kn_F:<18.4f} | {beta_deg_F:<15.2f} | {vn_F:<15.4f}"
            )

        print("-" * 80)


# --- Example Trajectory Run --------------------------------------------------

if __name__ == "__main__":
    # Leader Data (Robot 1): (Step, kn, beta_deg, vn)
    leader_data = [
        (1, 0.0000, 0.00, 0.6667),
        (2, 0.0100, 2.86, 0.6675),
        (3, 0.0306, 11.58, 0.6574),
        (4, 0.0522, 26.19, 0.6404),
        (5, 0.0762, 46.85, 0.6141),
        (6, 0.0715, 66.25, 0.6424),
        (7, 0.0297, 74.55, 0.6557),
        (8, -0.0302, 65.94, 0.6673),
        (9, -0.0308, 57.18, 0.6568),
        (10, -0.0197, 51.57, 0.6672),
    ]

    # Formation Offsets (dx, dy) relative to Leader:
    # R2, R3 = Left Followers; R4, R5 = Right Followers
    offsets = {"R2": (0, 3), "R3": (0, 6), "R4": (0, -3), "R5": (0, -6)}

    # Run and print results
    print_follower_trajectories(leader_data, offsets)