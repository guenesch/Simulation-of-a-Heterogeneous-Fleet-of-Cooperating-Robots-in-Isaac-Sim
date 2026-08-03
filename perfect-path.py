"""
Build the "perfect" follower paths from the leader's pose.

Given the leader's global pose at each step -- position (x, y) and heading theta
-- this computes where two followers should be to hold a rigid formation: one a
fixed distance d to the leader's LEFT, one the same distance to its RIGHT.

The offset is applied in the leader's OWN frame, not the world frame: "left"
means left relative to whichever way the leader is currently pointing. As the
leader rotates, the followers swing around with it, so the trio keeps its shape.
This is the position-level companion to the OCP work: instead of converting
motion commands, it just places each follower correctly at every step.
"""

import numpy as np
import pandas as pd


def calculate_perfect_paths(leader_data, d=3.0):
    """
    Transform the leader's global (x, y, theta) into follower positions.

    d : lateral offset in metres (used as +d for the left robot, -d via the
        mirrored formulas for the right robot).
    """
    perfect_coords = []
    for step in leader_data:
        x_l, y_l = step['pos']
        # Heading in radians for the trig functions.
        theta_rad = np.radians(step['theta'])

        # A unit vector pointing to the leader's LEFT is (-sin theta, cos theta):
        # it's the forward direction (cos, sin) rotated 90 deg counter-clockwise.
        # Robot 2 sits d along that left vector.
        x2 = x_l - d * np.sin(theta_rad)
        y2 = y_l + d * np.cos(theta_rad)

        # Robot 3 sits d along the RIGHT vector -- the exact opposite sign.
        x3 = x_l + d * np.sin(theta_rad)
        y3 = y_l - d * np.cos(theta_rad)

        perfect_coords.append({
            "Step": step['id'],
            "R1_X": round(x_l, 2), "R1_Y": round(y_l, 2),
            "R2_X": round(x2, 2), "R2_Y": round(y2, 2),
            "R3_X": round(x3, 2), "R3_Y": round(y3, 2)
        })
    return pd.DataFrame(perfect_coords)


# Leader (Robot 1) trajectory: global position and heading at each step.
leader_trajectory = [
    {'id': 'Initial', 'pos': (0.00, 0.00), 'theta': 0.0},
    {'id': 'Cmd 1', 'pos': (5.00, 0.00), 'theta': 0.0},
    {'id': 'Cmd 2', 'pos': (10.00, 0.25), 'theta': 5.7},
    {'id': 'Cmd 3', 'pos': (14.83, 1.24), 'theta': 17.2},
    {'id': 'Cmd 4', 'pos': (19.14, 3.36), 'theta': 34.4},
    {'id': 'Cmd 5', 'pos': (22.29, 6.72), 'theta': 57.3},
    {'id': 'Cmd 6', 'pos': (24.23, 11.13), 'theta': 68.8},
    {'id': 'Cmd 7', 'pos': (25.54, 15.87), 'theta': 68.8},
    {'id': 'Cmd 8', 'pos': (27.58, 20.44), 'theta': 63.0},
    {'id': 'Cmd 9', 'pos': (30.25, 24.58), 'theta': 51.6},
    {'id': 'Cmd 10', 'pos': (33.36, 28.50), 'theta': 51.6},
]

df = calculate_perfect_paths(leader_trajectory)
print(df.to_string(index=False))

# quick sanity check: distance from leader to each follower must equal d
d = 3.0
ok = True
for _, r in df.iterrows():
    d2 = np.hypot(r.R2_X - r.R1_X, r.R2_Y - r.R1_Y)
    d3 = np.hypot(r.R3_X - r.R1_X, r.R3_Y - r.R1_Y)
    if abs(d2 - d) > 0.01 or abs(d3 - d) > 0.01:
        ok = False
print("\nEvery follower exactly d=3.0 m from the leader?", ok)