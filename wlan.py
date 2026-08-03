## BOOTSTRAP: SimulationApp must be created before any other isaacsim import;
## this line launches the simulator (headless=False shows the viewport).
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import numpy as np
from scipy.spatial.transform import Rotation
from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.controllers.holonomic_controller import HolonomicController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.robots.holonomic_robot_usd_setup import HolonomicRobotUsdSetup
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdShade, Usd, UsdGeom, UsdPhysics
from collections import deque
import omni.usd
import omni.kit.app
import random
import csv
import os

# ============================================================
#  CASE DEFINITIONS
# ============================================================

## The 8 experiment cases as tuples: (label, use_wlan, use_friction, mode).
## Sweeps WLAN on/off x friction on/off x straight/curvature driving.
CASES = [
    ("WLAN_Straight",               True,  True,  'straight'  ),
    ("WLAN_Straight_NoFriction",    True,  False, 'straight'  ),
    ("NoWLAN_Straight",             False, True,  'straight'  ),
    ("NoWLAN_Straight_NoFriction",  False, False, 'straight'  ),
    ("WLAN_Curvature",              True,  True,  'curvature' ),
    ("WLAN_Curvature_NoFriction",   True,  False, 'curvature' ),
    ("NoWLAN_Curvature",            False, True,  'curvature' ),
    ("NoWLAN_Curvature_NoFriction", False, False, 'curvature' ),
]

## Run each case 300 steps; print a full report every 30 steps.
TOTAL_STEPS    = 300
PRINT_INTERVAL = 30

# ============================================================
#  WLAN PRESET
# ============================================================

## The lossy-link profile used when WLAN is ON: delay, jitter, packet-loss
## probability, command noise, and random link-failure probability.
WLAN_PRESET = {
    'delay_steps':    5,
    'jitter_steps':   2,
    'loss_prob':      0.05,
    'corrupt_std':    0.01,
    'link_fail_prob': 0.002,
}

# ============================================================
#  WLAN COMM CHANNEL
# ============================================================

## Simulates one robot's WLAN link: queues commands with delay/jitter, may
## drop or corrupt them, and can go down for random spells.
class CommChannel:
    def __init__(self, robot_name):
        self.robot_name      = robot_name
        self.delay_steps     = WLAN_PRESET['delay_steps']
        self.jitter_steps    = WLAN_PRESET['jitter_steps']
        self.loss_prob       = WLAN_PRESET['loss_prob']
        self.corrupt_std     = WLAN_PRESET['corrupt_std']
        self.link_fail_prob  = WLAN_PRESET['link_fail_prob']
        self.queue           = deque()
        self.last_cmd        = (0.0, 0.0, 0.0)
        self.link_down_steps = 0
        self.current_step    = 0
        self.stats = {'sent': 0, 'dropped': 0, 'corrupted': 0, 'link_fail': 0}

    ## Try to transmit a command: may be blocked (link down), dropped, or
    ## noised, otherwise queued for delayed delivery.
    def send(self, command):
        self.stats['sent'] += 1
        if self.link_down_steps > 0:
            self.link_down_steps -= 1
            self.stats['link_fail'] += 1
            return
        if random.random() < self.link_fail_prob:
            self.link_down_steps = random.randint(10, 50)
            self.stats['link_fail'] += 1
            return
        if random.random() < self.loss_prob:
            self.stats['dropped'] += 1
            return
        cmd = list(command)
        if self.corrupt_std > 0:
            noise = [random.gauss(0, self.corrupt_std) for _ in range(3)]
            cmd   = [c + n for c, n in zip(cmd, noise)]
            if any(abs(n) > self.corrupt_std * 0.5 for n in noise):
                self.stats['corrupted'] += 1
        jitter     = random.randint(-self.jitter_steps, self.jitter_steps)
        deliver_at = max(self.current_step + self.delay_steps + jitter,
                         self.current_step + 1)
        self.queue.append((deliver_at, tuple(cmd)))

    ## Advance one step; deliver any queued commands that are now due, keeping
    ## the most recent. Falls back to the last received command.
    def receive(self):
        self.current_step += 1
        received = None
        while self.queue and self.queue[0][0] <= self.current_step:
            _, cmd   = self.queue.popleft()
            received = cmd
        if received is not None:
            self.last_cmd = received
        return self.last_cmd

    ## Clear all channel state between cases.
    def reset(self):
        self.queue           = deque()
        self.last_cmd        = (0.0, 0.0, 0.0)
        self.link_down_steps = 0
        self.current_step    = 0
        self.stats           = {'sent': 0, 'dropped': 0,
                                 'corrupted': 0, 'link_fail': 0}

# ============================================================
#  ROBOT NAMES & POSITIONS
# ============================================================

## The 23 robots in three groups: I (inner), F (followers), L (line).
ROBOT_NAMES = [
    'I1', 'I2', 'I3', 'I4', 'I5',
    'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10',
    'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7', 'L8',
]

## Fixed starting (x, y, z) layout for all 23 robots.
ROBOT_INITIAL_POSITIONS = {
    'I1': (0.0,  0.0,  0.02), 'I2': (0.0,  2.0,  0.02),
    'I3': (0.0,  4.0,  0.02), 'I4': (0.0,  6.0,  0.02),
    'I5': (0.0,  8.0,  0.02),
    'F1': (4.0,  0.0,  0.02), 'F2': (4.0,  2.0,  0.02),
    'F3': (4.0,  4.0,  0.02), 'F4': (4.0,  6.0,  0.02),
    'F5': (4.0,  8.0,  0.02), 'F6': (6.0,  4.0,  0.02),
    'F7': (8.0,  4.0,  0.02), 'F8': (6.0,  8.0,  0.02),
    'F9': (8.0,  8.0,  0.02), 'F10':(10.0, 8.0,  0.02),
    'L1': (14.0, 0.0,  0.02), 'L2': (14.0, 2.0,  0.02),
    'L3': (14.0, 4.0,  0.02), 'L4': (14.0, 6.0,  0.02),
    'L5': (14.0, 8.0,  0.02), 'L6': (16.0, 0.0,  0.02),
    'L7': (18.0, 0.0,  0.02), 'L8': (20.0, 0.0,  0.02),
}

# ============================================================
#  COMMAND SETS
# ============================================================

## 'straight' mode: every slot is drive-forward, no turn.
STRAIGHT_CMD_SET = [(1.0, 0.0, 0.0)] * 10

## Five per-robot command sets for 'curvature' mode: each is 10 slots of
## [vx, vy, omega], tracing slightly different S-curves.
commands_robot1 = [
    (1.0,   0.0,   0.0   ), (1.0,   0.05,  0.001 ),
    (0.966, 0.198, 0.003 ), (0.862, 0.424, 0.005 ),
    (0.630, 0.672, 0.0070), (0.388, 0.882, 0.0069),
    (0.262, 0.948, 0.0029), (0.408, 0.914,-0.0030),
    (0.534, 0.828,-0.0030), (0.622, 0.784,-0.0020),
]
commands_robot2 = [
    (1.000, 0.000, 0.000 ), (0.970, 0.050, 0.0010),
    (0.876, 0.196, 0.0030), (0.712, 0.424, 0.0050),
    (0.420, 0.672, 0.0070), (0.181, 0.882, 0.0069),
    (0.174, 0.948, 0.0029), (0.498, 0.914,-0.0030),
    (0.624, 0.828,-0.0030), (0.682, 0.784,-0.0020),
]
commands_robot3 = [
    (1.000, 0.000, 0.000 ), (0.941, 0.049, 0.0010),
    (0.786, 0.191, 0.0031), (0.562, 0.419, 0.0050),
    (0.210, 0.672, 0.0070), (-0.026,0.882, 0.0069),
    (0.087, 0.948, 0.0029), (0.588, 0.914,-0.0030),
    (0.714, 0.828,-0.0030), (0.742, 0.784,-0.0020),
]
commands_robot4 = [
    (1.0,   0.0,   0.0   ), (1.030, 0.050, 0.0010),
    (1.056, 0.201, 0.0030), (1.012, 0.424, 0.0050),
    (0.840, 0.672, 0.0070), (0.595, 0.882, 0.0069),
    (0.350, 0.948, 0.0029), (0.318, 0.914,-0.0030),
    (0.444, 0.828,-0.0030), (0.562, 0.784,-0.0020),
]
commands_robot5 = [
    (1.0,   0.0,   0.0   ), (1.060, 0.051, 0.0010),
    (1.147, 0.203, 0.0030), (1.162, 0.424, 0.0050),
    (1.050, 0.672, 0.0070), (0.802, 0.882, 0.0069),
    (0.437, 0.948, 0.0029), (0.228, 0.914,-0.0030),
    (0.354, 0.828,-0.0030), (0.502, 0.784,-0.0020),
]

## Maps each of the 23 robots to one of the five command sets above.
CURVATURE_COMMANDS = {
    'I1': commands_robot3, 'I2': commands_robot2, 'I3': commands_robot1,
    'I4': commands_robot4, 'I5': commands_robot5,
    'F1': commands_robot3, 'F2': commands_robot2, 'F3': commands_robot1,
    'F4': commands_robot4, 'F5': commands_robot5, 'F6': commands_robot1,
    'F7': commands_robot1, 'F8': commands_robot4, 'F9': commands_robot5,
    'F10': commands_robot5,
    'L1': commands_robot3, 'L2': commands_robot2, 'L3': commands_robot1,
    'L4': commands_robot4, 'L5': commands_robot5, 'L6': commands_robot3,
    'L7': commands_robot3, 'L8': commands_robot3,
}

## Pick a robot's command for the current step: slot advances every 30
## steps (capped at 9); straight vs curvature chooses the source.
def get_command(name, step, mode):
    cmd_idx = min(step // 30, 9)
    if mode == 'straight':
        return STRAIGHT_CMD_SET[cmd_idx]
    else:
        return CURVATURE_COMMANDS[name][cmd_idx]

# ============================================================
#  NEIGHBOR PAIRS
# ============================================================

## Robot pairs whose separation is tracked; third field is the CSV column.
## Nominal spacing is 2.0 m.
NEIGHBOR_PAIRS = [
    ('I1', 'I2', 'dist_I1_I2'), ('I2', 'I3', 'dist_I2_I3'),
    ('I3', 'I4', 'dist_I3_I4'), ('I4', 'I5', 'dist_I4_I5'),
    ('F1', 'F2', 'dist_F1_F2'), ('F2', 'F3', 'dist_F2_F3'),
    ('F3', 'F4', 'dist_F3_F4'), ('F4', 'F5', 'dist_F4_F5'),
    ('F3', 'F6', 'dist_F3_F6'), ('F6', 'F7', 'dist_F6_F7'),
    ('F5', 'F8', 'dist_F5_F8'), ('F8', 'F9', 'dist_F8_F9'),
    ('F9', 'F10','dist_F9_F10'),
    ('L1', 'L2', 'dist_L1_L2'), ('L2', 'L3', 'dist_L2_L3'),
    ('L3', 'L4', 'dist_L3_L4'), ('L4', 'L5', 'dist_L4_L5'),
    ('L1', 'L6', 'dist_L1_L6'), ('L6', 'L7', 'dist_L6_L7'),
    ('L7', 'L8', 'dist_L7_L8'),
]

# ============================================================
#  HELPERS
# ============================================================

## Return a robot's world position and Euler angles (yaw in euler[2]).
## Isaac quaternion is [w,x,y,z]; scipy wants [x,y,z,w], hence the reorder.
def get_pose(robot):
    pos, orient = robot.get_world_pose()
    euler = Rotation.from_quat(
        [orient[1], orient[2], orient[3], orient[0]]
    ).as_euler('xyz', degrees=True)
    return pos, euler

## Straight-line XY distance between two points.
def euclidean_dist(a, b):
    return float(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))

## Wipe the 3D scene and tick a few frames so the clear fully applies.
def fresh_stage():
    omni.usd.get_context().new_stage()
    for _ in range(10):
        omni.kit.app.get_app().update()

# ============================================================
#  FRICTION HELPERS
# ============================================================

## Bind a zero-friction physics material to the ground plane (for the
## NoFriction cases).
def apply_zero_friction_to_ground(stage):
    ground_path = "/World/defaultGroundPlane"
    ground_prim = stage.GetPrimAtPath(ground_path)
    if not ground_prim or not ground_prim.IsValid():
        print("  [WARN] Ground plane not found.")
        return
    mat_path = f"{ground_path}/GroundPhysicsMaterial"
    mat      = UsdShade.Material.Define(stage, mat_path)
    api      = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr().Set(0.0)
    api.CreateDynamicFrictionAttr().Set(0.0)
    api.CreateRestitutionAttr().Set(0.0)
    UsdShade.MaterialBindingAPI(ground_prim).Bind(mat)
    print("  [Ground] zero friction applied.")

## Bind zero-friction material to every wheel/axle collision surface.
def apply_zero_friction_to_robots(stage, robot_paths):
    for robot_path in robot_paths:
        mat_path   = f"{robot_path}/WheelPhysicsMaterial"
        mat        = UsdShade.Material.Define(stage, mat_path)
        api        = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr().Set(0.0)
        api.CreateDynamicFrictionAttr().Set(0.0)
        api.CreateRestitutionAttr().Set(0.0)
        robot_prim = stage.GetPrimAtPath(robot_path)
        if not robot_prim or not robot_prim.IsValid():
            continue
        for prim in Usd.PrimRange(robot_prim):
            prim_name = prim.GetName().lower()
            if "axle" in prim_name or "wheel" in prim_name:
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdShade.MaterialBindingAPI(prim).Bind(mat)
                for child in prim.GetChildren():
                    if child.IsA(UsdGeom.Mesh) or child.HasAPI(UsdPhysics.CollisionAPI):
                        UsdShade.MaterialBindingAPI(child).Bind(mat)
    print(f"  [Wheels] zero friction applied to {len(robot_paths)} robots.")

# ============================================================
#  CSV LOGGER  — unchanged, exactly as before
# ============================================================

## Open a per-case CSV and build its header: step + per-robot x/y + neighbor
## distances + per-robot comm stats.
def make_csv_writer(case_label):
    filename  = f"case_{case_label}.csv"
    xy_cols   = [f'{ax}_{n}' for n in ROBOT_NAMES for ax in ('x', 'y')]
    dist_cols = [col for _, _, col in NEIGHBOR_PAIRS]
    stat_cols = [
        f'{s}_{n}'
        for n in ROBOT_NAMES
        for s in ('sent', 'dropped', 'loss_pct', 'corrupted', 'link_fail', 'queue_len')
    ]
    fieldnames = ['step'] + xy_cols + dist_cols + stat_cols
    fh         = open(filename, 'w', newline='')
    writer     = csv.DictWriter(fh, fieldnames=fieldnames)
    writer.writeheader()
    return fh, writer, filename

## Write one CSV row for this step: positions, neighbor gaps, and comm stats.
def write_csv_row(writer, step, positions, channels):
    row = {'step': step}
    for n in ROBOT_NAMES:
        row[f'x_{n}'] = round(float(positions[n][0]), 6)
        row[f'y_{n}'] = round(float(positions[n][1]), 6)
    for ra, rb, col in NEIGHBOR_PAIRS:
        row[col] = round(euclidean_dist(positions[ra], positions[rb]), 6)
    for n in ROBOT_NAMES:
        s     = channels[n].stats
        total = max(s['sent'], 1)
        row[f'sent_{n}']      = s['sent']
        row[f'dropped_{n}']   = s['dropped']
        row[f'loss_pct_{n}']  = round(100 * s['dropped'] / total, 2)
        row[f'corrupted_{n}'] = s['corrupted']
        row[f'link_fail_{n}'] = s['link_fail']
        row[f'queue_len_{n}'] = len(channels[n].queue)
    writer.writerow(row)

# ============================================================
#  PRINT — BLOCK 1: Positions + Commands + YAW
# ============================================================

## Console BLOCK 1: table of each robot's pose, yaw, intended vs actual
## command, and a status flag (ok / dropped / corrupt / waiting).
def print_positions_block(step, mode, positions, yaws, channels, use_wlan):
    cmd_idx = min(step // 30, 9)
    print(f"\n  ── ROBOT POSITIONS, YAW & COMMANDS  (step={step}, cmd_idx={cmd_idx}) ──")
    print(f"  {'Robot':<6} {'X':>9} {'Y':>9} {'Z':>7} {'Yaw(°)':>8}"
          f"  {'Int_vx':>7} {'Int_vy':>7} {'Int_w':>7}"
          f"  {'Act_vx':>7} {'Act_vy':>7} {'Act_w':>7}"
          f"  {'Status':<12}  Group")
    print(f"  {'-'*6} {'-'*9} {'-'*9} {'-'*7} {'-'*8}"
          f"  {'-'*7} {'-'*7} {'-'*7}"
          f"  {'-'*7} {'-'*7} {'-'*7}"
          f"  {'-'*12}  {'-'*5}")

    letter_groups = [
        ('I', ['I1','I2','I3','I4','I5']),
        ('F', ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10']),
        ('L', ['L1','L2','L3','L4','L5','L6','L7','L8']),
    ]
    for grp, members in letter_groups:
        for n in members:
            p    = positions[n]
            icmd = get_command(n, step, mode)
            acmd = channels[n].last_cmd if use_wlan else icmd
            if not use_wlan:
                status = 'no-wlan'
            elif channels[n].stats['sent'] == 0:
                status = 'waiting'
            elif acmd == (0.0, 0.0, 0.0):
                status = '*** DROP ***'
            elif any(abs(acmd[k] - icmd[k]) > 0.005 for k in range(3)):
                status = '~ corrupt'
            else:
                status = 'ok'
            print(f"  {n:<6} {p[0]:>9.4f} {p[1]:>9.4f} {p[2]:>7.4f} {yaws[n]:>8.2f}"
                  f"  {icmd[0]:>7.4f} {icmd[1]:>7.4f} {icmd[2]:>7.4f}"
                  f"  {acmd[0]:>7.4f} {acmd[1]:>7.4f} {acmd[2]:>7.4f}"
                  f"  {status:<12}  {grp}")
        print()

# ============================================================
#  PRINT — BLOCK 2: Neighbor Euclidean Distances
# ============================================================

## Console BLOCK 2: neighbor-pair distances and how far each drifts from 2 m.
def print_neighbor_distances(positions):
    print(f"  ── NEIGHBOR PAIR EUCLIDEAN DISTANCES ──")

    dist_groups = [
        ('I', [p for p in NEIGHBOR_PAIRS if p[0][0]=='I' and p[1][0]=='I']),
        ('F', [p for p in NEIGHBOR_PAIRS if p[0][0]=='F' or  p[1][0]=='F']),
        ('L', [p for p in NEIGHBOR_PAIRS if p[0][0]=='L' and p[1][0]=='L']),
    ]

    for letter, pairs in dist_groups:
        print(f"\n  [{letter}]  {'Pair':<14} {'Eucl. Dist (m)':>14}  "
              f"{'Dev from 2.0m':>14}  {'Status':<12}")
        print(f"       {'-'*14} {'-'*14}  {'-'*14}  {'-'*12}")
        for ra, rb, _ in pairs:
            dist      = euclidean_dist(positions[ra], positions[rb])
            deviation = dist - 2.0
            flag      = 'OK' if abs(deviation) < 0.5 else f'DRIFT {deviation:+.3f}m'
            print(f"       {ra} <-> {rb:<8} {dist:>14.4f}  "
                  f"{deviation:>+14.4f}  {flag:<12}")

# ============================================================
#  PRINT — BLOCK 3: WLAN Stats
# ============================================================

## Console BLOCK 3: per-robot and fleet-total WLAN stats (sent, dropped,
## loss %, corrupted, link failures, queue length).
def print_wlan_stats(channels):
    print(f"\n  ── WLAN COMMUNICATION STATS ──")
    print(f"  {'Robot':<6} {'Sent':>6} {'Dropped':>8} {'Loss%':>7} "
          f"{'Corrupted':>10} {'LinkFail':>9} {'QueueLen':>9}")
    print(f"  {'-'*6} {'-'*6} {'-'*8} {'-'*7} {'-'*10} {'-'*9} {'-'*9}")
    total_sent = total_dropped = total_corrupt = total_lf = 0
    for n in ROBOT_NAMES:
        s     = channels[n].stats
        total = max(s['sent'], 1)
        loss  = 100 * s['dropped'] / total
        qlen  = len(channels[n].queue)
        print(f"  {n:<6} {s['sent']:>6} {s['dropped']:>8} "
              f"{loss:>6.1f}% {s['corrupted']:>10} "
              f"{s['link_fail']:>9} {qlen:>9}")
        total_sent    += s['sent']
        total_dropped += s['dropped']
        total_corrupt += s['corrupted']
        total_lf      += s['link_fail']
    fleet_loss = 100 * total_dropped / max(total_sent, 1)
    print(f"  {'─'*6} {'─'*6} {'─'*8} {'─'*7} {'─'*10} {'─'*9} {'─'*9}")
    print(f"  {'FLEET':<6} {total_sent:>6} {total_dropped:>8} "
          f"{fleet_loss:>6.1f}% {total_corrupt:>10} "
          f"{total_lf:>9}   (totals)")

# ============================================================
#  MASTER PRINT — called every 30 steps
# ============================================================

## Master printer called every 30 steps: header + the three blocks above
## (WLAN stats only shown for WLAN cases).
def print_step_full(case_label, use_wlan, use_friction, mode,
                    step, positions, yaws, channels):
    cmd_idx = min(step // 30, 9)

    # Header
    wlan_str = 'WLAN=ON ' if use_wlan     else 'WLAN=OFF'
    fric_str = 'FRIC=DEFAULT' if use_friction else 'FRIC=ZERO   '
    print(f"\n{'='*100}")
    print(f"  CASE: {case_label:<38}  {wlan_str}  {fric_str}  "
          f"MODE={mode.upper():<10}  STEP={step:>4}  CMD_IDX={cmd_idx}")
    print(f"{'='*100}")

    # Block 1: positions + yaw + commands
    print_positions_block(step, mode, positions, yaws, channels, use_wlan)

    # Block 2: neighbor pair Euclidean distances
    print_neighbor_distances(positions)

    # Block 3: WLAN stats (WLAN cases only)
    if use_wlan:
        print_wlan_stats(channels)
    else:
        print(f"\n  ── WLAN STATS : N/A (perfect delivery, no channel) ──")

# ============================================================
#  FINAL SUMMARY
# ============================================================

## End-of-case report: each robot's final pose and displacement, then a
## ranking by total distance travelled.
def print_final_summary(case_label, mode, use_wlan, use_friction,
                        robots, initial_positions):
    print(f"\n{'='*85}")
    print(f"  FINAL RESULTS — {case_label}")
    print(f"  Mode={mode.upper()}  WLAN={'ON' if use_wlan else 'OFF'}  "
          f"Friction={'DEFAULT' if use_friction else 'ZERO'}  "
          f"Steps={TOTAL_STEPS}")
    print(f"{'='*85}")
    print(f"  {'Robot':<6} {'X_final':>9} {'Y_final':>9} "
          f"{'dX':>9} {'dY':>9} {'Dist(m)':>10} {'Yaw°':>8}  Group")
    print(f"  {'-'*6} {'-'*9} {'-'*9} "
          f"{'-'*9} {'-'*9} {'-'*10} {'-'*8}  {'-'*5}")
    letter_groups = [
        ('I', ['I1','I2','I3','I4','I5']),
        ('F', ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10']),
        ('L', ['L1','L2','L3','L4','L5','L6','L7','L8']),
    ]
    results = []
    for grp, members in letter_groups:
        for n in members:
            pos, euler = get_pose(robots[n])
            init = initial_positions[n]
            dx   = pos[0] - init[0]
            dy   = pos[1] - init[1]
            dist = np.sqrt(dx**2 + dy**2)
            results.append((n, dx, dy, dist, euler[2]))
            print(f"  {n:<6} {pos[0]:>9.4f} {pos[1]:>9.4f} "
                  f"{dx:>9.4f} {dy:>9.4f} {dist:>10.4f} "
                  f"{euler[2]:>8.2f}°  {grp}")
        print()
    print(f"  RANKING (by total displacement)")
    print(f"  {'-'*50}")
    for rank, (n, dx, dy, dist, yaw) in enumerate(
            sorted(results, key=lambda r: r[3], reverse=True), 1):
        print(f"  {rank:>2}. {n:<6}  {dist:>8.4f} m  "
              f"(dX={dx:>7.4f}, dY={dy:>7.4f}, yaw={yaw:>7.2f}°)")

# ============================================================
#  SINGLE CASE RUNNER
# ============================================================

## Run ONE case end to end: build the 23-robot scene, optionally zero the
## friction, drive 300 steps (commands through WLAN if enabled), log CSV
## and print reports, then summarize.
def run_case(case_label, use_wlan, use_friction, mode):

    print(f"\n\n{'#'*100}")
    print(f"#  CASE      : {case_label}")
    print(f"#  WLAN      : {'ON' if use_wlan else 'OFF'}")
    print(f"#  FRICTION  : {'DEFAULT' if use_friction else 'ZERO'}")
    print(f"#  MODE      : {mode.upper()}")
    print(f"#  STEPS     : {TOTAL_STEPS}  (10 cmd slots × 30 steps, "
          f"print every {PRINT_INTERVAL})")
    print(f"{'#'*100}")

    fresh_stage()

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        carb.log_error("Could not find Isaac Sim assets folder")
        return
    kaya_asset_path = assets_root_path + "/Isaac/Robots/Kaya/kaya.usd"

    robots      = {}
    controllers = {}
    channels    = {}
    robot_paths = []

    print(f"\n  Creating 23 robots ...")
    ## Spawn each Kaya at its start position, build its holonomic controller
    ## from USD geometry, and give it a comm channel.
    for name in ROBOT_NAMES:
        pos  = ROBOT_INITIAL_POSITIONS[name]
        path = f"/World/Kaya_{name}"
        robot_paths.append(path)

        robots[name] = world.scene.add(
            WheeledRobot(
                prim_path=path,
                name=f"kaya_{name}",
                wheel_dof_names=["axle_0_joint", "axle_1_joint", "axle_2_joint"],
                create_robot=True,
                usd_path=kaya_asset_path,
                position=np.array(pos),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )

        kaya_setup = HolonomicRobotUsdSetup(
            robot_prim_path=robots[name].prim_path,
            com_prim_path=f"{path}/base_link/control_offset"
        )
        (wheel_radius, wheel_positions, wheel_orientations,
         mecanum_angles, wheel_axis, up_axis
         ) = kaya_setup.get_holonomic_controller_params()

        controllers[name] = HolonomicController(
            name=f"holonomic_controller_{name}",
            wheel_radius=wheel_radius,
            wheel_positions=wheel_positions,
            wheel_orientations=wheel_orientations,
            mecanum_angles=mecanum_angles,
            wheel_axis=wheel_axis,
            up_axis=up_axis,
        )

        channels[name] = CommChannel(name)
        print(f"    {name:<5}  pos=({pos[0]:5.1f}, {pos[1]:4.1f}, {pos[2]:.2f})")

    ## Add the shared flat floor.
    world.scene.add_default_ground_plane()

    ## For NoFriction cases, override ground + wheel friction to zero.
    if not use_friction:
        apply_zero_friction_to_ground(stage)
        apply_zero_friction_to_robots(stage, robot_paths)
    else:
        print("  [Friction] Default physics friction retained.")

    ## Finalize physics; robots settle onto the ground.
    world.reset()

    initial_positions = {}
    print(f"\n  {'Robot':<6} {'X':>9} {'Y':>9} {'Z':>7} {'Yaw°':>8}  (initial)")
    for name in ROBOT_NAMES:
        pos, euler = get_pose(robots[name])
        initial_positions[name] = pos.copy()
        print(f"  {name:<6} {pos[0]:>9.4f} {pos[1]:>9.4f} "
              f"{pos[2]:>7.4f} {euler[2]:>8.2f}")

    fh, writer, csv_file = make_csv_writer(case_label)
    print(f"\n  [CSV] → {os.path.abspath(csv_file)}")
    print(f"  [SIM] Starting {TOTAL_STEPS} steps ...")

    ## MAIN DRIVE LOOP for this case: each step, route every robot's command
    ## (through its WLAN channel if enabled), apply it, log, and print periodically.
    for step in range(TOTAL_STEPS + 1):
        world.step(render=True)

        for name in ROBOT_NAMES:
            intended_cmd = get_command(name, step, mode)
            if use_wlan:
                channels[name].send(intended_cmd)
                actual_cmd = channels[name].receive()
            else:
                actual_cmd = intended_cmd
            robots[name].apply_wheel_actions(
                controllers[name].forward(
                    command=[actual_cmd[0], actual_cmd[1], actual_cmd[2]]
                )
            )

        # Collect positions AND yaw from physics engine every step
        positions = {}
        yaws      = {}
        for name in ROBOT_NAMES:
            pos, euler      = get_pose(robots[name])
            positions[name] = pos
            yaws[name]      = euler[2]   # true yaw in degrees from quaternion

        # CSV unchanged — x, y, neighbor dists, comm stats
        write_csv_row(writer, step, positions, channels)

        # Console print every 30 steps — now includes yaw + Euclidean dists
        if step % PRINT_INTERVAL == 0:
            print_step_full(
                case_label, use_wlan, use_friction, mode,
                step, positions, yaws, channels
            )

    fh.close()

    print_final_summary(case_label, mode, use_wlan, use_friction,
                        robots, initial_positions)
    print(f"\n  [CSV] {TOTAL_STEPS+1} rows → {os.path.abspath(csv_file)}")
    print(f"  [DONE] '{case_label}' complete.\n")

# ============================================================
#  MAIN
# ============================================================

print("\n" + "="*100)
## MAIN: describe the sweep, then run all 8 cases in turn.
print("  IFL FLEET SIMULATION  —  8-CASE SWEEP")
print("  Console prints every 30 steps:")
print("    [1] Robot X, Y, Z, Yaw(°), intended cmd, actual cmd, status")
print("    [2] Neighbor pair Euclidean distances + deviation from 2.0m nominal")
print("    [3] WLAN fleet comm stats (WLAN cases only)")
print("  CSV unchanged — x, y, neighbor dists, comm stats")
print("="*100)

## Run every case defined at the top of the file.
for case_label, use_wlan, use_friction, mode in CASES:
    run_case(case_label, use_wlan, use_friction, mode)

print("\n" + "="*100)
print("  ALL 8 CASES COMPLETE")
print("="*100)

## Shut down Isaac Sim cleanly.
simulation_app.close()