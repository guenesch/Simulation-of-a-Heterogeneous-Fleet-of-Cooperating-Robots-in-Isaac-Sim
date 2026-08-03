"""
=============================================================================
  ISAAC SIM — COMPLETE EXPERIMENT SUITE
  Experiments 1, 2, 3, 4, 5A, 5B, 6
  Each experiment runs 5 times (seeds 0–4).
  Only the FINAL STEP of each run is recorded.
  Statistics (mean, std, variance, min, max) printed + saved to CSV.
=============================================================================
"""

# ── Bootstrap ────────────────────────────────────────────────────────────────
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import csv
import os
import random
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation

from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.controllers.holonomic_controller import HolonomicController
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.robots.holonomic_robot_usd_setup import HolonomicRobotUsdSetup
from isaacsim.storage.native import get_assets_root_path

from pxr import UsdShade, Usd, UsdGeom, UsdPhysics, PhysxSchema, Sdf
import omni.usd
import omni.kit.app

# ── Asset paths ───────────────────────────────────────────────────────────────
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    exit()

KAYA_USD   = assets_root_path + "/Isaac/Robots/Kaya/kaya.usd"
JETBOT_USD = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_results")
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
#  SHARED HELPERS
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def fresh_stage():
    omni.usd.get_context().new_stage()
    for _ in range(10):
        omni.kit.app.get_app().update()


def get_pose(robot):
    pos, orient = robot.get_world_pose()
    euler = Rotation.from_quat(
        [orient[1], orient[2], orient[3], orient[0]]
    ).as_euler('xyz', degrees=True)
    return pos, euler


def euclidean_dist(a, b):
    return float(np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2))


def make_kaya(world, prim_path, name, position):
    robot = world.scene.add(
        WheeledRobot(
            prim_path=prim_path,
            name=name,
            wheel_dof_names=["axle_0_joint", "axle_1_joint", "axle_2_joint"],
            create_robot=True,
            usd_path=KAYA_USD,
            position=np.array(position),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
    )
    setup = HolonomicRobotUsdSetup(
        robot_prim_path=robot.prim_path,
        com_prim_path=f"{prim_path}/base_link/control_offset"
    )
    wr, wp, wo, ma, wa, ua = setup.get_holonomic_controller_params()
    ctrl = HolonomicController(
        name=f"ctrl_{name}",
        wheel_radius=wr, wheel_positions=wp, wheel_orientations=wo,
        mecanum_angles=ma, wheel_axis=wa, up_axis=ua,
    )
    return robot, ctrl


def apply_ground_material(stage, static_fric, dynamic_fric):
    ground_path = "/World/defaultGroundPlane"
    ground_prim = stage.GetPrimAtPath(ground_path)
    if not ground_prim or not ground_prim.IsValid():
        return
    mat_path = f"{ground_path}/PhysMat"
    mat = UsdShade.Material.Define(stage, mat_path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr().Set(float(static_fric))
    api.CreateDynamicFrictionAttr().Set(float(dynamic_fric))
    api.CreateRestitutionAttr().Set(0.0)
    UsdShade.MaterialBindingAPI(ground_prim).Bind(mat)


def apply_wheel_material(stage, robot_path, static_fric, dynamic_fric):
    mat_path = f"{robot_path}/WheelMat"
    mat = UsdShade.Material.Define(stage, mat_path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr().Set(float(static_fric))
    api.CreateDynamicFrictionAttr().Set(float(dynamic_fric))
    api.CreateRestitutionAttr().Set(0.0)
    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim or not robot_prim.IsValid():
        return
    for prim in Usd.PrimRange(robot_prim):
        nm = prim.GetName().lower()
        if "axle" in nm or "wheel" in nm:
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI(prim).Bind(mat)
            for child in prim.GetChildren():
                if child.IsA(UsdGeom.Mesh) or child.HasAPI(UsdPhysics.CollisionAPI):
                    UsdShade.MaterialBindingAPI(child).Bind(mat)


def apply_joint_damping(stage, robot_path, damping_value):
    """Apply damping to all axle joints of a Kaya robot."""
    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim or not robot_prim.IsValid():
        return
    for prim in Usd.PrimRange(robot_prim):
        nm = prim.GetName().lower()
        if "axle" in nm:
            if prim.HasAPI(UsdPhysics.DriveAPI):
                drive_api = UsdPhysics.DriveAPI(prim, "angular")
            else:
                drive_api = UsdPhysics.DriveAPI.Apply(prim, "angular")
            drive_api.CreateDampingAttr().Set(float(damping_value))


def apply_solver_iterations(stage, robot_path, pos_iter, vel_iter):
    """Apply solver iteration counts to articulation root."""
    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim:
        return
    for prim in Usd.PrimRange(robot_prim):
        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
            api = PhysxSchema.PhysxArticulationAPI(prim)
            api.CreateSolverPositionIterationCountAttr().Set(int(pos_iter))
            api.CreateSolverVelocityIterationCountAttr().Set(int(vel_iter))
            break


# =============================================================================
#  STATISTICS HELPERS
# =============================================================================

def compute_stats(values):
    """Return dict of mean, std, variance, min, max for a list of floats."""
    arr = np.array(values, dtype=float)
    return {
        'mean':     float(np.mean(arr)),
        'std':      float(np.std(arr)),
        'variance': float(np.var(arr)),
        'min':      float(np.min(arr)),
        'max':      float(np.max(arr)),
    }


def print_stats_table(title, stats_dict):
    """
    stats_dict: { metric_name: [val_run0, val_run1, ...] }
    Prints a table and returns computed stats.
    """
    print(f"\n{'='*90}")
    print(f"  STATISTICS — {title}")
    print(f"{'='*90}")
    print(f"  {'Metric':<45} {'Mean':>10} {'Std':>10} {'Var':>12} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")
    result = {}
    for metric, values in stats_dict.items():
        s = compute_stats(values)
        result[metric] = s
        print(f"  {metric:<45} {s['mean']:>10.4f} {s['std']:>10.4f} "
              f"{s['variance']:>12.6f} {s['min']:>10.4f} {s['max']:>10.4f}")
    sys.stdout.flush()  # show stats immediately after all 5 runs
    return result


def save_stats_csv(filename, run_data, stats_data):
    """
    run_data:   list of dicts, one per run  {metric: value}
    stats_data: {metric: {mean, std, variance, min, max}}
    """
    path = os.path.join(OUT_DIR, filename)
    all_metrics = list(run_data[0].keys()) if run_data else []
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['run'] + all_metrics)
        for i, row in enumerate(run_data):
            writer.writerow([i] + [row.get(m, '') for m in all_metrics])
        # Stats block
        writer.writerow([])
        writer.writerow(['STAT'] + all_metrics)
        for stat_name in ['mean', 'std', 'variance', 'min', 'max']:
            writer.writerow(
                [stat_name] + [stats_data[m][stat_name] if m in stats_data else ''
                               for m in all_metrics]
            )
    print(f"\n  [CSV] Saved → {path}")




# =============================================================================
#  STRUCTURED PER-RUN PRINT HELPERS
#  Each experiment has its own print function that formats results as a table.
#  Called immediately after each run so data appears on terminal right away.
# =============================================================================

def _sep(w=92): print('  ' + '─'*w)
def _header(*cols): print('  ' + '  '.join(f"{c[0]:<{c[1]}}" for c in cols))
def _row(*cols):    print('  ' + '  '.join(f"{c[0]:<{c[1]}}" if isinstance(c[0], str)
                                            else f"{c[0]:>{c[1]}.4f}" if isinstance(c[0], float)
                                            else f"{c[0]:>{c[1]}}" for c in cols))

def print_exp1_run(run_idx, result):
    """Exp1: Cable/WLAN/Bluetooth — per robot per medium."""
    print(f"\n  {'='*92}")
    print(f"  RUN {run_idx+1} RESULTS — EXP1  Communication Medium")
    print(f"  {'='*92}")
    for medium in ['cable', 'wlan', 'bluetooth']:
        print(f"\n  [{medium.upper()}]")
        print(f"  {'Robot':<16} {'dX':>9} {'dY':>9} {'Dist(m)':>9} {'Yaw°':>8} "
              f"{'Loss%':>7} {'Corrupt':>8} {'LinkFail':>9}")
        print(f"  {'-'*16} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*7} {'-'*8} {'-'*9}")
        for rname in ['Kaya_Center','Kaya_Left1','Kaya_Left2','Kaya_Right1','Kaya_Right2']:
            p = medium + '_' + rname
            dx   = result.get(f"{p}_dx",   0)
            dy   = result.get(f"{p}_dy",   0)
            dist = result.get(f"{p}_dist", 0)
            yaw  = result.get(f"{p}_yaw",  0)
            loss = result.get(f"{p}_loss_pct", 0)
            cor  = result.get(f"{p}_corrupt",  0)
            lf   = result.get(f"{p}_linkfail", 0)
            print(f"  {rname:<16} {dx:>9.4f} {dy:>9.4f} {dist:>9.4f} {yaw:>8.2f} "
                  f"{loss:>7.2f} {cor:>8} {lf:>9}")
        # neighbor distances
        pairs = [('dist_Left2_Left1','L2↔L1'),('dist_Left1_Center','L1↔C'),
                 ('dist_Center_Right1','C↔R1'),('dist_Right1_Right2','R1↔R2')]
        nd = '  '.join(f"{lab}={result.get(medium+'_'+col,0):.4f}m"
                       for col,lab in pairs)
        print(f"  Neighbor gaps: {nd}")
    sys.stdout.flush()


def print_exp2_run(run_idx, result):
    """Exp2: Friction — per ground condition per wheel config."""
    print(f"\n  {'='*92}")
    print(f"  RUN {run_idx+1} RESULTS — EXP2  Friction Comparison")
    print(f"  {'='*92}")
    ground_labels = ['Default','Gnd_s0_d0','Gnd_s1_d0','Gnd_s1_d1','Gnd_s1_d10']
    wheel_names   = ['Ice_Wheels','Low_Friction','Medium_Friction','High_Friction',
                     'Super_Grippy','LoStatic_HiDynamic','HiStatic_LoDynamic','UltraHi_Dynamic']
    for glabel in ground_labels:
        print(f"\n  [Ground: {glabel}]")
        print(f"  {'Wheel Config':<24} {'dX':>9} {'dY':>9} {'Dist(m)':>9} {'Yaw°':>8}")
        print(f"  {'-'*24} {'-'*9} {'-'*9} {'-'*9} {'-'*8}")
        for wname in wheel_names:
            p    = f"{glabel}_{wname}"
            dx   = result.get(f"{p}_dx",   0)
            dy   = result.get(f"{p}_dy",   0)
            dist = result.get(f"{p}_dist", 0)
            yaw  = result.get(f"{p}_yaw",  0)
            print(f"  {wname:<24} {dx:>9.4f} {dy:>9.4f} {dist:>9.4f} {yaw:>8.2f}")
    sys.stdout.flush()


def print_exp3_run(run_idx, result):
    """Exp3: 8-case sweep — per case summary (mean dist + mean yaw across robots)."""
    print(f"\n  {'='*92}")
    print(f"  RUN {run_idx+1} RESULTS — EXP3  8-Case Sweep (23 robots)")
    print(f"  {'='*92}")
    cases = ['WLAN_Straight','WLAN_Straight_NoFriction','NoWLAN_Straight',
             'NoWLAN_Straight_NoFriction','WLAN_Curvature','WLAN_Curvature_NoFriction',
             'NoWLAN_Curvature','NoWLAN_Curvature_NoFriction']
    robot_names = ['I1','I2','I3','I4','I5',
                   'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10',
                   'L1','L2','L3','L4','L5','L6','L7','L8']
    print(f"  {'Case':<34} {'Mean Dist(m)':>13} {'Std Dist':>10} "
          f"{'Mean Yaw°':>10} {'Min Dist':>10} {'Max Dist':>10}")
    print(f"  {'-'*34} {'-'*13} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for case in cases:
        dists = [result.get(f"{case}_{n}_dist", 0) for n in robot_names]
        yaws  = [result.get(f"{case}_{n}_yaw",  0) for n in robot_names]
        mean_d = float(np.mean(dists))
        std_d  = float(np.std(dists))
        mean_y = float(np.mean(yaws))
        min_d  = float(np.min(dists))
        max_d  = float(np.max(dists))
        print(f"  {case:<34} {mean_d:>13.4f} {std_d:>10.4f} "
              f"{mean_y:>10.2f} {min_d:>10.4f} {max_d:>10.4f}")
    sys.stdout.flush()


def print_exp4_run(run_idx, result):
    """Exp4: Heterogeneous WLAN delay — Scenario A and B."""
    print(f"\n  {'='*92}")
    print(f"  RUN {run_idx+1} RESULTS — EXP4  Heterogeneous WLAN Delay")
    print(f"  {'='*92}")
    for sc in ['A','B']:
        label = 'Front degradation (I5 worst)' if sc=='A' else 'Third robot worst (I3 worst)'
        print(f"\n  [Scenario {sc}: {label}]")
        print(f"  {'Robot':<6} {'Delay':>6} {'dX':>9} {'dY':>9} {'Dist(m)':>9} "
              f"{'Yaw°':>8} {'Loss%':>7} {'Corrupt':>8} {'LinkFail':>9}")
        print(f"  {'-'*6} {'-'*6} {'-'*9} {'-'*9} {'-'*9} {'-'*8} {'-'*7} {'-'*8} {'-'*9}")
        for n in ['I1','I2','I3','I4','I5']:
            p    = f"Sc{sc}_{n}"
            dx   = result.get(f"{p}_dx",       0)
            dy   = result.get(f"{p}_dy",       0)
            dist = result.get(f"{p}_dist",     0)
            yaw  = result.get(f"{p}_yaw",      0)
            loss = result.get(f"{p}_loss_pct", 0)
            cor  = result.get(f"{p}_corrupt",  0)
            lf   = result.get(f"{p}_linkfail", 0)
            dly  = result.get(f"{p}_delay",    0)
            print(f"  {n:<6} {dly:>6} {dx:>9.4f} {dy:>9.4f} {dist:>9.4f} "
                  f"{yaw:>8.2f} {loss:>7.2f} {cor:>8} {lf:>9}")
        pairs = [('dist_I1_I2','I1↔I2'),('dist_I2_I3','I2↔I3'),
                 ('dist_I3_I4','I3↔I4'),('dist_I4_I5','I4↔I5')]
        nd = '  '.join(f"{lab}={result.get('Sc'+sc+'_'+col,0):.4f}m "
                       f"(dev={result.get('Sc'+sc+'_'+col+'_dev',0):+.4f})"
                       for col,lab in pairs)
        print(f"  Gaps: {nd}")
    sys.stdout.flush()


def print_exp5a_run(run_idx, result):
    """Exp5A: Velocity effect."""
    print(f"\n  {'='*92}")
    print(f"  RUN {run_idx+1} RESULTS — EXP5A  Velocity Effect")
    print(f"  {'='*92}")
    print(f"  {'Speed':>8} {'dX':>9} {'dY':>9} {'Dist(m)':>9} {'Yaw°':>8}")
    print(f"  {'-'*8} {'-'*9} {'-'*9} {'-'*9} {'-'*8}")
    for vx in [0.2, 1.0]:
        label = f"vx{vx:.1f}"
        dx   = result.get(f"{label}_dx",   0)
        dy   = result.get(f"{label}_dy",   0)
        dist = result.get(f"{label}_dist", 0)
        yaw  = result.get(f"{label}_yaw",  0)
        print(f"  {vx:>8.1f} {dx:>9.4f} {dy:>9.4f} {dist:>9.4f} {yaw:>8.2f}")
    sys.stdout.flush()


def print_exp5b_run(run_idx, result):
    """Exp5B: Damping sweep."""
    print(f"\n  {'='*92}")
    print(f"  RUN {run_idx+1} RESULTS — EXP5B  Rigid Body Damping Sweep")
    print(f"  {'='*92}")
    print(f"  {'Config':<26} {'LinDamp':>8} {'AngDamp':>8} "
          f"{'dX':>9} {'dY':>9} {'Dist(m)':>9} {'Yaw°':>8}")
    print(f"  {'-'*26} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*9} {'-'*8}")
    for cfg in ['No_Damping','Low_Linear_Damping','Medium_Linear_Damping',
                'High_Linear_Damping','Low_Angular_Damping','Medium_Angular_Damping',
                'High_Angular_Damping','Both_High_Damping']:
        dx   = result.get(f"{cfg}_dx",       0)
        dy   = result.get(f"{cfg}_dy",       0)
        dist = result.get(f"{cfg}_dist",     0)
        yaw  = result.get(f"{cfg}_yaw",      0)
        lin  = result.get(f"{cfg}_lin_damp", 0)
        ang  = result.get(f"{cfg}_ang_damp", 0)
        print(f"  {cfg:<26} {lin:>8.2f} {ang:>8.2f} "
              f"{dx:>9.4f} {dy:>9.4f} {dist:>9.4f} {yaw:>8.2f}")
    sys.stdout.flush()




def print_exp6_run(run_idx, result):
    """Exp6: Kaya vs Jetbot."""
    print(f"\n  {'='*92}")
    print(f"  RUN {run_idx+1} RESULTS — EXP6  Kaya vs Jetbot at 0.2 m/s")
    print(f"  {'='*92}")
    print(f"  {'Robot':<10} {'dX':>9} {'dY':>9} {'Dist(m)':>9} {'Yaw°':>8}")
    print(f"  {'-'*10} {'-'*9} {'-'*9} {'-'*9} {'-'*8}")
    for prefix in ['kaya','jetbot']:
        dx   = result.get(f"{prefix}_dx",   0)
        dy   = result.get(f"{prefix}_dy",   0)
        dist = result.get(f"{prefix}_dist", 0)
        yaw  = result.get(f"{prefix}_yaw",  0)
        print(f"  {prefix:<10} {dx:>9.4f} {dy:>9.4f} {dist:>9.4f} {yaw:>8.2f}")
    sys.stdout.flush()

# =============================================================================
#  COMM CHANNEL  (used by Exp1, Exp3, Exp4)
# =============================================================================

class CommChannel:
    def __init__(self, robot_name, preset):
        self.robot_name      = robot_name
        self.delay_steps     = preset['delay_steps']
        self.jitter_steps    = preset['jitter_steps']
        self.loss_prob       = preset['loss_prob']
        self.corrupt_std     = preset['corrupt_std']
        self.link_fail_prob  = preset['link_fail_prob']
        self.queue           = deque()
        self.last_cmd        = (0.0, 0.0, 0.0)
        self.link_down_steps = 0
        self.current_step    = 0
        self.stats           = {'sent': 0, 'dropped': 0, 'corrupted': 0, 'link_fail': 0}

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

    def receive(self):
        self.current_step += 1
        received = None
        while self.queue and self.queue[0][0] <= self.current_step:
            _, cmd   = self.queue.popleft()
            received = cmd
        if received is not None:
            self.last_cmd = received
        return self.last_cmd

    def reset(self, preset):
        self.delay_steps     = preset['delay_steps']
        self.jitter_steps    = preset['jitter_steps']
        self.loss_prob       = preset['loss_prob']
        self.corrupt_std     = preset['corrupt_std']
        self.link_fail_prob  = preset['link_fail_prob']
        self.queue           = deque()
        self.last_cmd        = (0.0, 0.0, 0.0)
        self.link_down_steps = 0
        self.current_step    = 0
        self.stats           = {'sent': 0, 'dropped': 0, 'corrupted': 0, 'link_fail': 0}



# =============================================================================
#  EXPERIMENT 1 — Communication medium (cable / WLAN / bluetooth)
#  5 Kaya robots, curved path, 300 steps per medium
#
#  Mirrors your reference code (document 16) exactly:
#    — Single persistent World (no fresh_stage between mediums)
#    — Robots created in loop; HolonomicRobotUsdSetup called per robot
#      inside the same loop, BEFORE ground plane
#    — Ground plane added AFTER all robots
#    — world.reset() called ONCE after ground plane
#    — while simulation_app.is_running() state machine
#    — reset_needed flag switches between mediums without rebuilding world
# =============================================================================

EXP1_PRESETS = {
    'cable': {
        'delay_steps': 1, 'jitter_steps': 0,
        'loss_prob': 0.001, 'corrupt_std': 0.001, 'link_fail_prob': 0.0,
    },
    'wlan': {
        'delay_steps': 5, 'jitter_steps': 2,
        'loss_prob': 0.05, 'corrupt_std': 0.01, 'link_fail_prob': 0.002,
    },
    'bluetooth': {
        'delay_steps': 10, 'jitter_steps': 4,
        'loss_prob': 0.10, 'corrupt_std': 0.02, 'link_fail_prob': 0.005,
    },
}
EXP1_MEDIUMS = ['cable', 'wlan', 'bluetooth']

# Command sets — identical to reference (document 16)
EXP1_CMDS = {
    'Kaya_Center': [
        (1.0,0.0,0.0),(1.0,0.05,0.001),(0.966,0.198,0.003),(0.862,0.424,0.005),
        (0.630,0.672,0.007),(0.388,0.882,0.0069),(0.262,0.948,0.0029),
        (0.408,0.914,-0.003),(0.534,0.828,-0.003),(0.622,0.784,-0.002),
    ],
    'Kaya_Left1': [
        (1.0,0.0,0.0),(0.970,0.050,0.001),(0.876,0.196,0.003),(0.712,0.424,0.005),
        (0.420,0.672,0.007),(0.181,0.882,0.0069),(0.174,0.948,0.0029),
        (0.498,0.914,-0.003),(0.624,0.828,-0.003),(0.682,0.784,-0.002),
    ],
    'Kaya_Left2': [
        (1.0,0.0,0.0),(0.941,0.049,0.001),(0.786,0.191,0.0031),(0.562,0.419,0.005),
        (0.210,0.672,0.007),(-0.026,0.882,0.0069),(0.087,0.948,0.0029),
        (0.588,0.914,-0.003),(0.714,0.828,-0.003),(0.742,0.784,-0.002),
    ],
    'Kaya_Right1': [
        (1.0,0.0,0.0),(1.030,0.050,0.001),(1.056,0.201,0.003),(1.012,0.424,0.005),
        (0.840,0.672,0.007),(0.595,0.882,0.0069),(0.350,0.948,0.0029),
        (0.318,0.914,-0.003),(0.444,0.828,-0.003),(0.562,0.784,-0.002),
    ],
    'Kaya_Right2': [
        (1.0,0.0,0.0),(1.060,0.051,0.001),(1.147,0.203,0.003),(1.162,0.424,0.005),
        (1.050,0.672,0.007),(0.802,0.882,0.0069),(0.437,0.948,0.0029),
        (0.228,0.914,-0.003),(0.354,0.828,-0.003),(0.502,0.784,-0.002),
    ],
}

# robot_name → [x, y, z]  (same positions as reference)
EXP1_ROBOT_CFG = {
    'Kaya_Center': [0.0,  0.0, 0.02],
    'Kaya_Left1':  [0.0,  3.0, 0.02],
    'Kaya_Left2':  [0.0,  6.0, 0.02],
    'Kaya_Right1': [0.0, -3.0, 0.02],
    'Kaya_Right2': [0.0, -6.0, 0.02],
}
EXP1_NAMES = list(EXP1_ROBOT_CFG.keys())

EXP1_NEIGHBOR_PAIRS = [
    ('Kaya_Left2',  'Kaya_Left1',  'dist_Left2_Left1'),
    ('Kaya_Left1',  'Kaya_Center', 'dist_Left1_Center'),
    ('Kaya_Center', 'Kaya_Right1', 'dist_Center_Right1'),
    ('Kaya_Right1', 'Kaya_Right2', 'dist_Right1_Right2'),
]

EXP1_TOTAL_STEPS = 300


def run_exp1_single(seed):
    """
    Runs all 3 mediums sequentially inside ONE World,
    exactly like the reference while-loop state machine.
    Returns flat dict of final-step metrics for all 3 mediums.
    """
    set_seed(seed)
    fresh_stage()

    # ── 1. World ──────────────────────────────────────────────────────────────
    world = World(stage_units_in_meters=1.0)

    robots      = {}
    controllers = {}

    # ── 2. Create all robots + setup controllers inside same loop ─────────────
    #    (mirrors: for robot_name in robot_names: add robot → setup → controller)
    for rname, pos in EXP1_ROBOT_CFG.items():
        robot = world.scene.add(
            WheeledRobot(
                prim_path=f"/World/{rname}",
                name=rname.lower(),
                wheel_dof_names=["axle_0_joint", "axle_1_joint", "axle_2_joint"],
                create_robot=True,
                usd_path=KAYA_USD,
                position=np.array(pos),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        kaya_setup = HolonomicRobotUsdSetup(
            robot_prim_path=robot.prim_path,
            com_prim_path=f"/World/{rname}/base_link/control_offset"
        )
        (wheel_radius, wheel_positions, wheel_orientations,
         mecanum_angles, wheel_axis, up_axis
         ) = kaya_setup.get_holonomic_controller_params()

        controllers[rname] = HolonomicController(
            name=f"holonomic_controller_{rname}",
            wheel_radius=wheel_radius,
            wheel_positions=wheel_positions,
            wheel_orientations=wheel_orientations,
            mecanum_angles=mecanum_angles,
            wheel_axis=wheel_axis,
            up_axis=up_axis,
        )
        robots[rname] = robot

    # ── 3. Ground plane AFTER all robots ──────────────────────────────────────
    world.scene.add_default_ground_plane()

    # ── 4. Single reset ───────────────────────────────────────────────────────
    world.reset()

    # ── 5. State machine — mirrors while simulation_app.is_running() ──────────
    medium_idx     = 0
    current_medium = EXP1_MEDIUMS[medium_idx]
    channels       = {n: CommChannel(n, EXP1_PRESETS[current_medium])
                      for n in EXP1_NAMES}

    step_i       = 0          # steps within current medium
    reset_needed = True
    initial_data = {}
    run_result   = {}

    while medium_idx < len(EXP1_MEDIUMS):
        world.step(render=True)

        if world.is_stopped() and not reset_needed:
            reset_needed = True

        if world.is_playing():

            # ── Reset when switching medium ───────────────────────────────────
            if reset_needed:
                world.reset()
                for n in EXP1_NAMES:
                    controllers[n].reset()
                    channels[n].reset(EXP1_PRESETS[current_medium])

                initial_data = {}
                for n in EXP1_NAMES:
                    p, _ = get_pose(robots[n])
                    initial_data[n] = p.copy()

                step_i       = 0
                reset_needed = False

            # ── Command index: every 30 steps advance, max index 9 ───────────
            cmd_idx = min(step_i // 30, 9)

            # ── Apply commands through channel ────────────────────────────────
            for rname in EXP1_NAMES:
                intended = EXP1_CMDS[rname][cmd_idx]
                channels[rname].send(intended)
                actual = channels[rname].receive()
                robots[rname].apply_wheel_actions(
                    controllers[rname].forward(
                        command=[actual[0], actual[1], actual[2]]
                    )
                )

            step_i += 1

            # ── End of this medium — collect final step results ───────────────
            if step_i >= EXP1_TOTAL_STEPS:
                positions = {}
                for rname in EXP1_NAMES:
                    pos, euler = get_pose(robots[rname])
                    positions[rname] = pos
                    dx   = pos[0] - initial_data[rname][0]
                    dy   = pos[1] - initial_data[rname][1]
                    dist = float(np.sqrt(dx**2 + dy**2))
                    s    = channels[rname].stats
                    tot  = max(s['sent'], 1)
                    run_result[f"{current_medium}_{rname}_dx"]       = round(dx,   6)
                    run_result[f"{current_medium}_{rname}_dy"]       = round(dy,   6)
                    run_result[f"{current_medium}_{rname}_dist"]     = round(dist, 6)
                    run_result[f"{current_medium}_{rname}_yaw"]      = round(euler[2], 4)
                    run_result[f"{current_medium}_{rname}_loss_pct"] = round(
                        100 * s['dropped'] / tot, 4)
                    run_result[f"{current_medium}_{rname}_corrupt"]  = s['corrupted']
                    run_result[f"{current_medium}_{rname}_linkfail"] = s['link_fail']

                for ra, rb, col in EXP1_NEIGHBOR_PAIRS:
                    run_result[f"{current_medium}_{col}"] = round(
                        euclidean_dist(positions[ra], positions[rb]), 6)

                # Advance to next medium
                medium_idx += 1
                if medium_idx < len(EXP1_MEDIUMS):
                    current_medium = EXP1_MEDIUMS[medium_idx]
                    # Reset channels for new medium
                    channels = {n: CommChannel(n, EXP1_PRESETS[current_medium])
                                for n in EXP1_NAMES}
                    reset_needed = True

    return run_result


def experiment_1():
    print("\n" + "#"*90)
    print("  EXPERIMENT 1 — Communication Medium (Cable / WLAN / Bluetooth)")
    print("  5 Kaya robots, curved path, 300 steps per medium, 5 runs")
    print("  Single World per run — mediums switched via reset_needed state machine")
    print("#"*90)

    all_runs = []
    for run_idx in range(5):
        print(f"\n  [EXP1] Run {run_idx+1}/5  (seed={run_idx}) ...")
        result = run_exp1_single(seed=run_idx)
        all_runs.append(result)

        print_exp1_run(run_idx, result)

    # Statistics across 5 runs
    all_metrics = list(all_runs[0].keys())
    stats_input = {m: [r[m] for r in all_runs] for m in all_metrics}
    stats = print_stats_table("EXP1 — Communication Medium", stats_input)
    save_stats_csv("exp1_comm_medium.csv", all_runs, stats)
    sys.stdout.flush()

# =============================================================================

# =============================================================================

# =============================================================================
#  EXPERIMENT 2 — Friction comparison
#  All 40 robots (8 wheel configs × 5 ground conditions) in ONE scene
#  Ground conditions separated by X offset — each has its own ground cube
#  One World, one reset, 300 steps — no fresh stage between conditions
#
#  Layout (X axis):
#    Zone 0  x=  0  Default ground (Isaac Sim default plane)
#    Zone 1  x= 60  Ground static=0  dynamic=0
#    Zone 2  x=120  Ground static=1  dynamic=0
#    Zone 3  x=180  Ground static=1  dynamic=1
#    Zone 4  x=240  Ground static=1  dynamic=10
#
#  8 robots per zone at y=0,3,6,...,21
#  All robots run simultaneously, collect final step results
# =============================================================================

EXP2_TOTAL_STEPS = 300
EXP2_CMD         = (1.0, 0.0, 0.0)
EXP2_ZONE_OFFSET = 60.0   # X separation between ground zones

EXP2_WHEEL_CONFIGS = [
    {"name": "Ice_Wheels",       "static": 0.05, "dynamic": 0.03,  "restitution": 0.1},
    {"name": "Low_Friction",     "static": 0.3,  "dynamic": 0.25,  "restitution": 0.1},
    {"name": "Medium_Friction",  "static": 0.6,  "dynamic": 0.5,   "restitution": 0.2},
    {"name": "High_Friction",    "static": 1.0,  "dynamic": 0.8,   "restitution": 0.2},
    {"name": "Super_Grippy",     "static": 1.5,  "dynamic": 1.2,   "restitution": 0.1},
    {"name": "LoStatic_HiDynamic","static": 0.3, "dynamic": 0.5,   "restitution": 0.1},
    {"name": "HiStatic_LoDynamic","static": 1.2, "dynamic": 0.4,   "restitution": 0.2},
    {"name": "UltraHi_Dynamic",  "static": 1.2,  "dynamic": 200.0, "restitution": 0.2},
]

EXP2_GROUND_CONDITIONS = [
    {"label": "Default",    "override": False, "static": 0.0,  "dynamic": 0.0},
    {"label": "Gnd_s0_d0",  "override": True,  "static": 0.0,  "dynamic": 0.0},
    {"label": "Gnd_s1_d0",  "override": True,  "static": 1.0,  "dynamic": 0.0},
    {"label": "Gnd_s1_d1",  "override": True,  "static": 1.0,  "dynamic": 1.0},
    {"label": "Gnd_s1_d10", "override": True,  "static": 1.0,  "dynamic": 10.0},
]


def exp2_create_zone_ground(stage, zone_idx, gcfg, x_offset, n_robots=8):
    """
    Create a custom ground cube for one zone.
    For Default zone: use a plain cube with no material override.
    For override zones: apply Sdf-based friction material (mirrors doc 17).
    Ground cube centered on the zone's robot cluster.
    """
    ground_path = f"/World/Ground_Zone{zone_idx}"
    y_center    = ((n_robots - 1) * 3.0) / 2.0   # center of robot cluster

    ground_geom = UsdGeom.Cube.Define(stage, ground_path)
    ground_geom.CreateSizeAttr(1.0)
    ground_geom.AddTranslateOp().Set((x_offset, y_center, -0.5))
    ground_geom.AddScaleOp().Set((EXP2_ZONE_OFFSET - 5.0, 40.0, 1.0))

    ground_prim = stage.GetPrimAtPath(ground_path)
    UsdPhysics.CollisionAPI.Apply(ground_prim)
    rba = UsdPhysics.RigidBodyAPI.Apply(ground_prim)
    rba.CreateKinematicEnabledAttr().Set(True)

    if gcfg["override"]:
        # Mirrors doc 17 create_ground_plane_with_friction exactly
        mat_path         = f"{ground_path}/GroundMaterial"
        mat_prim         = stage.DefinePrim(mat_path, "Material")
        phys_mat_path    = f"{mat_path}/PhysicsMaterial"
        phys_mat_prim    = stage.DefinePrim(phys_mat_path, "PhysicsMaterial")

        s = float(gcfg["static"])
        d = float(gcfg["dynamic"])

        phys_mat_prim.CreateAttribute(
            "physics:staticFriction",        Sdf.ValueTypeNames.Float).Set(s)
        phys_mat_prim.CreateAttribute(
            "physics:dynamicFriction",       Sdf.ValueTypeNames.Float).Set(d)
        phys_mat_prim.CreateAttribute(
            "physics:restitution",           Sdf.ValueTypeNames.Float).Set(0.1)
        phys_mat_prim.CreateAttribute(
            "physxMaterial:staticFriction",  Sdf.ValueTypeNames.Float).Set(s)
        phys_mat_prim.CreateAttribute(
            "physxMaterial:dynamicFriction", Sdf.ValueTypeNames.Float).Set(d)
        phys_mat_prim.CreateAttribute(
            "physxMaterial:restitution",     Sdf.ValueTypeNames.Float).Set(0.1)

        material = UsdShade.Material(mat_prim)
        UsdShade.MaterialBindingAPI.Apply(ground_prim).Bind(
            material,
            UsdShade.Tokens.weakerThanDescendants,
            "physics"
        )


def exp2_apply_wheel_friction(stage, robot_path, wcfg):
    """Mirrors doc 17 apply_all_frictions — nested DefinePrim structure."""
    mat_path      = f"{robot_path}/WheelFrictionMaterial"
    mat_prim      = stage.DefinePrim(mat_path, "Material")
    phys_mat_path = f"{mat_path}/PhysicsMaterial"
    phys_mat_prim = stage.DefinePrim(phys_mat_path, "PhysicsMaterial")

    s = float(wcfg["static"])
    d = float(wcfg["dynamic"])
    r = float(wcfg["restitution"])

    phys_mat_prim.CreateAttribute(
        "physics:staticFriction",        Sdf.ValueTypeNames.Float).Set(s)
    phys_mat_prim.CreateAttribute(
        "physics:dynamicFriction",       Sdf.ValueTypeNames.Float).Set(d)
    phys_mat_prim.CreateAttribute(
        "physics:restitution",           Sdf.ValueTypeNames.Float).Set(r)
    phys_mat_prim.CreateAttribute(
        "physxMaterial:staticFriction",  Sdf.ValueTypeNames.Float).Set(s)
    phys_mat_prim.CreateAttribute(
        "physxMaterial:dynamicFriction", Sdf.ValueTypeNames.Float).Set(d)
    phys_mat_prim.CreateAttribute(
        "physxMaterial:restitution",     Sdf.ValueTypeNames.Float).Set(r)

    material   = UsdShade.Material(mat_prim)
    robot_prim = stage.GetPrimAtPath(robot_path)

    for prim in Usd.PrimRange(robot_prim):
        nm = prim.GetName().lower()
        if "axle" in nm or "wheel" in nm:
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    material,
                    UsdShade.Tokens.weakerThanDescendants,
                    "physics"
                )
            for child in prim.GetChildren():
                if child.IsA(UsdGeom.Mesh) or child.HasAPI(UsdPhysics.CollisionAPI):
                    UsdShade.MaterialBindingAPI.Apply(child).Bind(
                        material,
                        UsdShade.Tokens.weakerThanDescendants,
                        "physics"
                    )


def run_exp2_single(seed):
    """
    All 40 robots in one scene. 5 ground zones along X.
    One World, one reset, 300 steps — no fresh stage between conditions.
    """
    set_seed(seed)
    fresh_stage()

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    all_robots      = []   # list of (zone_label, wheel_name, robot, controller, init_pos)
    all_controllers = []

    # ── Create all robots and ground zones ────────────────────────────────────
    for zone_idx, gcfg in enumerate(EXP2_GROUND_CONDITIONS):
        x_offset = zone_idx * EXP2_ZONE_OFFSET

        # Create zone ground cube
        exp2_create_zone_ground(stage, zone_idx, gcfg, x_offset)

        # Create 8 robots in this zone
        for i, wcfg in enumerate(EXP2_WHEEL_CONFIGS):
            y_pos    = float(i * 3)
            rob_name = f"Z{zone_idx}_{wcfg['name']}"
            rob_path = f"/World/Kaya_{rob_name}"

            robot = world.scene.add(
                WheeledRobot(
                    prim_path=rob_path,
                    name=f"kaya_{rob_name}",
                    wheel_dof_names=["axle_0_joint","axle_1_joint","axle_2_joint"],
                    create_robot=True,
                    usd_path=KAYA_USD,
                    position=np.array([x_offset, y_pos, 0.02]),
                    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                )
            )
            kaya_setup = HolonomicRobotUsdSetup(
                robot_prim_path=robot.prim_path,
                com_prim_path=f"{rob_path}/base_link/control_offset"
            )
            (wr, wp, wo, ma, wa, ua) = kaya_setup.get_holonomic_controller_params()
            ctrl = HolonomicController(
                name=f"ctrl_{rob_name}",
                wheel_radius=wr, wheel_positions=wp, wheel_orientations=wo,
                mecanum_angles=ma, wheel_axis=wa, up_axis=ua,
            )

            # Apply wheel friction before reset
            exp2_apply_wheel_friction(stage, rob_path, wcfg)

            all_robots.append((gcfg["label"], wcfg["name"], robot, ctrl))
            all_controllers.append(ctrl)

    # ── Reset once ────────────────────────────────────────────────────────────
    world.reset()

    # ── Store initial positions ───────────────────────────────────────────────
    init_positions = {}
    for glabel, wname, robot, ctrl in all_robots:
        pos, _ = get_pose(robot)
        init_positions[(glabel, wname)] = pos.copy()

    # ── Run 300 steps — all 40 robots simultaneously ──────────────────────────
    for step in range(EXP2_TOTAL_STEPS + 1):
        for glabel, wname, robot, ctrl in all_robots:
            robot.apply_wheel_actions(ctrl.forward(command=list(EXP2_CMD)))
        world.step(render=True)

    # ── Collect final results ─────────────────────────────────────────────────
    result = {}
    for glabel, wname, robot, ctrl in all_robots:
        pos, euler = get_pose(robot)
        init = init_positions[(glabel, wname)]
        dx   = pos[0] - init[0]
        dy   = pos[1] - init[1]
        dist = float(np.sqrt(dx**2 + dy**2))
        prefix = f"{glabel}_{wname}"
        result[f"{prefix}_dx"]   = round(dx,       6)
        result[f"{prefix}_dy"]   = round(dy,       6)
        result[f"{prefix}_dist"] = round(dist,     6)
        result[f"{prefix}_yaw"]  = round(euler[2], 4)

    return result


def experiment_2():
    print("\n" + "#"*90)
    print("  EXPERIMENT 2 — Friction Comparison")
    print("  40 robots in ONE scene (8 wheel configs × 5 ground zones along X)")
    print("  300 steps, no fresh stage between conditions, 5 runs")
    print("#"*90)

    all_runs = []
    for run_idx in range(5):
        print(f"\n  [EXP2] Run {run_idx+1}/5  (seed={run_idx}) ...")
        result = run_exp2_single(seed=run_idx)
        all_runs.append(result)
        print_exp2_run(run_idx, result)

    all_metrics = list(all_runs[0].keys())
    stats_input = {m: [r[m] for r in all_runs] for m in all_metrics}
    stats = print_stats_table("EXP2 — Friction Comparison", stats_input)
    save_stats_csv("exp2_friction.csv", all_runs, stats)
    sys.stdout.flush()


# =============================================================================
#  EXPERIMENT 3 — 8-case sweep (23 robots)
#  All 184 robots (8 cases × 23 robots) in ONE scene simultaneously
#  Cases separated by X offset — each case has its own ground zone
#  WLAN cases: stochastic channels per robot independently
#  NoFriction cases: zero friction applied only to robots in that zone
#
#  Layout (X axis):
#    Case 0  x=  0   WLAN_Straight
#    Case 1  x= 30   WLAN_Straight_NoFriction
#    Case 2  x= 60   NoWLAN_Straight
#    Case 3  x= 90   NoWLAN_Straight_NoFriction
#    Case 4  x=120   WLAN_Curvature
#    Case 5  x=150   WLAN_Curvature_NoFriction
#    Case 6  x=180   NoWLAN_Curvature
#    Case 7  x=210   NoWLAN_Curvature_NoFriction
#
#  23 robots per case at their standard Y positions
#  All 184 robots step simultaneously, collect final results per case
# =============================================================================

EXP3_CASES = [
    ("WLAN_Straight",               True,  True,  'straight'),
    ("WLAN_Straight_NoFriction",    True,  False, 'straight'),
    ("NoWLAN_Straight",             False, True,  'straight'),
    ("NoWLAN_Straight_NoFriction",  False, False, 'straight'),
    ("WLAN_Curvature",              True,  True,  'curvature'),
    ("WLAN_Curvature_NoFriction",   True,  False, 'curvature'),
    ("NoWLAN_Curvature",            False, True,  'curvature'),
    ("NoWLAN_Curvature_NoFriction", False, False, 'curvature'),
]

EXP3_WLAN_PRESET = {
    'delay_steps': 5, 'jitter_steps': 2,
    'loss_prob': 0.05, 'corrupt_std': 0.01, 'link_fail_prob': 0.002,
}

EXP3_ROBOT_NAMES = [
    'I1','I2','I3','I4','I5',
    'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10',
    'L1','L2','L3','L4','L5','L6','L7','L8',
]

EXP3_INIT_POS = {
    'I1':(0.0,0.0,0.02),'I2':(0.0,2.0,0.02),'I3':(0.0,4.0,0.02),
    'I4':(0.0,6.0,0.02),'I5':(0.0,8.0,0.02),
    'F1':(4.0,0.0,0.02),'F2':(4.0,2.0,0.02),'F3':(4.0,4.0,0.02),
    'F4':(4.0,6.0,0.02),'F5':(4.0,8.0,0.02),'F6':(6.0,4.0,0.02),
    'F7':(8.0,4.0,0.02),'F8':(6.0,8.0,0.02),'F9':(8.0,8.0,0.02),
    'F10':(10.0,8.0,0.02),
    'L1':(14.0,0.0,0.02),'L2':(14.0,2.0,0.02),'L3':(14.0,4.0,0.02),
    'L4':(14.0,6.0,0.02),'L5':(14.0,8.0,0.02),'L6':(16.0,0.0,0.02),
    'L7':(18.0,0.0,0.02),'L8':(20.0,0.0,0.02),
}

_c1=[(1.0,0.0,0.0),(1.0,0.05,0.001),(0.966,0.198,0.003),(0.862,0.424,0.005),
     (0.630,0.672,0.007),(0.388,0.882,0.0069),(0.262,0.948,0.0029),
     (0.408,0.914,-0.003),(0.534,0.828,-0.003),(0.622,0.784,-0.002)]
_c2=[(1.0,0.0,0.0),(0.970,0.050,0.001),(0.876,0.196,0.003),(0.712,0.424,0.005),
     (0.420,0.672,0.007),(0.181,0.882,0.0069),(0.174,0.948,0.0029),
     (0.498,0.914,-0.003),(0.624,0.828,-0.003),(0.682,0.784,-0.002)]
_c3=[(1.0,0.0,0.0),(0.941,0.049,0.001),(0.786,0.191,0.0031),(0.562,0.419,0.005),
     (0.210,0.672,0.007),(-0.026,0.882,0.0069),(0.087,0.948,0.0029),
     (0.588,0.914,-0.003),(0.714,0.828,-0.003),(0.742,0.784,-0.002)]
_c4=[(1.0,0.0,0.0),(1.030,0.050,0.001),(1.056,0.201,0.003),(1.012,0.424,0.005),
     (0.840,0.672,0.007),(0.595,0.882,0.0069),(0.350,0.948,0.0029),
     (0.318,0.914,-0.003),(0.444,0.828,-0.003),(0.562,0.784,-0.002)]
_c5=[(1.0,0.0,0.0),(1.060,0.051,0.001),(1.147,0.203,0.003),(1.162,0.424,0.005),
     (1.050,0.672,0.007),(0.802,0.882,0.0069),(0.437,0.948,0.0029),
     (0.228,0.914,-0.003),(0.354,0.828,-0.003),(0.502,0.784,-0.002)]
_straight=[(1.0,0.0,0.0)]*10

EXP3_CURVATURE_CMDS = {
    'I1':_c3,'I2':_c2,'I3':_c1,'I4':_c4,'I5':_c5,
    'F1':_c3,'F2':_c2,'F3':_c1,'F4':_c4,'F5':_c5,
    'F6':_c1,'F7':_c1,'F8':_c4,'F9':_c5,'F10':_c5,
    'L1':_c3,'L2':_c2,'L3':_c1,'L4':_c4,'L5':_c5,
    'L6':_c3,'L7':_c3,'L8':_c3,
}

EXP3_NEIGHBOR_PAIRS = [
    ('I1','I2','dist_I1_I2'),('I2','I3','dist_I2_I3'),
    ('I3','I4','dist_I3_I4'),('I4','I5','dist_I4_I5'),
    ('F1','F2','dist_F1_F2'),('F2','F3','dist_F2_F3'),
    ('F3','F4','dist_F3_F4'),('F4','F5','dist_F4_F5'),
    ('F3','F6','dist_F3_F6'),('F6','F7','dist_F6_F7'),
    ('F5','F8','dist_F5_F8'),('F8','F9','dist_F8_F9'),
    ('F9','F10','dist_F9_F10'),
    ('L1','L2','dist_L1_L2'),('L2','L3','dist_L2_L3'),
    ('L3','L4','dist_L3_L4'),('L4','L5','dist_L4_L5'),
    ('L1','L6','dist_L1_L6'),('L6','L7','dist_L6_L7'),
    ('L7','L8','dist_L7_L8'),
]

EXP3_TOTAL_STEPS = 300
EXP3_CASE_OFFSET = 30.0   # X separation between cases


def exp3_get_cmd(name, step, mode):
    idx = min(step // 30, 9)
    return _straight[idx] if mode == 'straight' else EXP3_CURVATURE_CMDS[name][idx]


def exp3_create_case_ground(stage, case_idx, use_friction, x_offset):
    """
    Create ground cube for one case zone.
    NoFriction cases: zero friction material.
    Default friction cases: plain cube with no material (Isaac Sim default).
    """
    ground_path = f"/World/Ground_Case{case_idx}"
    y_center    = 10.0   # rough center of 23-robot cluster

    ground_geom = UsdGeom.Cube.Define(stage, ground_path)
    ground_geom.CreateSizeAttr(1.0)
    ground_geom.AddTranslateOp().Set((x_offset + 10.0, y_center, -0.5))
    ground_geom.AddScaleOp().Set((EXP3_CASE_OFFSET - 2.0, 30.0, 1.0))

    ground_prim = stage.GetPrimAtPath(ground_path)
    UsdPhysics.CollisionAPI.Apply(ground_prim)
    rba = UsdPhysics.RigidBodyAPI.Apply(ground_prim)
    rba.CreateKinematicEnabledAttr().Set(True)

    if not use_friction:
        # Apply zero friction to this zone's ground
        mat_path = f"{ground_path}/ZeroFrictionMat"
        mat      = UsdShade.Material.Define(stage, mat_path)
        api      = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        api.CreateStaticFrictionAttr().Set(0.0)
        api.CreateDynamicFrictionAttr().Set(0.0)
        api.CreateRestitutionAttr().Set(0.0)
        UsdShade.MaterialBindingAPI(ground_prim).Bind(mat)


def exp3_apply_zero_friction_robot(stage, robot_path):
    """Apply zero friction to one robot's wheels."""
    mat_path = f"{robot_path}/ZeroWheelMat"
    mat      = UsdShade.Material.Define(stage, mat_path)
    api      = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr().Set(0.0)
    api.CreateDynamicFrictionAttr().Set(0.0)
    api.CreateRestitutionAttr().Set(0.0)

    robot_prim = stage.GetPrimAtPath(robot_path)
    if not robot_prim or not robot_prim.IsValid():
        return
    for prim in Usd.PrimRange(robot_prim):
        nm = prim.GetName().lower()
        if "axle" in nm or "wheel" in nm:
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI(prim).Bind(mat)
            for child in prim.GetChildren():
                if child.IsA(UsdGeom.Mesh) or child.HasAPI(UsdPhysics.CollisionAPI):
                    UsdShade.MaterialBindingAPI(child).Bind(mat)


def run_exp3_single(seed):
    """
    All 184 robots (8 cases × 23) in ONE scene.
    Cases separated by X offset. One World, one reset, 300 steps.
    WLAN channels are independent and stochastic per robot.
    """
    set_seed(seed)
    fresh_stage()

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    # all_robots: list of dicts with case info + robot + controller + channel
    all_robots = []

    # ── Create all 8 case zones + 23 robots each ──────────────────────────────
    for case_idx, (case_label, use_wlan, use_friction, mode) in enumerate(EXP3_CASES):
        x_offset = case_idx * EXP3_CASE_OFFSET

        # Ground cube for this case zone
        exp3_create_case_ground(stage, case_idx, use_friction, x_offset)

        for rname in EXP3_ROBOT_NAMES:
            base_pos = EXP3_INIT_POS[rname]
            # Offset X by case zone, keep Y and Z from standard positions
            pos      = (base_pos[0] + x_offset, base_pos[1], base_pos[2])
            rob_id   = f"C{case_idx}_{rname}"
            rob_path = f"/World/Kaya_{rob_id}"

            robot = world.scene.add(
                WheeledRobot(
                    prim_path=rob_path,
                    name=f"kaya_{rob_id}",
                    wheel_dof_names=["axle_0_joint","axle_1_joint","axle_2_joint"],
                    create_robot=True,
                    usd_path=KAYA_USD,
                    position=np.array(pos),
                    orientation=np.array([1.0,0.0,0.0,0.0]),
                )
            )
            kaya_setup = HolonomicRobotUsdSetup(
                robot_prim_path=robot.prim_path,
                com_prim_path=f"{rob_path}/base_link/control_offset"
            )
            (wr, wp, wo, ma, wa, ua) = kaya_setup.get_holonomic_controller_params()
            ctrl = HolonomicController(
                name=f"ctrl_{rob_id}",
                wheel_radius=wr, wheel_positions=wp, wheel_orientations=wo,
                mecanum_angles=ma, wheel_axis=wa, up_axis=ua,
            )

            # Apply zero friction before reset if needed
            if not use_friction:
                exp3_apply_zero_friction_robot(stage, rob_path)

            # WLAN channel — independent stochastic per robot
            channel = CommChannel(rname, EXP3_WLAN_PRESET) if use_wlan else None

            all_robots.append({
                'case_label': case_label,
                'case_idx':   case_idx,
                'rname':      rname,
                'robot':      robot,
                'ctrl':       ctrl,
                'channel':    channel,
                'use_wlan':   use_wlan,
                'mode':       mode,
            })

    # ── Single reset ──────────────────────────────────────────────────────────
    world.reset()

    # ── Store initial positions ───────────────────────────────────────────────
    for rd in all_robots:
        pos, _ = get_pose(rd['robot'])
        rd['init_pos'] = pos.copy()

    # ── Run 300 steps — all 184 robots simultaneously ─────────────────────────
    for step in range(EXP3_TOTAL_STEPS + 1):
        for rd in all_robots:
            intended = exp3_get_cmd(rd['rname'], step, rd['mode'])
            if rd['use_wlan']:
                rd['channel'].send(intended)
                actual = rd['channel'].receive()
            else:
                actual = intended
            rd['robot'].apply_wheel_actions(
                rd['ctrl'].forward(command=list(actual))
            )
        world.step(render=True)

    # ── Collect final results per case ────────────────────────────────────────
    result = {}

    # Group by case
    for case_idx, (case_label, use_wlan, use_friction, mode) in enumerate(EXP3_CASES):
        case_robots = [rd for rd in all_robots if rd['case_idx'] == case_idx]

        positions = {}
        for rd in case_robots:
            pos, euler = get_pose(rd['robot'])
            rname = rd['rname']
            positions[rname] = pos
            dx   = pos[0] - rd['init_pos'][0]
            dy   = pos[1] - rd['init_pos'][1]
            dist = float(np.sqrt(dx**2 + dy**2))
            result[f"{case_label}_{rname}_dx"]   = round(dx,       6)
            result[f"{case_label}_{rname}_dy"]   = round(dy,       6)
            result[f"{case_label}_{rname}_dist"] = round(dist,     6)
            result[f"{case_label}_{rname}_yaw"]  = round(euler[2], 4)

        # Neighbor distances
        for ra, rb, col in EXP3_NEIGHBOR_PAIRS:
            result[f"{case_label}_{col}"] = round(
                euclidean_dist(positions[ra], positions[rb]), 6)

    return result


def experiment_3():
    print("\n" + "#"*90)
    print("  EXPERIMENT 3 — 8-Case Sweep (184 robots in ONE scene)")
    print("  8 cases × 23 robots, separated by X offset")
    print("  One World, one reset, 300 steps — WLAN stochastic per robot")
    print("#"*90)

    all_runs = []
    for run_idx in range(5):
        print(f"\n  [EXP3] Run {run_idx+1}/5  (seed={run_idx}) ...")
        result = run_exp3_single(seed=run_idx)
        all_runs.append(result)
        print_exp3_run(run_idx, result)

    all_metrics = list(all_runs[0].keys())
    stats_input = {m: [r[m] for r in all_runs] for m in all_metrics}
    stats = print_stats_table("EXP3 — 8-Case Sweep", stats_input)
    save_stats_csv("exp3_8case_sweep.csv", all_runs, stats)
    sys.stdout.flush()

#  EXPERIMENT 4 — Heterogeneous WLAN delay
#
#  Mirrors reference code (document 18) exactly:
#    — Single persistent World created once (no fresh_stage between scenarios)
#    — Robots + controllers in loop BEFORE ground plane
#    — world.reset() once after ground plane
#    — while simulation_app.is_running() with reset_needed state machine
#    — CommChannel takes per-robot delay_steps from SHARED_WLAN dict
#    — Scenarios switched by resetting world + channels without rebuilding
# =============================================================================

EXP4_SHARED_WLAN = dict(
    jitter_steps   = 2,
    loss_prob      = 0.05,
    corrupt_std    = 0.01,
    link_fail_prob = 0.002,
)

EXP4_SCENARIOS = {
    'A': {
        'label': 'Front degradation (I5 worst)',
        'delays': {'I1':2,'I2':4,'I3':6,'I4':8,'I5':10},
    },
    'B': {
        'label': 'Third robot worst (I3 worst)',
        'delays': {'I1':3,'I2':3,'I3':10,'I4':3,'I5':3},
    },
}

EXP4_ROBOT_NAMES = ['I1','I2','I3','I4','I5']
EXP4_INIT_POS = {
    'I1':(0.0,0.0,0.02),'I2':(0.0,2.0,0.02),
    'I3':(0.0,4.0,0.02),'I4':(0.0,6.0,0.02),'I5':(0.0,8.0,0.02),
}
EXP4_NEIGHBOR_PAIRS = [
    ('I1','I2','dist_I1_I2'),('I2','I3','dist_I2_I3'),
    ('I3','I4','dist_I3_I4'),('I4','I5','dist_I4_I5'),
]
EXP4_TOTAL_STEPS = 300
EXP4_FORWARD_CMD = (1.0, 0.0, 0.0)
EXP4_INIT_GAP    = 2.0


class Exp4HeterogeneousWlan:
    """
    Mirrors document 18 structure exactly.
    One instance per run. Runs both scenarios inside the same World
    using the while simulation_app.is_running() state machine.
    """

    def __init__(self, kaya_usd):
        # ── 1. World ─────────────────────────────────────────────────────────
        self.world       = World(stage_units_in_meters=1.0)
        self.kaya_usd    = kaya_usd
        self.robots      = {}
        self.controllers = {}

        # ── 2. Robots + controllers in same loop BEFORE ground ────────────────
        for name in EXP4_ROBOT_NAMES:
            pos  = EXP4_INIT_POS[name]
            path = f"/World/Kaya_{name}"
            robot = self.world.scene.add(
                WheeledRobot(
                    prim_path=path,
                    name=f"kaya_{name}",
                    wheel_dof_names=["axle_0_joint","axle_1_joint","axle_2_joint"],
                    create_robot=True,
                    usd_path=kaya_usd,
                    position=np.array(pos),
                    orientation=np.array([1.0,0.0,0.0,0.0]),
                )
            )
            setup = HolonomicRobotUsdSetup(
                robot_prim_path=robot.prim_path,
                com_prim_path=f"{path}/base_link/control_offset",
            )
            wr, wp, wo, ma, wa, ua = setup.get_holonomic_controller_params()
            self.controllers[name] = HolonomicController(
                name=f"ctrl_{name}",
                wheel_radius=wr, wheel_positions=wp, wheel_orientations=wo,
                mecanum_angles=ma, wheel_axis=wa, up_axis=ua,
            )
            self.robots[name] = robot

        # ── 3. Ground plane AFTER robots ──────────────────────────────────────
        self.world.scene.add_default_ground_plane()

        # ── 4. Single reset ───────────────────────────────────────────────────
        self.world.reset()

    def run_both_scenarios(self):
        """
        Runs Scenario A then B in the same World.
        Mirrors the for-loop over SCENARIOS + while simulation_app.is_running()
        from document 18 exactly.
        Returns combined result dict for both scenarios.
        """
        combined = {}

        for sc_key, sc_cfg in EXP4_SCENARIOS.items():
            # Build fresh channels with this scenario's delay profile
            channels = {
                n: CommChannel(n, dict(EXP4_SHARED_WLAN,
                                       delay_steps=sc_cfg['delays'][n]))
                for n in EXP4_ROBOT_NAMES
            }

            # Reset world + controllers for this scenario
            self.world.reset()
            for n in EXP4_ROBOT_NAMES:
                self.controllers[n].reset()

            # Store initial positions after reset
            init_pos = {}
            for n in EXP4_ROBOT_NAMES:
                p, _ = get_pose(self.robots[n])
                init_pos[n] = p.copy()

            step         = 0
            reset_needed = False
            done         = False

            while simulation_app.is_running() and not done:
                self.world.step(render=True)

                if self.world.is_stopped() and not reset_needed:
                    reset_needed = True

                if self.world.is_playing():
                    if reset_needed:
                        self.world.reset()
                        for n in EXP4_ROBOT_NAMES:
                            self.controllers[n].reset()
                            channels[n].reset()
                        reset_needed = False

                    # Send + receive + apply
                    for n in EXP4_ROBOT_NAMES:
                        channels[n].send(EXP4_FORWARD_CMD)
                        actual = channels[n].receive()
                        self.robots[n].apply_wheel_actions(
                            self.controllers[n].forward(command=list(actual))
                        )

                    step += 1
                    if step >= EXP4_TOTAL_STEPS:
                        done = True

            # ── Collect final step results ────────────────────────────────────
            positions = {}
            for n in EXP4_ROBOT_NAMES:
                pos, euler = get_pose(self.robots[n])
                positions[n] = pos
                dx   = pos[0] - init_pos[n][0]
                dy   = pos[1] - init_pos[n][1]
                dist = float(np.sqrt(dx**2 + dy**2))
                s    = channels[n].stats
                tot  = max(s['sent'], 1)
                combined[f"Sc{sc_key}_{n}_dx"]       = round(dx,   6)
                combined[f"Sc{sc_key}_{n}_dy"]       = round(dy,   6)
                combined[f"Sc{sc_key}_{n}_dist"]     = round(dist, 6)
                combined[f"Sc{sc_key}_{n}_yaw"]      = round(euler[2], 4)
                combined[f"Sc{sc_key}_{n}_loss_pct"] = round(
                    100*s['dropped']/tot, 4)
                combined[f"Sc{sc_key}_{n}_corrupt"]  = s['corrupted']
                combined[f"Sc{sc_key}_{n}_linkfail"] = s['link_fail']
                combined[f"Sc{sc_key}_{n}_delay"]    = sc_cfg['delays'][n]

            for ra, rb, col in EXP4_NEIGHBOR_PAIRS:
                dist = euclidean_dist(positions[ra], positions[rb])
                dev  = dist - EXP4_INIT_GAP
                combined[f"Sc{sc_key}_{col}"]     = round(dist, 6)
                combined[f"Sc{sc_key}_{col}_dev"] = round(dev,  6)

        return combined


def run_exp4_single(seed):
    set_seed(seed)
    fresh_stage()
    exp4 = Exp4HeterogeneousWlan(KAYA_USD)
    return exp4.run_both_scenarios()


def experiment_4():
    print("\n" + "#"*90)
    print("  EXPERIMENT 4 — Heterogeneous WLAN Delay (Scenario A vs B)")
    print("  5 robots, 600 steps, 5 runs")
    print("  Mirrors doc 18: single World, while loop, reset_needed state machine")
    print("#"*90)

    all_runs = []
    for run_idx in range(5):
        print(f"\n  [EXP4] Run {run_idx+1}/5  (seed={run_idx}) ...")
        result = run_exp4_single(seed=run_idx)
        all_runs.append(result)
        print_exp4_run(run_idx, result)

    all_metrics = list(all_runs[0].keys())
    stats_input = {m: [r[m] for r in all_runs] for m in all_metrics}
    stats = print_stats_table("EXP4 — Heterogeneous WLAN Delay", stats_input)
    save_stats_csv("exp4_wlan_delay.csv", all_runs, stats)
    sys.stdout.flush()


# =============================================================================
#  EXPERIMENT 5A — Velocity effect (SLOW 0.2 vs FAST 1.0 m/s)
#  Default simulation values, straight path, 4500 steps
#  Single robot, fresh World per speed
# =============================================================================

EXP5A_SPEEDS = [0.2, 1.0]
EXP5A_STEPS  = 300


def run_exp5a_single(seed):
    set_seed(seed)
    result = {}

    for vx in EXP5A_SPEEDS:
        fresh_stage()
        world = World(stage_units_in_meters=1.0)
        path  = "/World/Kaya_test"

        robot = world.scene.add(
            WheeledRobot(
                prim_path=path,
                name="kaya_test",
                wheel_dof_names=["axle_0_joint","axle_1_joint","axle_2_joint"],
                create_robot=True,
                usd_path=KAYA_USD,
                position=np.array([0.0,0.0,0.02]),
                orientation=np.array([1.0,0.0,0.0,0.0]),
            )
        )
        kaya_setup = HolonomicRobotUsdSetup(
            robot_prim_path=robot.prim_path,
            com_prim_path=f"{path}/base_link/control_offset"
        )
        (wr, wp, wo, ma, wa, ua) = kaya_setup.get_holonomic_controller_params()
        ctrl = HolonomicController(
            name="ctrl_test",
            wheel_radius=wr, wheel_positions=wp, wheel_orientations=wo,
            mecanum_angles=ma, wheel_axis=wa, up_axis=ua,
        )

        world.scene.add_default_ground_plane()
        world.reset()

        init_p, _ = get_pose(robot)

        for step in range(EXP5A_STEPS + 1):
            robot.apply_wheel_actions(ctrl.forward(command=[vx, 0.0, 0.0]))
            world.step(render=True)

        pos, euler = get_pose(robot)
        dx   = pos[0] - init_p[0]
        dy   = pos[1] - init_p[1]
        dist = float(np.sqrt(dx**2 + dy**2))
        label = f"vx{vx:.1f}"
        result[f"{label}_dx"]   = round(dx,       6)
        result[f"{label}_dy"]   = round(dy,       6)
        result[f"{label}_dist"] = round(dist,     6)
        result[f"{label}_yaw"]  = round(euler[2], 4)

    return result


def experiment_5a():
    print("\n" + "#"*90)
    print("  EXPERIMENT 5A — Velocity Effect (0.2 vs 1.0 m/s)")
    print("  Default simulation values, 4500 steps, 5 runs")
    print("#"*90)

    all_runs = []
    for run_idx in range(5):
        print(f"\n  [EXP5A] Run {run_idx+1}/5  (seed={run_idx}) ...")
        result = run_exp5a_single(seed=run_idx)
        all_runs.append(result)
        print_exp5a_run(run_idx, result)

    all_metrics = list(all_runs[0].keys())
    stats_input = {m: [r[m] for r in all_runs] for m in all_metrics}
    stats = print_stats_table("EXP5A — Velocity Effect", stats_input)
    save_stats_csv("exp5a_velocity.csv", all_runs, stats)
    sys.stdout.flush()


# =============================================================================

# =============================================================================
#  EXPERIMENT 5B — Rigid body damping sweep
#  All 8 damping configs in ONE scene simultaneously
#  One World, one reset, 300 steps — no fresh stage between configs
#
#  Layout: 8 robots at y=0,3,6,...,21  all at x=0
#  Each robot has same friction but different linear/angular damping
#  Damping applied to ALL RigidBodyAPI prims AFTER reset (mirrors doc 20)
# =============================================================================

EXP5B_DAMPING_CONFIGS = [
    {"name": "No_Damping",             "linear": 0.0, "angular": 0.0,  "friction_static": 1.0, "friction_dynamic": 0.8},
    {"name": "Low_Linear_Damping",     "linear": 0.1, "angular": 0.0,  "friction_static": 1.0, "friction_dynamic": 0.8},
    {"name": "Medium_Linear_Damping",  "linear": 0.5, "angular": 0.0,  "friction_static": 1.0, "friction_dynamic": 0.8},
    {"name": "High_Linear_Damping",    "linear": 1.0, "angular": 0.0,  "friction_static": 1.0, "friction_dynamic": 0.8},
    {"name": "Low_Angular_Damping",    "linear": 0.0, "angular": 0.05, "friction_static": 1.0, "friction_dynamic": 0.8},
    {"name": "Medium_Angular_Damping", "linear": 0.0, "angular": 0.2,  "friction_static": 1.0, "friction_dynamic": 0.8},
    {"name": "High_Angular_Damping",   "linear": 0.0, "angular": 0.5,  "friction_static": 1.0, "friction_dynamic": 0.8},
    {"name": "Both_High_Damping",      "linear": 1.0, "angular": 0.5,  "friction_static": 1.0, "friction_dynamic": 0.8},
]

EXP5B_STEPS = 300
EXP5B_CMD   = (1.0, 0.0, 0.0)


def exp5b_create_ground(stage):
    """Custom ground cube with friction — mirrors doc 20 exactly."""
    ground_path = "/World/GroundPlane"
    ground_geom = UsdGeom.Cube.Define(stage, ground_path)
    ground_geom.CreateSizeAttr(1.0)
    ground_geom.AddTranslateOp().Set((0.0, 12.0, -0.5))
    ground_geom.AddScaleOp().Set((200.0, 50.0, 1.0))
    ground_prim = stage.GetPrimAtPath(ground_path)
    UsdPhysics.CollisionAPI.Apply(ground_prim)
    rba = UsdPhysics.RigidBodyAPI.Apply(ground_prim)
    rba.CreateKinematicEnabledAttr().Set(True)

    mat_path = f"{ground_path}/GroundPhysicsMaterial"
    material  = UsdShade.Material.Define(stage, mat_path)
    mat_prim  = material.GetPrim()

    phys_api = UsdPhysics.MaterialAPI.Apply(mat_prim)
    phys_api.CreateStaticFrictionAttr().Set(1.0)
    phys_api.CreateDynamicFrictionAttr().Set(1.0)
    phys_api.CreateRestitutionAttr().Set(0.0)

    PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
    mat_prim.CreateAttribute("physxMaterial:staticFriction",
                             Sdf.ValueTypeNames.Float).Set(1.0)
    mat_prim.CreateAttribute("physxMaterial:dynamicFriction",
                             Sdf.ValueTypeNames.Float).Set(1.0)
    mat_prim.CreateAttribute("physxMaterial:restitution",
                             Sdf.ValueTypeNames.Float).Set(0.0)
    mat_prim.CreateAttribute("physxMaterial:frictionCombineMode",
                             Sdf.ValueTypeNames.Token).Set("average")

    UsdShade.MaterialBindingAPI(ground_prim).Bind(material)


def exp5b_apply_wheel_friction(stage, robot_path, config):
    """Apply wheel friction BEFORE reset — mirrors doc 20 apply_all_frictions."""
    mat_path = f"{robot_path}/WheelPhysicsMaterial"
    material  = UsdShade.Material.Define(stage, mat_path)
    mat_prim  = material.GetPrim()

    phys_api = UsdPhysics.MaterialAPI.Apply(mat_prim)
    phys_api.CreateStaticFrictionAttr().Set(config['friction_static'])
    phys_api.CreateDynamicFrictionAttr().Set(config['friction_dynamic'])
    phys_api.CreateRestitutionAttr().Set(0.0)

    PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
    mat_prim.CreateAttribute("physxMaterial:staticFriction",
                             Sdf.ValueTypeNames.Float).Set(config['friction_static'])
    mat_prim.CreateAttribute("physxMaterial:dynamicFriction",
                             Sdf.ValueTypeNames.Float).Set(config['friction_dynamic'])
    mat_prim.CreateAttribute("physxMaterial:restitution",
                             Sdf.ValueTypeNames.Float).Set(0.0)
    mat_prim.CreateAttribute("physxMaterial:frictionCombineMode",
                             Sdf.ValueTypeNames.Token).Set("average")

    robot_prim = stage.GetPrimAtPath(robot_path)
    for prim in Usd.PrimRange(robot_prim):
        nm = prim.GetName().lower()
        if "axle" in nm or "wheel" in nm:
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI(prim).Bind(material)
            for child in prim.GetChildren():
                if child.IsA(UsdGeom.Mesh) or child.HasAPI(UsdPhysics.CollisionAPI):
                    UsdShade.MaterialBindingAPI(child).Bind(material)


def exp5b_apply_damping(stage, robot_path, config):
    """Apply damping to all RigidBodyAPI prims AFTER reset — mirrors doc 20."""
    robot_prim = stage.GetPrimAtPath(robot_path)
    for prim in Usd.PrimRange(robot_prim):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            lin_attr = prim.GetAttribute("physics:linearDamping")
            ang_attr = prim.GetAttribute("physics:angularDamping")
            if lin_attr:
                lin_attr.Set(config['linear'])
            else:
                prim.CreateAttribute("physics:linearDamping",
                                     Sdf.ValueTypeNames.Float).Set(config['linear'])
            if ang_attr:
                ang_attr.Set(config['angular'])
            else:
                prim.CreateAttribute("physics:angularDamping",
                                     Sdf.ValueTypeNames.Float).Set(config['angular'])


def run_exp5b_single(seed):
    """
    All 8 damping configs in ONE scene.
    Friction BEFORE reset, damping AFTER reset — mirrors doc 20 order exactly.
    """
    set_seed(seed)
    fresh_stage()

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()

    all_robots = []   # list of (config, robot, controller, robot_path)

    # ── Create all 8 robots + apply wheel friction BEFORE reset ───────────────
    for i, config in enumerate(EXP5B_DAMPING_CONFIGS):
        y_pos    = float(i * 3)
        rob_path = f"/World/Kaya_{config['name']}"

        robot = world.scene.add(
            WheeledRobot(
                prim_path=rob_path,
                name=f"kaya_{config['name']}",
                wheel_dof_names=["axle_0_joint","axle_1_joint","axle_2_joint"],
                create_robot=True,
                usd_path=KAYA_USD,
                position=np.array([0.0, y_pos, 0.02]),
                orientation=np.array([1.0,0.0,0.0,0.0]),
            )
        )
        kaya_setup = HolonomicRobotUsdSetup(
            robot_prim_path=robot.prim_path,
            com_prim_path=f"{rob_path}/base_link/control_offset"
        )
        (wr, wp, wo, ma, wa, ua) = kaya_setup.get_holonomic_controller_params()
        ctrl = HolonomicController(
            name=f"ctrl_{config['name']}",
            wheel_radius=wr, wheel_positions=wp, wheel_orientations=wo,
            mecanum_angles=ma, wheel_axis=wa, up_axis=ua,
        )

        # Friction BEFORE reset
        exp5b_apply_wheel_friction(stage, rob_path, config)

        all_robots.append((config, robot, ctrl, rob_path))

    # ── Ground plane BEFORE reset ─────────────────────────────────────────────
    exp5b_create_ground(stage)

    # ── Reset ─────────────────────────────────────────────────────────────────
    world.reset()

    # ── Damping AFTER reset — mirrors doc 20 exactly ──────────────────────────
    for config, robot, ctrl, rob_path in all_robots:
        exp5b_apply_damping(stage, rob_path, config)

    # ── Store initial positions ───────────────────────────────────────────────
    init_positions = {}
    for config, robot, ctrl, rob_path in all_robots:
        pos, _ = get_pose(robot)
        init_positions[config['name']] = pos.copy()

    # ── Run 300 steps — all 8 robots simultaneously ───────────────────────────
    for step in range(EXP5B_STEPS + 1):
        for config, robot, ctrl, rob_path in all_robots:
            robot.apply_wheel_actions(ctrl.forward(command=list(EXP5B_CMD)))
        world.step(render=True)

    # ── Collect final results ─────────────────────────────────────────────────
    result = {}
    for config, robot, ctrl, rob_path in all_robots:
        pos, euler = get_pose(robot)
        init = init_positions[config['name']]
        dx   = pos[0] - init[0]
        dy   = pos[1] - init[1]
        dist = float(np.sqrt(dx**2 + dy**2))
        label = config['name']
        result[f"{label}_dx"]       = round(dx,              6)
        result[f"{label}_dy"]       = round(dy,              6)
        result[f"{label}_dist"]     = round(dist,            6)
        result[f"{label}_yaw"]      = round(euler[2],        4)
        result[f"{label}_lin_damp"] = config['linear']
        result[f"{label}_ang_damp"] = config['angular']

    return result


def experiment_5b():
    print("\n" + "#"*90)
    print("  EXPERIMENT 5B — Rigid Body Damping Sweep")
    print("  8 damping configs in ONE scene, one reset, 300 steps, 5 runs")
    print("  Friction BEFORE reset, damping AFTER reset — mirrors doc 20")
    print("#"*90)

    all_runs = []
    for run_idx in range(5):
        print(f"\n  [EXP5B] Run {run_idx+1}/5  (seed={run_idx}) ...")
        result = run_exp5b_single(seed=run_idx)
        all_runs.append(result)
        print_exp5b_run(run_idx, result)

    all_metrics = list(all_runs[0].keys())
    stats_input = {m: [r[m] for r in all_runs] for m in all_metrics}
    stats = print_stats_table("EXP5B — Damping Sweep", stats_input)
    save_stats_csv("exp5b_damping.csv", all_runs, stats)
    sys.stdout.flush()

#  EXPERIMENT 6 — Kaya vs Jetbot at 0.2 m/s linear
#  600 steps, default simulation values
#
#  Setup order mirrors your working reference exactly:
#    1. World()
#    2. scene.add(Kaya)
#    3. scene.add(Jetbot)
#    4. scene.add_default_ground_plane()
#    5. HolonomicRobotUsdSetup → fully unpack → HolonomicController
#    6. DifferentialController (wheel_radius=0.03, wheel_base=0.1125)
#    7. world.reset()  — single reset at end of setup
# =============================================================================

EXP6_STEPS = 600
EXP6_VX    = 0.2


def run_exp6_single(seed):
    set_seed(seed)
    fresh_stage()

    # ── 1. World ──────────────────────────────────────────────────────────────
    world = World(stage_units_in_meters=1.0)

    # ── 2. Kaya robot ─────────────────────────────────────────────────────────
    my_kaya_1 = world.scene.add(
        WheeledRobot(
            prim_path="/World/Kaya_1",
            name="my_kaya_1",
            wheel_dof_names=["axle_0_joint","axle_1_joint","axle_2_joint"],
            create_robot=True,
            usd_path=KAYA_USD,
            position=np.array([-3.0, 0.0, 0.02]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
    )

    # ── 3. Jetbot robot ───────────────────────────────────────────────────────
    my_jetbot = world.scene.add(
        WheeledRobot(
            prim_path="/World/Jetbot",
            name="my_jetbot",
            wheel_dof_names=["left_wheel_joint","right_wheel_joint"],
            create_robot=True,
            usd_path=JETBOT_USD,
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),  # 180° around Z
        )
    )

    # ── 4. Ground plane ───────────────────────────────────────────────────────
    world.scene.add_default_ground_plane()

    # ── 5 & 6. Controllers ────────────────────────────────────────────────────
    kaya_setup_1 = HolonomicRobotUsdSetup(
        robot_prim_path=my_kaya_1.prim_path,
        com_prim_path="/World/Kaya_1/base_link/control_offset"
    )
    (
        wheel_radius_1,
        wheel_positions_1,
        wheel_orientations_1,
        mecanum_angles_1,
        wheel_axis_1,
        up_axis_1,
    ) = kaya_setup_1.get_holonomic_controller_params()
    my_controller_1 = HolonomicController(
        name="holonomic_controller_1",
        wheel_radius=wheel_radius_1,
        wheel_positions=wheel_positions_1,
        wheel_orientations=wheel_orientations_1,
        mecanum_angles=mecanum_angles_1,
        wheel_axis=wheel_axis_1,
        up_axis=up_axis_1,
    )

    wheel_radius_jetbot = 0.03
    wheel_base_jetbot   = 0.1125
    my_controller_jetbot = DifferentialController(
        name="differential_controller_jetbot",
        wheel_radius=wheel_radius_jetbot,
        wheel_base=wheel_base_jetbot,
    )

    # ── 7. Single reset ───────────────────────────────────────────────────────
    world.reset()

    kaya_init,   _ = get_pose(my_kaya_1)
    jetbot_init, _ = get_pose(my_jetbot)

    # ── Simulation loop ───────────────────────────────────────────────────────
    reset_needed = False
    step         = 0

    while step < EXP6_STEPS:
        world.step(render=True)

        if world.is_stopped() and not reset_needed:
            reset_needed = True

        if world.is_playing():
            if reset_needed:
                world.reset()
                my_controller_1.reset()
                reset_needed = False

            my_kaya_1.apply_wheel_actions(
                my_controller_1.forward(command=[EXP6_VX, 0.0, 0.0])
            )
            my_jetbot.apply_wheel_actions(
                my_controller_jetbot.forward(command=[EXP6_VX, 0.0])
            )
            step += 1

    kaya_pos,   kaya_euler   = get_pose(my_kaya_1)
    jetbot_pos, jetbot_euler = get_pose(my_jetbot)

    def metrics(pos, init, euler, prefix):
        dx   = pos[0] - init[0]
        dy   = pos[1] - init[1]
        dist = float(np.sqrt(dx**2 + dy**2))
        return {
            f"{prefix}_dx":   round(dx,       6),
            f"{prefix}_dy":   round(dy,       6),
            f"{prefix}_dist": round(dist,     6),
            f"{prefix}_yaw":  round(euler[2], 4),
        }

    result = {}
    result.update(metrics(kaya_pos,   kaya_init,   kaya_euler,   "kaya"))
    result.update(metrics(jetbot_pos, jetbot_init, jetbot_euler, "jetbot"))
    return result


def experiment_6():
    print("\n" + "#"*90)
    print("  EXPERIMENT 6 — Kaya vs Jetbot at 0.2 m/s Linear")
    print("  600 steps, default values, 5 runs")
    print("#"*90)

    all_runs = []
    for run_idx in range(5):
        print(f"\n  [EXP6] Run {run_idx+1}/5  (seed={run_idx}) ...")
        result = run_exp6_single(seed=run_idx)
        all_runs.append(result)
        print_exp6_run(run_idx, result)

    all_metrics = list(all_runs[0].keys())
    stats_input = {m: [r[m] for r in all_runs] for m in all_metrics}
    stats = print_stats_table("EXP6 — Kaya vs Jetbot 0.2 m/s", stats_input)
    save_stats_csv("exp6_kaya_vs_jetbot.csv", all_runs, stats)
    sys.stdout.flush()

# =============================================================================
#  MAIN — run all experiments sequentially
# =============================================================================

print("\n" + "="*90)
print("  ISAAC SIM COMPLETE EXPERIMENT SUITE")
print("  Experiments: 1, 2, 3, 4, 5A, 5B, 6")
print("  Each runs 5 times (seeds 0–4)")
print("  Only final step captured — statistics printed + saved to CSV")
print(f"  Output directory: {OUT_DIR}")
print("="*90)

experiment_1()
experiment_2()
experiment_3()
experiment_4()
experiment_5a()
experiment_5b()
experiment_6()

print("\n" + "="*90)
print("  ALL EXPERIMENTS COMPLETE")
print(f"  Results saved to: {OUT_DIR}")
print("="*90)

simulation_app.close()