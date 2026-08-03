"""
=============================================================================
  COMBINED EXPERIMENT FILE
  EXP 7  — Kaya + Jetbot fleet (60 cases, 120 robots, 300 steps)
  EXP 4B — Heterogeneous WLAN delay 3 scenarios (15 robots, 300 steps)
  EXP 8  — 23-robot Kaya fleet (doc 23, 300 steps)
  Each experiment: 5 runs, seeds 0-4
  All results printed immediately + statistics after 5 runs + CSV saved
=============================================================================
"""

# ── Bootstrap (must be first) ─────────────────────────────────────────────────
## BOOTSTRAP: SimulationApp must be created before any other isaacsim import;
## this line launches the simulator (headless=False shows the viewport).
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# ── Standard imports ──────────────────────────────────────────────────────────
import carb
import csv
import os
import sys
import random
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation

# ── Isaac Sim imports ─────────────────────────────────────────────────────────
from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.controllers.holonomic_controller import HolonomicController
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.robots.holonomic_robot_usd_setup import HolonomicRobotUsdSetup
from isaacsim.storage.native import get_assets_root_path

# ── USD / Omni imports ────────────────────────────────────────────────────────
from pxr import UsdShade, Usd, UsdGeom, UsdPhysics, Sdf
import omni.usd
import omni.kit.app

sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
#  ASSET PATHS
# =============================================================================

## Locate Isaac's bundled robot assets; abort if missing.
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    exit()

KAYA_USD   = assets_root_path + "/Isaac/Robots/Kaya/kaya.usd"
JETBOT_USD = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"

## Create a 'results' folder next to this script for the CSV output.
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
#  SHARED HELPERS
# =============================================================================

## Seed both RNGs so each run index is reproducible.
def set_seed(s):
    random.seed(s)
    np.random.seed(s)


## Wipe the 3D scene and tick a few frames so the clear fully applies.
def fresh_stage():
    omni.usd.get_context().new_stage()
    for _ in range(10):
        omni.kit.app.get_app().update()


## Return a robot's world position + Euler angles (yaw = euler[2]). Isaac
## quaternion is [w,x,y,z]; scipy wants [x,y,z,w], hence the reorder.
def get_pose(robot):
    pos, q = robot.get_world_pose()
    euler = Rotation.from_quat(
        [q[1], q[2], q[3], q[0]]
    ).as_euler('xyz', degrees=True)
    return pos, euler


## Straight-line XY distance between two points.
def euclid2d(a, b):
    return float(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))


## mean/std/variance/min/max for one metric across the 5 runs.
def compute_stats(values):
    a = np.array(values, dtype=float)
    return {
        'mean':     float(np.mean(a)),
        'std':      float(np.std(a)),
        'variance': float(np.var(a)),
        'min':      float(np.min(a)),
        'max':      float(np.max(a)),
    }


## Print a stats table; sd maps each metric to its list of 5 run-values.
def print_stats_table(title, sd):
    print(f"\n{'='*95}")
    print(f"  STATISTICS — {title}")
    print(f"{'='*95}")
    print(f"  {'Metric':<55} {'Mean':>9} {'Std':>9} {'Var':>11} {'Min':>9} {'Max':>9}")
    print(f"  {'-'*55} {'-'*9} {'-'*9} {'-'*11} {'-'*9} {'-'*9}")
    result = {}
    for metric, values in sd.items():
        s = compute_stats(values)
        result[metric] = s
        print(f"  {metric:<55} {s['mean']:>9.4f} {s['std']:>9.4f} "
              f"{s['variance']:>11.6f} {s['min']:>9.4f} {s['max']:>9.4f}")
    sys.stdout.flush()
    return result


## Write per-run values plus the summary stats to a CSV in OUT_DIR.
def save_stats_csv(filename, run_data, stats_data):
    path = os.path.join(OUT_DIR, filename)
    metrics = list(run_data[0].keys()) if run_data else []
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run'] + metrics)
        for i, row in enumerate(run_data):
            w.writerow([i] + [row.get(m, '') for m in metrics])
        w.writerow([])
        w.writerow(['STAT'] + metrics)
        for sn in ['mean', 'std', 'variance', 'min', 'max']:
            w.writerow([sn] + [
                stats_data[m][sn] if m in stats_data else ''
                for m in metrics
            ])
    print(f"\n  [CSV] → {path}")
    sys.stdout.flush()


## Spawn one Kaya and build its HolonomicController ([vx,vy,omega] command),
## reading wheel geometry from USD.
def make_kaya_ctrl(world, prim_path, name, position):
    """Create Kaya robot + HolonomicController, return (robot, ctrl)."""
    robot = world.scene.add(WheeledRobot(
        prim_path=prim_path,
        name=name,
        wheel_dof_names=["axle_0_joint", "axle_1_joint", "axle_2_joint"],
        create_robot=True,
        usd_path=KAYA_USD,
        position=np.array(position),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    ))
    ks = HolonomicRobotUsdSetup(
        robot_prim_path=robot.prim_path,
        com_prim_path=f"{prim_path}/base_link/control_offset"
    )
    wr, wp, wo, ma, wa, ua = ks.get_holonomic_controller_params()
    ctrl = HolonomicController(
        name=f"hctrl_{name}",
        wheel_radius=wr, wheel_positions=wp, wheel_orientations=wo,
        mecanum_angles=ma, wheel_axis=wa, up_axis=ua,
    )
    return robot, ctrl


## Spawn one Jetbot and build its DifferentialController ([forward,turn]),
## with wheel radius/base hardcoded (0.03 / 0.1125).
def make_jetbot_ctrl(world, prim_path, name, position):
    """Create Jetbot robot + DifferentialController, return (robot, ctrl).
    Hardcoded physical params from doc 25 — wheel_radius=0.03, wheel_base=0.1125
    Orientation [0,0,0,1] = 180° around Z — faces same direction as Kaya (doc 25).
    """
    robot = world.scene.add(WheeledRobot(
        prim_path=prim_path,
        name=name,
        wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
        create_robot=True,
        usd_path=JETBOT_USD,
        position=np.array(position),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),  # 180° around Z — doc 25
    ))
    ctrl = DifferentialController(
        name=f"dctrl_{name}",
        wheel_radius=0.03,
        wheel_base=0.1125,
    )
    return robot, ctrl


# =============================================================================
#  SHARED FRICTION HELPERS — doc 17 Sdf style
# =============================================================================

## Create a physics material with given static (s) / dynamic (d) friction.
def _fric_prim(stage, base_path, s, d):
    """Create nested Material/PhysicsMaterial and return UsdShade.Material."""
    mat_p  = stage.DefinePrim(f"{base_path}/Mat",     "Material")
    phys_p = stage.DefinePrim(f"{base_path}/Mat/Phy", "PhysicsMaterial")
    for attr, val in [
        ("physics:staticFriction",        s),
        ("physics:dynamicFriction",        d),
        ("physics:restitution",           0.1),
        ("physxMaterial:staticFriction",   s),
        ("physxMaterial:dynamicFriction",  d),
        ("physxMaterial:restitution",     0.1),
    ]:
        phys_p.CreateAttribute(attr, Sdf.ValueTypeNames.Float).Set(val)
    return UsdShade.Material(mat_p)


## Bind a friction material to a ground cube (unless override is off).
def apply_ground_fric(stage, ground_path, fric):
    """Apply friction to a ground cube prim."""
    if not fric['override']:
        return
    mat = _fric_prim(stage, ground_path, fric['static'], fric['dynamic'])
    UsdShade.MaterialBindingAPI.Apply(
        stage.GetPrimAtPath(ground_path)
    ).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


## Bind the friction material to every wheel/axle collision surface.
def apply_wheel_fric(stage, robot_path, fric, suffix="WM"):
    """Apply wheel friction to all collision prims of a robot."""
    if not fric['override']:
        return
    mat    = _fric_prim(stage, f"{robot_path}/{suffix}",
                        fric['static'], fric['dynamic'])
    rprim  = stage.GetPrimAtPath(robot_path)
    if not rprim or not rprim.IsValid():
        return
    for prim in Usd.PrimRange(rprim):
        nm = prim.GetName().lower()
        if any(k in nm for k in ["axle", "wheel", "left_wheel", "right_wheel"]):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    mat, UsdShade.Tokens.weakerThanDescendants, "physics")
            for ch in prim.GetChildren():
                if ch.IsA(UsdGeom.Mesh) or ch.HasAPI(UsdPhysics.CollisionAPI):
                    UsdShade.MaterialBindingAPI.Apply(ch).Bind(
                        mat, UsdShade.Tokens.weakerThanDescendants, "physics")


## Build one rectangular ground tile (scaled cube, kinematic collider).
def make_zone_ground(stage, prim_path, cx, cy, sx, sy):
    """Create a kinematic ground cube at center (cx,cy) with scale (sx,sy)."""
    g = UsdGeom.Cube.Define(stage, prim_path)
    g.CreateSizeAttr(1.0)
    g.AddTranslateOp().Set((cx, cy, -0.5))
    g.AddScaleOp().Set((sx, sy, 1.0))
    prim = stage.GetPrimAtPath(prim_path)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr().Set(True)
    return prim


# =============================================================================
#  SHARED COMM CHANNEL
#  Works for both Exp7 (preset dict) and Exp4B (delay_steps int + WLAN shared)
# =============================================================================

## Stochastic link model shared by EXP7 and EXP4B: delay, jitter, packet
## loss, command noise, and random outages. A zero-delay/zero-loss preset
## is treated as a perfect link.
class CommChannel:
    """
    Stochastic communication channel.
    preset: dict with delay_steps, jitter_steps, loss_prob, corrupt_std, link_fail_prob
    """
    def __init__(self, preset):
        self.delay   = preset['delay_steps']
        self.jitter  = preset['jitter_steps']
        self.loss    = preset['loss_prob']
        self.corrupt = preset['corrupt_std']
        self.lf_prob = preset['link_fail_prob']
        self.perfect = (self.delay == 0 and self.loss == 0.0)
        self.queue   = deque()
        self.last    = None
        self.ldown   = 0
        self.step    = 0
        self.stats   = {'sent': 0, 'dropped': 0, 'corrupted': 0, 'link_fail': 0}

    ## Try to transmit a command; may be dropped, corrupted, blocked, or queued
    ## for delayed delivery.
    def send(self, cmd):
        self.stats['sent'] += 1
        if self.perfect:
            self.last = tuple(cmd)
            return
        if self.ldown > 0:
            self.ldown -= 1
            self.stats['link_fail'] += 1
            return
        if random.random() < self.lf_prob:
            self.ldown = random.randint(10, 50)
            self.stats['link_fail'] += 1
            return
        if random.random() < self.loss:
            self.stats['dropped'] += 1
            return
        c     = list(cmd)
        noise = [random.gauss(0, self.corrupt) for _ in range(len(c))]
        c     = [x + n for x, n in zip(c, noise)]
        if any(abs(n) > self.corrupt * 0.5 for n in noise):
            self.stats['corrupted'] += 1
        jit = random.randint(-self.jitter, self.jitter) if self.jitter > 0 else 0
        deliver = max(self.step + self.delay + jit, self.step + 1)
        self.queue.append((deliver, tuple(c)))

    ## Advance one step; deliver any queued commands now due (keep the latest);
    ## fall back to the last received command or the default.
    def receive(self, default):
        self.step += 1
        if self.perfect:
            return self.last if self.last is not None else tuple(default)
        rec = None
        while self.queue and self.queue[0][0] <= self.step:
            _, rec = self.queue.popleft()
        if rec is not None:
            self.last = rec
        return self.last if self.last is not None else tuple(default)

    ## Clear all channel state.
    def reset(self):
        self.queue = deque()
        self.last  = None
        self.ldown = 0
        self.step  = 0
        self.stats = {'sent': 0, 'dropped': 0, 'corrupted': 0, 'link_fail': 0}


# =============================================================================
#  EXP 7 — Combined Kaya + Jetbot Fleet
#  60 cases = 4 comm × 5 friction × 3 velocity
#  120 robots in one scene, 300 steps (10 slots × 30 steps), 5 runs
# =============================================================================

## EXP7 config: 300 steps in 10 command slots of 30, cases spaced 6 m on X.
E7_TOTAL_STEPS   = 300
E7_STEPS_PER_CMD = 30
E7_CASE_X_OFFSET = 6.0  # 4m ground patch + 2m gap between cases

## Four link qualities, from a perfect wired link to flaky bluetooth.
E7_COMM_PRESETS = {
    'no_wlan':   {'delay_steps':  0, 'jitter_steps': 0,
                  'loss_prob': 0.0,   'corrupt_std': 0.0,   'link_fail_prob': 0.0},
    'cable':     {'delay_steps':  1, 'jitter_steps': 0,
                  'loss_prob': 0.001, 'corrupt_std': 0.001, 'link_fail_prob': 0.0},
    'wlan':      {'delay_steps':  5, 'jitter_steps': 2,
                  'loss_prob': 0.05,  'corrupt_std': 0.01,  'link_fail_prob': 0.002},
    'bluetooth': {'delay_steps': 10, 'jitter_steps': 4,
                  'loss_prob': 0.10,  'corrupt_std': 0.02,  'link_fail_prob': 0.005},
}
E7_COMM_LABELS = ['no_wlan', 'cable', 'wlan', 'bluetooth']

## Five friction settings, from frictionless to very sticky.
E7_FRICTION = [
    {"label": "Default", "override": False, "static": 0.0, "dynamic":  0.0},
    {"label": "s0_d0",   "override": True,  "static": 0.0, "dynamic":  0.0},
    {"label": "s1_d1",   "override": True,  "static": 1.0, "dynamic":  1.0},
    {"label": "s1_d5",   "override": True,  "static": 1.0, "dynamic":  5.0},
    {"label": "s1_d10",  "override": True,  "static": 1.0, "dynamic": 10.0},
]

E7_VEL_LABELS = ['scurve', 'straight_02', 'straight_01']

## Kaya S-curve commands [vx, vy, omega], one per slot.
E7_KAYA_SCURVE = [
    (0.200, 0.000,  0.0000), (0.200, 0.010,  0.0100),
    (0.194, 0.040,  0.0305), (0.172, 0.084,  0.0492),
    (0.126, 0.134,  0.0705), (0.078, 0.178,  0.0699),
    (0.052, 0.188,  0.0287), (0.082, 0.184, -0.0303),
    (0.106, 0.166, -0.0294), (0.124, 0.156, -0.0208),
]

## Matching Jetbot S-curve commands [left_wheel, right_wheel].
E7_JETBOT_SCURVE = [
    (0.2000, 0.2000), (0.1997, 0.2008), (0.1963, 0.1999), (0.1885, 0.1943),
    (0.1798, 0.1881), (0.1902, 0.1984), (0.1934, 0.1967), (0.2032, 0.1997),
    (0.1987, 0.1952), (0.2005, 0.1981),
]

# Build 60 cases
## Build the 60 EXP7 cases as the cross-product comm x friction x velocity.
E7_CASES = []
for _comm in E7_COMM_LABELS:
    for _fric in E7_FRICTION:
        for _vel in E7_VEL_LABELS:
            E7_CASES.append({
                'comm':  _comm,
                'fric':  _fric,
                'vel':   _vel,
                'label': f"{_comm}__{_fric['label']}__{_vel}",
            })
assert len(E7_CASES) == 60


## Kaya command for a velocity mode + slot.
def e7_kaya_cmd(vel, slot):
    if vel == 'scurve':        return E7_KAYA_SCURVE[slot]
    elif vel == 'straight_02': return (0.2, 0.0, 0.0)
    else:                      return (0.1, 0.0, 0.0)


## Jetbot command for a velocity mode + slot.
def e7_jetbot_cmd(vel, slot):
    if vel == 'scurve':        return E7_JETBOT_SCURVE[slot]
    elif vel == 'straight_02': return (0.2, 0.0)
    else:                      return (0.1, 0.0)


## Print one EXP7 run grouped by velocity mode.
def e7_print_run(run_idx, result):
    print(f"\n  {'='*95}")
    print(f"  RUN {run_idx+1} — EXP7  Kaya+Jetbot Fleet (60 cases, 300 steps)")
    print(f"  {'='*95}")
    for vel in E7_VEL_LABELS:
        print(f"\n  [Mode: {vel}]")
        print(f"  {'Comm':<12} {'Friction':<10} "
              f"{'K_dist':>8} {'K_yaw°':>7} "
              f"{'J_dist':>8} {'J_yaw°':>7} {'KJ_euclid':>10}")
        print(f"  {'-'*12} {'-'*10} {'-'*8} {'-'*7} {'-'*8} {'-'*7} {'-'*10}")
        for case in E7_CASES:
            if case['vel'] != vel:
                continue
            lb = case['label']
            print(f"  {case['comm']:<12} {case['fric']['label']:<10} "
                  f"{result.get(lb+'_kaya_dist',   0):>8.4f} "
                  f"{result.get(lb+'_kaya_yaw',    0):>7.2f} "
                  f"{result.get(lb+'_jetbot_dist',  0):>8.4f} "
                  f"{result.get(lb+'_jetbot_yaw',   0):>7.2f} "
                  f"{result.get(lb+'_kj_euclid',   0):>10.4f}")
    sys.stdout.flush()


## One EXP7 run: build all 60 Kaya+Jetbot pairs on their own ground tiles,
## drive 300 steps sending each command through its comm channel, then measure.
def run_exp7(seed):
    set_seed(seed)
    fresh_stage()

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    pairs = []

    for idx, case in enumerate(E7_CASES):
        x_off  = float(idx) * E7_CASE_X_OFFSET
        x_ctr  = x_off + E7_CASE_X_OFFSET / 2.0
        fric   = case['fric']
        preset = E7_COMM_PRESETS[case['comm']]

        # Kaya at (x_off - 3, 0, 0.02), Jetbot at (x_off, 0, 0)
        # Mirrors doc 25: Kaya behind Jetbot on X axis, same Y=0 line
        kaya_x  = x_off - 3.0
        jetbt_x = x_off

        # Ground zone covers both robots: center at x_off-1.5, width 5m
        gp_cx = x_off - 1.5
        gp = make_zone_ground(stage, f"/World/G7_{idx}",
                              gp_cx, 0.0, 5.0, 3.0)
        apply_ground_fric(stage, f"/World/G7_{idx}", fric)

        # Kaya
        kp = f"/World/K7_{idx}"
        kaya, kctrl = make_kaya_ctrl(world, kp, f"k7_{idx}",
                                     [kaya_x, 0.0, 0.02])
        apply_wheel_fric(stage, kp, fric, "KWM")

        # Jetbot — orientation [0,0,0,1] set inside make_jetbot_ctrl (doc 25)
        jp = f"/World/J7_{idx}"
        jetbot, jctrl = make_jetbot_ctrl(world, jp, f"j7_{idx}",
                                         [jetbt_x, 0.0, 0.0])
        apply_wheel_fric(stage, jp, fric, "JWM")

        pairs.append({
            'case':   case,
            'kaya':   kaya,   'kctrl': kctrl,
            'kch':    CommChannel(preset),
            'jetbot': jetbot, 'jctrl': jctrl,
            'jch':    CommChannel(preset),
        })

    world.reset()

    # Initial positions
    for p in pairs:
        ki, _ = get_pose(p['kaya'])
        ji, _ = get_pose(p['jetbot'])
        p['ki'] = ki.copy()
        p['ji'] = ji.copy()

    # 300 steps
    ## EXP7 drive loop: route each robot's command through its channel and apply it.
    for step in range(E7_TOTAL_STEPS + 1):
        slot = min(step // E7_STEPS_PER_CMD, 9)
        for p in pairs:
            vel = p['case']['vel']
            ki  = e7_kaya_cmd(vel, slot)
            p['kch'].send(ki)
            ka = p['kch'].receive(ki)
            p['kaya'].apply_wheel_actions(p['kctrl'].forward(command=list(ka)))

            ji = e7_jetbot_cmd(vel, slot)
            p['jch'].send(ji)
            ja = p['jch'].receive(ji)
            p['jetbot'].apply_wheel_actions(p['jctrl'].forward(command=list(ja)))

        world.step(render=True)

    # Collect results
    result = {}
    for p in pairs:
        lb   = p['case']['label']
        kpos, ke = get_pose(p['kaya'])
        jpos, je = get_pose(p['jetbot'])
        kdx  = kpos[0] - p['ki'][0]; kdy = kpos[1] - p['ki'][1]
        jdx  = jpos[0] - p['ji'][0]; jdy = jpos[1] - p['ji'][1]
        result[f"{lb}_kaya_dist"]   = round(float(np.sqrt(kdx**2+kdy**2)), 6)
        result[f"{lb}_kaya_yaw"]    = round(ke[2], 4)
        result[f"{lb}_jetbot_dist"] = round(float(np.sqrt(jdx**2+jdy**2)), 6)
        result[f"{lb}_jetbot_yaw"]  = round(je[2], 4)
        result[f"{lb}_kj_euclid"]   = round(euclid2d(kpos, jpos), 6)
    return result


## Run EXP7 five times, aggregate stats, and save the CSV.
def experiment_7():
    print("\n" + "#"*95)
    print("  EXP 7 — Combined Kaya+Jetbot Fleet")
    print("  60 cases | 120 robots | 300 steps (10×30) | 5 runs")
    print("#"*95)
    all_runs = []
    for run_idx in range(5):
        print(f"\n  [EXP7] Run {run_idx+1}/5  (seed={run_idx}) ...")
        res = run_exp7(seed=run_idx)
        all_runs.append(res)
        e7_print_run(run_idx, res)
    metrics     = list(all_runs[0].keys())
    stats_input = {m: [r[m] for r in all_runs] for m in metrics}
    stats       = print_stats_table("EXP7 — Kaya+Jetbot Fleet", stats_input)
    save_stats_csv("exp7_combined_fleet.csv", all_runs, stats)


# =============================================================================
#  EXP 4B — Heterogeneous WLAN Delay — 3 Scenarios
#  15 robots (3 scenarios × 5) in one scene, 300 steps, 5 runs
# =============================================================================

## EXP4B config: 300 steps; the 3 scenarios are offset 25 m apart on X.
E4B_TOTAL_STEPS    = 300
E4B_STEPS_PER_CMD  = 30
E4B_INIT_GAP       = 2.0
E4B_FORWARD_CMD    = (1.0, 0.0, 0.0)
E4B_SCENARIO_XOFF  = 25.0

E4B_ROBOT_NAMES = ['I1', 'I2', 'I3', 'I4', 'I5']

E4B_INIT_POS = {
    'I1': (0.0, 0.0, 0.02), 'I2': (0.0, 2.0, 0.02),
    'I3': (0.0, 4.0, 0.02), 'I4': (0.0, 6.0, 0.02),
    'I5': (0.0, 8.0, 0.02),
}

E4B_NEIGHBOR_PAIRS = [
    ('I1','I2','dist_I1_I2'), ('I2','I3','dist_I2_I3'),
    ('I3','I4','dist_I3_I4'), ('I4','I5','dist_I4_I5'),
]

## WLAN params shared by all EXP4B robots (only per-robot DELAY varies).
E4B_SHARED_WLAN = {
    'jitter_steps': 2, 'loss_prob': 0.05,
    'corrupt_std': 0.01, 'link_fail_prob': 0.002,
}

## Three delay scenarios: the worst 10-step delay is on the back (A),
## front (B), or middle (C) robot; the rest get 3.
E4B_SCENARIOS = {
    'A': {'label': 'Back robot worst (I1=10)',
          'delays': {'I1':10,'I2':3,'I3':3,'I4':3,'I5':3}},
    'B': {'label': 'Front robot worst (I5=10)',
          'delays': {'I1':3,'I2':3,'I3':3,'I4':3,'I5':10}},
    'C': {'label': 'Middle robot worst (I3=10)',
          'delays': {'I1':3,'I2':3,'I3':10,'I4':3,'I5':3}},
}


## Build a CommChannel preset from the shared WLAN params + this robot's delay.
def e4b_make_preset(delay_steps):
    """Build CommChannel preset for Exp4B from shared WLAN + per-robot delay."""
    return {
        'delay_steps':    delay_steps,
        'jitter_steps':   E4B_SHARED_WLAN['jitter_steps'],
        'loss_prob':      E4B_SHARED_WLAN['loss_prob'],
        'corrupt_std':    E4B_SHARED_WLAN['corrupt_std'],
        'link_fail_prob': E4B_SHARED_WLAN['link_fail_prob'],
    }


## Print one EXP4B run: per-robot distance/yaw/comm stats + neighbor gaps,
## grouped by scenario.
def e4b_print_run(run_idx, result):
    print(f"\n  {'='*90}")
    print(f"  RUN {run_idx+1} — EXP4B  Heterogeneous WLAN Delay (3 Scenarios)")
    print(f"  {'='*90}")
    for sc_key, sc_cfg in E4B_SCENARIOS.items():
        print(f"\n  [Scenario {sc_key}: {sc_cfg['label']}]")
        print(f"  {'Robot':<6} {'Delay':>6} {'Dist(m)':>9} {'Yaw°':>8} "
              f"{'Loss%':>7} {'Corrupt':>8} {'LinkFail':>9}")
        print(f"  {'-'*6} {'-'*6} {'-'*9} {'-'*8} {'-'*7} {'-'*8} {'-'*9}")
        for n in E4B_ROBOT_NAMES:
            pfx  = f"Sc{sc_key}_{n}"
            dist = result.get(f"{pfx}_dist",     0)
            yaw  = result.get(f"{pfx}_yaw",      0)
            loss = result.get(f"{pfx}_loss_pct", 0)
            cor  = result.get(f"{pfx}_corrupt",  0)
            lf   = result.get(f"{pfx}_linkfail", 0)
            dly  = sc_cfg['delays'][n]
            print(f"  {n:<6} {dly:>6} {dist:>9.4f} {yaw:>8.2f} "
                  f"{loss:>7.2f} {cor:>8} {lf:>9}")
        gap_str = '  '.join(
            f"{ra}↔{rb}="
            f"{result.get(f'Sc{sc_key}_{col}', 0):.4f}m"
            f"(dev={result.get(f'Sc{sc_key}_{col}_dev', 0):+.4f})"
            for ra, rb, col in E4B_NEIGHBOR_PAIRS
        )
        print(f"  Gaps: {gap_str}")
    sys.stdout.flush()


## One EXP4B run: 3 scenarios x 5 Kayas in one scene, all driving straight,
## each robot's link delayed per its scenario; measure drift and comm stats.
def run_exp4b(seed):
    set_seed(seed)
    fresh_stage()

    world = World(stage_units_in_meters=1.0)
    all_sc = []   # (sc_key, sc_cfg, robots, ctrls, channels)

    for sc_idx, (sc_key, sc_cfg) in enumerate(E4B_SCENARIOS.items()):
        x_off    = sc_idx * E4B_SCENARIO_XOFF
        robots   = {}
        ctrls    = {}
        channels = {}

        for n in E4B_ROBOT_NAMES:
            base = E4B_INIT_POS[n]
            pos  = (base[0] + x_off, base[1], base[2])
            path = f"/World/Sc{sc_key}_{n}"
            robot, ctrl = make_kaya_ctrl(world, path, f"sc{sc_key}_{n}", pos)
            robots[n]   = robot
            ctrls[n]    = ctrl
            channels[n] = CommChannel(e4b_make_preset(sc_cfg['delays'][n]))

        all_sc.append((sc_key, sc_cfg, robots, ctrls, channels))

    world.scene.add_default_ground_plane()
    world.reset()

    # Initial positions
    init_pos = {}
    for sc_key, sc_cfg, robots, ctrls, channels in all_sc:
        init_pos[sc_key] = {}
        for n in E4B_ROBOT_NAMES:
            p, _ = get_pose(robots[n])
            init_pos[sc_key][n] = p.copy()

    # 300 steps
    ## EXP4B drive loop: every robot sends/receives its delayed forward command.
    for step in range(E4B_TOTAL_STEPS + 1):
        for sc_key, sc_cfg, robots, ctrls, channels in all_sc:
            for n in E4B_ROBOT_NAMES:
                channels[n].send(E4B_FORWARD_CMD)
                actual = channels[n].receive(E4B_FORWARD_CMD)
                robots[n].apply_wheel_actions(
                    ctrls[n].forward(command=list(actual))
                )
        world.step(render=True)

    # Collect results
    result = {}
    for sc_key, sc_cfg, robots, ctrls, channels in all_sc:
        positions = {}
        for n in E4B_ROBOT_NAMES:
            pos, euler = get_pose(robots[n])
            positions[n] = pos
            dx   = pos[0] - init_pos[sc_key][n][0]
            dy   = pos[1] - init_pos[sc_key][n][1]
            dist = float(np.sqrt(dx**2 + dy**2))
            s    = channels[n].stats
            tot  = max(s['sent'], 1)
            pfx  = f"Sc{sc_key}_{n}"
            result[f"{pfx}_dist"]     = round(dist,                   6)
            result[f"{pfx}_yaw"]      = round(euler[2],               4)
            result[f"{pfx}_loss_pct"] = round(100*s['dropped']/tot,   4)
            result[f"{pfx}_corrupt"]  = s['corrupted']
            result[f"{pfx}_linkfail"] = s['link_fail']
            result[f"{pfx}_delay"]    = sc_cfg['delays'][n]
        for ra, rb, col in E4B_NEIGHBOR_PAIRS:
            d   = euclid2d(positions[ra], positions[rb])
            dev = d - E4B_INIT_GAP
            result[f"Sc{sc_key}_{col}"]     = round(d,   6)
            result[f"Sc{sc_key}_{col}_dev"] = round(dev, 6)
    return result


## Run EXP4B five times, aggregate stats, and save the CSV.
def experiment_4b():
    print("\n" + "#"*90)
    print("  EXP 4B — Heterogeneous WLAN Delay (3 Scenarios)")
    print("  15 robots (3×5) in ONE scene | 300 steps (10×30) | 5 runs")
    print("#"*90)
    all_runs = []
    for run_idx in range(5):
        print(f"\n  [EXP4B] Run {run_idx+1}/5  (seed={run_idx}) ...")
        res = run_exp4b(seed=run_idx)
        all_runs.append(res)
        e4b_print_run(run_idx, res)
    metrics     = list(all_runs[0].keys())
    stats_input = {m: [r[m] for r in all_runs] for m in metrics}
    stats       = print_stats_table("EXP4B — Heterogeneous WLAN Delay", stats_input)
    save_stats_csv("exp4b_wlan_delay.csv", all_runs, stats)


# =============================================================================
#  EXP 8 — 23-Robot Kaya Fleet (doc 23)
#  while simulation_app.is_running() loop, command_idx = min(i // 30, 9)
#  300 total steps, 5 runs
# =============================================================================

## EXP8 config: 300 steps total.
E8_TOTAL_STEPS = 300

## Five per-robot command sets for EXP8 (like the curvature sets, with
## larger omega values).
E8_CMD1 = [
    (1.0,   0.0,   0.0   ),(1.0,   0.05,  0.002 ),
    (0.966, 0.198, 0.006 ),(0.862, 0.424, 0.010 ),
    (0.630, 0.672, 0.0140),(0.388, 0.882, 0.014 ),
    (0.262, 0.948, 0.006 ),(0.408, 0.914,-0.0060),
    (0.534, 0.828,-0.0060),(0.622, 0.784,-0.0040),
]
E8_CMD2 = [
    (1.000, 0.000, 0.000 ),(0.970, 0.050, 0.0020),
    (0.876, 0.196, 0.0060),(0.712, 0.424, 0.0100),
    (0.420, 0.672, 0.0140),(0.181, 0.882, 0.014 ),
    (0.174, 0.948, 0.006 ),(0.498, 0.914,-0.0060),
    (0.624, 0.828,-0.0060),(0.682, 0.784,-0.0040),
]
E8_CMD3 = [
    (1.000, 0.000, 0.000 ),(0.941, 0.049, 0.0020),
    (0.786, 0.191, 0.0062),(0.562, 0.419, 0.0100),
    (0.210, 0.672, 0.0140),(-0.026,0.882, 0.0140),
    (0.087, 0.948, 0.006 ),(0.588, 0.914,-0.0060),
    (0.714, 0.828,-0.0060),(0.742, 0.784,-0.0040),
]
E8_CMD4 = [
    (1.0,   0.0,   0.0   ),(1.030, 0.050, 0.0020),
    (1.056, 0.201, 0.0060),(1.012, 0.424, 0.0100),
    (0.840, 0.672, 0.0140),(0.595, 0.882, 0.0140),
    (0.350, 0.948, 0.006 ),(0.318, 0.914,-0.0060),
    (0.444, 0.828,-0.0060),(0.562, 0.784,-0.0040),
]
E8_CMD5 = [
    (1.0,   0.0,   0.0   ),(1.060, 0.051, 0.0020),
    (1.147, 0.203, 0.0060),(1.162, 0.424, 0.0100),
    (1.050, 0.672, 0.0140),(0.802, 0.882, 0.014 ),
    (0.437, 0.948, 0.006 ),(0.228, 0.914,-0.0060),
    (0.354, 0.828,-0.0060),(0.502, 0.784,-0.0040),
]

## Maps each of the 23 robots to one of the five command sets.
E8_ROBOT_CMDS = {
    'I1':E8_CMD3,'I2':E8_CMD2,'I3':E8_CMD1,'I4':E8_CMD4,'I5':E8_CMD5,
    'F1':E8_CMD3,'F2':E8_CMD2,'F3':E8_CMD1,'F4':E8_CMD4,'F5':E8_CMD5,
    'F6':E8_CMD1,'F7':E8_CMD1,'F8':E8_CMD4,'F9':E8_CMD5,'F10':E8_CMD5,
    'L1':E8_CMD3,'L2':E8_CMD2,'L3':E8_CMD1,'L4':E8_CMD4,'L5':E8_CMD5,
    'L6':E8_CMD3,'L7':E8_CMD3,'L8':E8_CMD3,
}

## Fixed starting layout for the 23 EXP8 robots (I / F / L groups).
E8_INIT_POS = {
    'F1': (4.0,  0.0,  0.02),'F2': (4.0,  2.0,  0.02),
    'F3': (4.0,  4.0,  0.02),'F4': (4.0,  6.0,  0.02),
    'F5': (4.0,  8.0,  0.02),'F6': (6.0,  4.0,  0.02),
    'F7': (8.0,  4.0,  0.02),'F8': (6.0,  8.0,  0.02),
    'F9': (8.0,  8.0,  0.02),'F10':(10.0, 8.0,  0.02),
    'I1': (0.0,  0.0,  0.02),'I2': (0.0,  2.0,  0.02),
    'I3': (0.0,  4.0,  0.02),'I4': (0.0,  6.0,  0.02),
    'I5': (0.0,  8.0,  0.02),
    'L1': (14.0, 0.0,  0.02),'L2': (14.0, 2.0,  0.02),
    'L3': (14.0, 4.0,  0.02),'L4': (14.0, 6.0,  0.02),
    'L5': (14.0, 8.0,  0.02),'L6': (16.0, 0.0,  0.02),
    'L7': (18.0, 0.0,  0.02),'L8': (20.0, 0.0,  0.02),
}

## The 23 robot names in order.
E8_ROBOT_NAMES = [
    'I1','I2','I3','I4','I5',
    'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10',
    'L1','L2','L3','L4','L5','L6','L7','L8',
]


## Print one EXP8 run: each robot's distance/yaw, then a ranking by distance.
def e8_print_run(run_idx, result):
    print(f"\n  {'='*70}")
    print(f"  RUN {run_idx+1} — EXP8  23-Robot Kaya Fleet")
    print(f"  {'='*70}")
    print(f"  {'Robot':<8} {'Dist(m)':>9} {'Yaw°':>8}")
    print(f"  {'-'*8} {'-'*9} {'-'*8}")
    for n in E8_ROBOT_NAMES:
        print(f"  {n:<8} {result.get(n+'_dist',0):>9.4f} "
              f"{result.get(n+'_yaw',0):>8.2f}")
    ranked = sorted([(n, result.get(n+'_dist',0)) for n in E8_ROBOT_NAMES],
                    key=lambda x: x[1], reverse=True)
    print(f"\n  RANKING:")
    for rank, (n, d) in enumerate(ranked, 1):
        print(f"  {rank:>2}. {n:<8} {d:.4f} m")
    sys.stdout.flush()


## One EXP8 run: 23 Kayas driven by their command sets for 300 steps via a
## while-loop gated on the sim playing; measure final displacement.
def run_exp8(seed):
    """
    Mirrors doc 23 while-loop exactly.
    Uses simulation_app.is_running() but breaks when TOTAL_STEPS reached.
    """
    set_seed(seed)
    fresh_stage()

    world       = World(stage_units_in_meters=1.0)
    robots      = {}
    controllers = {}

    ## Spawn each of the 23 Kayas at its start position with a controller.
    for name in E8_ROBOT_NAMES:
        path = f"/World/Kaya_{name}"
        robot, ctrl = make_kaya_ctrl(world, path, f"kaya_{name}",
                                     E8_INIT_POS[name])
        robots[name]      = robot
        controllers[name] = ctrl

    world.scene.add_default_ground_plane()
    world.reset()

    # Initial positions
    initial = {}
    for name in E8_ROBOT_NAMES:
        pos, _ = get_pose(robots[name])
        initial[name] = pos.copy()

    i            = 0
    reset_needed = False
    final_result = {}

    ## EXP8 main loop: step physics; only act while playing; stop at 300 steps.
    while simulation_app.is_running():
        world.step(render=True)

        if world.is_stopped() and not reset_needed:
            reset_needed = True

        if world.is_playing():
            if reset_needed:
                world.reset()
                for name in E8_ROBOT_NAMES:
                    controllers[name].reset()
                reset_needed = False

            command_idx = min(i // 30, 9)

            for name in E8_ROBOT_NAMES:
                cmd = E8_ROBOT_CMDS[name][command_idx]
                robots[name].apply_wheel_actions(
                    controllers[name].forward(command=[cmd[0], cmd[1], cmd[2]])
                )

            if i >= E8_TOTAL_STEPS:
                for name in E8_ROBOT_NAMES:
                    pos, euler = get_pose(robots[name])
                    dx   = pos[0] - initial[name][0]
                    dy   = pos[1] - initial[name][1]
                    dist = float(np.sqrt(dx**2 + dy**2))
                    final_result[f"{name}_dist"] = round(dist,    6)
                    final_result[f"{name}_yaw"]  = round(euler[2],4)
                break

            i += 1

    return final_result


## Run EXP8 five times, aggregate stats, and save the CSV.
def experiment_8():
    print("\n" + "#"*90)
    print("  EXP 8 — 23-Robot Kaya Fleet (doc 23)")
    print("  300 steps (10×30) | while-loop | 5 runs")
    print("#"*90)
    all_runs = []
    for run_idx in range(5):
        print(f"\n  [EXP8] Run {run_idx+1}/5  (seed={run_idx}) ...")
        res = run_exp8(seed=run_idx)
        all_runs.append(res)
        e8_print_run(run_idx, res)
    metrics     = list(all_runs[0].keys())
    stats_input = {m: [r[m] for r in all_runs] for m in metrics}
    stats       = print_stats_table("EXP8 — 23-Robot Kaya Fleet", stats_input)
    save_stats_csv("exp8_kaya_fleet.csv", all_runs, stats)


# =============================================================================
#  MAIN — run all three experiments sequentially
# =============================================================================

print("\n" + "="*95)
print("  COMBINED EXPERIMENT SUITE")
print("  EXP7  — Kaya+Jetbot Fleet (60 cases, 120 robots, 300 steps)")
print("  EXP4B — WLAN Delay 3 Scenarios (15 robots, 300 steps)")
print("  EXP8  — 23-Robot Kaya Fleet (doc 23, 300 steps)")
print("  Each: 5 runs | results printed immediately | stats + CSV after 5 runs")
print(f"  Output: {OUT_DIR}")
print("="*95)

## MAIN: run the three experiments in sequence, then close the simulator.
experiment_7()
experiment_4b()
experiment_8()

print("\n" + "="*95)
print("  ALL EXPERIMENTS COMPLETE")
print(f"  Results in: {OUT_DIR}")
print("="*95)

simulation_app.close()