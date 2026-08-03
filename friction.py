"""
=============================================================================
  MERGED EXPERIMENT SUITE
  EXP 6  — Kaya vs Jetbot, 0.2 m/s straight, 300 steps (10x30), 5 runs
           (CORRECTED: Jetbot wheel_radius/wheel_base read from USD)
  EXP D  — 5 Kaya robots (I1..I5), new commands_robot1..5, 300 steps,
           snapshot every 30, full position + neighbor-gap statistics
  EXP E  — 23 Kaya robots (I/F/L), new commands_robot1..5 mapping,
           300 steps, snapshot every 30, full position + neighbor-gap stats
  EXP 7  — Kaya+Jetbot fleet, 60 cases (4 comm x 5 friction x 3 vel),
           120 robots, 300 steps, 5 runs
           (CORRECTED: Jetbot wheel_radius/wheel_base read from USD)

  Each experiment: 5 runs (seeds 0-4)
  Results printed immediately after each run; statistics
  (mean/std/var/min/max) printed after all 5 runs of each experiment.
=============================================================================
"""

# ── Bootstrap (must be first) ───────────────────────────────────────────────
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# ── Standard imports ─────────────────────────────────────────────────────────
import carb
import sys
import random
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation

# ── Isaac Sim imports ────────────────────────────────────────────────────────
from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.controllers.holonomic_controller import HolonomicController
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.robots.holonomic_robot_usd_setup import HolonomicRobotUsdSetup
from isaacsim.storage.native import get_assets_root_path

# ── USD / Omni imports ───────────────────────────────────────────────────────
from pxr import UsdShade, Usd, UsdGeom, UsdPhysics, Sdf, Gf
import omni.usd
import omni.kit.app

sys.stdout.reconfigure(line_buffering=True)

# =============================================================================
#  ASSET PATHS
# =============================================================================

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    exit()

KAYA_USD   = assets_root_path + "/Isaac/Robots/Kaya/kaya.usd"
JETBOT_USD = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"


# =============================================================================
#  SHARED HELPERS
# =============================================================================

def set_seed(s):
    random.seed(s)
    np.random.seed(s)


def fresh_stage():
    omni.usd.get_context().new_stage()
    for _ in range(10):
        omni.kit.app.get_app().update()


def get_pose(robot):
    pos, q = robot.get_world_pose()
    euler = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler('xyz', degrees=True)
    return pos, euler


def euclid2d(a, b):
    return float(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))


def compute_stats(values):
    a = np.array(values, dtype=float)
    return {
        'mean':     float(np.mean(a)),
        'std':      float(np.std(a)),
        'variance': float(np.var(a)),
        'min':      float(np.min(a)),
        'max':      float(np.max(a)),
    }


def print_stats_table(title, sd, fmt=4):
    print(f"\n{'='*92}")
    print(f"  STATISTICS — {title}")
    print(f"{'='*92}")
    print(f"  {'Metric':<48} {'Mean':>10} {'Std':>10} {'Var':>12} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*48} {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")
    result = {}
    for metric, values in sd.items():
        s = compute_stats(values)
        result[metric] = s
        print(f"  {metric:<48} {s['mean']:>10.{fmt}f} {s['std']:>10.{fmt}f} "
              f"{s['variance']:>12.{fmt+2}f} {s['min']:>10.{fmt}f} {s['max']:>10.{fmt}f}")
    sys.stdout.flush()
    return result


# =============================================================================
#  ROBOT FACTORY HELPERS (Kaya & Jetbot, params read from USD — no hardcoding)
# =============================================================================

def get_jetbot_wheel_params(stage, robot_prim_path):
    """Read Jetbot wheel_radius and wheel_base directly from USD geometry.

    DifferentialRobotUsdSetup is not available in this Isaac Sim version
    (isaacsim.robot.wheeled_robots-4.0.3 only ships HolonomicRobotUsdSetup).
    Instead, we measure:
      - wheel_radius: from the bounding box of a wheel mesh
                      (radius = half of the extent perpendicular to the
                       wheel's rotation axis)
      - wheel_base:   the actual world-space distance between the
                      left_wheel and right_wheel prims

    No values are hardcoded — everything is derived from the loaded USD.
    """
    rprim = stage.GetPrimAtPath(robot_prim_path)
    if not rprim or not rprim.IsValid():
        raise RuntimeError(f"Invalid robot prim: {robot_prim_path}")

    xf_cache   = UsdGeom.XformCache()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                    ['default', 'render'], useExtentsHint=True)

    left_prim  = None
    right_prim = None
    for prim in Usd.PrimRange(rprim):
        nm = prim.GetName().lower()
        if 'wheel' in nm and 'caster' not in nm and prim.IsA(UsdGeom.Xformable):
            if 'left' in nm and left_prim is None:
                left_prim = prim
            elif 'right' in nm and right_prim is None:
                right_prim = prim

    if left_prim is None or right_prim is None:
        raise RuntimeError(
            f"Could not find left/right wheel prims under {robot_prim_path}"
        )

    left_pos  = xf_cache.GetLocalToWorldTransform(left_prim).ExtractTranslation()
    right_pos = xf_cache.GetLocalToWorldTransform(right_prim).ExtractTranslation()
    wheel_base = (Gf.Vec3d(left_pos) - Gf.Vec3d(right_pos)).GetLength()

    bbox = bbox_cache.ComputeWorldBound(left_prim)
    rng  = bbox.GetRange()
    size = rng.GetSize()
    # Wheel rotates about the lateral (Y) axis -> radius from X/Z extent
    wheel_radius = max(size[0], size[2]) / 2.0

    return float(wheel_radius), float(wheel_base)


def make_kaya_ctrl(world, prim_path, name, position):
    """Create Kaya robot + HolonomicController, params read from USD."""
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
    return robot, ctrl, wr


def make_jetbot_ctrl(world, prim_path, name, position,
                      orientation=np.array([0.0, 0.0, 0.0, 1.0])):
    """Create Jetbot robot + DifferentialController.

    CORRECTED: wheel_radius and wheel_base are measured directly from the
    Jetbot USD geometry via get_jetbot_wheel_params() (no hardcoded
    0.03 / 0.1125, and no dependency on DifferentialRobotUsdSetup which
    is unavailable in this Isaac Sim version).
    """
    robot = world.scene.add(WheeledRobot(
        prim_path=prim_path,
        name=name,
        wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
        create_robot=True,
        usd_path=JETBOT_USD,
        position=np.array(position),
        orientation=np.array(orientation),
    ))

    stage = omni.usd.get_context().get_stage()
    wheel_radius, wheel_base = get_jetbot_wheel_params(stage, robot.prim_path)

    ctrl = DifferentialController(
        name=f"dctrl_{name}",
        wheel_radius=wheel_radius,
        wheel_base=wheel_base,
    )
    return robot, ctrl, wheel_radius, wheel_base


# =============================================================================
#  EXP 6 — Kaya vs Jetbot, 0.2 m/s straight (CORRECTED, auto wheel params)
#  300 steps (10 x 30), 5 runs
# =============================================================================

EXP6_VX            = 0.2
EXP6_STEPS_PER_CMD = 30
EXP6_NUM_CMDS      = 10
EXP6_TOTAL_STEPS   = EXP6_STEPS_PER_CMD * EXP6_NUM_CMDS  # 300


def print_exp6_run(run_idx, result):
    print(f"\n  {'='*60}")
    print(f"  RUN {run_idx+1} RESULTS — EXP6  Kaya vs Jetbot 0.2 m/s (corrected)")
    print(f"  {'='*60}")
    print(f"  {'Robot':<10} {'Dist(m)':>10} {'Yaw°':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'kaya':<10} {result['kaya_dist']:>10.6f} {result['kaya_yaw']:>10.4f}")
    print(f"  {'jetbot':<10} {result['jetbot_dist']:>10.6f} {result['jetbot_yaw']:>10.4f}")
    print(f"  Wheel radius (kaya)   : {result['kaya_wheel_radius']:.6f} m")
    print(f"  Wheel radius (jetbot) : {result['jetbot_wheel_radius']:.6f} m")
    print(f"  Wheel base   (jetbot) : {result['jetbot_wheel_base']:.6f} m")
    sys.stdout.flush()


def run_exp6_single(seed):
    set_seed(seed)
    fresh_stage()

    world = World(stage_units_in_meters=1.0)

    my_kaya = world.scene.add(
        WheeledRobot(
            prim_path="/World/Kaya_1",
            name="my_kaya_1",
            wheel_dof_names=["axle_0_joint", "axle_1_joint", "axle_2_joint"],
            create_robot=True,
            usd_path=KAYA_USD,
            position=np.array([0.0, 0.0, 0.02]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
    )

    my_jetbot = world.scene.add(
        WheeledRobot(
            prim_path="/World/Jetbot",
            name="my_jetbot",
            wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
            create_robot=True,
            usd_path=JETBOT_USD,
            position=np.array([0.0, 3.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),  # 180 deg about Z (validated config)
        )
    )

    world.scene.add_default_ground_plane()

    kaya_setup = HolonomicRobotUsdSetup(
        robot_prim_path=my_kaya.prim_path,
        com_prim_path="/World/Kaya_1/base_link/control_offset",
    )
    (
        wheel_radius_kaya, wheel_positions_kaya, wheel_orientations_kaya,
        mecanum_angles_kaya, wheel_axis_kaya, up_axis_kaya,
    ) = kaya_setup.get_holonomic_controller_params()

    kaya_controller = HolonomicController(
        name="holonomic_controller_kaya",
        wheel_radius=wheel_radius_kaya,
        wheel_positions=wheel_positions_kaya,
        wheel_orientations=wheel_orientations_kaya,
        mecanum_angles=mecanum_angles_kaya,
        wheel_axis=wheel_axis_kaya,
        up_axis=up_axis_kaya,
    )

    stage = omni.usd.get_context().get_stage()
    wheel_radius_jetbot, wheel_base_jetbot = get_jetbot_wheel_params(
        stage, my_jetbot.prim_path
    )

    jetbot_controller = DifferentialController(
        name="differential_controller_jetbot",
        wheel_radius=wheel_radius_jetbot,
        wheel_base=wheel_base_jetbot,
    )

    world.reset()

    kaya_init,   _ = get_pose(my_kaya)
    jetbot_init, _ = get_pose(my_jetbot)

    reset_needed = False
    step = 0

    while step < EXP6_TOTAL_STEPS:
        world.step(render=True)

        if world.is_stopped() and not reset_needed:
            reset_needed = True

        if world.is_playing():
            if reset_needed:
                world.reset()
                kaya_controller.reset()
                jetbot_controller.reset()
                reset_needed = False

            my_kaya.apply_wheel_actions(
                kaya_controller.forward(command=[EXP6_VX, 0.0, 0.0])
            )
            my_jetbot.apply_wheel_actions(
                jetbot_controller.forward(command=[EXP6_VX, 0.0])
            )
            step += 1

    kaya_pos,   kaya_euler   = get_pose(my_kaya)
    jetbot_pos, jetbot_euler = get_pose(my_jetbot)

    kaya_dx   = kaya_pos[0] - kaya_init[0]
    kaya_dy   = kaya_pos[1] - kaya_init[1]
    kaya_dist = float(np.sqrt(kaya_dx**2 + kaya_dy**2))

    jetbot_dx   = jetbot_pos[0] - jetbot_init[0]
    jetbot_dy   = jetbot_pos[1] - jetbot_init[1]
    jetbot_dist = float(np.sqrt(jetbot_dx**2 + jetbot_dy**2))

    return {
        'kaya_dist':           round(kaya_dist, 6),
        'kaya_yaw':            round(kaya_euler[2], 4),
        'jetbot_dist':         round(jetbot_dist, 6),
        'jetbot_yaw':          round(jetbot_euler[2], 4),
        'kaya_wheel_radius':   float(wheel_radius_kaya) if np.isscalar(wheel_radius_kaya)
                                else float(np.mean(wheel_radius_kaya)),
        'jetbot_wheel_radius': float(wheel_radius_jetbot),
        'jetbot_wheel_base':   float(wheel_base_jetbot),
    }


def experiment_6():
    print("\n" + "#" * 90)
    print("  EXP 6 (CORRECTED) — Kaya vs Jetbot 0.2 m/s straight")
    print(f"  {EXP6_TOTAL_STEPS} steps ({EXP6_NUM_CMDS} x {EXP6_STEPS_PER_CMD}), 5 runs, seeds 0-4")
    print("  Jetbot wheel_radius/wheel_base read from USD (was hardcoded 0.03/0.1125)")
    print("#" * 90)

    all_runs = []
    for run_idx in range(5):
        print(f"\n  [EXP6] Run {run_idx+1}/5  (seed={run_idx}) ...")
        result = run_exp6_single(seed=run_idx)
        all_runs.append(result)
        print_exp6_run(run_idx, result)

    stats_input = {
        'kaya_dist':   [r['kaya_dist']   for r in all_runs],
        'kaya_yaw':    [r['kaya_yaw']    for r in all_runs],
        'jetbot_dist': [r['jetbot_dist'] for r in all_runs],
        'jetbot_yaw':  [r['jetbot_yaw']  for r in all_runs],
    }
    print_stats_table("EXP6 (CORRECTED) — Kaya vs Jetbot 0.2 m/s", stats_input, fmt=6)


# =============================================================================
#  NEW COMMAND SETS (commands_robot1 .. commands_robot5) — for EXP D & EXP E
# =============================================================================

commands_robot1 = [
    (1.0,   0.0,   0.00000),
    (1.0,   0.05,  0.00010),
    (0.966, 0.198, 0.00030),
    (0.862, 0.424, 0.00050),
    (0.630, 0.672, 0.00070),
    (0.388, 0.882, 0.00069),
    (0.262, 0.948, 0.00029),
    (0.408, 0.914,-0.00030),
    (0.534, 0.828,-0.00030),
    (0.622, 0.784,-0.00020),
]
commands_robot2 = [
    (1.000, 0.000, 0.00000),
    (0.970, 0.050, 0.00010),
    (0.876, 0.196, 0.00030),
    (0.712, 0.424, 0.00050),
    (0.420, 0.672, 0.00070),
    (0.181, 0.882, 0.00069),
    (0.174, 0.948, 0.00029),
    (0.498, 0.914,-0.00030),
    (0.624, 0.828,-0.00030),
    (0.682, 0.784,-0.00020),
]
commands_robot3 = [
    (1.000, 0.000, 0.00000),
    (0.941, 0.049, 0.00010),
    (0.786, 0.191, 0.00031),
    (0.562, 0.419, 0.00050),
    (0.210, 0.672, 0.00070),
    (-0.026,0.882, 0.00069),
    (0.087, 0.948, 0.00029),
    (0.588, 0.914,-0.00030),
    (0.714, 0.828,-0.00030),
    (0.742, 0.784,-0.00020),
]
commands_robot4 = [
    (1.0,   0.0,   0.00000),
    (1.030, 0.050, 0.00010),
    (1.056, 0.201, 0.00030),
    (1.012, 0.424, 0.00050),
    (0.840, 0.672, 0.00070),
    (0.595, 0.882, 0.00069),
    (0.350, 0.948, 0.00029),
    (0.318, 0.914,-0.00030),
    (0.444, 0.828,-0.00030),
    (0.562, 0.784,-0.00020),
]
commands_robot5 = [
    (1.0,   0.0,   0.00000),
    (1.060, 0.051, 0.00010),
    (1.147, 0.203, 0.00030),
    (1.162, 0.424, 0.00050),
    (1.050, 0.672, 0.00070),
    (0.802, 0.882, 0.00069),
    (0.437, 0.948, 0.00029),
    (0.228, 0.914,-0.00030),
    (0.354, 0.828,-0.00030),
    (0.502, 0.784,-0.00020),
]


# =============================================================================
#  EXP D — 5 robots (I1..I5), I-column command mapping
#  EXP E — 23 robots (I/F/L), full robot_commands mapping
#  300 steps (10x30), snapshot every 30, 5 runs
# =============================================================================

DE_TOTAL_STEPS   = 300
DE_STEPS_PER_CMD = 30
DE_MAX_CMD_IDX   = 9
DE_SNAP_INTERVAL = 30

EXP_D_NAMES = ['I1', 'I2', 'I3', 'I4', 'I5']

EXP_D_POSITIONS = {
    'I1': (0.0, 0.0, 0.02),
    'I2': (0.0, 2.0, 0.02),
    'I3': (0.0, 4.0, 0.02),
    'I4': (0.0, 6.0, 0.02),
    'I5': (0.0, 8.0, 0.02),
}

EXP_D_CMD_MAP = {
    'I1': commands_robot3,
    'I2': commands_robot2,
    'I3': commands_robot1,
    'I4': commands_robot4,
    'I5': commands_robot5,
}

EXP_D_PAIRS    = [('I1','I2'), ('I2','I3'), ('I3','I4'), ('I4','I5')]
EXP_D_INIT_GAP = 2.0


EXP_E_NAMES = [
    'I1','I2','I3','I4','I5',
    'F1','F2','F3','F4','F5','F6','F7','F8','F9','F10',
    'L1','L2','L3','L4','L5','L6','L7','L8',
]

EXP_E_POSITIONS = {
    'F1':  (4.0,  0.0,  0.02), 'F2':  (4.0,  2.0,  0.02),
    'F3':  (4.0,  4.0,  0.02), 'F4':  (4.0,  6.0,  0.02),
    'F5':  (4.0,  8.0,  0.02), 'F6':  (6.0,  4.0,  0.02),
    'F7':  (8.0,  4.0,  0.02), 'F8':  (6.0,  8.0,  0.02),
    'F9':  (8.0,  8.0,  0.02), 'F10': (10.0, 8.0,  0.02),
    'I1':  (0.0,  0.0,  0.02), 'I2':  (0.0,  2.0,  0.02),
    'I3':  (0.0,  4.0,  0.02), 'I4':  (0.0,  6.0,  0.02),
    'I5':  (0.0,  8.0,  0.02),
    'L1':  (14.0, 0.0,  0.02), 'L2':  (14.0, 2.0,  0.02),
    'L3':  (14.0, 4.0,  0.02), 'L4':  (14.0, 6.0,  0.02),
    'L5':  (14.0, 8.0,  0.02), 'L6':  (16.0, 0.0,  0.02),
    'L7':  (18.0, 0.0,  0.02), 'L8':  (20.0, 0.0,  0.02),
}

EXP_E_CMD_MAP = {
    'I1': commands_robot3,   'I2': commands_robot2,   'I3': commands_robot1,
    'I4': commands_robot4,   'I5': commands_robot5,
    'F1': commands_robot3,   'F2': commands_robot2,   'F3': commands_robot1,
    'F4': commands_robot4,   'F5': commands_robot5,   'F6': commands_robot1,
    'F7': commands_robot1,   'F8': commands_robot4,   'F9': commands_robot5,
    'F10': commands_robot5,
    'L1': commands_robot3,   'L2': commands_robot2,   'L3': commands_robot1,
    'L4': commands_robot4,   'L5': commands_robot5,   'L6': commands_robot3,
    'L7': commands_robot3,   'L8': commands_robot3,
}

EXP_E_PAIRS = [(EXP_E_NAMES[i], EXP_E_NAMES[i+1]) for i in range(len(EXP_E_NAMES)-1)]


def exp_e_initial_gap(ra, rb):
    pa = np.array(EXP_E_POSITIONS[ra])
    pb = np.array(EXP_E_POSITIONS[rb])
    return float(np.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2))


def make_world_and_kaya_robots(positions_dict):
    world       = World(stage_units_in_meters=1.0)
    robots      = {}
    controllers = {}
    for name, pos in positions_dict.items():
        path  = f"/World/Kaya_{name}"
        robot, ctrl, _wr = make_kaya_ctrl(world, path, f"kaya_{name}", pos)
        robots[name]      = robot
        controllers[name] = ctrl
    world.scene.add_default_ground_plane()
    world.reset()
    return world, robots, controllers


def run_de_simulation(world, robots, controllers, cmd_map, robot_names,
                      neighbor_pairs,
                      steps_per_cmd=DE_STEPS_PER_CMD, max_cmd_idx=DE_MAX_CMD_IDX,
                      snap_interval=DE_SNAP_INTERVAL, total_steps=DE_TOTAL_STEPS):

    initial = {}
    for name in robot_names:
        pos, _ = get_pose(robots[name])
        initial[name] = pos.copy()

    snapshots    = {}
    i            = 0
    reset_needed = False

    while simulation_app.is_running():
        world.step(render=True)

        if world.is_stopped() and not reset_needed:
            reset_needed = True

        if world.is_playing():
            if reset_needed:
                world.reset()
                for name in robot_names:
                    controllers[name].reset()
                reset_needed = False

            cmd_idx = min(i // steps_per_cmd, max_cmd_idx)

            for name in robot_names:
                cmd = cmd_map[name][cmd_idx]
                robots[name].apply_wheel_actions(
                    controllers[name].forward(command=[cmd[0], cmd[1], cmd[2]]))

            if i % snap_interval == 0:
                rsnap = {}
                pos_cache = {}
                for name in robot_names:
                    pos, euler = get_pose(robots[name])
                    pos_cache[name] = pos
                    rsnap[name] = {'x': round(pos[0],4),
                                   'y': round(pos[1],4),
                                   'yaw': round(euler[2],2)}
                gsnap = {}
                for ra, rb in neighbor_pairs:
                    gsnap[(ra,rb)] = round(euclid2d(pos_cache[ra], pos_cache[rb]), 4)
                snapshots[i] = {'robots': rsnap, 'gaps': gsnap}

            if i >= total_steps:
                final = {}
                for name in robot_names:
                    pos, euler = get_pose(robots[name])
                    dx   = pos[0] - initial[name][0]
                    dy   = pos[1] - initial[name][1]
                    dist = float(np.sqrt(dx**2 + dy**2))
                    final[name] = {'dist': round(dist,6),
                                   'yaw':  round(euler[2],4),
                                   'pos':  pos}
                return final, snapshots

            i += 1

    return {}, {}


def print_de_snapshots(snapshots, robot_names, neighbor_pairs):
    print(f"\n  Position snapshots (X, Y, Yaw):")
    print(f"  {'Step':>5}  ", end="")
    for n in robot_names:
        print(f"  {n+'_X':>9}  {n+'_Y':>9}  {n+'_Yaw':>9}", end="")
    print()
    print("  " + "-"*(7 + len(robot_names)*33))
    for step in sorted(snapshots.keys()):
        d = snapshots[step]['robots']
        print(f"  {step:>5}  ", end="")
        for n in robot_names:
            r = d[n]
            print(f"  {r['x']:>9.4f}  {r['y']:>9.4f}  {r['yaw']:>9.2f}", end="")
        print()

    if neighbor_pairs:
        print(f"\n  Neighbor-pair gap snapshots (m):")
        print(f"  {'Step':>5}  ", end="")
        for ra, rb in neighbor_pairs:
            print(f"  {ra+'-'+rb:>10}", end="")
        print()
        print("  " + "-"*(7 + len(neighbor_pairs)*12))
        for step in sorted(snapshots.keys()):
            g = snapshots[step]['gaps']
            print(f"  {step:>5}  ", end="")
            for ra, rb in neighbor_pairs:
                print(f"  {g[(ra,rb)]:>10.4f}", end="")
            print()
    sys.stdout.flush()


def print_de_final(run_idx, exp_label, final, gaps, robot_names, neighbor_pairs, init_gap_fn):
    print(f"\n  {'='*75}")
    print(f"  RUN {run_idx+1} — {exp_label}")
    print(f"  {'='*75}")
    print(f"  {'Robot':<8} {'Dist(m)':>9} {'Yaw°':>8}")
    print(f"  {'-'*8} {'-'*9} {'-'*8}")
    for name in robot_names:
        print(f"  {name:<8} {final[name]['dist']:>9.4f} {final[name]['yaw']:>8.2f}")
    if neighbor_pairs:
        print(f"\n  Neighbor gaps (final):")
        for ra, rb in neighbor_pairs:
            d   = gaps[(ra,rb)]
            ig  = init_gap_fn(ra, rb)
            dev = d - ig
            print(f"    {ra}↔{rb} = {d:.4f}m  (initial={ig:.4f}m, dev={dev:+.4f}m)")
    sys.stdout.flush()


def run_de_experiment(exp_label, names, positions, cmd_map, neighbor_pairs, init_gap_fn):
    print("\n"+"#"*85)
    print(f"  {exp_label}")
    print(f"  {len(names)} robots | {DE_TOTAL_STEPS} steps ({DE_MAX_CMD_IDX+1}x{DE_STEPS_PER_CMD}) "
          f"| snapshot every {DE_SNAP_INTERVAL} | 5 runs")
    print("#"*85)

    all_final = []
    all_snaps = []
    all_gaps  = []

    for run_idx in range(5):
        set_seed(run_idx)
        fresh_stage()
        world, robots, controllers = make_world_and_kaya_robots(positions)
        final, snaps = run_de_simulation(world, robots, controllers,
                                         cmd_map, names, neighbor_pairs)
        gaps = {(ra,rb): euclid2d(final[ra]['pos'], final[rb]['pos'])
                for ra,rb in neighbor_pairs}
        all_final.append(final)
        all_snaps.append(snaps)
        all_gaps.append(gaps)

        print_de_final(run_idx, exp_label, final, gaps, names, neighbor_pairs, init_gap_fn)
        print_de_snapshots(snaps, names, neighbor_pairs)

    stats = {}

    for n in names:
        stats[f"{n}_dist"] = [r[n]['dist'] for r in all_final]
        stats[f"{n}_yaw"]  = [r[n]['yaw']  for r in all_final]

    for ra, rb in neighbor_pairs:
        ig = init_gap_fn(ra, rb)
        stats[f"{ra}_{rb}_gap_final"]     = [g[(ra,rb)] for g in all_gaps]
        stats[f"{ra}_{rb}_gap_final_dev"] = [g[(ra,rb)] - ig for g in all_gaps]

    steps = sorted(all_snaps[0].keys())
    for step in steps:
        for n in names:
            stats[f"step{step:03d}_{n}_X"]   = [s[step]['robots'][n]['x']   for s in all_snaps]
            stats[f"step{step:03d}_{n}_Y"]   = [s[step]['robots'][n]['y']   for s in all_snaps]
            stats[f"step{step:03d}_{n}_Yaw"] = [s[step]['robots'][n]['yaw'] for s in all_snaps]

    for step in steps:
        for ra, rb in neighbor_pairs:
            stats[f"step{step:03d}_{ra}_{rb}_gap"] = [s[step]['gaps'][(ra,rb)] for s in all_snaps]

    print_stats_table(exp_label, stats)


def experiment_d():
    run_de_experiment(
        "EXP D — 5 robots (I1..I5), new commands",
        EXP_D_NAMES, EXP_D_POSITIONS, EXP_D_CMD_MAP, EXP_D_PAIRS,
        init_gap_fn=lambda ra, rb: EXP_D_INIT_GAP,
    )


def experiment_e():
    run_de_experiment(
        "EXP E — 23 robots (I/F/L), new commands",
        EXP_E_NAMES, EXP_E_POSITIONS, EXP_E_CMD_MAP, EXP_E_PAIRS,
        init_gap_fn=exp_e_initial_gap,
    )


# =============================================================================
#  COMM CHANNEL — for EXP 7
# =============================================================================

class CommChannel:
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

    def reset(self):
        self.queue = deque()
        self.last  = None
        self.ldown = 0
        self.step  = 0
        self.stats = {'sent': 0, 'dropped': 0, 'corrupted': 0, 'link_fail': 0}


# =============================================================================
#  FRICTION HELPERS — for EXP 7
# =============================================================================

def _fric_prim(stage, base_path, s, d):
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


def apply_ground_fric(stage, ground_path, fric):
    if not fric['override']:
        return
    mat = _fric_prim(stage, ground_path, fric['static'], fric['dynamic'])
    UsdShade.MaterialBindingAPI.Apply(
        stage.GetPrimAtPath(ground_path)
    ).Bind(mat, UsdShade.Tokens.weakerThanDescendants, "physics")


def apply_wheel_fric(stage, robot_path, fric, suffix="WM"):
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


def make_zone_ground(stage, prim_path, cx, cy, sx, sy):
    g = UsdGeom.Cube.Define(stage, prim_path)
    g.CreateSizeAttr(1.0)
    g.AddTranslateOp().Set((cx, cy, -0.5))
    g.AddScaleOp().Set((sx, sy, 1.0))
    prim = stage.GetPrimAtPath(prim_path)
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr().Set(True)
    return prim


# =============================================================================
#  EXP 7 (CORRECTED) — Kaya + Jetbot Fleet
#  60 cases = 4 comm x 5 friction x 3 velocity, 120 robots, 300 steps, 5 runs
# =============================================================================

E7_TOTAL_STEPS   = 300
E7_STEPS_PER_CMD = 30
E7_CASE_X_OFFSET = 25.0

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

E7_FRICTION = [
    {"label": "Default", "override": False, "static": 0.0, "dynamic":  0.0},
    {"label": "s0_d0",   "override": True,  "static": 0.0, "dynamic":  0.0},
    {"label": "s1_d1",   "override": True,  "static": 1.0, "dynamic":  1.0},
    {"label": "s1_d5",   "override": True,  "static": 1.0, "dynamic":  5.0},
    {"label": "s1_d10",  "override": True,  "static": 1.0, "dynamic": 10.0},
]

E7_VEL_LABELS = ['scurve', 'straight_02', 'straight_01']

E7_KAYA_SCURVE = [
    (0.200, 0.000,  0.00000), (0.200, 0.010,  0.00100),
    (0.194, 0.040,  0.00305), (0.172, 0.084,  0.00492),
    (0.126, 0.134,  0.00705), (0.078, 0.178,  0.00699),
    (0.052, 0.188,  0.00287), (0.082, 0.184, -0.00303),
    (0.106, 0.166, -0.00294), (0.124, 0.156, -0.00208),
]

E7_JETBOT_SCURVE = [
    (0.2000, 0.2000), (0.1997, 0.2008), (0.1963, 0.1999), (0.1885, 0.1943),
    (0.1798, 0.1881), (0.1902, 0.1984), (0.1934, 0.1967), (0.2032, 0.1997),
    (0.1987, 0.1952), (0.2005, 0.1981),
]

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


def e7_kaya_cmd(vel, slot):
    if vel == 'scurve':        return E7_KAYA_SCURVE[slot]
    elif vel == 'straight_02': return (0.2, 0.0, 0.0)
    else:                      return (0.1, 0.0, 0.0)


def e7_jetbot_cmd(vel, slot):
    if vel == 'scurve':        return E7_JETBOT_SCURVE[slot]
    elif vel == 'straight_02': return (0.2, 0.0)
    else:                      return (0.1, 0.0)


def e7_print_run(run_idx, result):
    print(f"\n  {'='*95}")
    print(f"  RUN {run_idx+1} — EXP7 (CORRECTED)  Kaya+Jetbot Fleet (60 cases, 300 steps)")
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


def run_exp7(seed):
    set_seed(seed)
    fresh_stage()

    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    pairs = []

    jetbot_wheel_radius = None
    jetbot_wheel_base   = None

    for idx, case in enumerate(E7_CASES):
        x_off  = float(idx) * E7_CASE_X_OFFSET
        fric   = case['fric']
        preset = E7_COMM_PRESETS[case['comm']]

        kaya_x  = x_off - 3.0
        jetbt_x = x_off

        gp_cx = x_off - 1.5
        make_zone_ground(stage, f"/World/G7_{idx}", gp_cx, 0.0, 20.0, 15.0)
        apply_ground_fric(stage, f"/World/G7_{idx}", fric)

        kp = f"/World/K7_{idx}"
        kaya, kctrl, _kr = make_kaya_ctrl(world, kp, f"k7_{idx}", [kaya_x, 0.0, 0.02])
        apply_wheel_fric(stage, kp, fric, "KWM")

        jp = f"/World/J7_{idx}"
        jetbot, jctrl, jr, jb = make_jetbot_ctrl(world, jp, f"j7_{idx}", [jetbt_x, 0.0, 0.0])
        apply_wheel_fric(stage, jp, fric, "JWM")

        if jetbot_wheel_radius is None:
            jetbot_wheel_radius = float(jr)
            jetbot_wheel_base   = float(jb)

        pairs.append({
            'case':   case,
            'kaya':   kaya,   'kctrl': kctrl,
            'kch':    CommChannel(preset),
            'jetbot': jetbot, 'jctrl': jctrl,
            'jch':    CommChannel(preset),
        })

    world.reset()

    for p in pairs:
        ki, _ = get_pose(p['kaya'])
        ji, _ = get_pose(p['jetbot'])
        p['ki'] = ki.copy()
        p['ji'] = ji.copy()

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

    result['_jetbot_wheel_radius'] = jetbot_wheel_radius
    result['_jetbot_wheel_base']   = jetbot_wheel_base
    return result


def experiment_7():
    print("\n" + "#"*95)
    print("  EXP 7 (CORRECTED) — Kaya+Jetbot Fleet")
    print("  60 cases | 120 robots | 300 steps (10x30) | 5 runs")
    print("  Jetbot wheel_radius/wheel_base measured from USD geometry (no hardcoding)")
    print("  (was hardcoded 0.03 / 0.1125)")
    print("#"*95)
    all_runs = []
    for run_idx in range(5):
        print(f"\n  [EXP7] Run {run_idx+1}/5  (seed={run_idx}) ...")
        res = run_exp7(seed=run_idx)
        print(f"  Jetbot wheel_radius = {res['_jetbot_wheel_radius']:.6f} m, "
              f"wheel_base = {res['_jetbot_wheel_base']:.6f} m  "
              f"(previously hardcoded: 0.030000 / 0.112500)")
        all_runs.append(res)
        e7_print_run(run_idx, res)

    metrics = [m for m in all_runs[0].keys() if not m.startswith('_')]
    stats_input = {m: [r[m] for r in all_runs] for m in metrics}
    print_stats_table("EXP7 (CORRECTED) — Kaya+Jetbot Fleet", stats_input)


# =============================================================================
#  MAIN — run all experiments sequentially
# =============================================================================

print("\n" + "="*95)
print("  MERGED EXPERIMENT SUITE")
print("  EXP6 — Kaya vs Jetbot 0.2 m/s straight (corrected, 300 steps, 5 runs)")
print("  EXP D — 5-robot fleet (I1..I5), new commands (300 steps, snap 30, 5 runs)")
print("  EXP E — 23-robot fleet (I/F/L), new commands (300 steps, snap 30, 5 runs)")
print("  EXP7 — Kaya+Jetbot 60-case fleet (corrected, 300 steps, 5 runs)")
print("="*95)

experiment_6()
experiment_d()
experiment_e()
experiment_7()

print("\n" + "="*95)
print("  ALL EXPERIMENTS COMPLETE")
print("="*95)

simulation_app.close()