"""
=============================================================================
  THREE KAYA FLEET EXPERIMENTS
  Exp A — 2 robots, same curvature (CMD_R1), 300 steps, snapshot every 30
  Exp B — 5 robots, COMBINED commands, 300 steps, snapshot every 15
  Exp C — 5 robots, doc 29 commands, 300 steps, snapshot every 30
  Each: 5 runs, seeds 0-4
  Results + snapshots printed immediately after each run
  Statistics (mean/std/var/min/max) after all 5 runs
=============================================================================
"""

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import sys
import random
import numpy as np
from scipy.spatial.transform import Rotation

from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.controllers.holonomic_controller import HolonomicController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.robots.holonomic_robot_usd_setup import HolonomicRobotUsdSetup
from isaacsim.storage.native import get_assets_root_path

import omni.usd
import omni.kit.app

sys.stdout.reconfigure(line_buffering=True)

assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    simulation_app.close()
    exit()

KAYA_USD = assets_root_path + "/Isaac/Robots/Kaya/kaya.usd"
TOTAL_STEPS = 300

# =============================================================================
#  COMMAND SETS
# =============================================================================

CMD_R1 = [
    (1.0,   0.0,   0.000000),
    (1.0,   0.05,  0.000100),
    (0.966, 0.198, 0.000300),
    (0.862, 0.424, 0.000500),
    (0.630, 0.672, 0.000700),
    (0.388, 0.882, 0.000690),
    (0.262, 0.948, 0.000290),
    (0.408, 0.914,-0.000300),
    (0.534, 0.828,-0.000300),
    (0.622, 0.784,-0.000200),
]

COMBINED_R1 = [
    (0.6672, 0.0570, 0.000000),(0.6672, 0.1710, 0.003360),
    (0.6672, 0.2850, 0.003130),(0.6672, 0.3990, 0.002800),
    (0.6672, 0.5130, 0.002420),(0.6672, 0.6270, 0.002060),
    (0.6672, 0.7410, 0.001740),(0.6672, 0.8550, 0.001470),
    (0.6672, 0.9690, 0.001240),(0.6672, 1.0830, 0.001060),
    (0.6486, 0.9792, 0.000060),(0.6113, 0.8283, 0.000940),
    (0.5672, 0.6480, 0.002980),(0.5264, 0.5265, 0.004890),
    (0.5165, 0.4391, 0.006810),(0.5441, 0.3801, 0.007680),
    (0.5939, 0.3410, 0.007490),(0.6426, 0.3120, 0.006820),
    (0.6820, 0.2890, 0.005970),(0.7060, 0.2700, 0.005100),
]
COMBINED_R2 = [
    (0.6672, 0.0570, 0.000000),(0.5666, 0.1711, 0.003350),
    (0.5732, 0.2850, 0.003130),(0.5833, 0.3990, 0.002800),
    (0.5945, 0.5131, 0.002420),(0.6054, 0.6270, 0.002060),
    (0.6151, 0.7410, 0.001730),(0.6233, 0.8549, 0.001460),
    (0.6298, 0.9690, 0.001250),(0.6356, 1.0829, 0.001060),
    (0.6213, 0.9792, 0.000060),(0.5920, 0.8283, 0.000940),
    (0.5551, 0.6480, 0.002980),(0.5201, 0.5265, 0.004890),
    (0.5129, 0.4391, 0.006810),(0.5369, 0.3801, 0.007680),
    (0.5816, 0.3410, 0.007490),(0.6269, 0.3120, 0.006820),
    (0.6636, 0.2890, 0.005970),(0.6864, 0.2700, 0.005100),
]
COMBINED_R3 = [
    (0.6672, 0.0570, 0.000000),(0.4660, 0.1711, 0.003350),
    (0.4792, 0.2850, 0.003130),(0.4993, 0.3990, 0.002800),
    (0.5218, 0.5131, 0.002420),(0.5436, 0.6270, 0.002060),
    (0.5631, 0.7410, 0.001730),(0.5794, 0.8549, 0.001460),
    (0.5924, 0.9690, 0.001250),(0.6039, 1.0829, 0.001060),
    (0.5940, 0.9792, 0.000060),(0.5727, 0.8283, 0.000940),
    (0.5430, 0.6480, 0.002980),(0.5138, 0.5265, 0.004890),
    (0.5093, 0.4391, 0.006810),(0.5297, 0.3801, 0.007680),
    (0.5693, 0.3410, 0.007490),(0.6112, 0.3120, 0.006820),
    (0.6452, 0.2890, 0.005970),(0.6668, 0.2700, 0.005100),
]
COMBINED_R4 = [
    (0.6672, 0.0570, 0.000000),(0.7679, 0.1711, 0.003350),
    (0.7613, 0.2850, 0.003130),(0.7512, 0.3990, 0.002800),
    (0.7399, 0.5131, 0.002420),(0.7290, 0.6270, 0.002060),
    (0.7192, 0.7410, 0.001730),(0.7112, 0.8549, 0.001460),
    (0.7046, 0.9690, 0.001250),(0.6989, 1.0829, 0.001060),
    (0.6759, 0.9792, 0.000060),(0.6306, 0.8283, 0.000940),
    (0.5793, 0.6480, 0.002980),(0.5327, 0.5265, 0.004890),
    (0.5201, 0.4391, 0.006810),(0.5513, 0.3801, 0.007680),
    (0.6062, 0.3410, 0.007490),(0.6583, 0.3120, 0.006820),
    (0.7004, 0.2890, 0.005970),(0.7256, 0.2700, 0.005100),
]
COMBINED_R5 = [
    (0.6672, 0.0570, 0.000000),(0.8685, 0.1711, 0.003350),
    (0.8553, 0.2850, 0.003130),(0.8352, 0.3990, 0.002800),
    (0.8126, 0.5131, 0.002420),(0.7908, 0.6270, 0.002060),
    (0.7712, 0.7410, 0.001730),(0.7551, 0.8549, 0.001460),
    (0.7420, 0.9690, 0.001250),(0.7306, 1.0829, 0.001060),
    (0.7032, 0.9792, 0.000060),(0.6499, 0.8283, 0.000940),
    (0.5914, 0.6480, 0.002980),(0.5453, 0.5265, 0.004890),
    (0.5265, 0.4391, 0.006810),(0.5641, 0.3801, 0.007680),
    (0.6239, 0.3410, 0.007490),(0.6826, 0.3120, 0.006820),
    (0.7272, 0.2890, 0.005970),(0.7484, 0.2700, 0.005100),
]

DOC29_R1 = [
    (1.0,  0.0,      0.000000),(1.12, 0.05,     0.002000),
    (1.22, 0.10008,  0.004000),(1.31, 0.15036,  0.006000),
    (1.38, 0.20076,  0.008000),(1.19, 0.15207,  0.004000),
    (0.98, 0.10072,  0.000000),(0.88, -0.04968,-0.002000),
    (0.74, -0.09962,-0.004000),(1.0,  0.00038,  0.000000),
]
DOC29_R2 = [
    (1.0,  0.0,      0.000000),(1.06, 0.05,     0.002000),
    (1.1,  0.10004,  0.004000),(1.13, 0.15018,  0.006000),
    (1.14, 0.20038,  0.008000),(1.07, 0.15069,  0.004000),
    (0.98, 0.10036,  0.000000),(0.94, -0.04984,-0.002000),
    (0.86, -0.09981,-0.004000),(1.0,  0.00019,  0.000000),
]
DOC29_R3 = [
    (1.0,  0.0,    0.000000),(1.0,  0.05,   0.002000),
    (0.98, 0.1,    0.004000),(0.95, 0.15,   0.006000),
    (0.9,  0.2,    0.008000),(0.95, 0.15,   0.004000),
    (0.98, 0.1,    0.000000),(1.0,  -0.05, -0.002000),
    (0.98, -0.1,  -0.004000),(1.0,  0.0,    0.000000),
]
DOC29_R4 = [
    (1.0,  0.0,      0.000000),(0.94, 0.05,     0.002000),
    (0.86, 0.09996,  0.004000),(0.77, 0.14982,  0.006000),
    (0.66, 0.19962,  0.008000),(0.83, 0.14931,  0.004000),
    (0.98, 0.09964,  0.000000),(1.06, -0.05016,-0.002000),
    (1.1,  -0.10019,-0.004000),(1.0,  -0.00019, 0.000000),
]
DOC29_R5 = [
    (1.0,  0.0,      0.000000),(0.88, 0.05,     0.002000),
    (0.74, 0.09992,  0.004000),(0.59, 0.14964,  0.006000),
    (0.42, 0.19924,  0.008000),(0.71, 0.14793,  0.004000),
    (0.98, 0.09928,  0.000000),(1.12, -0.05032,-0.002000),
    (1.22, -0.10038,-0.004000),(1.0,  -0.00038, 0.000000),
]

# =============================================================================
#  HELPERS
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
    euler = Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_euler('xyz', degrees=True)
    return pos, euler


def euclid2d(a, b):
    return float(np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2))


def make_world_and_robots(positions_dict):
    world       = World(stage_units_in_meters=1.0)
    robots      = {}
    controllers = {}
    for name, pos in positions_dict.items():
        path  = f"/World/Kaya_{name}"
        robot = world.scene.add(WheeledRobot(
            prim_path=path, name=f"kaya_{name}",
            wheel_dof_names=["axle_0_joint","axle_1_joint","axle_2_joint"],
            create_robot=True, usd_path=KAYA_USD,
            position=np.array(pos),
            orientation=np.array([1.0,0.0,0.0,0.0]),
        ))
        ks = HolonomicRobotUsdSetup(
            robot_prim_path=robot.prim_path,
            com_prim_path=f"{path}/base_link/control_offset")
        wr,wp,wo,ma,wa,ua = ks.get_holonomic_controller_params()
        controllers[name] = HolonomicController(
            name=f"hctrl_{name}",
            wheel_radius=wr, wheel_positions=wp, wheel_orientations=wo,
            mecanum_angles=ma, wheel_axis=wa, up_axis=ua)
        robots[name] = robot
    world.scene.add_default_ground_plane()
    world.reset()
    return world, robots, controllers


def compute_stats(values):
    a = np.array(values, dtype=float)
    return {'mean':float(np.mean(a)),'std':float(np.std(a)),
            'variance':float(np.var(a)),'min':float(np.min(a)),'max':float(np.max(a))}


def print_stats_table(title, data):
    print(f"\n{'='*90}")
    print(f"  STATISTICS — {title}")
    print(f"{'='*90}")
    print(f"  {'Metric':<42} {'Mean':>9} {'Std':>9} {'Var':>12} {'Min':>9} {'Max':>9}")
    print(f"  {'-'*42} {'-'*9} {'-'*9} {'-'*12} {'-'*9} {'-'*9}")
    for metric, values in data.items():
        s = compute_stats(values)
        print(f"  {metric:<42} {s['mean']:>9.4f} {s['std']:>9.4f} "
              f"{s['variance']:>12.6f} {s['min']:>9.4f} {s['max']:>9.4f}")
    sys.stdout.flush()

# =============================================================================
#  SIMULATION RUNNER
#  Returns: (final_result, snapshots)
#  final_result: {name: {dist, yaw, pos}}
#  snapshots:    {step: {name: {x, y, yaw}}}
# =============================================================================

def run_simulation(world, robots, controllers, cmd_map, robot_names,
                   steps_per_cmd=30, max_cmd_idx=9, snap_interval=30):

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
                    controllers[name].forward(command=[cmd[0],cmd[1],cmd[2]]))

            # Snapshot at every snap_interval steps
            if i % snap_interval == 0:
                snap = {}
                for name in robot_names:
                    pos, euler = get_pose(robots[name])
                    snap[name] = {'x': round(pos[0],4),
                                  'y': round(pos[1],4),
                                  'yaw': round(euler[2],2)}
                snapshots[i] = snap

            if i >= TOTAL_STEPS:
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

# =============================================================================
#  PRINT HELPERS
# =============================================================================

def print_snapshots(snapshots, robot_names, snap_interval):
    print(f"\n  Position snapshots (every {snap_interval} steps):")
    header = f"  {'Step':>5}  " + "  ".join(f"{n:>5}(X)  {n:>5}(Y)  {n:>5}(Yaw)" for n in robot_names)
    print(f"  {'Step':>5}  ", end="")
    for n in robot_names:
        print(f"  {n+'_X':>9}  {n+'_Y':>9}  {n+'_Yaw':>9}", end="")
    print()
    print("  " + "-"*( 7 + len(robot_names)*33))
    for step in sorted(snapshots.keys()):
        print(f"  {step:>5}  ", end="")
        for n in robot_names:
            d = snapshots[step][n]
            print(f"  {d['x']:>9.4f}  {d['y']:>9.4f}  {d['yaw']:>9.2f}", end="")
        print()
    sys.stdout.flush()


def print_final(run_idx, exp_label, final, gaps, robot_names, neighbor_pairs):
    print(f"\n  {'='*75}")
    print(f"  RUN {run_idx+1} — {exp_label}")
    print(f"  {'='*75}")
    print(f"  {'Robot':<8} {'Dist(m)':>9} {'Yaw°':>8}")
    print(f"  {'-'*8} {'-'*9} {'-'*8}")
    for name in robot_names:
        print(f"  {name:<8} {final[name]['dist']:>9.4f} {final[name]['yaw']:>8.2f}")
    if neighbor_pairs:
        print(f"\n  Neighbor gaps (initial = 3.0m):")
        for ra, rb in neighbor_pairs:
            d   = gaps[(ra,rb)]
            dev = d - 3.0
            print(f"    {ra}↔{rb} = {d:.4f}m  (dev={dev:+.4f}m)")
    sys.stdout.flush()

# =============================================================================
#  EXP A — 2 robots, CMD_R1, snapshot every 30
# =============================================================================

def experiment_a():
    print("\n"+"#"*80)
    print("  EXP A — 2 Kaya robots, same curvature (CMD_R1)")
    print("  R1(0,0)  R2(0,3) | 300 steps (10×30) | snapshot every 30 | 5 runs")
    print("#"*80)

    positions = {'R1':(0.0,0.0,0.02), 'R2':(0.0,3.0,0.02)}
    cmd_map   = {'R1':CMD_R1, 'R2':CMD_R1}
    names     = ['R1','R2']

    all_final = []
    all_snaps = []
    all_gaps  = []

    for run_idx in range(5):
        set_seed(run_idx)
        fresh_stage()
        world, robots, controllers = make_world_and_robots(positions)
        final, snaps = run_simulation(world, robots, controllers,
                                      cmd_map, names,
                                      steps_per_cmd=30, max_cmd_idx=9,
                                      snap_interval=30)
        gap = euclid2d(final['R1']['pos'], final['R2']['pos'])
        all_final.append(final)
        all_snaps.append(snaps)
        all_gaps.append(gap)

        print_final(run_idx, "EXP A — 2 robots same curvature",
                    final, {}, names, [])
        print(f"\n  R1↔R2 final distance: {gap:.4f}m  (dev={gap-3.0:+.4f}m)")
        print_snapshots(snaps, names, 30)

    # Statistics
    stats = {}
    for n in names:
        stats[f"{n}_dist"] = [r[n]['dist'] for r in all_final]
        stats[f"{n}_yaw"]  = [r[n]['yaw']  for r in all_final]
    stats['R1_R2_gap'] = all_gaps
    steps = sorted(all_snaps[0].keys())
    for step in steps:
        for n in names:
            stats[f"step{step:03d}_{n}_X"]   = [s[step][n]['x']   for s in all_snaps]
            stats[f"step{step:03d}_{n}_Y"]   = [s[step][n]['y']   for s in all_snaps]
            stats[f"step{step:03d}_{n}_Yaw"] = [s[step][n]['yaw'] for s in all_snaps]
    print_stats_table("EXP A — Two robots same curvature", stats)

# =============================================================================
#  EXP B — 5 robots, COMBINED, snapshot every 15
# =============================================================================

def experiment_b():
    print("\n"+"#"*80)
    print("  EXP B — 5 Kaya robots, COMBINED commands")
    print("  R1(0,0) R2(0,3) R3(0,6) R4(0,-3) R5(0,-6) | 300 steps (20×15) | snapshot every 15 | 5 runs")
    print("#"*80)

    positions = {'R1':(0.0,0.0,0.02),'R2':(0.0,3.0,0.02),'R3':(0.0,6.0,0.02),
                 'R4':(0.0,-3.0,0.02),'R5':(0.0,-6.0,0.02)}
    cmd_map   = {'R1':COMBINED_R1,'R2':COMBINED_R2,'R3':COMBINED_R3,
                 'R4':COMBINED_R4,'R5':COMBINED_R5}
    names     = ['R1','R2','R3','R4','R5']
    pairs     = [('R1','R2'),('R2','R3'),('R3','R4'),('R4','R5')]

    all_final = []
    all_snaps = []
    all_gaps  = []

    for run_idx in range(5):
        set_seed(run_idx)
        fresh_stage()
        world, robots, controllers = make_world_and_robots(positions)
        final, snaps = run_simulation(world, robots, controllers,
                                      cmd_map, names,
                                      steps_per_cmd=15, max_cmd_idx=19,
                                      snap_interval=15)
        gaps = {(ra,rb): euclid2d(final[ra]['pos'],final[rb]['pos'])
                for ra,rb in pairs}
        all_final.append(final)
        all_snaps.append(snaps)
        all_gaps.append(gaps)

        print_final(run_idx, "EXP B — 5 robots COMBINED", final, gaps, names, pairs)
        print_snapshots(snaps, names, 15)

    # Statistics
    stats = {}
    for n in names:
        stats[f"{n}_dist"] = [r[n]['dist'] for r in all_final]
        stats[f"{n}_yaw"]  = [r[n]['yaw']  for r in all_final]
    for ra,rb in pairs:
        stats[f"{ra}_{rb}_gap"] = [g[(ra,rb)] for g in all_gaps]
    steps = sorted(all_snaps[0].keys())
    for step in steps:
        for n in names:
            stats[f"step{step:03d}_{n}_X"]   = [s[step][n]['x']   for s in all_snaps]
            stats[f"step{step:03d}_{n}_Y"]   = [s[step][n]['y']   for s in all_snaps]
            stats[f"step{step:03d}_{n}_Yaw"] = [s[step][n]['yaw'] for s in all_snaps]
    print_stats_table("EXP B — 5 robots COMBINED commands", stats)

# =============================================================================
#  EXP C — 5 robots, doc 29, snapshot every 30
# =============================================================================

def experiment_c():
    print("\n"+"#"*80)
    print("  EXP C — 5 Kaya robots, doc 29 commands")
    print("  R1(0,0) R2(0,3) R3(0,6) R4(0,-3) R5(0,-6) | 300 steps (10×30) | snapshot every 30 | 5 runs")
    print("#"*80)

    positions = {'R1':(0.0,0.0,0.02),'R2':(0.0,3.0,0.02),'R3':(0.0,6.0,0.02),
                 'R4':(0.0,-3.0,0.02),'R5':(0.0,-6.0,0.02)}
    cmd_map   = {'R1':DOC29_R1,'R2':DOC29_R2,'R3':DOC29_R3,
                 'R4':DOC29_R4,'R5':DOC29_R5}
    names     = ['R1','R2','R3','R4','R5']
    pairs     = [('R1','R2'),('R2','R3'),('R3','R4'),('R4','R5')]

    all_final = []
    all_snaps = []
    all_gaps  = []

    for run_idx in range(5):
        set_seed(run_idx)
        fresh_stage()
        world, robots, controllers = make_world_and_robots(positions)
        final, snaps = run_simulation(world, robots, controllers,
                                      cmd_map, names,
                                      steps_per_cmd=30, max_cmd_idx=9,
                                      snap_interval=30)
        gaps = {(ra,rb): euclid2d(final[ra]['pos'],final[rb]['pos'])
                for ra,rb in pairs}
        all_final.append(final)
        all_snaps.append(snaps)
        all_gaps.append(gaps)

        print_final(run_idx, "EXP C — 5 robots doc 29", final, gaps, names, pairs)
        print_snapshots(snaps, names, 30)

    # Statistics
    stats = {}
    for n in names:
        stats[f"{n}_dist"] = [r[n]['dist'] for r in all_final]
        stats[f"{n}_yaw"]  = [r[n]['yaw']  for r in all_final]
    for ra,rb in pairs:
        stats[f"{ra}_{rb}_gap"] = [g[(ra,rb)] for g in all_gaps]
    steps = sorted(all_snaps[0].keys())
    for step in steps:
        for n in names:
            stats[f"step{step:03d}_{n}_X"]   = [s[step][n]['x']   for s in all_snaps]
            stats[f"step{step:03d}_{n}_Y"]   = [s[step][n]['y']   for s in all_snaps]
            stats[f"step{step:03d}_{n}_Yaw"] = [s[step][n]['yaw'] for s in all_snaps]
    print_stats_table("EXP C — 5 robots doc 29 commands", stats)

# =============================================================================
#  MAIN
# =============================================================================

print("\n"+"="*80)
print("  THREE KAYA FLEET EXPERIMENTS")
print("  Exp A — 2 robots CMD_R1     | 300 steps | snapshot every 30")
print("  Exp B — 5 robots COMBINED   | 300 steps | snapshot every 15")
print("  Exp C — 5 robots doc29      | 300 steps | snapshot every 30")
print("  Each: 5 runs | printed immediately | stats after 5 runs")
print("="*80)

experiment_a()
experiment_b()
experiment_c()

print("\n"+"="*80)
print("  ALL EXPERIMENTS COMPLETE")
print("="*80)

simulation_app.close()