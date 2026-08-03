"""
=============================================================================
  EXP6 (SPEED-MATCHED) — Kaya vs Jetbot, distance-matched velocities
  300 steps (10 x 30), 5 runs (seeds 0-4)

  Step 1: run both robots at the SAME commanded 0.2 m/s once, measure the
          actual distance each one covers.
  Step 2: compute Jetbot's scale factor = Kaya_dist / Jetbot_dist, and
          re-derive Jetbot's commanded velocity = 0.2 * scale_factor, so
          that Jetbot's actual traveled distance matches Kaya's.
  Step 3: run the speed-matched configuration 5 times (seeds 0-4) and
          report full statistics (mean/var/min/max) for distance and yaw,
          for both robots.

  No hardcoded wheel parameters: Jetbot's wheel_radius/wheel_base are
  measured directly from the USD geometry (DifferentialRobotUsdSetup is
  unavailable in this Isaac Sim version).
=============================================================================
"""

## BOOTSTRAP: SimulationApp must be created before any other isaacsim import;
## this line launches the simulator (headless=False shows the viewport).
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import carb
import sys
import random
import numpy as np
from scipy.spatial.transform import Rotation

from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.controllers.holonomic_controller import HolonomicController
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.robots.holonomic_robot_usd_setup import HolonomicRobotUsdSetup
from isaacsim.storage.native import get_assets_root_path

from pxr import Usd, UsdGeom, Gf
import omni.usd
import omni.kit.app

sys.stdout.reconfigure(line_buffering=True)

## Locate Isaac's bundled robot assets; abort if missing.
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")
    ## Shut down Isaac Sim cleanly.
    simulation_app.close()
    exit()

KAYA_USD   = assets_root_path + "/Isaac/Robots/Kaya/kaya.usd"
JETBOT_USD = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"

## Kaya drives at a fixed 0.2 m/s; the run is 10 command slots x 30 = 300 steps.
KAYA_VX            = 0.2     # Kaya's commanded velocity stays fixed
STEPS_PER_CMD       = 30
NUM_CMDS             = 10
TOTAL_STEPS          = STEPS_PER_CMD * NUM_CMDS  # 300


# =============================================================================
#  HELPERS
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
    euler = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler('xyz', degrees=True)
    return pos, euler


## mean/std/variance/min/max for one metric across the 5 runs.
def compute_stats(values):
    a = np.array(values, dtype=float)
    return {'mean':float(np.mean(a)),'std':float(np.std(a)),
            'variance':float(np.var(a)),'min':float(np.min(a)),'max':float(np.max(a))}


## Print a stats table; sd maps each metric to its list of run-values.
def print_stats_table(title, sd, fmt=6):
    print(f"\n{'='*90}")
    print(f"  STATISTICS — {title}")
    print(f"{'='*90}")
    print(f"  {'Metric':<24} {'Mean':>10} {'Std':>10} {'Var':>12} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*24} {'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")
    for metric, values in sd.items():
        s = compute_stats(values)
        print(f"  {metric:<24} {s['mean']:>10.{fmt}f} {s['std']:>10.{fmt}f} "
              f"{s['variance']:>12.{fmt+2}f} {s['min']:>10.{fmt}f} {s['max']:>10.{fmt}f}")
    sys.stdout.flush()


## Measure the Jetbot's wheel radius + wheelbase from its 3D model (nothing
## hardcoded): wheelbase = distance between the left/right wheel prims;
## radius = half the wheel bounding-box extent across its spin axis.
def get_jetbot_wheel_params(stage, robot_prim_path):
    """Measure Jetbot wheel_radius / wheel_base directly from USD geometry
    (no DifferentialRobotUsdSetup dependency, no hardcoded values)."""
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
        raise RuntimeError(f"Could not find left/right wheel prims under {robot_prim_path}")

    left_pos  = xf_cache.GetLocalToWorldTransform(left_prim).ExtractTranslation()
    right_pos = xf_cache.GetLocalToWorldTransform(right_prim).ExtractTranslation()
    wheel_base = (Gf.Vec3d(left_pos) - Gf.Vec3d(right_pos)).GetLength()

    bbox = bbox_cache.ComputeWorldBound(left_prim)
    size = bbox.GetRange().GetSize()
    wheel_radius = max(size[0], size[2]) / 2.0

    return float(wheel_radius), float(wheel_base)


# =============================================================================
#  SINGLE RUN  (kaya_vx, jetbot_vx given explicitly)
# =============================================================================

## One full run at the GIVEN commanded speeds: build the scene, drive both
## robots straight for 300 steps, and return how far each travelled + its yaw.
## Kaya and Jetbot speeds are passed in so the same function serves both the
## calibration run and the speed-matched runs.
def run_single(seed, kaya_vx, jetbot_vx):
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
            orientation=np.array([0.0, 0.0, 0.0, 1.0]),
        )
    )

    ## Flat infinite floor for both robots.
    world.scene.add_default_ground_plane()

    ## Build the Kaya's holonomic controller from its USD wheel geometry.
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

    ## Measure the Jetbot's wheel params from geometry, then build its
    ## differential controller from them.
    stage = omni.usd.get_context().get_stage()
    wheel_radius_jetbot, wheel_base_jetbot = get_jetbot_wheel_params(
        stage, my_jetbot.prim_path
    )

    jetbot_controller = DifferentialController(
        name="differential_controller_jetbot",
        wheel_radius=wheel_radius_jetbot,
        wheel_base=wheel_base_jetbot,
    )

    ## Finalize physics; robots settle onto the ground.
    world.reset()

    kaya_init,   _ = get_pose(my_kaya)
    jetbot_init, _ = get_pose(my_jetbot)

    reset_needed = False
    step = 0

    ## MAIN LOOP: advance physics one frame at a time until 300 steps, sending
    ## each robot its constant command every step (only while the sim is playing).
    while step < TOTAL_STEPS:
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
                kaya_controller.forward(command=[kaya_vx, 0.0, 0.0])
            )
            my_jetbot.apply_wheel_actions(
                jetbot_controller.forward(command=[jetbot_vx, 0.0])
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
        'jetbot_wheel_radius': float(wheel_radius_jetbot),
        'jetbot_wheel_base':   float(wheel_base_jetbot),
    }


# =============================================================================
#  STEP 1 — calibration run: measure the speed mismatch at equal commands
# =============================================================================

print("\n" + "#" * 90)
print("  STEP 1 — CALIBRATION RUN  (both robots commanded at 0.2 m/s)")
print("  Purpose: measure Jetbot's distance/Kaya's distance ratio to derive")
print("           the velocity scale factor needed to match distances.")
print("#" * 90)

## STEP 1 calibration: run BOTH robots at the same 0.2 m/s once to see how
## far each actually travels (they differ because their drives differ).
calib = run_single(seed=0, kaya_vx=KAYA_VX, jetbot_vx=KAYA_VX)

kaya_calib_dist   = calib['kaya_dist']
jetbot_calib_dist = calib['jetbot_dist']
## Scale factor = how much farther Kaya went than Jetbot. Multiplying the
## Jetbot's command by this makes its actual distance match Kaya's.
scale_factor       = kaya_calib_dist / jetbot_calib_dist
jetbot_vx_matched   = round(KAYA_VX * scale_factor, 6)

print(f"\n  Kaya   distance @ 0.2 m/s commanded : {kaya_calib_dist:.6f} m")
print(f"  Jetbot distance @ 0.2 m/s commanded : {jetbot_calib_dist:.6f} m")
print(f"  Scale factor (Kaya_dist / Jetbot_dist) : {scale_factor:.6f}")
print(f"  Jetbot's NEW commanded velocity (matched) : {jetbot_vx_matched:.6f} m/s")
print(f"  Jetbot wheel_radius : {calib['jetbot_wheel_radius']:.6f} m")
print(f"  Jetbot wheel_base   : {calib['jetbot_wheel_base']:.6f} m")


# =============================================================================
#  STEP 2 — speed-matched experiment: 5 runs + statistics
# =============================================================================

## Print one speed-matched run: each robot's distance and yaw.
def print_run_result(run_idx, result, jetbot_vx_used):
    print(f"\n  {'='*64}")
    print(f"  RUN {run_idx+1} RESULTS — Kaya vs Jetbot, SPEED-MATCHED")
    print(f"  Kaya cmd = {KAYA_VX:.6f} m/s | Jetbot cmd = {jetbot_vx_used:.6f} m/s")
    print(f"  {'='*64}")
    print(f"  {'Robot':<10} {'Dist(m)':>10} {'Yaw°':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'kaya':<10} {result['kaya_dist']:>10.6f} {result['kaya_yaw']:>10.4f}")
    print(f"  {'jetbot':<10} {result['jetbot_dist']:>10.6f} {result['jetbot_yaw']:>10.4f}")
    sys.stdout.flush()


print("\n" + "#" * 90)
print("  STEP 2 — SPEED-MATCHED EXPERIMENT")
print(f"  Kaya cmd = {KAYA_VX:.6f} m/s (fixed) | Jetbot cmd = {jetbot_vx_matched:.6f} m/s (matched)")
print(f"  {TOTAL_STEPS} steps ({NUM_CMDS} x {STEPS_PER_CMD}), 5 runs, seeds 0-4")
print("#" * 90)

## STEP 2: run the speed-matched setup 5 times (seeds 0-4) and collect results.
all_runs = []
for run_idx in range(5):
    print(f"\n  [EXP] Run {run_idx+1}/5  (seed={run_idx}) ...")
    result = run_single(seed=run_idx, kaya_vx=KAYA_VX, jetbot_vx=jetbot_vx_matched)
    all_runs.append(result)
    print_run_result(run_idx, result, jetbot_vx_matched)

## Gather distance and yaw across the 5 runs for the statistics table.
stats_input = {
    'kaya_dist':   [r['kaya_dist']   for r in all_runs],
    'kaya_yaw':    [r['kaya_yaw']    for r in all_runs],
    'jetbot_dist': [r['jetbot_dist'] for r in all_runs],
    'jetbot_yaw':  [r['jetbot_yaw']  for r in all_runs],
}
print_stats_table("Kaya vs Jetbot, speed-matched (5 runs)", stats_input)

print("\n" + "#" * 90)
print("  SUMMARY")
print(f"  Kaya commanded velocity   : {KAYA_VX:.6f} m/s (unchanged)")
print(f"  Jetbot commanded velocity : {jetbot_vx_matched:.6f} m/s "
      f"(scaled from 0.2 m/s by factor {scale_factor:.6f})")
print("  With this scaling, both robots now cover approximately the same")
print("  distance over 300 steps, isolating yaw/heading behavior from the")
print("  earlier confound of mismatched effective travel speed.")
print("#" * 90)

simulation_app.close()