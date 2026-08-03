### BOOTSTRAP: SimulationApp must be created before any other isaacsim
### import. Creating it launches the simulator process.
from isaacsim import SimulationApp

### headless=False shows the 3D viewport window.
simulation_app = SimulationApp({"headless": False})
import carb
import numpy as np
from scipy.spatial.transform import Rotation
from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.controllers.holonomic_controller import HolonomicController
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.robots.holonomic_robot_usd_setup import HolonomicRobotUsdSetup
from isaacsim.storage.native import get_assets_root_path

### Create the simulation world; 1 unit = 1 metre.
my_world = World(stage_units_in_meters=1.0)
### Locate Isaac's bundled robot assets (error-logged if missing).
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")

### Paths to the two robot models: Kaya (3 omni-wheels, holonomic) and
### Jetbot (2 wheels, differential drive).
kaya_asset_path = assets_root_path + "/Isaac/Robots/Kaya/kaya.usd"
jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"

# Create Kaya robot at x = -1.5
### Spawn the Kaya at x=-3, lifted 0.02 m above the floor. Its three
### axle_*_joint names are the omni-wheels; identity quaternion = no rotation.
my_kaya_1 = my_world.scene.add(
    WheeledRobot(
        prim_path="/World/Kaya_1",
        name="my_kaya_1",
        wheel_dof_names=["axle_0_joint", "axle_1_joint", "axle_2_joint"],
        create_robot=True,
        usd_path=kaya_asset_path,
        position=np.array([-3.0, 0.0, 0.02]),
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    )
)

# Create Jetbot at x = +1.5 (3m apart on X axis)
# Rotated 180° around Z to face same direction as Kaya
### Spawn the Jetbot at the origin, 3 m from the Kaya along X. Its two
### wheel joints drive it; the [0,0,0,1] quaternion faces it like the Kaya.
my_jetbot = my_world.scene.add(
    WheeledRobot(
        prim_path="/World/Jetbot",
        name="my_jetbot",
        wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
        create_robot=True,
        usd_path=jetbot_asset_path,
        position=np.array([0.0, 0.0, 0.0]),
        orientation=np.array([0.0, 0.0, 0.0, 1.0]),  # 180° around Z
    )
)

### Add a flat infinite floor for both robots to drive on.
my_world.scene.add_default_ground_plane()

# --- Kaya controller (holonomic) ---
### Read the Kaya's wheel geometry from its USD model, then build a
### HolonomicController that turns a [vx, vy, omega] command into 3 wheel speeds.
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

# --- Jetbot controller (differential) ---
# Hardcoded physical params — no DifferentialRobotUsdSetup needed
### Jetbot wheel params are HARDCODED here (radius 0.03 m, base 0.1125 m)
### rather than measured from the model.
wheel_radius_jetbot = 0.03
wheel_base_jetbot   = 0.1125

### DifferentialController turns a [forward_speed, turn] command into
### left/right wheel speeds.
my_controller_jetbot = DifferentialController(
    name="differential_controller_jetbot",
    wheel_radius=wheel_radius_jetbot,
    wheel_base=wheel_base_jetbot,
)

### Finalize physics; robots settle onto the ground.
my_world.reset()


# ── pose helper ──────────────────────────────────────────────────────────────
### Helper: return a robot's world position and Euler angles. Isaac gives
### the quaternion as [w,x,y,z]; scipy wants [x,y,z,w], hence the reorder.
### euler[2] is the yaw (heading) in degrees.
def get_pose_info(robot):
    """Return (position, euler_xyz_degrees) for a WheeledRobot."""
    pos, orient = robot.get_world_pose()
    # Isaac Sim quaternion convention: [w, x, y, z]
    euler = Rotation.from_quat(
        [orient[1], orient[2], orient[3], orient[0]]
    ).as_euler('xyz', degrees=True)
    return pos, euler


### Print both robots' X/Y/Z position and yaw at a given step.
def print_positions(step_number, kaya, jetbot):
    print(f"\n{'=' * 80}")
    print(f"STEP {step_number}")
    print(f"{'=' * 80}")
    for name, robot in [("Kaya_1 (holonomic)", kaya), ("Jetbot (differential)", jetbot)]:
        pos, euler = get_pose_info(robot)
        print(
            f"  {name:<28} "
            f"X={pos[0]:8.4f}  Y={pos[1]:8.4f}  Z={pos[2]:8.4f}  "
            f"Yaw={euler[2]:7.2f}°"
        )


### Print each robot's net displacement, total distance, and final yaw
### relative to where it started.
def print_final_comparison(kaya, jetbot, initial_data):
    print(f"\n{'=' * 80}")
    print("FINAL COMPARISON")
    print(f"{'=' * 80}")
    print(
        f"\n{'Robot':<28} {'X-Disp':>10} {'Y-Disp':>10} "
        f"{'Total Dist':>12} {'Final Yaw':>10}"
    )
    print("-" * 72)
    for name, robot in [("Kaya_1 (holonomic)", kaya), ("Jetbot (differential)", jetbot)]:
        pos, euler = get_pose_info(robot)
        init_pos = initial_data[name]
        dx = pos[0] - init_pos[0]
        dy = pos[1] - init_pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        print(
            f"  {name:<26} {dx:>10.4f} {dy:>10.4f} "
            f"{dist:>12.4f} {euler[2]:>9.2f}°"
        )


# ── store initial positions ───────────────────────────────────────────────────
print("\n" + "=" * 80)
print("INITIAL ROBOT POSITIONS")
print("=" * 80)

### Record each robot's starting position so displacement can be measured later.
initial_data = {}
for name, robot in [("Kaya_1 (holonomic)", my_kaya_1), ("Jetbot (differential)", my_jetbot)]:
    pos, euler = get_pose_info(robot)
    initial_data[name] = pos.copy()
    print(
        f"  {name:<28} "
        f"X={pos[0]:8.4f}  Y={pos[1]:8.4f}  Z={pos[2]:8.4f}  "
        f"Yaw={euler[2]:7.2f}°"
    )

# Kaya commands (holonomic: vx, vy, omega)
### Kaya command sequence (holonomic): each tuple is [vx, vy, omega], one
### per 30-step interval, tracing an S-curve. omega is 0 (no self-rotation).
kaya_commands = [
    (0.2000, 0.0000, 0.0),
    (0.1997, 0.0150, 0.0),
    (0.1872, 0.0640, 0.0),
    (0.1358, 0.1343, 0.0),
    (0.0237, 0.1814, 0.0),
    (-0.1032, 0.1635, 0.0),
    (-0.1588, 0.1130, 0.0),
    (-0.1442, 0.1404, 0.0),
    (-0.0945, 0.1726, 0.0),
    (-0.0535, 0.1919, 0.0)
]

# Jetbot commands (differential: v_linear, omega)
### Jetbot command sequence (differential): each tuple is [forward, turn],
### one per interval, meant to follow a similar curve.
jetbot_commands = [
    (0.1085, 0.1115),  # step 0 - straight
    (0.1119, 0.1151),  # step 1 - gentle left
    (0.1052, 0.1118),  # step 2 - more left
    (0.1003, 0.1143),  # step 3 - turning more
    (0.0939, 0.1137),  # step 4 - sharpest turn
    (0.0974, 0.1145),  # step 5 - still turning
    (0.1052, 0.1147),  # step 6 - easing off
    (0.1052, 0.1155),  # step 7 - now turning right
    (0.1074, 0.1146),  # step 8 - gentle right
    (0.1071, 0.1116),  # step 9 - gentle right
]

### Print positions every 30 steps; run for 300 steps total.
PRINT_INTERVAL = 30   # print positions every N steps
TOTAL_STEPS    = 300  # total steps for one full run

### i indexes the command intervals; global_step counts all sim steps.
i = 0
global_step  = 0
reset_needed = False

print("\n" + "=" * 80)
print("STARTING SIMULATION")
print("=" * 80)

### MAIN LOOP: advance the physics one frame at a time until 300 steps,
### while the sim window is open. Only acts while the sim is playing.
while simulation_app.is_running() and global_step <= TOTAL_STEPS:
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            my_controller_1.reset()
            my_controller_jetbot.reset()
            reset_needed = False

        # Each interval is 30 steps; select command index from the lists above
        ### Pick the current command: advance every 30 steps, hold on the last one.
        cmd_index = min(i // 30, len(kaya_commands) - 1)
        kaya_cmd   = kaya_commands[cmd_index]
        jetbot_cmd = jetbot_commands[cmd_index]

        ### Convert each robot's command into wheel actions and apply them this step.
        my_kaya_1.apply_wheel_actions(my_controller_1.forward(command=list(kaya_cmd)))
        my_jetbot.apply_wheel_actions(my_controller_jetbot.forward(command=list(jetbot_cmd)))

        # Periodic position printing
        ### Every 30 steps, print a position snapshot.
        if global_step % PRINT_INTERVAL == 0:
            print_positions(global_step, my_kaya_1, my_jetbot)

        ### On the final step, print the start-vs-end comparison and stop.
        if global_step == TOTAL_STEPS:
            # Print final comparison once the run is complete, then stop
            print_final_comparison(my_kaya_1, my_jetbot, initial_data)
            break

        global_step += 1
        i += 1

### Shut down Isaac Sim cleanly.
simulation_app.close()