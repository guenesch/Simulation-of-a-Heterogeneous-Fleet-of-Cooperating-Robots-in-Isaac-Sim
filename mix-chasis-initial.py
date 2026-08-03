from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})
import carb
import numpy as np
from isaacsim.core.api import World
from isaacsim.robot.wheeled_robots.controllers.holonomic_controller import HolonomicController
from isaacsim.robot.wheeled_robots.controllers.differential_controller import DifferentialController
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from isaacsim.robot.wheeled_robots.robots.holonomic_robot_usd_setup import HolonomicRobotUsdSetup
from isaacsim.storage.native import get_assets_root_path

my_world = World(stage_units_in_meters=1.0)
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder")

kaya_asset_path = assets_root_path + "/Isaac/Robots/Kaya/kaya.usd"
jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"

# Create Kaya robot at x = -1.5
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

my_world.scene.add_default_ground_plane()

# --- Kaya controller (holonomic) ---
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
wheel_radius_jetbot = 0.03
wheel_base_jetbot   = 0.1125

my_controller_jetbot = DifferentialController(
    name="differential_controller_jetbot",
    wheel_radius=wheel_radius_jetbot,
    wheel_base=wheel_base_jetbot,
)

my_world.reset()

# Kaya commands (holonomic: vx, vy, omega)
kaya_commands = [
    (0.200, 0.000, 0.000),
    (0.200, 0.010, 0.00100),
    (0.194, 0.040, 0.00305),
    (0.172, 0.084, 0.00492),
    (0.126, 0.134, 0.00705),
    (0.078, 0.178, 0.00699),
    (0.052, 0.188, 0.00287),
    (0.082, 0.184, -0.00303),
    (0.106, 0.166, -0.00294),
    (0.124, 0.156, -0.00208),
]

# Jetbot commands (differential: v_linear, omega)
jetbot_commands = [
    (0.2000, 0.2000),
    (0.1997, 0.2008),
    (0.1963, 0.1999),
    (0.1885, 0.1943),
    (0.1798, 0.1881),
    (0.1902, 0.1984),
    (0.1934, 0.1967),
    (0.2032, 0.1997),
    (0.1987, 0.1952),
    (0.2005, 0.1981),
]

i = 0
reset_needed = False

while simulation_app.is_running():
    my_world.step(render=True)
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            my_controller_1.reset()
            my_controller_jetbot.reset()
            reset_needed = False

        # Each interval is 50 steps; select command index from the lists above
        cmd_index = min(i // 30, len(kaya_commands) - 1)
        kaya_cmd   = kaya_commands[cmd_index]
        jetbot_cmd = jetbot_commands[cmd_index]

        my_kaya_1.apply_wheel_actions(my_controller_1.forward(command=list(kaya_cmd)))
        my_jetbot.apply_wheel_actions(my_controller_jetbot.forward(command=list(jetbot_cmd)))

        if i == 300:
            i = 0

        i += 1

simulation_app.close()