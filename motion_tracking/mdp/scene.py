import numpy as np
from copy import deepcopy

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg

import sys
sys.path.append('../..')
import config


def make_actuators_from_joint_params(joint_params):
    return {
        joint_name: ImplicitActuatorCfg(
            joint_names_expr=[joint_name],
            stiffness=params["STIFFNESS"],
            damping=params["DAMPING"],
            effort_limit_sim=params["TORQUE_LIM"],
            velocity_limit_sim=params["VEL_LIM"],
        )
        for joint_name, params in joint_params.items()
    }

G1_CFG = ArticulationCfg(
    prim_path=None,
    spawn=sim_utils.UsdFileCfg(
        usd_path=config.G1_USD_PATH,
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        joint_pos=config.G1_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators=make_actuators_from_joint_params(config.G1_JOINT_PARAMS)
)

@configclass
class G1SceneCfg(InteractiveSceneCfg):

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )

    g1 = G1_CFG.replace(
        prim_path="{ENV_REGEX_NS}/g1",
    )

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/g1/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=False,
    )
