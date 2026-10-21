import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp


def randomize_joint_mounting_offset(
    env,
    env_ids,
    offset_range: tuple[float, float],
    action_name: str = "joint_position",
):
    action_term = env.action_manager.get_term(action_name)

    if env_ids is None:
        env_ids = torch.arange(
            env.num_envs,
            device=env.device,
            dtype=torch.long,
        )

    mounting_error = torch.empty(
        (len(env_ids), action_term.action_dim),
        device=env.device,
    ).uniform_(*offset_range)

    action_term._offset[env_ids] += mounting_error


@configclass
class G1EventCfg:

    #
    # Since the robot state is resetted in command resetting, 
    # the event terms are mainly domain randomization.
    #

    joint_mounting_randomization = EventTermCfg(
        func=randomize_joint_mounting_offset,
        mode="startup",
        params={
            "action_name": "joint_position",
            "offset_range": (-0.05, 0.05),  
        },
    )

    physical_material_randomization = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        min_step_count_between_reset=0,
        params={
            "asset_cfg": SceneEntityCfg("g1", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0., 0.5),
            "num_buckets": 128,
            "make_consistent": True
        }
    )

    body_mass_randomization = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("g1", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True
        }
    )

    body_com_randomization = EventTermCfg(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "g1",
                body_names=["pelvis", "torso_link"],
            ),
            "com_range": {
                "x": (-0.075, 0.075),
                "y": (-0.075, 0.075),
                "z": (-0.075, 0.075),
            },
        },
    )

    actuator_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("g1", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform"
        }
    )

    joint_params_randomization = EventTermCfg(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("g1", joint_names=".*"),
            "armature_distribution_params": (0.9, 1.1),
            "friction_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform"
        }
    )



    random_push = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "asset_cfg": SceneEntityCfg("g1"),
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            }
        },
    )
