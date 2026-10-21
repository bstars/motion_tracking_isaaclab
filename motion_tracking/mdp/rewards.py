from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.sensors import ContactSensor
from isaaclab.assets import Articulation
from isaaclab.utils.math import axis_angle_from_quat, quat_mul, quat_inv, quat_apply_inverse

import sys
sys.path.append("..")
import config
from motion_tracking.mdp.commands import MotionCommand

def reward_anchor_pos_tracking(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, std: float):
    robot: Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos_diff = robot.data.body_pos_w[:, asset_cfg.body_ids[0], :] - command.ref_anchor_pos_w # [num_envs, 3]
    pos_diff = torch.sum(pos_diff**2, dim=-1)
    return torch.exp(-pos_diff / std**2)

def reward_anchor_pos_tracking_z(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, std: float):
    robot: Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos_diff = robot.data.body_pos_w[:, asset_cfg.body_ids[0], -1] - command.ref_anchor_pos_w[:, -1] # [num_envs, 3]
    pos_diff = pos_diff**2
    return torch.exp(-pos_diff / std**2)

def reward_anchor_ori_tracking(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, std: float):
    robot: Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)

    quat = quat_mul(
        robot.data.body_quat_w[:, asset_cfg.body_ids[0], :],
        quat_inv(command.ref_anchor_quat_w)
    ) # [num_env, 4]

    ori_diff = axis_angle_from_quat(quat) # [num_env, 3]
    ori_diff = torch.sum(ori_diff**2, dim=-1)

    return torch.exp(-ori_diff / std**2)

def reward_anchor_ori_tracking_rp(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, std: float):
    robot: Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)

    ref_gravity = quat_apply_inverse(command.ref_anchor_quat_w, robot.data.GRAVITY_VEC_W)
    rob_gravity = quat_apply_inverse(robot.data.body_quat_w[:, asset_cfg.body_ids[0], :], robot.data.GRAVITY_VEC_W)
    ori_diff = (ref_gravity[:, :2] - rob_gravity[:, :2]).square().sum(-1)
    return torch.exp(-ori_diff / std**2)

    

def reward_pos_tracking(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, std: float):
    robot: Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos_diff = robot.data.body_pos_w[:, asset_cfg.body_ids, :] - command.target_body_pos_w
    pos_diff = torch.mean(
        torch.sum(
            pos_diff**2, dim=-1
        ),
        dim=-1
    )
    return torch.exp(-pos_diff / std**2)

def reward_ori_tracking(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, std: float):
    robot: Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)

    quat = quat_mul(
        robot.data.body_quat_w[:, asset_cfg.body_ids, :],
        quat_inv(command.target_body_quat_w)
    ) # [num_env, num_body, 4]

    ori_diff = axis_angle_from_quat(quat) # [num_env, num_body, 3]
    ori_diff = torch.mean(
        torch.sum(
            ori_diff**2, dim=-1
        ),
        dim=-1
    )

    return torch.exp(-ori_diff / std**2)

def reward_lin_vel_tracking(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, frame: str, std: float):
    robot: Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)

    if frame == "body":
        ref_lin_vel_b = quat_apply_inverse(
            command.ref_body_quat_w,
            command.ref_body_lin_vel_w
        )

        lin_vel_b = quat_apply_inverse(
            robot.data.body_quat_w[:, asset_cfg.body_ids, :],
            robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
        )
        lin_vel_diff = ref_lin_vel_b - lin_vel_b
    else:
        lin_vel_diff = command.ref_body_lin_vel_w - robot.data.body_lin_vel_w[:, asset_cfg.body_ids, :]

    lin_vel_diff = torch.mean(
        torch.sum(
            lin_vel_diff**2, dim=-1
        ),
        dim=-1
    )
    return torch.exp(-lin_vel_diff / std**2)

def reward_ang_vel_tracking(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, frame: str, std: float):
    robot: Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)

    if frame == "body":
        ref_ang_vel_b = quat_apply_inverse(
            command.ref_body_quat_w,
            command.ref_body_ang_vel_w
        )

        lin_ang_b = quat_apply_inverse(
            robot.data.body_quat_w[:, asset_cfg.body_ids, :],
            robot.data.body_ang_vel_w[:, asset_cfg.body_ids, :]
        )

        lin_ang_diff = ref_ang_vel_b - lin_ang_b
    else:
        lin_ang_diff = command.ref_body_ang_vel_w - robot.data.body_ang_vel_w[:, asset_cfg.body_ids, :]

    lin_ang_diff = torch.mean(
        torch.sum(
            lin_ang_diff**2, dim=-1
        ),
        dim=-1
    )
    return torch.exp(-lin_ang_diff / std**2)

@configclass
class G1MotionRewardCfg:
    r_anchor_pos = RewardTermCfg(
        func=reward_anchor_pos_tracking,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("g1", body_names=[config.G1_ANCHOR_BODY_NAME], preserve_order=True),
            "command_name": "motion_command",
            "std": 0.2
        }
    )

    r_anchor_ori = RewardTermCfg(
        func=reward_anchor_ori_tracking,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("g1", body_names=[config.G1_ANCHOR_BODY_NAME], preserve_order=True),
            "command_name": "motion_command",
            "std": 0.25
        }
    )


    r_pos = RewardTermCfg(
        func=reward_pos_tracking,
        weight=1.,
        params={
            "asset_cfg": SceneEntityCfg("g1", body_names=config.G1_TRACKED_BODY_NAMES, preserve_order=True),
            "command_name": "motion_command",
            "std": 0.1
        }
    )

    r_ori = RewardTermCfg(
        func=reward_ori_tracking,
        weight=1.,
        params={
            "asset_cfg": SceneEntityCfg("g1", body_names=config.G1_TRACKED_BODY_NAMES, preserve_order=True),
            "command_name": "motion_command",
            "std": 0.25
        }
    )

    r_lin_vel = RewardTermCfg(
        func=reward_lin_vel_tracking,
        weight=1.,
        params={
            "asset_cfg": SceneEntityCfg("g1", body_names=config.G1_TRACKED_BODY_NAMES, preserve_order=True),
            "command_name": "motion_command",
            "frame": "world",
            "std": 0.3
        }
    )

    r_ang_vel = RewardTermCfg(
        func=reward_ang_vel_tracking,
        weight=1.,
        params={
            "asset_cfg": SceneEntityCfg("g1", body_names=config.G1_TRACKED_BODY_NAMES, preserve_order=True),
            "command_name": "motion_command",
            "frame": "world",
            "std": 0.5 * math.pi
        }
    )

    p_action_rate = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.05
    )

    p_soft_joint_lim = RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-1.,
        params={
            "asset_cfg": SceneEntityCfg("g1", joint_names=config.G1_MJ_JOINT_NAMES, preserve_order=True)
        }
    )

    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)"
                    r"(?!right_ankle_roll_link$)"
                    r"(?!left_wrist_yaw_link$)"
                    r"(?!right_wrist_yaw_link$)"
                    r"(?!left_elbow_link$)"
                    r"(?!right_elbow_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )