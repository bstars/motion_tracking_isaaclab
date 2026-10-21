from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import TerminationTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul, matrix_from_quat


import sys
sys.path.append("../..")

import config
from motion_tracking.mdp.commands import MotionCommand


def motion_clip_finished(env: ManagerBasedRLEnv, command_name: str):
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.motion_finished()

def bad_anchor_ori(env: ManagerBasedRLEnv, asset_cfg:SceneEntityCfg, command_name: str, threshold: float):
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[asset_cfg.name]

    ref_gravity = quat_apply_inverse(command.ref_anchor_quat_w, robot.data.GRAVITY_VEC_W)
    rob_gravity = quat_apply_inverse(robot.data.body_quat_w[:, asset_cfg.body_ids[0], :], robot.data.GRAVITY_VEC_W)

    return (ref_gravity[:, 2] - rob_gravity[:, 2]).abs() > threshold

def bad_anchor_z(env: ManagerBasedRLEnv, asset_cfg:SceneEntityCfg, command_name: str, threshold: float):
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[asset_cfg.name]

    return torch.abs(
        command.ref_anchor_pos_w[:, -1] - robot.data.body_pos_w[:, asset_cfg.body_ids[0], -1]
    ) > threshold

def bad_eef_z(env: ManagerBasedRLEnv, asset_cfg:SceneEntityCfg, command_name: str, threshold: float):
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot: Articulation = env.scene[asset_cfg.name]

    error = torch.abs(
        command.ref_body_pos_w[:, config.G1_END_EFFECTOR_BODY_IDXES, -1]
        - robot.data.body_pos_w[:, asset_cfg.body_ids, -1]
    )

    return torch.any(error > threshold, dim=-1)



@configclass
class G1MotionTerminationCfg:

    timeout = TerminationTermCfg(func=mdp.time_out, time_out=True)

    motion_end = TerminationTermCfg(
        func=motion_clip_finished,
        params={"command_name": "motion_command"},
        time_out=True,
    )

    anchor_ori = TerminationTermCfg(
        func=bad_anchor_ori,
        params={
            "asset_cfg": SceneEntityCfg("g1", body_names=[config.G1_ANCHOR_BODY_NAME]),
            "command_name": "motion_command",
            "threshold": 0.8
        },
        time_out=False
    )

    anchor_z = TerminationTermCfg(
        func=bad_anchor_z,
        params={
            "asset_cfg": SceneEntityCfg("g1", body_names=[config.G1_ANCHOR_BODY_NAME]),
            "command_name": "motion_command",
            "threshold": 0.25
        },
        time_out=False
    )

    eef_z = TerminationTermCfg(
        func=bad_eef_z,
        params={
            "asset_cfg": SceneEntityCfg("g1", body_names=config.G1_END_EFFECTOR_BODY_NAMES, preserve_order=True),
            "command_name": "motion_command",
            "threshold": 0.25
        },
        time_out=False
    )