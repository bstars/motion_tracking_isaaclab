import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg
from isaaclab.managers import ObservationTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise, GaussianNoiseCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.assets import Articulation
from isaaclab.utils.math import quat_apply_inverse, quat_inv, quat_mul, matrix_from_quat


import sys
sys.path.append("../..")
import config
from motion_tracking.mdp.commands import MotionCommand


#
# Proprioception
#
@configclass
class G1ProprioObsGroupCfg(ObservationGroupCfg):

    root_ang_vel = ObservationTermCfg(
        func=mdp.base_ang_vel,
        params={
            "asset_cfg": SceneEntityCfg("g1")
        },
        noise=Unoise(n_min=-0.1, n_max=0.1)
    )

    projected_gravity = ObservationTermCfg(
        func=mdp.projected_gravity,
        params={
            "asset_cfg" : SceneEntityCfg("g1")
        },
        noise=Unoise(n_min=-0.1, n_max=0.1)
    )

    joint_pos = ObservationTermCfg(
        func=mdp.joint_pos_rel,
        params={
            "asset_cfg": SceneEntityCfg("g1", joint_names=config.G1_MJ_JOINT_NAMES, preserve_order=True)
        },
        noise=Unoise(n_min=-0.05, n_max=0.05)
    )

    joint_vel = ObservationTermCfg(
        func=mdp.joint_vel_rel,
        params={
            "asset_cfg": SceneEntityCfg("g1", joint_names=config.G1_MJ_JOINT_NAMES, preserve_order=True)
        },
        noise=Unoise(n_min=-0.25, n_max=0.25)
    )

    last_action = ObservationTermCfg(
        func=mdp.last_action
    )

@configclass
class G1LinVelProprioObsGroupCfg(ObservationGroupCfg):
    root_lin_vel = ObservationTermCfg(
        func=mdp.base_lin_vel,
        params={
            "asset_cfg": SceneEntityCfg("g1")
        },
        noise=Unoise(n_min=-0.25, n_max=0.25)
    )

#
# Motion Command
#
@configclass
class G1MotionCommandObsGroupCfg(ObservationGroupCfg):
    motion_command = ObservationTermCfg(
        mdp.generated_commands,
        params={"command_name": "motion_command"}
    )



#
# The relative position/orientation reference anchor to robot anchor
# This is used to correct global drift
#

def ref_anchor_pos_rel_anchor(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str):
    robot: Articulation = env.scene[asset_cfg.name]
    anchor_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids[0], :]   # [num_envs, 3]
    anchor_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids[0], :] # [num_envs, 4]

    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_anchor_pos_w = command.ref_anchor_pos_w     # [num_envs, 3]

    rel_pos = quat_apply_inverse(anchor_quat_w, ref_anchor_pos_w - anchor_pos_w)    # [num_envs, 3]
    return rel_pos

def ref_anchor_ori_rel_anchor(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str):
    robot: Articulation = env.scene[asset_cfg.name]
    anchor_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids[0], :] # [num_envs, 4]

    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_anchor_quat_w = command.ref_anchor_quat_w   # [num_envs, 4]

    rel_ori = quat_mul(quat_inv(anchor_quat_w), ref_anchor_quat_w)  # [num_envs, 4]
    rel_ori = matrix_from_quat(rel_ori) # [num_envs, 3, 3]
    rel_ori = rel_ori[:, :, 1:].reshape(-1, 6)  # [num_envs, 6]
    return rel_ori

@configclass
class G1RefAnchorPoseRelAnchorObsGroupCfg(ObservationGroupCfg):
   rel_pos = ObservationTermCfg(
       func=ref_anchor_pos_rel_anchor,
       params={
           "asset_cfg": SceneEntityCfg("g1", body_names=[config.G1_ANCHOR_BODY_NAME], preserve_order=True),
           "command_name": "motion_command"
       },
       noise=Unoise(n_min=-0.05, n_max=0.05)
   )

   rel_ori = ObservationTermCfg(
        func=ref_anchor_ori_rel_anchor,
        params={
            "asset_cfg": SceneEntityCfg("g1", body_names=[config.G1_ANCHOR_BODY_NAME], preserve_order=True),
            "command_name": "motion_command"
        },
        noise=Unoise(n_min=-0.05, n_max=0.05)
   )






#
# The relative position/orientation target tracked bodies (except anchor itself) to robot anchor
# This is privileged knowledge to enable the critic to estimate the tracking error
# (This is not the one used in BeyondMimic)
#

def tar_body_pos_rel_anchor(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str):
    num_envs = env.num_envs

    command: MotionCommand = env.command_manager.get_term(command_name)
    target_body_pos_w = command.target_body_pos_w[:, 1:, :] # [num_evns, num_body-1, 3]

    robot: Articulation = env.scene[asset_cfg.name]
    anchor_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids[0], :] # [num_env, 4]
    anchor_quat_w = anchor_quat_w[:, None, :].expand(-1, target_body_pos_w.shape[1], -1) # [num_env, num_body-1, 4]
    anchor_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids[0], :] # [num_env, 3]
    anchor_pos_w = anchor_pos_w[:, None, :].expand(-1, target_body_pos_w.shape[1], -1) # [num_env, num_body-1, 3]

    rel_pos = quat_apply_inverse(anchor_quat_w, target_body_pos_w - anchor_pos_w) # [num_env, num_body-1, 3]
    rel_pos = rel_pos.reshape(num_envs, -1)
    return rel_pos

def tar_body_ori_rel_anchor(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str):
    num_envs = env.num_envs
    
    command: MotionCommand = env.command_manager.get_term(command_name)
    tar_body_quat_w = command.target_body_quat_w[:, 1:, :] # [num_evns, num_body-1, 4]

    robot: Articulation = env.scene[asset_cfg.name]
    anchor_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids[0], :] # [num_env, 4]
    anchor_quat_w = anchor_quat_w[:, None, :].expand(-1, tar_body_quat_w.shape[1], -1) # [num_env, num_body-1, 4]

    rel_ori = quat_mul(quat_inv(anchor_quat_w), tar_body_quat_w)
    rel_ori = matrix_from_quat(rel_ori)# [num_env, num_body-1, 3, 3]
    rel_ori = rel_ori[:, :, :, 1:].reshape(num_envs, -1) # [num_env, (num_body-1)*6]
    return rel_ori



@configclass
class G1TarBodyPoseRelAnchorObsGroupCfg(ObservationGroupCfg):  
    rel_pos = ObservationTermCfg(
        func=tar_body_pos_rel_anchor,
        params={
            "asset_cfg": SceneEntityCfg("g1", body_names=[config.G1_ANCHOR_BODY_NAME], preserve_order=True),
            "command_name": "motion_command"
        },
        noise=Unoise(n_min=-0.05, n_max=0.05)
    )

    rel_ori = ObservationTermCfg(
        func=tar_body_ori_rel_anchor,
        params={
            "asset_cfg": SceneEntityCfg("g1", body_names=[config.G1_ANCHOR_BODY_NAME], preserve_order=True),
            "command_name": "motion_command"
        },
        noise=Unoise(n_min=-0.05, n_max=0.05)
    )


#
# The relative position/orientation of tracked bodies (except to anchor itself) relative to anchor.
# This is privileged knowledge that describe the robot's pose in cartesion space.
# (This is in what BeyondMimic uses: https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py#L32)
#

def body_pos_rel_anchor(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    num_envs = env.num_envs
    robot: Articulation = env.scene[asset_cfg.name]

    body_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :] # [num_env, num_body, 3]
    anchor_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids[0], :] # [num_env, 3]

    anchor_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids[0], :] # [num_env, 4]
    anchor_quat_w = anchor_quat_w[:, None, :].expand(-1, body_pos_w.shape[1]-1, -1) # [num_env, num_body-1, 4]

    rel_pos = quat_apply_inverse(anchor_quat_w, body_pos_w[:, 1:, :] - anchor_pos_w[:, None, :]) # [num_env, num_body-1, 3]
    rel_pos = rel_pos.reshape(num_envs, -1)
    return rel_pos

def body_ori_rel_anchor(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg):
    num_envs = env.num_envs
    robot: Articulation = env.scene[asset_cfg.name]

    body_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids, :] # [num_env, num_body, 4]
    anchor_quat_w = body_quat_w[:, :1, :].expand(-1, body_quat_w.shape[1]-1, -1) # [num_env, num_body-1, 4]

    rel_ori = quat_mul(quat_inv(anchor_quat_w), body_quat_w[:, 1:, :])
    rel_ori = matrix_from_quat(rel_ori)# [num_env, num_body-1, 3, 3]
    rel_ori = rel_ori[:, :, :, 1:].reshape(num_envs, -1) # [num_env, (num_body-1)*6]
    return rel_ori

@configclass
class G1BodyPoseRelAnchorObsGroupCfg(ObservationGroupCfg):  
    rel_pos = ObservationTermCfg(
        func=body_pos_rel_anchor,
        params={
            "asset_cfg": SceneEntityCfg("g1", body_names=config.G1_TRACKED_BODY_NAMES, preserve_order=True),
        },
        noise=Unoise(n_min=-0.05, n_max=0.05)
    )

    rel_ori = ObservationTermCfg(
        func=body_ori_rel_anchor,
        params={
            "asset_cfg": SceneEntityCfg("g1", body_names=config.G1_TRACKED_BODY_NAMES, preserve_order=True),
        },
        noise=Unoise(n_min=-0.05, n_max=0.05)
    )