
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R
import re
import torch
from rsl_rl.modules import ActorCritic
import glfw
import time
from scipy.io import loadmat
from tensordict import TensorDict

import os
import sys
sys.path.append("../..")

import config


class Sim2SimEnv:
    def __init__(self, scene_file, motion, start_frame=0, num_frames=-1, num_future_frames=1):
        self.model = mujoco.MjModel.from_xml_path(scene_file)
        self.data = mujoco.MjData(self.model)


        motion_file=os.path.join(config.LAFAN1_INTERPOLATED_PATH, motion+".mat")
        data = loadmat(motion_file)
        end_frame = None if num_frames == -1 else start_frame + num_frames
        self.q_hist = np.asarray(data["q_hist"][start_frame:end_frame])          # [num_frame, 7+29]
        self.qdot_hist = np.asarray(data["qdot_hist"][start_frame:end_frame])    # [num_frame, 6+29]
        self.poses_hist = np.asarray(data["poses_hist"][start_frame:end_frame])  # [num_frame, num_body, 7]
        self.vels_hist = np.array(data["vels_hist"][start_frame:end_frame])  # [num_frame, num_body, 6]
        self.num_frames = self.q_hist.shape[0]
        self.frame_idx = 0

        self.sim_steps = 4
        self.model.opt.timestep = self.dt = 0.005
        self.control_dt = self.sim_steps * self.dt
        self.anchor_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            config.G1_ANCHOR_BODY_NAME,
        )

        self.gravity = np.array([0.0, 0.0, -1.0])
        self.action_dim = len(config.G1_MJ_JOINT_NAMES)
        self.prev_action = np.zeros(self.action_dim, dtype=np.float32)
        self.num_future_frames = num_future_frames
        

        self.default_joint_pos= np.array([
            next(
                (
                    value
                    for pattern, value
                    in config.G1_DEFAULT_JOINT_POS.items()
                    if re.fullmatch(pattern, joint_name)
                ),
                0.0,
            )
            for joint_name in config.G1_MJ_JOINT_NAMES
        ])

        # Unitree names actuators "left_hip_pitch", while the associated
        # joint is named "left_hip_pitch_joint". Map using transmission
        # joint IDs rather than relying on actuator names.
        id_joint_to_actuator = {}
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)

            # print(joint_name, actuator_name)
            id_joint_to_actuator[joint_id] = actuator_id

        # map from mujoco_joint_names[i] to index of xxx
        self.joint_ids = np.empty(self.action_dim, dtype=np.int32)
        self.qpos_ids = np.empty(self.action_dim, dtype=np.int32)
        self.qvel_ids = np.empty(self.action_dim, dtype=np.int32)
        self.actuator_ids = np.empty(self.action_dim, dtype=np.int32)

        self.kp = np.empty(self.action_dim, dtype=np.float32)
        self.kd = np.empty(self.action_dim, dtype=np.float32)
        self.torque_limit = np.empty(self.action_dim, dtype=np.float32)

        for mj_index, joint_name in enumerate(config.G1_MJ_JOINT_NAMES):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator_id = id_joint_to_actuator[joint_id]
            params = config.G1_JOINT_PARAMS[joint_name]

            self.joint_ids[mj_index] = joint_id
            self.qpos_ids[mj_index] = self.model.jnt_qposadr[joint_id]
            self.qvel_ids[mj_index] = self.model.jnt_dofadr[joint_id]
            self.actuator_ids[mj_index] = actuator_id
            self.kp[mj_index] = params["STIFFNESS"]
            self.kd[mj_index] = params["DAMPING"]
            self.torque_limit[mj_index] = params["TORQUE_LIM"]

            dof_id = self.qvel_ids[mj_index]
            self.model.dof_frictionloss[dof_id] = 0.0
            # self.model.dof_damping[dof_id] = params["DAMPING"]
            self.model.dof_armature[dof_id] = params["ARMATURE"]

        # Action scale is stored in policy/Isaac order.
        self.action_scale = np.array([
            0.25
            * config.G1_JOINT_PARAMS[joint_name]["TORQUE_LIM"]
            / config.G1_JOINT_PARAMS[joint_name]["STIFFNESS"]
            for joint_name in config.G1_MJ_JOINT_NAMES
        ])


        #
        # Contact friction
        #
        for geom_id in range(self.model.ngeom):
            self.model.geom_friction[geom_id, 0] = 1.0

    def get_obs(self):

        #
        # proprioception
        #
        root_ang_vel = self.data.qvel[3:6].copy()

        quat = R.from_quat(self.data.qpos[3:7].copy(), scalar_first=True)
        projected_gravity = quat.inv().apply(self.gravity)

        joint_pos = self.data.qpos[self.qpos_ids]- self.default_joint_pos
        joint_vel = self.data.qvel[self.qvel_ids].copy()

        last_action = self.prev_action.copy()

        proprio = np.concatenate([root_ang_vel, projected_gravity, joint_pos, joint_vel, last_action])

        #
        # motion command
        #
        idx = self.frame_idx + np.arange(self.num_future_frames)

        ref_joint_pos = self.q_hist[idx, 7:]    # [num_future_frames, 29]
        ref_joint_vel = self.qdot_hist[idx, 6:] # [num_future_frames, 29]

        root_quat = self.q_hist[idx, 3:7]   # [num_future_frames, 4]
        gravity = np.broadcast_to(self.gravity,(self.num_future_frames, 3))
        ref_gravity = R.from_quat(root_quat, scalar_first=True).inv().apply(gravity)

        command = np.concatenate([ref_joint_pos, ref_joint_vel, ref_gravity], axis=-1).flatten()

        #
        # pose of reference anchor to anchor
        #
        anchor_pos_w = self.data.xpos[self.anchor_body_id].copy()
        anchor_quat_w = R.from_quat(
            self.data.xquat[self.anchor_body_id].copy(),
            scalar_first=True,
        )

        ref_anchor_quat_w = R.from_quat(self.poses_hist[self.frame_idx, 0, 3:7], scalar_first=True)
        ref_anchor_pos_w = self.poses_hist[self.frame_idx, 0, 0:3]

        rel_ori = anchor_quat_w.inv() * ref_anchor_quat_w
        rel_ori = rel_ori.as_matrix()[:,1:].reshape(6)

        rel_pos = anchor_quat_w.inv().apply(ref_anchor_pos_w - anchor_pos_w)
        
        rel_anchor_pose = np.concatenate([rel_pos, rel_ori])

        return TensorDict(
            {
                "shared_command" : command[None,:],
                "actor_proprio" : proprio[None,:],
                "actor_ref_anchor_rel_anchor": rel_anchor_pose[None,:],

                "critic_proprio" : proprio[None,:],
                "critic_lin_vel": np.zeros([1,3]),
                "critic_ref_anchor_rel_anchor": rel_anchor_pose[None,:],
                "critic_body_rel_anchor": np.zeros([1, 13*9])
            },
            batch_size=[1,]
        ).float32()

    def apply_position_target(self, target_joint_pos):
        """Convert position targets into torque commands."""

        joint_pos = self.data.qpos[self.qpos_ids]
        joint_vel = self.data.qvel[self.qvel_ids]

        torque = (
            self.kp * (target_joint_pos - joint_pos)
            - self.kd * joint_vel
        )
        torque = np.clip(
            torque,
            -self.torque_limit,
            self.torque_limit,
        )

        # actuator_ids and torque are both in MuJoCo joint order.
        self.data.ctrl[self.actuator_ids] = torque


    def reset(self, start_idx):
        self.frame_idx = start_idx

        self.data.qpos[0:7] = self.q_hist[self.frame_idx, 0:7].copy()
        self.data.qpos[self.qpos_ids] = self.q_hist[self.frame_idx, 7:].copy()

        self.data.qvel[0:6] = self.qdot_hist[self.frame_idx, 0:6].copy()
        self.data.qvel[self.qvel_ids] = self.qdot_hist[self.frame_idx, 6:].copy()
        self.apply_position_target(self.q_hist[self.frame_idx, 7:])

        mujoco.mj_forward(self.model, self.data)
        self.prev_action[:] = (self.q_hist[self.frame_idx, 7:] - self.default_joint_pos) / self.action_scale

        return self.get_obs()

    def step(self, action):
        self.frame_idx += 1
        target_joint_pos = self.default_joint_pos + self.action_scale * action
        for _ in range(self.sim_steps):
            self.apply_position_target(target_joint_pos)
            mujoco.mj_step(self.model, self.data)
            self.prev_action[:] = action

        return self.get_obs(), self.frame_idx + self.num_future_frames >= self.num_frames


def load_teacher_policy_rsl(obs, checkpoint_path, device="cpu"):
    obs_groups = {
        "policy": [
            "shared_command",
            "actor_proprio",
            "actor_ref_anchor_rel_anchor",
        ],
        "critic": [
            "shared_command",
            "critic_proprio",
            "critic_lin_vel",
            "critic_ref_anchor_rel_anchor",
            "critic_body_rel_anchor",
        ],
    }

    policy = ActorCritic(
        obs=obs,
        obs_groups=obs_groups,
        num_actions=29,
        actor_hidden_dims=[1024, 512, 128],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
        init_noise_std=0.75,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    policy.eval()

    return policy

if __name__ == '__main__':
    motion = "dance1_subject2"

    env = Sim2SimEnv(
        scene_file=config.G1_XML_PATH,
        motion=motion,
        start_frame=0,
        num_frames=2000,
        num_future_frames=5
    )
    obs = env.reset(0)

    policy = load_teacher_policy_rsl(obs, checkpoint_path=os.path.join("./ckpts", "temp.pth"))

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()

            # action = np.zeros([29,])
            with torch.inference_mode():
                action = policy.act_inference(obs).squeeze(0).detach().numpy()

            obs, done = env.step(action)

            viewer.cam.lookat[:] = env.data.qpos[0:3]
            viewer.sync()

            remaining = env.control_dt - (time.perf_counter() - step_start)

            # if remaining > 0.0:
            #     time.sleep(remaining)
            time.sleep(env.control_dt)

            if done: 
                break