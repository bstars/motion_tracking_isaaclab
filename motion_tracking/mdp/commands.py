from __future__ import annotations


import numpy as np
import torch
from scipy.io import loadmat
import os
from dataclasses import MISSING
from abc import ABC, abstractmethod

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_inv,quat_mul, yaw_quat
from isaaclab.envs import ManagerBasedRLEnv

import sys
sys.path.append("../..")
import config


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        #
        # Robot
        #
        self.g1: Articulation = env.scene[cfg.asset_name]
        anchor_body_ids, anchor_body_names = self.g1.find_bodies(
            [config.G1_ANCHOR_BODY_NAME],
            preserve_order=True,
        )

        # isaac joint ids in mujoco order
        self.joint_ids, joint_names = self.g1.find_joints(config.G1_MJ_JOINT_NAMES,preserve_order=True)
        self.anchor_body_id = anchor_body_ids[0]


        #
        # Motion
        #
        motion_files = [
            os.path.join(config.LAFAN1_INTERPOLATED_PATH, motion_name + ".mat")
            for motion_name in self.cfg.motions
        ]

        q_hist_all = []
        qdot_hist_all = []
        poses_hist_all = []
        vels_hist_all = []
        motion_lengths = []

        start_frame = self.cfg.start_frame
        end_frame = None if self.cfg.num_frames == -1 else start_frame + self.cfg.num_frames

        for motion_file in motion_files:
            data = loadmat(motion_file)
            q_hist = torch.tensor(data["q_hist"][start_frame:end_frame], dtype=torch.float32, device=self.device)    # [num_frame, 7+29]
            qdot_hist = torch.tensor(data["qdot_hist"][start_frame:end_frame], dtype=torch.float32, device=self.device)   # [num_frame, 6+29]
            poses_hist = torch.tensor(data["poses_hist"][start_frame:end_frame], dtype=torch.float32, device=self.device) # [num_frame, num_body, 7]
            vels_hist = torch.tensor(data["vels_hist"][start_frame:end_frame], dtype=torch.float32, device=self.device)   # [num_frame, num_body, 6]

            q_hist_all.append(q_hist)
            qdot_hist_all.append(qdot_hist)
            poses_hist_all.append(poses_hist)
            vels_hist_all.append(vels_hist)
            motion_lengths.append(q_hist.shape[0])

        self.q_hist = torch.cat(q_hist_all, dim=0)
        self.qdot_hist = torch.cat(qdot_hist_all, dim=0)
        self.poses_hist = torch.cat(poses_hist_all, dim=0)
        self.vels_hist = torch.cat(vels_hist_all, dim=0)
        self.num_frames = torch.tensor(motion_lengths, dtype=torch.long, device=self.device)
        self.motion_starts = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=self.device),
                self.num_frames.cumsum(dim=0),
            ]
        )[:-1]
        self.num_motions = len(motion_files)

        #
        # Commands
        #
        self._command = torch.zeros([self.num_envs, (29+29+3) * self.cfg.num_future_frames], dtype=torch.float32, device=self.device) # (joint_pos + joint_vel + projected_gravity) * num_frames
        self._motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._frame_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        num_bodies = self.poses_hist.shape[1]
        self._target_body_pos_w = torch.zeros(
            [self.num_envs, num_bodies, 3], dtype=torch.float32, device=self.device
        )
        self._target_body_quat_w = torch.zeros(
            [self.num_envs, num_bodies, 4], dtype=torch.float32, device=self.device
        )

        #
        # Adaptive Sampling
        #
        self.adaptive_sampling_cfg = self.cfg.adaptive_sampling_cfg

        if self.adaptive_sampling_cfg is not None:

            bin_starts_all = []
            bin_ends_all = []
            for motion_id, motion_frames in enumerate(self.num_frames.tolist()):
                num_valid_starts = motion_frames - self.cfg.num_future_frames
                bin_starts = torch.arange(
                    0, num_valid_starts, self.adaptive_sampling_cfg["bin_size"],
                    dtype=torch.long, device=self.device
                )
                bin_ends = torch.minimum(
                    bin_starts + self.adaptive_sampling_cfg["bin_size"],
                    torch.tensor(num_valid_starts, dtype=torch.long, device=self.device),
                )

                motion_ids = torch.ones_like(bin_starts, dtype=torch.long, device=self.device) * motion_id

                bin_starts_all.append(self.get_frame_ids(motion_ids, bin_starts))
                bin_ends_all.append(self.get_frame_ids(motion_ids, bin_ends))

            self.bin_starts = torch.cat(bin_starts_all) # frame start of bin in whole motion dataset
            self.bin_ends = torch.cat(bin_ends_all)     # frame end of bin in whole motion dataset
            self.bin_lengths = self.bin_ends - self.bin_starts
            self.exposures = torch.ones_like(self.bin_starts, dtype=torch.float32, device=self.q_hist.device)
            self.failures = torch.ones_like(self.bin_starts, dtype=torch.float32, device=self.q_hist.device)
            self._bin_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self._has_motion_sample = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    #
    # Joints
    #
    @property   # [num_envs, 29]
    def ref_joint_pos(self):
        return self.q_hist[self._frame_ids, 7:]

    @property   # [num_envs, 29]
    def ref_joint_vel(self):
        return self.qdot_hist[self._frame_ids, 6:]


    #
    # Poses after anchor transform
    #
    @property  # [num_envs, num_body, 3]
    def target_body_pos_w(self):
        return self._target_body_pos_w

    @property  # [num_envs, num_body, 4]
    def target_body_quat_w(self):
        return self._target_body_quat_w


    #
    # Root Body
    #
    @property  # [num_envs, 3]
    def ref_root_pos_w(self):
        return self.q_hist[self._frame_ids, :3] + self._env.scene.env_origins

    @property  # [num_envs, 4]
    def ref_root_quat_w(self):
        return self.q_hist[self._frame_ids, 3:7]

    @property  # [num_envs, 3]
    def ref_root_lin_vel_w(self):
        return self.qdot_hist[self._frame_ids, :3]

    @property  # [num_envs, 3]
    def ref_root_ang_vel_b(self):
        return self.qdot_hist[self._frame_ids, 3:6]


    #
    # Tracked Bodies
    #
    @property  # [num_envs, num_body, 3]
    def ref_body_pos_w(self):
        return self.poses_hist[self._frame_ids, :, :3] + self._env.scene.env_origins[:, None, :]

    @property  # [num_envs, num_body, 4]
    def ref_body_quat_w(self):
        return self.poses_hist[self._frame_ids, :, 3:7]

    @property  # [num_envs, num_body, 3]
    def ref_body_lin_vel_w(self):
        return self.vels_hist[self._frame_ids, :, :3]

    @property  # [num_envs, num_body, 3]
    def ref_body_ang_vel_w(self):
        return self.vels_hist[self._frame_ids, :, 3:]

    #
    # Anchor Bodies
    #
    @property  # [num_envs, 3]
    def ref_anchor_pos_w(self):
        return self.poses_hist[self._frame_ids, 0, :3] + self._env.scene.env_origins

    @property  # [num_envs, 4]
    def ref_anchor_quat_w(self):
        return self.poses_hist[self._frame_ids, 0, 3:]

    @property  # [num_envs, 3]
    def ref_anchor_lin_vel_w(self):
        return self.vels_hist[self._frame_ids, 0, :3]

    @property  # [num_envs, 3]
    def ref_anchor_ang_vel_w(self):
        return self.vels_hist[self._frame_ids, 0, 3:]

    def get_frame_ids(self, motion_ids, time_steps):
        return self.motion_starts[motion_ids] + time_steps

    #
    # Adaptive sampling
    #
    @torch.no_grad()
    def update_exposure(self, bin_ids: torch.Tensor, valid_mask: torch.Tensor):
        counts = torch.bincount(
            bin_ids,
            weights=valid_mask.to(torch.float32),
            minlength=self.bin_starts.numel(),
        )

        self.exposures.add_(counts / self.bin_lengths.to(torch.float32))

    @torch.no_grad()
    def update_failure(self, bin_ids: torch.Tensor, failure_mask: torch.Tensor):
        counts = torch.bincount(
            bin_ids,
            weights=failure_mask.to(torch.float32),
            minlength=self.bin_starts.numel(),
        )

        multiplier = self.adaptive_sampling_cfg.get("failure_counts_multiplier", 1.0)
        self.failures.add_(multiplier * counts)
        


    #
    # Core updates
    #
    @torch.no_grad()
    def _update_command_tensor(self, env_ids):
        env_ids = env_ids if env_ids is not None else slice(None)
        frame_offsets = torch.arange(self.cfg.num_future_frames, device=self.device) # [num_future_frames]
        frame_ids = self._frame_ids[env_ids, None] + frame_offsets[None, :]    # [batch_envs, num_future_frames]

        root_quat = self.q_hist[frame_ids, 3:7]      # [batch_envs, num_future_frames, 4]
        joint_pos = self.q_hist[frame_ids, 7:]       # [batch_envs, num_future_frames, 29]
        joint_vel = self.qdot_hist[frame_ids, 6:]    # [batch_envs, num_future_frames, 29]

        gravity_w = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32).expand(*root_quat.shape[:-1], 3)    # [batch_envs, num_future_frames, 3]
        projected_gravity = quat_apply_inverse(root_quat, gravity_w)    # [batch_envs, num_future_frames, 3]
        self._command[env_ids] = torch.cat([joint_pos, joint_vel, projected_gravity], dim=-1).flatten(start_dim=1)

    @torch.no_grad()
    def _update_target_pose_tensor(self, env_ids, anchor_pos_w, anchor_quat_w):
        """
        For env_ids, given the anchor pose, compute and update target body poses.

        env_ids: torch.Tensor, [batch_envs]
        anchor_pos_w: torch.Tensor, [batch_envs, 3]
        anchor_quat_w: torch.Tensor, [batch_envs, 4]
        """

        env_ids = env_ids if env_ids is not None else slice(None)
        
        target_anchor_pos_w = anchor_pos_w.clone()                      # [batch_envs, 3]
        target_anchor_pos_w[:, 2] = self.ref_anchor_pos_w[env_ids, 2]   # [batch_envs, 3]

        delta_quat_w = yaw_quat(
            quat_mul(
                anchor_quat_w,
                quat_inv(self.ref_anchor_quat_w[env_ids,:]),
            )
        )   # [batch_envs, 4]

        num_body = self.ref_body_pos_w.shape[1]
        delta_quat_all = delta_quat_w[:, None, :].expand(-1, num_body, -1)  # [batch_envs, num_body, 4]

        self._target_body_quat_w[env_ids] = quat_mul(
            delta_quat_all,
            self.ref_body_quat_w[env_ids, :, :],
        ) # [batch_envs, num_body, 4]

        self._target_body_pos_w[env_ids] = (
            target_anchor_pos_w[:, None, :]
            + quat_apply(
                delta_quat_all,
                self.ref_body_pos_w[env_ids, :, :] - self.ref_anchor_pos_w[env_ids, None, :],
            )
        )  # [batch_envs, num_body, 3]

    @torch.no_grad()
    def _resample_command(self, env_ids):
        if len(env_ids) == 0:
            return


        if self.adaptive_sampling_cfg is None:
            self._motion_ids[env_ids] = torch.randint(
                low=0,
                high=self.num_motions,
                size=(len(env_ids),),
                device=self.device,
            )

            motion_num_frames = self.num_frames[self._motion_ids[env_ids]]
            num_valid_starts = motion_num_frames - self.cfg.num_future_frames

            if self.cfg.start_idx is not None: 
                self._time_steps[env_ids] = self.cfg.start_idx
            else:
                self._time_steps[env_ids] = (
                    torch.rand(len(env_ids), device=self.device) * num_valid_starts
                ).long()

            self._frame_ids[env_ids] = self.get_frame_ids(self._motion_ids[env_ids], self._time_steps[env_ids])
        
        else:
            termination_manager = self._env.termination_manager

            completed_mask = self._has_motion_sample[env_ids] & termination_manager.dones[env_ids]
            failure_mask = completed_mask & termination_manager.terminated[env_ids]
            
            self.update_failure(bin_ids=self._bin_ids[env_ids], failure_mask=failure_mask)

            # sonic-style adaptive sampling
            failure_rate = self.failures / self.exposures.clamp_min(1e-8)
            failure_rate = torch.clamp(failure_rate, 0., failure_rate.mean() * self.adaptive_sampling_cfg["beta"])
            dist = failure_rate / failure_rate.sum()
            dist = self.adaptive_sampling_cfg["alpha"] * dist + (1 - self.adaptive_sampling_cfg["alpha"]) / self.bin_starts.numel()
            
            sampled_bin_ids = torch.multinomial(dist, num_samples=env_ids.numel(), replacement=True)
            sampled_frame_ids = (
                torch.rand(len(env_ids), device=self.device) * self.bin_lengths[sampled_bin_ids]
            ).floor().long() + self.bin_starts[sampled_bin_ids]

            self._bin_ids[env_ids] = sampled_bin_ids
            self._frame_ids[env_ids] = sampled_frame_ids
            self._motion_ids[env_ids] = torch.searchsorted(self.motion_starts, self._frame_ids[env_ids], right=True)-1
            self._time_steps[env_ids] = sampled_frame_ids - self.motion_starts[self._motion_ids[env_ids]]
            self._has_motion_sample[env_ids] = True
            

        # TODO: Randomize robot start state
        root_pos_w = self.ref_root_pos_w[env_ids].clone()
        root_quat_w = self.ref_root_quat_w[env_ids].clone()
        root_lin_vel_w = self.ref_root_lin_vel_w[env_ids].clone()
        root_ang_vel_w = quat_apply(root_quat_w, self.ref_root_ang_vel_b[env_ids].clone())

        self.g1.write_root_link_state_to_sim(
            torch.cat([root_pos_w, root_quat_w, root_lin_vel_w, root_ang_vel_w], dim=-1),
            env_ids=env_ids,
        )

        # TODO: Randomize joints state
        joint_pos = self.ref_joint_pos[env_ids].clone()
        joint_vel = self.ref_joint_vel[env_ids].clone()
        self.g1.write_joint_state_to_sim(
            joint_pos, joint_vel, env_ids=env_ids, joint_ids=self.joint_ids
        )

        #
        # Update
        #
        self._update_command_tensor(env_ids)
        self._target_body_pos_w[env_ids] = self.ref_body_pos_w[env_ids]
        self._target_body_quat_w[env_ids] = self.ref_body_quat_w[env_ids]

        # So far we are doing perfect initialization to a frame,
        # so we dont need to compute the target body poses
        # 
        # If we add domain randomization,
        # we need to compute the target motion

    @torch.no_grad()
    def _update_command(self):
        # update adaptive sampling
        if self.adaptive_sampling_cfg is not None:
            self.update_exposure(bin_ids=self._bin_ids, valid_mask=self._has_motion_sample)

        # update command
        reset_mask = self._env.reset_buf
        env_ids = (~reset_mask).nonzero(as_tuple=False).squeeze(-1)

        if env_ids.numel() > 0:
            self._time_steps[env_ids] += 1
            self._frame_ids[env_ids] = self.get_frame_ids(self._motion_ids[env_ids], self._time_steps[env_ids])

            if self.adaptive_sampling_cfg is not None:
                self._bin_ids[env_ids] = torch.searchsorted(self.bin_starts, self._frame_ids[env_ids],right=True) - 1
    
            self._update_target_pose_tensor(
                env_ids=env_ids,
                anchor_pos_w=self.g1.data.body_pos_w[env_ids, self.anchor_body_id, :],
                anchor_quat_w=self.g1.data.body_quat_w[env_ids, self.anchor_body_id, :],
            )

        self._update_command_tensor(None)


    def motion_finished(self):
        motion_num_frames = self.num_frames[self._motion_ids]
        return self._time_steps + self.cfg.num_future_frames >= motion_num_frames
    
    @property
    def command(self):  return self._command

    def _update_metrics(self): pass

        
@configclass
class MotionCommandCfg(CommandTermCfg):
    class_type: type = MotionCommand
    asset_name: str = MISSING
    motions: list[str] = MISSING
    num_future_frames:int = MISSING
    start_frame: int = 0        
    num_frames: int = -1
    start_idx: int | None = None
    adaptive_sampling_cfg: dict | None = None # Add this later