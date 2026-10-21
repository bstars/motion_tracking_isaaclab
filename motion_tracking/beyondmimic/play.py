import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--motion", type=str, default="dance1_subject2")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--ckpt", type=str, default="temp.pth")



AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import torch

import os
import sys
sys.path.append("../..")

from motion_tracking.beyondmimic.env import G1BeyondMimicEnvCfg
from motion_tracking.beyondmimic.ppo_cfg import G1BeyondMimicPPORunnerCfg

if __name__ == "__main__":
    env_cfg: G1BeyondMimicEnvCfg = G1BeyondMimicEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.commands.motion_command.start_frame = 0
    env_cfg.commands.motion_command.num_frames = 2000
    env_cfg.commands.motion_command.num_future_frames = 5
    env_cfg.commands.motion_command.start_idx = 0
    env_cfg.commands.motion_command.motions = [args.motion]
    env_cfg.commands.motion_command.adaptive_sampling_cfg = None


    env_cfg.decimation = 2
    env_cfg.sim.dt = 0.01
    env_cfg.episode_length_s = 1e9
    env_cfg.terminations.timeout = None
    env_cfg.terminations.anchor_ori = None
    env_cfg.terminations.anchor_z = None
    env_cfg.terminations.eef_z = None

    # Evaluation does not need observation noise
    env_cfg.observations.actor_proprio.enable_corruption = False
    env_cfg.observations.actor_ref_anchor_rel_anchor.enable_corruption = False

    # Evaluation does not need domain randomization
    env_cfg.events.joint_mounting_randomization = None
    env_cfg.events.physical_material_randomization = None
    env_cfg.events.body_mass_randomization = None
    env_cfg.events.body_com_randomization = None
    env_cfg.events.actuator_gains = None
    env_cfg.events.joint_params_randomization = None
    env_cfg.events.random_push = None


    # Rewards are unused during inference.
    env_cfg.rewards.r_anchor_pos = None
    env_cfg.rewards.r_anchor_ori = None
    env_cfg.rewards.r_pos = None
    env_cfg.rewards.r_ori = None
    env_cfg.rewards.r_lin_vel = None
    env_cfg.rewards.r_ang_vel = None
    env_cfg.rewards.p_action_rate = None
    env_cfg.rewards.p_soft_joint_lim = None
    env_cfg.rewards.undesired_contacts = None

    # Contact measurements are unnecessary after disabling contact rewards.
    env_cfg.scene.contact_forces = None
    env_cfg.scene.g1.spawn.activate_contact_sensors = False

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    

    ppo_cfg: G1BeyondMimicPPORunnerCfg = G1BeyondMimicPPORunnerCfg()
    runner = OnPolicyRunner(
        env,
        ppo_cfg.to_dict(),
        log_dir="./logs",
        device=ppo_cfg.device,
    )

    
    runner.load(os.path.join("./ckpts", args.ckpt), load_optimizer=True)
    actor_critic = runner.alg.policy
    actor_critic.eval()
        

    obs, _ = env.reset()

    try:
        
        while app.is_running():
            with torch.inference_mode():
                actions = actor_critic.act_inference(obs)
                obs, _, _, _ = env.step(actions)
    finally:
    

        env.close()
        app.close()