import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import carb
settings = carb.settings.get_settings()
settings.set("/app/runLoops/main/rateLimitEnabled", True)
settings.set("/app/runLoops/main/rateLimitFrequency", 50)

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.envs import ManagerBasedRLEnv
import torch
from isaaclab.utils.math import quat_apply

import sys
sys.path.append("../..")


import config
from motion_tracking.mdp.scene import G1SceneCfg
from motion_tracking.mdp.commands import MotionCommand
from motion_tracking.beyondmimic.env import G1BeyondMimicEnvCfg


def check_scene():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))

    sim.set_camera_view(
        eye=(4.0, 4.0, 3.0),
        target=(0.0, 0.0, 0.8),
    )

    scene_cfg = G1SceneCfg(num_envs=4, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # print(scene.env_origins)

    g1: Articulation = scene["g1"]

    root_state = g1.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins

    joint_pos = g1.data.default_joint_pos.clone()
    joint_vel = g1.data.default_joint_vel.clone()
    # print(root_state[3:], g1.data.projected_gravity_b)
    # joint_pos[:] = 0.

    g1.write_root_pose_to_sim(root_state[:, :7])
    g1.write_root_velocity_to_sim(root_state[:, 7:])
    g1.write_joint_state_to_sim(joint_pos, joint_vel)

    scene.reset()

    # print("Joint names:")
    # for name in g1.joint_names:
    #     print("  ", name)

    print("\nBody names:")
    for name in g1.body_names:
        print("  ", name)

    
    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        print(g1.data.projected_gravity_b)

    # while simulation_app.is_running():
    #     sim.render()

def check_env():

    env_cfg = G1BeyondMimicEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env_cfg.commands.motion_command.motions = ["walk1_subject1", "dance1_subject2"]
    env_cfg.commands.motion_command.start_frame = 300
    env_cfg.commands.motion_command.num_frames = -1
    env_cfg.commands.motion_command.start_idx = None
    # env_cfg.sim.render_interval = 1_000_000 # env.step() still runs physics, but does not refresh the viewport.
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    command: MotionCommand = env.command_manager.get_term("motion_command")
    g1: Articulation = env.scene["g1"]
    joint_ids, joint_names = g1.find_joints(
        config.G1_MJ_JOINT_NAMES,
        preserve_order=True,
    )

    print(1)
    obs, extras = env.reset()
    print(2)

    zero_actions = torch.zeros(
        env.num_envs,
        env.action_manager.total_action_dim,
        device=env.device,
    )

    for i in range(500):
        print(3)
        obs, rew, terminated, truncated, extras = env.step(zero_actions)
        print(obs.keys())
        print(obs["shared_command"].shape)

        #
        # write reference motion for sanity check
        #

        # root_pos_w = command.ref_root_pos_w.clone()
        # root_quat_w = command.ref_root_quat_w.clone()
        # root_lin_vel_w = command.ref_root_lin_vel_w.clone()
        # root_ang_vel_w = quat_apply(root_quat_w, command.ref_root_ang_vel_b.clone())

        # g1.write_root_link_state_to_sim(
        #     torch.cat([
        #         root_pos_w, root_quat_w, root_lin_vel_w, root_ang_vel_w
        #     ], dim=-1)
        # )

        # joint_pos = command.ref_joint_pos.clone()
        # joint_vel = command.ref_joint_vel.clone()
        # g1.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids)

        # env.sim.forward()
        env.sim.render()

    env.close()

if __name__ == "__main__":
    try:
        # check_scene()
        check_env()
        
    finally:
        simulation_app.close()
