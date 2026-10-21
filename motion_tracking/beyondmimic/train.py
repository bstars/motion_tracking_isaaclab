import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--motions", nargs="+", type=str, default=[])
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_env_steps", type=float, default=1e7)
parser.add_argument("--ent_coef", type=float, default=5e-3)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--ckpt", type=str, default="temp.pth")
parser.add_argument("--push", action="store_true")


AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import os
import sys
sys.path.append("../..")
from motion_tracking.beyondmimic.env import G1BeyondMimicEnvCfg
from motion_tracking.beyondmimic.ppo_cfg import G1BeyondMimicPPORunnerCfg

def main():
    env_cfg: G1BeyondMimicEnvCfg = G1BeyondMimicEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.commands.motion_command.start_frame = 0
    env_cfg.commands.motion_command.num_frames = 2000 # all frames
    env_cfg.commands.motion_command.num_future_frames = 5
    env_cfg.commands.motion_command.motions = args.motions
    if not args.push:
        env_cfg.events.random_push= None
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    

    ppo_cfg: G1BeyondMimicPPORunnerCfg = G1BeyondMimicPPORunnerCfg()
    ppo_cfg.device = args.device
    ppo_cfg.algorithm.entropy_coef = args.ent_coef

    ppo_cfg.max_iterations = math.ceil(
        args.max_env_steps / (ppo_cfg.num_steps_per_env * args.num_envs)
    )
    ppo_cfg.save_interval = ppo_cfg.max_iterations + 2 # Only save the final ckpt
    runner = OnPolicyRunner(
        env,
        ppo_cfg.to_dict(),
        log_dir="./logs",
        device=ppo_cfg.device,
    )

    if args.resume:
        runner.load(os.path.join("./ckpts", args.ckpt), load_optimizer=True)
        runner.alg.entropy_coef = args.ent_coef

    runner.learn(num_learning_iterations=ppo_cfg.max_iterations)
    runner.save(os.path.join("./ckpts", args.ckpt))
    env.close()

if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()