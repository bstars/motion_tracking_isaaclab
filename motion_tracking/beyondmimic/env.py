
#
# RL Environment similar to BeyondMimic
#   

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass


import sys
sys.path.append("../..")


import motion_tracking.mdp.scene as mt_scenes
import motion_tracking.mdp.events as mt_events
import motion_tracking.mdp.commands as mt_commands
import motion_tracking.mdp.observations as mt_observations
import motion_tracking.mdp.actions as mt_actions
import motion_tracking.mdp.rewards as mt_rewards
import motion_tracking.mdp.terminations as mt_terminations

@configclass
class G1BeyondMimicCommandCfg:
    # motion_command = mt_commands.MotionCommandCfg(
    #     class_type=mt_commands_single.SingleMotionCommand,
    #     asset_name="g1",
    #     motion="walk1_subject1",
    #     num_future_frames=1,
    #     num_frames=-1, 
    #     resampling_time_range=(1.0e9, 1.0e9)
    # )

    motion_command = mt_commands.MotionCommandCfg(
        class_type=mt_commands.MotionCommand,
        asset_name="g1",
        motions=["walk1_subject1"],
        num_future_frames=5,
        start_frame=0,
        num_frames=-1,
        start_idx=None,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        adaptive_sampling_cfg={
            "bin_size": 50,
            "alpha": 0.75,
            "beta": 100,
            "failure_counts_multiplier": 1.0
        }
    )


@configclass
class G1BeyondMimicObsCfg:

    #
    # Shared observations
    #
    shared_command = mt_observations.G1MotionCommandObsGroupCfg(concatenate_terms=True, concatenate_dim=-1, enable_corruption=False)


    #
    # Actor-specific observations
    #
    actor_proprio = mt_observations.G1ProprioObsGroupCfg(concatenate_terms=True, concatenate_dim=-1, enable_corruption=True)
    actor_ref_anchor_rel_anchor = mt_observations.G1RefAnchorPoseRelAnchorObsGroupCfg(concatenate_terms=True, concatenate_dim=-1, enable_corruption=True)

    #
    # Critic-specific observation
    #
    critic_proprio = mt_observations.G1ProprioObsGroupCfg(concatenate_terms=True, concatenate_dim=-1, enable_corruption=False)
    critic_lin_vel = mt_observations.G1LinVelProprioObsGroupCfg(concatenate_terms=True, concatenate_dim=-1, enable_corruption=False)
    critic_ref_anchor_rel_anchor = mt_observations.G1RefAnchorPoseRelAnchorObsGroupCfg(concatenate_terms=True, concatenate_dim=-1, enable_corruption=False)
    critic_body_rel_anchor = mt_observations.G1BodyPoseRelAnchorObsGroupCfg(concatenate_terms=True, concatenate_dim=-1, enable_corruption=False)

@configclass
class G1BeyondMimicEnvCfg(ManagerBasedRLEnvCfg):
    scene = mt_scenes.G1SceneCfg(num_envs=4, env_spacing=2.5)
    events = mt_events.G1EventCfg()
    commands = G1BeyondMimicCommandCfg()
    observations = G1BeyondMimicObsCfg()
    actions = mt_actions.G1ActionCfg()
    rewards = mt_rewards.G1MotionRewardCfg()
    terminations = mt_terminations.G1MotionTerminationCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation