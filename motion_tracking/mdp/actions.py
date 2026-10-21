import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

import sys
sys.path.append("../..")
import config

@configclass
class G1ActionCfg:
    joint_position = mdp.JointPositionActionCfg(
        asset_name="g1",
        joint_names=config.G1_MJ_JOINT_NAMES, # using mujoco joint order for action
        scale=config.G1_ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True
    )