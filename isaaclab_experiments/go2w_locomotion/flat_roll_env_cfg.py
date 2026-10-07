"""Unitree Go2W flat-terrain locomotion with a commanded base roll.

Extends :class:`~.flat_env_cfg.Go2WFlatEnvCfg` with a uniformly sampled base
roll command that the policy must track while following velocity commands.
The configuration reproduces the ``unitree_go2w_flat_z`` training runs
(``logs/rsl_rl/unitree_go2w_flat_z``), including the observation layout, so
their checkpoints remain loadable.
"""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from robot_lab.tasks.manager_based.locomotion.velocity.config.wheeled.unitree_go2w.rough_env_cfg import (
    UnitreeGo2WRewardsCfg,
)
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import CommandsCfg

import isaaclab_experiments.go2w_locomotion.mdp as mdp

from .flat_env_cfg import Go2WFlatEnvCfg

WHEEL_JOINT_NAMES = ["FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint"]
WHEEL_BODY_NAMES = ["FR_foot", "FL_foot", "RR_foot", "RL_foot"]


@configclass
class Go2WFlatRollCommandsCfg(CommandsCfg):
    """Velocity command plus a uniformly sampled target base roll."""

    base_roll = mdp.UniformBaseRollCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        debug_vis=True,
        ranges=mdp.UniformBaseRollCommandCfg.Ranges(min_roll=-0.3, max_roll=0.3) 
       # if max_roll change, change max_roll of _roll_scale function in rewards.py too
       # range roll: Range [-0.75~0.75] (rad); pitch: Range [-0.75~0.75] (rad); yaw: Range [-0.6~0.6] (rad).
       # https://support.unitree.com/home/en/developer/Motion_Services_Interface_V2.04
    )

@configclass
class Go2WFlatRollObservationsCfg:
    """Observation groups with the roll command placed after the velocity command.

    The term order defines the observation vector layout and must not change:
    base_lin_vel, base_ang_vel, projected_gravity, velocity_commands,
    roll_command, joint_pos, joint_vel, actions.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1), clip=(-100.0, 100.0), scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-100.0, 100.0), scale=0.25)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05), clip=(-100.0, 100.0), scale=1.0
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}, clip=(-100.0, 100.0), scale=1.0
        )
        roll_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_roll"}, clip=(-100.0, 100.0), scale=1.0
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
            },
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        # kept so that the robot_lab base configuration can disable it
        height_scan: ObsTerm | None = None

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=1.0)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}, clip=(-100.0, 100.0), scale=1.0
        )
        roll_command = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_roll"}, clip=(-100.0, 100.0), scale=1.0
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_without_wheel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        height_scan: ObsTerm | None = None

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Go2WFlatRollRewardsCfg(UnitreeGo2WRewardsCfg):
    """Go2W rewards plus the terms specific to roll tracking."""

    go2w_joint_mirror_error = RewTerm(
        func=mdp.go2w_joint_mirror,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "hip_reference_weight": 0.5,
            "min_symmetry_scale": 0.25,
            "yaw_max": 1.0,
        },
    )

    wheel_position_penalty = RewTerm(
        func=mdp.wheel_position_penalty,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=WHEEL_BODY_NAMES)},
    )

@configclass
class Go2WFlatRollEnvCfg(Go2WFlatEnvCfg):
    commands: Go2WFlatRollCommandsCfg = Go2WFlatRollCommandsCfg()
    observations: Go2WFlatRollObservationsCfg = Go2WFlatRollObservationsCfg()
    rewards: Go2WFlatRollRewardsCfg = Go2WFlatRollRewardsCfg()

    def __post_init__(self):
        # post init of parent (rough -> flat -> IL4OP flat)
        super().__post_init__()

        self.sim.enable_scene_query_support = False

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # ------------------------------Observations------------------------------
        # the base configuration removes the linear velocity from the policy; restore it
        self.observations.policy.base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1), clip=(-100.0, 100.0), scale=2.0
        )

        # ------------------------------Rewards------------------------------
        """# track the commanded base height, measured from the terrain under the base
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_penalty,
            weight=-100.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=self.base_link_name),
                "sensor_cfg": SceneEntityCfg("height_scanner_base"),
                "target_height": None,
                "command_name": "base_height",
            },
        )"""

        # track the commanded base roll
        self.rewards.roll_l2 = RewTerm(
               func=mdp.track_base_roll_penalty,
               weight=-10,
               params={
                   "asset_cfg": SceneEntityCfg("robot", body_names=self.base_link_name),
                   "target_roll": None,
                   "command_name": "base_roll"
                },
            )

        # replaced by go2w_joint_mirror_error and wheel_position_penalty
        
        self.rewards.joint_mirror.weight = 0
        self.rewards.joint_pos_penalty.weight = 0
        self.rewards.joint_acc_wheel_l2.weight = 0
        self.rewards.go2w_joint_mirror_error.weight = 0
        self.rewards.wheel_position_penalty.weight = 0

        # On peut en ajouter plus mais on verra après

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "Go2WFlatRollEnvCfg":
            self.disable_zero_weight_rewards()
