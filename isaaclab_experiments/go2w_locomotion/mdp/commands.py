"""Command terms for the Go2W locomotion task."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz, wrap_to_pi,quat_mul, yaw_quat

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def base_quat_to_roll(robot) -> torch.Tensor:
    """Roll of the basis in the world coordinate system, in radians, in the interval [-pi, pi]."""
    roll, _, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    return wrap_to_pi(roll)


class UniformBaseHeightCommand(CommandTerm):
    """Samples a target base height uniformly within a range.

    The command is a single value per environment, resampled every
    ``resampling_time_range`` seconds. It is meant to be consumed by
    :func:`~isaaclab_experiments.go2w_locomotion.mdp.rewards.base_height_penalty`
    and exposed to the policy through ``mdp.generated_commands``.
    """

    cfg: UniformBaseHeightCommandCfg

    def __init__(self, cfg: UniformBaseHeightCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            "UniformBaseHeightCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}\n"
            f"\tHeight range: ({self.cfg.ranges.min_height}, {self.cfg.ranges.max_height})"
        )

    @property
    def command(self) -> torch.Tensor:
        """Target base height. Shape is (num_envs, 1)."""
        return self.height_command

    def _update_metrics(self):
        self.metrics["error_height"] = torch.abs(self.height_command[:, 0] - self.robot.data.root_pos_w[:, 2])

    def _resample_command(self, env_ids: Sequence[int]):
        self.height_command[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(
            self.cfg.ranges.min_height, self.cfg.ranges.max_height
        )

    def _update_command(self):
        pass


@configclass
class UniformBaseHeightCommandCfg(CommandTermCfg):
    """Configuration for :class:`UniformBaseHeightCommand`."""

    class_type: type = UniformBaseHeightCommand

    asset_name: str = MISSING
    """Name of the articulation whose base height is commanded."""

    @configclass
    class Ranges:
        min_height: float = MISSING
        max_height: float = MISSING

    ranges: Ranges = MISSING
    """Uniform sampling range of the target height (in meters)."""












class UniformBaseRollCommand(CommandTerm):
    """Samples a target base roll uniformly within a range.

    The command is a single value per environment, resampled every
    ``resampling_time_range`` seconds. It is meant to be consumed by
    :func:`~isaaclab_experiments.go2w_locomotion.mdp.rewards.base_roll_penalty`
    and exposed to the policy through ``mdp.generated_commands``.
    """

    cfg: UniformBaseRollCommandCfg

    def __init__(self, cfg: UniformBaseRollCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.roll_command = torch.zeros(self.num_envs, 1, device=self.device)
        self.metrics["error_roll"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (
            "UniformBaseRollCommand:\n"
            f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
            f"\tResampling time range: {self.cfg.resampling_time_range}\n"
            f"\tRoll range: ({self.cfg.ranges.min_roll}, {self.cfg.ranges.max_roll})"
        )

    @property
    def command(self) -> torch.Tensor:
        """Target base roll. Shape is (num_envs, 1)."""
        return self.roll_command

    def _update_metrics(self):
        self.metrics["error_roll"] = torch.abs(wrap_to_pi(self.roll_command[:, 0] - base_quat_to_roll(self.robot)))

    def _resample_command(self, env_ids: Sequence[int]):
        self.roll_command[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(
            self.cfg.ranges.min_roll, self.cfg.ranges.max_roll
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        self.cfg.current_roll_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.cfg.goal_roll_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        if debug_vis:
            if not hasattr(self, "current_roll_visualizer"):
                self.current_roll_visualizer = VisualizationMarkers(self.cfg.current_roll_visualizer_cfg)
                self.goal_roll_visualizer = VisualizationMarkers(self.cfg.goal_roll_visualizer_cfg)
            self.current_roll_visualizer.set_visibility(True)
            self.goal_roll_visualizer.set_visibility(True)
        elif hasattr(self, "current_roll_visualizer"):
            self.current_roll_visualizer.set_visibility(False)
            self.goal_roll_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        pos = self.robot.data.root_pos_w.clone()
        quat = self.robot.data.root_quat_w
        cmd = self.command[:, 0]
        zeros = torch.zeros_like(cmd)
        goal_quat = quat_mul(yaw_quat(quat), quat_from_euler_xyz(cmd, zeros, zeros))
        current_pos = pos.clone()
        current_pos[:, 2] += 0.4
        goal_pos = pos.clone()
        goal_pos[:, 2] += 0.8
        self.current_roll_visualizer.visualize(goal_pos, quat)
        self.goal_roll_visualizer.visualize(goal_pos, goal_quat)

    def _update_command(self):
        pass


@configclass
class UniformBaseRollCommandCfg(CommandTermCfg):
    """Configuration for :class:`UniformBaseRollCommand`."""

    class_type: type = UniformBaseRollCommand

    asset_name: str = MISSING
    """Name of the articulation whose base roll is commanded."""

    @configclass
    class Ranges:
        min_roll: float = MISSING
        max_roll: float = MISSING

    current_roll_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/roll_current")
    goal_roll_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/roll_goal")

    ranges: Ranges = MISSING
    """Uniform sampling range of the target roll (in radians)."""
