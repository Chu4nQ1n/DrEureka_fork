import torch
import numpy as np
from forward_locomotion.go1_gym.utils.math_utils import quat_apply_yaw, wrap_to_pi, get_scale_shift
from isaacgym.torch_utils import *

class EurekaReward():
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    def compute_reward(self):
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.
        
        # Constants
        target_velocity_x = 2.0  # m/s
        target_torso_height = 0.34  # meters
    
        # Velocity Reward Component
        forward_velocity = env.root_states[:, 7]  # Linear velocity in x of the base
        velocity_error = torch.abs(forward_velocity - target_velocity_x)
        velocity_reward = torch.exp(-velocity_error)
    
        # Height Reward Component
        z_position = env.root_states[:, 2]  # Z position of the robot base
        height_error = torch.abs(z_position - target_torso_height)
        height_reward = torch.exp(-height_error * 10)  # Scaled to make the robot maintain closer to the target height
    
        # Orientation Reward Component
        projected_gravity_norm = torch.norm(env.projected_gravity, dim=1)
        orientation_reward = torch.exp(-torch.abs(projected_gravity_norm - 1.0))  # Encouraging to be perpendicular to gravity
    
        # Smoothness Reward Component
        dof_vel_change = torch.abs(env.dof_vel - env.last_dof_vel)
        action_change = torch.abs(env.actions - env.last_actions)
        smoothness_reward = torch.exp(-torch.mean(dof_vel_change, dim=1)) * torch.exp(-torch.mean(action_change, dim=1))
    
        # DOF Limits Avoidance
        dof_limit_min = env.dof_pos_limits[:, 0]
        dof_limit_max = env.dof_pos_limits[:, 1]
        dof_position = env.dof_pos
        dof_limits_penalty = torch.mean((dof_position < dof_limit_min) | (dof_position > dof_limit_max), dim=1) * -1.0
    
        # Aggregate the rewards and penalties
        total_reward = velocity_reward + height_reward + orientation_reward + smoothness_reward + dof_limits_penalty
    
        # Collate the individual components for logging or debugging purposes
        reward_components = {
            "velocity": velocity_reward,
            "height": height_reward,
            "orientation": orientation_reward,
            "smoothness": smoothness_reward,
            "dof_limits": dof_limits_penalty
        }
    
        return total_eward, reward_components
    
    # Success criteria as forward velocity
    def compute_success(self):
        target_velocity = 2.0
        lin_vel_error = torch.square(target_velocity - self.env.root_states[:, 7])
        return torch.exp(-lin_vel_error / 0.25)

