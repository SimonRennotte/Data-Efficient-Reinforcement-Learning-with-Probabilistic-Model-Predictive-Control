import sys
import os
file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(file_path), '../../'))

import gym

from rl_gp_mpc.config_classes.visu_config import VisuConfig
from rl_gp_mpc.run_env_function import run_env
from config_pendulum import get_config


def run_pendulum():
	# choose the configuration file to load the corresponding env
	# open(os.path.join('params', 'main_parameters_mountain_car.json'))
	# open(os.path.join('params', 'main_parameters_my_env.json'))
    num_steps = 150
    random_actions_init = 10
    num_repeat_actions = 1
    include_time_model = False
    len_horizon = 15
    verbose = True

    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    control_config = get_config(len_horizon=len_horizon, include_time_model=include_time_model)
    
    visu_config = VisuConfig()
    costs = run_env(env, control_config, visu_config, random_actions_init=random_actions_init, num_repeat_actions=num_repeat_actions, num_steps=num_steps, verbose=verbose) 
    return costs


if __name__ == '__main__':
	costs = run_pendulum()
