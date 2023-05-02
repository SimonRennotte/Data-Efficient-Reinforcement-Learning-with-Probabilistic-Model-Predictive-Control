import sys
import os
file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(file_path), '../../'))

import gym

from rl_gp_mpc.config_classes.visu_config import VisuConfig
from config_pendulum import get_config

from rl_gp_mpc.run_env_function import run_env_multiple

if __name__ == '__main__':
    num_runs = 20

    num_steps = 300
    random_actions_init = 10
    num_repeat_actions = 1
    include_time_model = False
    len_horizon = 15
    verbose = True

    env_name = 'Pendulum-v0'
    env = gym.make(env_name)
    control_config = get_config(len_horizon=len_horizon, include_time_model=include_time_model)
    
    visu_config = VisuConfig()

    run_env_multiple(env, control_config, visu_config, num_runs, random_actions_init=random_actions_init, num_repeat_actions=num_repeat_actions, num_steps=num_steps, verbose=verbose)
