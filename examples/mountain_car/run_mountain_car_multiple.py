import sys
import os
file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(file_path), '../../'))

import gym

from rl_gp_mpc.config_classes.visu_config import VisuConfig
from config_mountaincar import get_config

from rl_gp_mpc.run_env_function import run_env_multiple

if __name__ == '__main__':
    num_runs = 20
    random_actions_init=10
    num_repeat_actions=5
    num_steps=500
    len_horizon = 12
    verbose = False

    env_name = 'MountainCarContinuous-v0'
    env = gym.make(env_name)
    control_config = get_config(len_horizon=len_horizon, num_repeat_actions=num_repeat_actions)
    
    visu_config = VisuConfig(render_live_plot_2d=False, render_env=True, save_render_env=True)

    run_env_multiple(env, env_name, control_config, visu_config, num_runs, random_actions_init=random_actions_init, num_steps=num_steps, verbose=verbose)
