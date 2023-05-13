import sys
import os
file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(file_path), '../../'))

from rl_gp_mpc.config_classes.visu_config import VisuConfig
from rl_gp_mpc.envs.process_control import ProcessControl
from rl_gp_mpc.run_env_function import run_env
from config_process_control import get_config


def run_process_control():
	# choose the configuration file to load the corresponding env
	# open(os.path.join('params', 'main_parameters_mountain_car.json'))
	# open(os.path.join('params', 'main_parameters_my_env.json'))
    num_steps = 500
    random_actions_init = 100
    num_repeat_actions = 5
    include_time_model = False
    len_horizon = 5
    verbose = True

    env = ProcessControl(
        dt=1, 
        s_range=(20, 30), 
        fi_range=(0.15, 0.3), ci_range=(0.15, 0.2), 
        cr_range=(0.8, 1.0), 
        noise_l_prop_range=(5e-3, 1e-2), noise_co_prop_range=(5e-3, 1e-2), 
        sp_l_range=(0.4, 0.6), sp_co_range=(0.4, 0.6), 
        change_params=False, period_change=200
    )
    control_config = get_config(len_horizon=len_horizon, include_time_model=include_time_model, num_repeat_actions=num_repeat_actions)
    
    visu_config = VisuConfig(render_live_plot_2d=True, render_env=False, save_render_env=False, save_live_plot_2d=True)
    costs = run_env(env, control_config, visu_config, random_actions_init=random_actions_init, num_steps=num_steps, verbose=verbose) 
    return costs

if __name__ == '__main__':
	costs = run_process_control()
