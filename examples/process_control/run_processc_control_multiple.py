import sys
import os
file_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(file_path), '../../'))

from rl_gp_mpc.envs.process_control import ProcessControl

from rl_gp_mpc.config_classes.visu_config import VisuConfig
from config_process_control import get_config

from rl_gp_mpc.run_env_function import run_env_multiple

if __name__ == '__main__':
    num_runs = 10

    num_steps = 1000
    random_actions_init = 10
    num_repeat_actions = 10
    include_time_model = True
    len_horizon = 5
    verbose = True

    env = ProcessControl(
        dt=1, 
        s_range=(20, 30), 
        fi_range=(0.2, 0.3), ci_range=(0.1, 0.2), 
        cr_range=(0.9, 1.0), 
        noise_l_prop_range=(3e-3, 1e-2), noise_co_prop_range=(3e-3, 1e-2), 
        sp_l_range=(0.4, 0.6), sp_co_range=(0.4, 0.6), 
        change_params=True, period_change=500
    )
    env_name = 'process_control'
    control_config = get_config(len_horizon=len_horizon, include_time_model=include_time_model, num_repeat_actions=num_repeat_actions)
    
    visu_config = VisuConfig(render_live_plot_2d=False, render_env=True, save_render_env=True)

    run_env_multiple(env, env_name, control_config, visu_config, num_runs, random_actions_init=random_actions_init, num_steps=num_steps, verbose=verbose)
