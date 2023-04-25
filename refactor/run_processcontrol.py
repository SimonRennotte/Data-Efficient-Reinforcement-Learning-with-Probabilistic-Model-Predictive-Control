import time

import torch

from rl_gp_mpc import GpMpcController
from rl_gp_mpc import ControlVisualizations
from rl_gp_mpc.config_classes.visu_config import VisuConfig
from rl_gp_mpc.envs.process_control import ProcessControl

from configs.config_processcontrol import get_config

random_actions_init = 25
num_steps = 2500
num_repeat_actions = 25
verbose = True
render_env = True
save_plots_2d = False
save_plots_model_3d = False
freq_iter_save_plots = 25

def main():
	# choose the configuration file to load the corresponding env
	# open(os.path.join('params', 'main_parameters_mountain_car.json'))
	# open(os.path.join('params', 'main_parameters_my_env.json'))
    env = ProcessControl(
        dt=1, 
        s_range=(8, 15), 
        fi_range=(0.25, 0.45), ci_range=(0.0, 0.2), 
        cr_range=(0.8, 1.0), 
        noise_l_prop_range=(3e-3, 3e-2), noise_co_prop_range=(3e-3, 3e-2), 
        sp_l_range=(0.4, 0.6), sp_co_range=(0.4, 0.6), 
        change_params=True, period_change=250
    )
    config = get_config(num_repeat_actions=num_repeat_actions, 
                        len_horizon=3,
                        include_time_model=True)

    ctrl_obj = GpMpcController(observation_space=env.observation_space, action_space=env.action_space, config=config)

    control_config = get_config(num_repeat_actions=num_repeat_actions, len_horizon=5)
    visu_config = VisuConfig()

    visu_obj = ControlVisualizations(env=env, num_steps=num_steps, control_config=control_config, visu_config=visu_config)
    ctrl_obj = GpMpcController(observation_space=env.observation_space, action_space=env.action_space, config=control_config)

    obs = env.reset()
    
    for idx_init in range(random_actions_init // num_repeat_actions):
        action = env.action_space.sample()
        for idx_action_repeat in range(num_repeat_actions):
            obs_new, reward, done, info = env.step(action)
            cost, cost_var = ctrl_obj.compute_cost_unnormalized(obs_new, action)
            visu_obj.update(obs=obs_new, cost=cost, action=action)
            
        ctrl_obj.add_memory(obs=obs, action=action, obs_new=obs_new,
                            reward=-cost, check_storage=False)
        obs = obs_new
        # Necessary to store the last action in case that the parameter limit_action_change is set to True
        ctrl_obj.action_previous_iter = torch.Tensor(action)

    for idx_ctrl in range(random_actions_init // num_repeat_actions, num_steps//num_repeat_actions):
        time_start = time.time()
        action = ctrl_obj.compute_action(obs_mu=obs)
        iter_info = ctrl_obj.get_iter_info()

        for idx_action_repeat in range(num_repeat_actions):
            obs_new, reward, done, info = env.step(action)
            cost, cost_var = ctrl_obj.compute_cost_unnormalized(obs_new, action)
            visu_obj.update(obs=obs_new, cost=cost, action=action, iter_info=iter_info)

        ctrl_obj.add_memory(obs=obs, action=action, obs_new=obs_new,
                            reward=-cost, check_storage=True,
                            predicted_state=iter_info.predicted_states[0],
                            predicted_state_std=iter_info.predicted_states_std[0])
        if verbose:
            iter_info_dict = iter_info.__dict__
            for key in iter_info_dict:
                print(key + ': ' + str(iter_info_dict[key]))

        obs = obs_new
        print('time loop: ' + str(time.time() - time_start) + ' s\n')

    #visu_obj.save(random_actions_init=random_actions_init)
    env.__exit__()
    ctrl_obj.check_and_close_processes()
    visu_obj.close() 


if __name__ == '__main__':
	main()
