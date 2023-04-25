import time

import torch

from rl_gp_mpc import GpMpcController
from rl_gp_mpc.utils.utils import close_run, init_visu_and_folders
from rl_gp_mpc.envs.process_control import ProcessControl

from configs.config_processcontrol import get_config

random_actions_init = 25
num_steps = 250
num_repeat_actions = 5
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
        s_range=(8, 10), 
        fi_range=(0.3, 0.4), ci_range=(0.0, 0.1), 
        cr_range=(0.9, 1.0), 
        noise_l_prop_range=(1e-4, 1e-3), noise_co_prop_range=(1e-4, 1e-3), 
        sp_l_range=(0.4, 0.6), sp_co_range=(0.4, 0.6), 
        change_params=False, period_change=250
    )
    config = get_config(num_repeat_actions=num_repeat_actions, 
                        len_horizon=5,
                        include_time_model=True)

    live_plot_obj, rec, folder_save = init_visu_and_folders(
                                    env=env, num_steps=num_steps, config=config, 
                                    render_live_plot_2d=True, 
                                    run_live_graph_parallel_process=True, 
                                    save_render_env=True)
    obs_lst = []
    actions_lst = []
    rewards_lst = []
    obs = env.reset()
    obs_prev_ctrl = None
    action = None
    cost = None

    ctrl_obj = GpMpcController(observation_space=env.observation_space, action_space=env.action_space, config=config)
    
    for idx_action in range(random_actions_init):
        if idx_action % num_repeat_actions == 0:
            if obs_prev_ctrl is not None and action is not None and cost is not None:
                ctrl_obj.add_memory(obs=obs_prev_ctrl, action=action, obs_new=obs,
                                    reward=-cost, check_storage=False)
            obs_prev_ctrl = obs
            action = env.action_space.sample()
        obs_new, reward, done, info = env.step(action)

        cost, cost_var = ctrl_obj.compute_cost_unnormalized(obs_new, action)
        obs_lst.append(obs)
        actions_lst.append(action)
        rewards_lst.append(-cost)

        obs = obs_new
        if live_plot_obj is not None:
            live_plot_obj.update(obs=obs, cost=cost, action=action)
        # Necessary to store the last action in case that the parameter limit_action_change is set to True
        ctrl_obj.action_previous_iter = torch.Tensor(action)


    iter_info = None
    for iter_ctrl in range(random_actions_init, num_steps):
        time_start = time.time()
        if iter_ctrl % num_repeat_actions == 0:
            if iter_info is not None:
                predicted_state = iter_info.predicted_states[0]
                predicted_state_std = iter_info.predicted_states_std[0]
                check_storage = True
            else:
                predicted_state = None
                predicted_state_std = None
                check_storage = False
            # If num_repeat_actions != 1, the gaussian process models predict that much step ahead,
            # For iteration k, the memory holds obs(k - step), action (k - step), obs(k), reward(k)
            # Add memory is put before compute action because it uses data from step before
            ctrl_obj.add_memory(obs=obs_prev_ctrl, action=action, obs_new=obs,
                                reward=-cost, check_storage=check_storage,
                                predicted_state=predicted_state,
                                predicted_state_std=predicted_state_std)
            action = ctrl_obj.compute_action(obs_mu=obs)
            iter_info = ctrl_obj.get_iter_info()

            if verbose:
                iter_info_dict = iter_info.__dict__
                for key in iter_info_dict:
                    print(key + ': ' + str(iter_info_dict[key]))
            obs_prev_ctrl = obs

        obs_new, reward, done, info = env.step(action)

        cost, cost_var = ctrl_obj.compute_cost_unnormalized(obs, action)
        obs_lst.append(obs)
        actions_lst.append(action)
        rewards_lst.append(-cost)

        if render_env:
            try: env.render()
            except: pass
        if rec is not None:
            try: rec.capture_frame()
            except: pass

        if save_plots_2d and \
                (iter_ctrl % freq_iter_save_plots == 0):
            ctrl_obj.save_plots_2d(states=obs_lst, actions=actions_lst, rewards=rewards_lst,
                                    random_actions_init=random_actions_init)

        if save_plots_model_3d and \
                (iter_ctrl % freq_iter_save_plots == 0):
            ctrl_obj.save_plots_model_3d()

        if live_plot_obj is not None:
            live_plot_obj.update(obs=obs, cost=cost, action=action, iter_info=iter_info)

        obs = obs_new
        print('time loop: ' + str(time.time() - time_start) + ' s\n')

    close_run(ctrl_obj=ctrl_obj, env=env, obs_lst=obs_lst, actions_lst=actions_lst,
                    rewards_lst=rewards_lst, random_actions_init=random_actions_init,
                    live_plot_obj=live_plot_obj, rec=rec, save_plots_2d=save_plots_2d,
                    save_plots_3d=save_plots_model_3d)


if __name__ == '__main__':
	main()
