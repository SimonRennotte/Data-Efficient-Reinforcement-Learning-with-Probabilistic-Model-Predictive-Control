import time
import os

import numpy as np
import gym
import json

from control_objects.gp_mpc_controller import GpMpcController
from utils.utils import init_control, init_visu_and_folders, plot_costs, close_run
from utils.env_example import CustomEnv


def main():
	# choose the configuration file to load the corresponding env
	# open(os.path.join('params', 'main_parameters_mountain_car.json'))
	# open(os.path.join('params', 'main_parameters_my_env.json'))
	with open(os.path.join('params', 'main_parameters_pendulum.json')) as json_data_file:
		params_general = json.load(json_data_file)

	params_control_path = os.path.join('params', 'params_' + params_general['env_to_control'] + '.json')

	if not (os.path.exists(params_control_path)):
		raise FileNotFoundError('The parameter file ' + params_control_path + ' does not exist. You must first create it.')

	with open(params_control_path) as json_data_file:
		params_controller_dict = json.load(json_data_file)

	num_steps = params_general['num_steps_env']
	num_tests = params_general['number_tests_to_run']
	num_repeat_actions = params_controller_dict['controller']['num_repeat_actions']
	random_actions_init = params_general['random_actions_init']

	costs_runs = np.ones((num_tests, num_steps))
	for idx_test in range(num_tests):
		if params_general['env_to_control'] == 'my_env':
			env = CustomEnv()
			env_str = 'MyEnv'
		else:
			try:
				env = gym.make(params_general['env_to_control'])
			except:
				raise ValueError('Could not find env ' + params_general['env_to_control'] +
								 '. Check the name and try again.')
			env_str = env.env.spec.entry_point.replace('-', '_').replace(':', '_').replace('.', '_')

		live_plot_obj, rec, folder_save = init_visu_and_folders(env=env, num_steps=num_steps, env_str=env_str,
										params_general=params_general, params_controller_dict=params_controller_dict)

		ctrl_obj = GpMpcController(observation_space=env.observation_space, action_space=env.action_space,
										params_dict=params_controller_dict, folder_save=folder_save)

		ctrl_obj, env, live_plot_obj, rec, obs, action, cost, \
			obs_prev_ctrl, costs_runs, obs_lst, actions_lst, rewards_lst = init_control(
					ctrl_obj=ctrl_obj, env=env, live_plot_obj=live_plot_obj, rec=rec,
					params_general=params_general, random_actions_init=random_actions_init,
					costs_tests=costs_runs, idx_test=idx_test, num_repeat_actions=num_repeat_actions)

		info_dict = None
		for iter_ctrl in range(random_actions_init, num_steps):
			time_start = time.time()
			if iter_ctrl % num_repeat_actions == 0:
				if info_dict is not None:
					predicted_state = info_dict['predicted states'][0]
					predicted_state_std = info_dict['predicted states std'][0]
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
				action, info_dict = ctrl_obj.compute_action(obs_mu=obs)

				if params_general['verbose']:
					for key in info_dict:
						print(key + ': ' + str(info_dict[key]))
				obs_prev_ctrl = obs

			obs_new, reward, done, info = env.step(action)

			cost, cost_var = ctrl_obj.compute_cost_unnormalized(obs, action)
			costs_runs[idx_test, iter_ctrl] = cost
			obs_lst.append(obs)
			actions_lst.append(action)
			rewards_lst.append(-cost)

			if params_general['render_env']:
				try: env.render()
				except: pass
			if rec is not None:
				try: rec.capture_frame()
				except: pass

			if params_general['save_plots_2d'] and \
					(iter_ctrl % params_general['freq_iter_save_plots'] == 0):
				ctrl_obj.save_plots_2d(states=obs_lst, actions=actions_lst, rewards=rewards_lst,
										random_actions_init=random_actions_init)

			if params_general['save_plots_model_3d'] and \
					(iter_ctrl % params_general['freq_iter_save_plots'] == 0):
				ctrl_obj.save_plots_model_3d()

			if live_plot_obj is not None:
				live_plot_obj.update(obs=obs, cost=cost, action=action, info_dict=info_dict)

			obs = obs_new
			print('time loop: ' + str(time.time() - time_start) + ' s\n')

		close_run(ctrl_obj=ctrl_obj, env=env, obs_lst=obs_lst, actions_lst=actions_lst,
					rewards_lst=rewards_lst, random_actions_init=random_actions_init,
					live_plot_obj=live_plot_obj, rec=rec, save_plots_2d=params_general['save_plots_2d'],
					save_plots_3d=params_general['save_plots_model_3d'])

	if num_tests > 1:
		plot_costs(costs=costs_runs, env_str=env_str)


if __name__ == '__main__':
	main()
