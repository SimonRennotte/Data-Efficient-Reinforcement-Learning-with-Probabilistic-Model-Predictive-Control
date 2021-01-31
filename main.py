import time
import os
import datetime

import numpy as np
import matplotlib.pyplot as plt
import gym
import json
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from control_objects.probabilistic_gp_mpc_controller import ProbabiliticGpMpcController


def main():
	with open(os.path.join("params", "global_parameters.json")) as json_data_file:
		params_json_general = json.load(json_data_file)
	params_general = params_json_general['general_parameters']
	env_to_control = params_general['env_to_control']  # Pendulum-v0 or MountainCarContinuous-v0
	json_file_params = os.path.join("params", "params_" + env_to_control + ".json")
	if not (os.path.exists(json_file_params)):
		raise FileNotFoundError('The parameter file ' + json_file_params + ' does not exist. You must first create it.')
	with open(json_file_params) as json_data_file:
		params_json_env = json.load(json_data_file)
	hyperparameters_init = params_json_env['hyperparameters_init']
	params_constraints_hyperparams = params_json_env['params_constraints_hyperparams']
	params_constraints_states = params_json_env['params_constraints_states']
	params_controller = params_json_env['params_controller']
	params_train = params_json_env['params_train']
	params_actions_optimizer = params_json_env['params_actions_optimizer']
	params_init = params_json_env['params_init']
	params_memory = params_json_env['params_memory']

	num_steps = params_json_env['num_steps_env']
	num_random_actions = params_init['num_random_actions_init']
	num_repeat_actions = params_controller["num_repeat_actions"]
	num_tests = params_general['number_tests_to_run']

	losses_tests = np.ones((num_tests, num_steps // num_repeat_actions - num_random_actions))
	for test_idx in range(num_tests):
		env = gym.make(env_to_control)

		datetime_now = datetime.datetime.now()
		folder_save = os.path.join('folder_save', env_to_control, 'y' + str(datetime_now.year) \
																  + '_mon' + str(datetime_now.month) + '_d' + str(
			datetime_now.day) + '_h' + str(datetime_now.hour) \
																  + '_min' + str(datetime_now.minute) + '_s' + str(
			datetime_now.second))
		if not os.path.exists(folder_save):
			os.makedirs(folder_save)

		if params_general['save_render_env']:
			if not os.path.exists(os.path.join('folder_save', env_to_control)):
				os.makedirs(os.path.join('folder_save', env_to_control))
			rec = VideoRecorder(env, path=os.path.join(folder_save, 'anim' + env_to_control + '.mp4'))

		env.reset()

		target = np.array(params_controller['target'])
		weights_target = np.diag(params_controller['weights_target'])
		weights_target_terminal_cost = np.diag(params_controller['weights_target_terminal_cost'])
		s_observation = np.diag(params_controller['s_observation'])

		control_object = ProbabiliticGpMpcController(env.observation_space, env.action_space, params_controller,
			params_train, params_actions_optimizer, params_constraints_states,
			hyperparameters_init, target, weights_target, weights_target_terminal_cost, params_constraints_hyperparams,
			env_to_control,
			folder_save, num_repeat_actions)

		action = env.action_space.sample()
		# low + (env.action_space.high - env.action_space.low) * np.random.uniform(0, 1)
		observation, reward, done, info = env.step(action)
		for idx_random_action in range(num_random_actions):
			control_object.action = env.action_space.sample()
			action = env.action_space.sample()
			for idx_action in range(num_repeat_actions):
				new_observation, reward, done, info = env.step(action)
				if params_general['render_env']:
					try:
						env.render()
					except:
						pass
				if params_general['save_render_env']:
					try:
						rec.capture_frame()
					except:
						pass
			control_object.add_point_memory(observation, action, new_observation, reward)
			observation = new_observation

		for index_iter in range((num_steps // num_repeat_actions) - params_init['num_random_actions_init']):
			time_start = time.time()
			action, add_info_dict = control_object.compute_prediction_action(observation, s_observation)
			for idx_action in range(num_repeat_actions):
				new_observation, reward, done, info = env.step(action)
				if params_general['render_env']:
					try:
						env.render()
					except:
						pass
				if params_general['save_render_env']:
					try:
						rec.capture_frame()
					except:
						pass
			losses_tests[test_idx, index_iter] = add_info_dict['cost']
			control_object.add_point_memory(observation, action, new_observation, reward,
											add_info_dict=add_info_dict, params_memory=params_memory)
			if params_general['verbose']:
				for key in add_info_dict:
					print(key + ': ' + str(add_info_dict[key]))
			observation = new_observation

			if params_general['save_plot_history'] and \
					(control_object.num_points_memory % params_general["frequency_iter_save"] == 0):
				control_object.save_plot_history()

			if params_general['save_plot_model_3d'] and \
					(control_object.num_points_memory % params_general["frequency_iter_save"] == 0):
				control_object.save_plot_model_3d()

			# TODO: Allow dynamic visualizations
			if params_general['render_live_plot_model_3d']:
				pass  # control_object.render_plot_model3d()

			if params_general['render_live_plot_history']:
				pass  # control_object.render_plot_model3d()

			print("time loop: " + str(time.time() - time_start) + " s\n")
		env.__exit__()
		if params_general['save_render_env']:
			rec.close()

		# save plots at the end
		control_object.check_and_close_processes()
		control_object.save_plot_model_3d()
		control_object.save_plot_history()
		# wait for the processes to be finished
		control_object.p_save_plot_model_3d.join()
		control_object.p_save_plot_model_3d.close()
		control_object.p_save_plot_history.join()
		control_object.p_save_plot_history.close()

	if num_tests > 1:
		mean_cost_runs = losses_tests.mean(0)
		std_cost_runs = losses_tests.std(0)
		plt.figure()
		indexes = (np.arange(losses_tests.shape[1]) + num_random_actions) * num_repeat_actions
		plt.plot(indexes, mean_cost_runs, label='mean cost run')
		plt.fill_between(indexes, mean_cost_runs - 2 * std_cost_runs, mean_cost_runs + 2 * std_cost_runs,
			label='cost runs 2 std', alpha=0.6)
		plt.title('Cost of the different runs for env ' + env_to_control)
		plt.ylabel('Cost')
		plt.xlabel('Number of environment steps')
		plt.savefig(os.path.join('folder_save', env_to_control, 'Cost_runs_' + env_to_control))
		plt.show()


if __name__ == '__main__':
	main()
