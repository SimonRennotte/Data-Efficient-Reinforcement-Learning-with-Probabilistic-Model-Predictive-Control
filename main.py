import time
import os

import numpy as np
import gym
import json

from control_objects.probabilistic_gp_mpc_controller import ProbabiliticGpMpcController


def main():
	with open("parameters.json") as json_data_file:
		params_json_general = json.load(json_data_file)
	params_general = params_json_general['general_parameters']
	env_to_control = params_general['env_to_control']
	json_file_params = "params_" + env_to_control + ".json"
	if not(os.path.exists(json_file_params)):
		raise FileNotFoundError('The parameter file ' + json_file_params + ' does not exist. Create it.')
	with open(json_file_params) as json_data_file:
		params_json_env = json.load(json_data_file)
	hyperparameters_init = params_json_env['hyperparameters_init']
	params_constraints = params_json_env['params_constraints']
	params_controller = params_json_env['params_controller']
	params_train = params_json_env['params_train']
	params_actions_optimizer = params_json_env['params_actions_optimizer']
	params_init = params_json_env['params_init']
	params_memory = params_json_env['params_memory']
	env = gym.make(env_to_control)
	env.reset()
	target = np.array(params_controller['target'])
	weights_target = np.diag(params_controller['weights_target'])
	weights_target_terminal_cost = np.diag(params_controller['weights_target_terminal_cost'])
	s_observation = np.diag(params_controller['s_observation'])
	control_object = ProbabiliticGpMpcController(env.observation_space, env.action_space, params_controller,
									params_train, params_actions_optimizer, hyperparameters_init, target,
									weights_target, weights_target_terminal_cost, params_constraints, env_to_control)
	observation, reward, done, info = \
		env.step(env.action_space.low + (env.action_space.high - env.action_space.low) * np.random.uniform(0, 1))

	for idx_random_action in range(params_init['num_random_actions_init']):
		control_object.action = np.random.uniform(0, 1)
		action = env.action_space.low + (env.action_space.high - env.action_space.low) * control_object.action
		new_observation, reward, done, info = env.step(action)
		if params_general['render_env']:
			try:
				env.render()
			except:
				pass
		control_object.add_point_memory(observation, action, new_observation, reward)
		observation = new_observation

	while True:
		time_start = time.time()

		action, add_info_dict = control_object.compute_prediction_action(observation, s_observation)
		new_observation, reward, done, info = env.step(action)
		control_object.add_point_memory(observation, action, new_observation, reward,
										add_info_dict=add_info_dict, params_memory=params_memory)
		if params_general['verbose']:
			for key in add_info_dict:
				print(key + ': ' + str(add_info_dict[key]))
		observation = new_observation

		if params_general['render_env']:
			try:
				env.render()
			except:
				pass

		if params_general['save_plot_history'] and \
				(control_object.num_points_memory % params_general["frequency_iter_save"] == 0):
			control_object.save_plot_history()

		if params_general['save_plot_model_3d'] and \
				(control_object.num_points_memory % params_general["frequency_iter_save"] == 0):
			control_object.save_plot_model_3d()

		# TODO: Allow dynamic visualizations
		if params_general['render_live_plot_model_3d']:
			control_object.render_plot_model3d()

		if params_general['render_live_plot_history']:
			control_object.render_plot_model3d()

		print("time loop: " + str(time.time() - time_start) + " s\n")


if __name__ == '__main__':
	main()
