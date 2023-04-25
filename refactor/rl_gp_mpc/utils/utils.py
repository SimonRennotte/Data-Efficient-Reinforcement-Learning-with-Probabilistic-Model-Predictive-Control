import datetime
import os
import time
import threading
import queue
import multiprocessing

from gym.wrappers.monitoring.video_recorder import VideoRecorder
import torch
import gpytorch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

from rl_gp_mpc.control_objects.utils.gp_models import ExactGPModelMonoTask
from rl_gp_mpc.control_objects.utils.iteration_info_class import IterationInformation
from rl_gp_mpc.config_classes.config import Config

matplotlib.rc('font', size='6')
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
ALPHA_CONFIDENCE_BOUNDS = 0.3
FONTSIZE = 6

# Values used for graph visualizations only
MEAN_PRED_COST_INIT = 1
STD_MEAN_PRED_COST_INIT = 10


def get_init_action_change(len_horizon, max_change_action_norm):
	return np.dot(np.expand_dims(np.random.uniform(low=-1, high=1, size=(len_horizon)), 1), np.expand_dims(max_change_action_norm, 0))


def get_init_action(len_horizon, num_actions):
	return np.random.uniform(low=0, high=1, size=(len_horizon, num_actions))


def create_models(train_inputs, train_targets, params, constraints_gp, num_models=None, num_inputs=None):
	"""
	Define gaussian process models used for predicting state transition,
	using constraints and init values for (outputscale, noise, lengthscale).

	Args:
		train_inputs (torch.Tensor or None): Input values in the memory of the gps
		train_targets (torch.Tensor or None): target values in the memory of the gps.
												Represent the change in state values
		params (dict or list of dict): Value of the hyper-parameters of the gaussian processes.
										Dict type is used to create the models with init values from the json file.
										List of dict type is used to create the models
													with exported parameters, such as in the parallel training process.
		constraints_gp (dict): See the ReadMe about parameters for information about keys
		num_models (int or None): Must be provided when train_inputs or train_targets are None.
									The number of models should be equal to the dimension of state,
									so that the transition for each state can be predicted with a different gp.
									Default=None
		num_inputs (int or None): Must be provided when train_inputs or train_targets are None.
									The number of inputs should be equal to the sum of the dimension of state
									and dimension of action. Default=None
		include_time (bool): If True, gp will have one additional input corresponding to the time of the observation.
								This is usefull if the env change with time,
								as more recent points will be trusted more than past points
								(time closer to the point to make inference at).
								It is to be specified only if

	Returns:
		models (list of gpytorch.models.ExactGP): models containing the parameters, memory,
													constraints of the gps and functions for exact predictions
	"""
	if train_inputs is not None and train_targets is not None:
		num_models = len(train_targets[0])
		models = [ExactGPModelMonoTask(train_inputs, train_targets[:, idx_model], len(train_inputs[0]))
			for idx_model in range(num_models)]
	else:
		if num_models is None or num_inputs is None:
			raise(ValueError('If train_inputs or train_targets are None, num_models and num_inputs must be defined'))
		else:
			models = [ExactGPModelMonoTask(None, None, num_inputs) for _ in range(num_models)]

	for idx_model in range(num_models):
		if constraints_gp is not None:
			if "min_std_noise" in constraints_gp.keys():
				if type(constraints_gp['min_std_noise']) != float and \
						type(constraints_gp['min_std_noise']) != int:
					min_var_noise = np.power(constraints_gp['min_std_noise'][idx_model], 2)
				else:
					min_var_noise = np.power(constraints_gp['min_std_noise'], 2)
				if type(constraints_gp['max_std_noise']) != float and \
						type(constraints_gp['max_std_noise']) != int:
					max_var_noise = np.power(constraints_gp['max_std_noise'][idx_model], 2)
				else:
					max_var_noise = np.power(constraints_gp['max_std_noise'], 2)
				models[idx_model].likelihood.noise_covar.register_constraint("raw_noise",
					gpytorch.constraints.Interval(lower_bound=min_var_noise, upper_bound=max_var_noise))

			if "min_outputscale" in constraints_gp.keys():
				if type(constraints_gp['min_outputscale']) != float and \
						type(constraints_gp['min_outputscale']) != int:
					min_outputscale = constraints_gp['min_outputscale'][idx_model]
				else:
					min_outputscale = constraints_gp['min_outputscale']

				if type(constraints_gp['max_outputscale']) != float and \
						type(constraints_gp['max_outputscale']) != int:
					max_outputscale = constraints_gp['max_outputscale'][idx_model]
				else:
					max_outputscale = constraints_gp['max_outputscale']
				models[idx_model].covar_module.register_constraint("raw_outputscale",
					gpytorch.constraints.Interval(lower_bound=min_outputscale, upper_bound=max_outputscale))

			if "min_lengthscale" in constraints_gp.keys():
				if type(constraints_gp['min_lengthscale']) == float or type(constraints_gp['min_lengthscale']) == int:
					min_lengthscale = constraints_gp['min_lengthscale']
				else:
					min_lengthscale = constraints_gp['min_lengthscale'][idx_model]
				if type(constraints_gp['min_lengthscale']) == float or type(constraints_gp['min_lengthscale']) == int:
					max_lengthscale = constraints_gp['max_lengthscale']
				else:
					max_lengthscale = constraints_gp['max_lengthscale'][idx_model]
				models[idx_model].covar_module.base_kernel.register_constraint("raw_lengthscale",
					gpytorch.constraints.Interval(lower_bound=min_lengthscale, upper_bound=max_lengthscale))
		# load parameters
		# dict type is used when initializing the models from the json config file
		# list type is used when initializing the models in the parallel training process
		# using the exported parameters
		if type(params) == dict:
			hypers = {'base_kernel.lengthscale': params['base_kernel.lengthscale'][idx_model],
				'outputscale': params['outputscale'][idx_model]}
			hypers_likelihood = {'noise_covar.noise': params['noise_covar.noise'][idx_model]}
			models[idx_model].likelihood.initialize(**hypers_likelihood)
			models[idx_model].covar_module.initialize(**hypers)
		elif type(params) == list:
			models[idx_model].load_state_dict(params[idx_model])
	return models


def init_control(ctrl_obj, env, live_plot_obj, rec, params_general, random_actions_init,
					costs_tests, idx_test, num_repeat_actions=1):
	"""
	Init the gym env and memory. Define the lists containing the points for visualisations.
	Control the env with random actions for a number of steps.
	Args:
		ctrl_obj (control_objects.gp_mpc_controller.GpMpcController):
																Object containing the control functions.
																Used here to compute cost and add_memory.
		env (gym env): environment used to get the next observation given current observation and action
		live_plot_obj (object): object used for visualisation in real time of the control in a 2d graph.
								Must contain a function called update, that adds new points on the graph
		rec (object): object used to visualise the gym env in real-time.
						Must heve a function called capture_frame that updates the visualisation
		params_general (dict): see parameters.md for more information
		random_actions_init (int): number of random actions to apply for the initialization.
		costs_tests (numpy.array): contains the costs of all the tests realized, which is used to compute
									the mean cost over a large number of control. Parameter not relevant if num_tests=1.
									Dim=(num_runs, num_timesteps)
		idx_test (int): index of the test realized
		num_repeat_actions (int): number of consecutive actions that are constant.
									The GPs only store points at which the control change.
	"""
	# Lists used for plotting values
	obs_lst = []
	actions_lst = []
	rewards_lst = []
	env.reset()
	obs, reward, done, info = env.step(env.action_space.sample())
	obs_prev_ctrl = None
	action = None
	cost = None
	for idx_action in range(random_actions_init):
		if idx_action % num_repeat_actions == 0:
			if obs_prev_ctrl is not None and action is not None and cost is not None:
				ctrl_obj.add_memory(obs=obs_prev_ctrl, action=action, obs_new=obs,
									reward=-cost, check_storage=False)
			obs_prev_ctrl = obs
			action = env.action_space.sample()
		obs_new, reward, done, info = env.step(action)
		if params_general['render_env']:
			try: env.render()
			except: pass
		if params_general['save_render_env']:
			try: rec.capture_frame()
			except: pass

		cost, cost_var = ctrl_obj.compute_cost_unnormalized(obs_new, action)
		costs_tests[idx_test, idx_action] = cost
		obs_lst.append(obs)
		actions_lst.append(action)
		rewards_lst.append(-cost)

		if params_general['render_live_plots_2d']:
			live_plot_obj.update(obs=obs, action=action, cost=cost, info_dict=None)

		obs = obs_new
		# Necessary to store the last action in case that the parameter limit_action_change is set to True
		ctrl_obj.action_previous_iter = torch.Tensor(action)

	return ctrl_obj, env, live_plot_obj, rec, obs, action, cost, obs_prev_ctrl, \
			costs_tests, obs_lst, actions_lst, rewards_lst


def init_visu_and_folders(env, num_steps, config:Config, render_live_plot_2d=True, run_live_graph_parallel_process=True, save_render_env=True):
	"""
	Create and return the objects used for visualisation in real-time.
	Also create the folder where to save the figures.
	Args:
		env (gym env): gym environment
		num_steps (int): number of steps during which the action is maintained constant
		env_str (str): string that describes the env name
		params_general (dict): general parameters (see parameters.md for more info)
		params_controller_dict (dict): controller parameters (see parameters.md for more info)

	Returns:
		live_plot_obj (object): object used to visualize the control and observation evolution in a 2d graph

		rec (object): object used to visualize the gym env

		folder_save (str): name of the folder where the figures will be saved
	"""
	if render_live_plot_2d:
		if run_live_graph_parallel_process:
			live_plot_obj = LivePlotParallel(num_steps,
				env.observation_space, env.action_space,
				step_pred=config.controller.num_repeat_actions,
				use_constraints=bool(config.reward.use_constraints),
				state_min=config.reward.state_min,
				state_max=config.reward.state_max)
		else:
			live_plot_obj = LivePlotSequential(num_steps,
				env.observation_space, env.action_space,
				step_pred=config.controller.num_repeat_actions,
				use_constraints=bool(config.reward.use_constraints),
				state_min=config.reward.state_min,
				state_max=config.reward.state_max)
	else:
		live_plot_obj = None

	datetime_now = datetime.datetime.now()
	try:
		env_str = env.env.spec.entry_point.replace('-', '_').replace(':', '_').replace('.', '_')
	except:
		env_str = env.name
	folder_save = os.path.join('folder_save', env_str, datetime_now.strftime("%Y_%m_%d_%H_%M_%S"))
	if not os.path.exists(folder_save):
		os.makedirs(folder_save)

	if save_render_env:
		try:
			rec = VideoRecorder(env, path=os.path.join(folder_save, 'anim' + env_str + '.mp4'))
		except:
			pass
	else:
		rec = None
	return live_plot_obj, rec, folder_save


def close_run(ctrl_obj, env, obs_lst, actions_lst, rewards_lst, random_actions_init,
					live_plot_obj=None, rec=None, save_plots_3d=False, save_plots_2d=False):
	"""
	Close all visualisations and parallel processes that are still running.
	Save all visualisations one last time if save args set to True
	Args:
		ctrl_obj:
		env (gym env): gym environment
		obs_lst (list of numpy array): Contains all the past observations
		actions_lst (list of numpy array): Contains all the past actions
		rewards_lst(list): Contains all the past rewards (or -cost)
		random_actions_init (int): number of initial actions where the controller was not used (random actions)
		live_plot_obj (object or None): object used for visualisation in real time of the control in a 2d graph.
								If None, not need to close it. Default=None
		rec (object or None): object used to visualise the gym env in real-time.
								If None, no need to close it. default=None
		save_plots_3d (bool): If set to True, will save the 3d plots. Default=False
		save_plots_2d (bool): If set to True, will save the 2d plots. Default=False
	"""

	env.__exit__()
	if rec is not None:
		rec.close()
	if live_plot_obj is not None:
		live_plot_obj.graph_p.terminate()
	# save plots at the end
	ctrl_obj.check_and_close_processes()
	if save_plots_3d:
		ctrl_obj.save_plots_model_3d()
		# wait for the process to be finished
		ctrl_obj.p_save_plots_model_3d.join()
		ctrl_obj.p_save_plots_model_3d.close()
	if save_plots_2d:
		ctrl_obj.save_plots_2d(states=obs_lst, actions=actions_lst, rewards=rewards_lst,
								random_actions_init=random_actions_init)
		# wait for the process to be finished
		ctrl_obj.p_save_plots_2d.join()
		ctrl_obj.p_save_plots_2d.close()
	plt.close()


def plot_costs(costs, env_str):
	"""
	Plot the mean and standard deviation of the costs of all the runs.
	Args:
		costs (numpy.array): contains all the costs (or -rewards) of all the runs. Dim=(num_runs, num_timesteps)
		env_str (str): name of the env, used to save the plot in the right folder
	"""
	mean_cost_runs = costs.mean(0)
	std_cost_runs = costs.std(0)
	plt.figure()
	indexes = (np.arange(costs.shape[1]))
	plt.plot(indexes, mean_cost_runs, label='mean cost run')
	plt.fill_between(indexes, mean_cost_runs - 2 * std_cost_runs, mean_cost_runs + 2 * std_cost_runs,
		label='cost runs 2 std', alpha=0.6)
	plt.grid()
	plt.title('Cost of the different runs for env ' + env_str)
	plt.ylabel('Cost')
	plt.xlabel('Number of environment steps')
	plt.savefig(os.path.join('../folder_save', env_str, 'Cost_runs_' + env_str))
	plt.show()


def draw_plots_2d(states, actions, costs, pred_info, num_repeat_actions=1, random_actions_init=0,
		state_min_constraint=None, state_max_constraint=None, iter_ahead_show=3, fig=None, axes=None,
		update_x_limit=True, mul_std_bounds=3):
	idxs_x = np.arange(0, len(states))
	pred_iters = np.array(pred_info['iteration'])
	idxs_ctrl = pred_iters + random_actions_init
	try:
		states_pred_show = np.zeros((len(idxs_ctrl) - iter_ahead_show, states.shape[1]))
		states_pred_show[:] = [pred[iter_ahead_show - 1].numpy() for pred in pred_info['predicted states']][:-iter_ahead_show]

		states_pred_std_show = np.ones((len(idxs_ctrl) - iter_ahead_show, states.shape[1]))
		states_pred_std_show[:] = [pred[iter_ahead_show - 1].numpy() for pred in pred_info['predicted states std']][:-iter_ahead_show]
	except ValueError:
		states_pred_show = None
		states_pred_std_show = None

	mean_cost_pred = np.array([element for element in pred_info['mean predicted cost']])
	mean_cost_std_pred = np.array([element for element in pred_info['mean predicted cost std']])
	np.nan_to_num(states_pred_show, copy=False, nan=-1, posinf=-1, neginf=-1)
	np.nan_to_num(states_pred_std_show, copy=False, nan=np.max(states_pred_std_show), posinf=99, neginf=0)
	np.nan_to_num(mean_cost_pred, copy=False, nan=-1, posinf=-1, neginf=-1)
	np.nan_to_num(states_pred_std_show, copy=False, nan=np.max(mean_cost_std_pred), posinf=99, neginf=0)
	for state_idx in range(len(states[0])):
		axes[0].plot(idxs_x, np.array(states)[:, state_idx], label='state' + str(state_idx),
			color=plt.get_cmap('tab20').colors[2 * state_idx])
		if states_pred_show is not None:
			axes[0].plot(idxs_ctrl[iter_ahead_show:], states_pred_show[:, state_idx],
				label=str(iter_ahead_show * num_repeat_actions) + 'step_' + 'prediction' + str(state_idx),
				color=plt.get_cmap('tab20').colors[2 * state_idx], linestyle='--')
		if states_pred_std_show is not None:
			axes[0].fill_between(idxs_ctrl[iter_ahead_show:],
				states_pred_show[:, state_idx] - mul_std_bounds * states_pred_std_show[:, state_idx],
				states_pred_show[:, state_idx] + mul_std_bounds * states_pred_std_show[:, state_idx],
				color=plt.get_cmap('tab20').colors[2 * state_idx], alpha=ALPHA_CONFIDENCE_BOUNDS)
		# label=str(num_iter_prediction_ahead_show * num_repeat_actions) + 'step_' + 'bounds_prediction' + str(
		# 				state_idx)
		if state_min_constraint is not None and state_max_constraint is not None:
			axes[0].axhline(y=state_min_constraint[state_idx],
				color=plt.get_cmap('tab20').colors[2 * state_idx], linestyle='-.')
			axes[0].axhline(y=state_max_constraint[state_idx],
				color=plt.get_cmap('tab20').colors[2 * state_idx], linestyle='-.')
	# costs = np.array([element for element in pred_info['cost']])
	axes[2].plot(idxs_x, costs, label='cost', color='k')

	axes[2].plot(list(idxs_ctrl), mean_cost_pred, label='mean predicted cost', color='orange')
	axes[2].fill_between(list(idxs_ctrl),
							mean_cost_pred - mul_std_bounds * mean_cost_std_pred,
							mean_cost_pred + mul_std_bounds * mean_cost_std_pred,
							color='orange', alpha=ALPHA_CONFIDENCE_BOUNDS)

	for idx_action in range(len(actions[0])):
		axes[1].step(idxs_x, np.array(actions)[:, idx_action], label='a_' + str(idx_action),
			color=plt.get_cmap('tab20').colors[1 + 2 * idx_action])

	if update_x_limit is True:
		axes[2].set_xlim(0, np.max(idxs_x))
	axes[2].set_ylim(0, np.max([np.max(mean_cost_pred), np.max(costs)]) * 1.2)
	plt.xlabel('Env steps')
	return fig, axes


def save_plot_2d(states, actions, costs, pred_info, folder_save,
		num_repeat_actions=1, random_actions_init=0,
		state_min_constraint=None, state_max_constraint=None,
		iter_ahead_show=3, mul_std_bounds=3):
	fig, axes = plt.subplots(nrows=3, figsize=(6, 5), sharex=True)
	axes[0].set_title('Normed states and predictions')
	axes[1].set_title('Normed actions')
	axes[2].set_title('Cost and horizon cost')
	plt.xlabel("time")
	max_state_plot = np.max([states.max(), 1.03])
	min_state_plot = np.min([states.min(), -0.03])
	axes[0].set_ylim(min_state_plot, max_state_plot)
	axes[1].set_ylim(0, 1.02)
	plt.tight_layout()
	fig, axes = draw_plots_2d(states=states, actions=actions, costs=costs,
							pred_info=pred_info, num_repeat_actions=num_repeat_actions,
							random_actions_init=random_actions_init,
							state_min_constraint=state_min_constraint,
							state_max_constraint=state_max_constraint,
							iter_ahead_show=iter_ahead_show, fig=fig, axes=axes,
							mul_std_bounds=mul_std_bounds)
	axes[0].legend()
	axes[0].grid()
	axes[1].legend()
	axes[1].grid()
	axes[2].legend()
	axes[2].grid()
	hour = datetime.datetime.now().hour
	minute = datetime.datetime.now().minute
	second = datetime.datetime.now().second
	fig.savefig(os.path.join(folder_save, 'history' + '_h' +
										  str(hour) + '_m' +
										  str(minute) + '_s' +
										  str(second) + '.png'))
	plt.close()


def anim_plots_2d_p(queue, num_steps_total, step_pred, obs_space, action_space,
					use_constraints=False, state_min=None, state_max=None, time_between_updates=0.5, mul_std_bounds=3):
	fig, axes = plt.subplots(nrows=3, figsize=(6, 5), sharex=True)
	axes[0].set_title('Normed states and predictions')
	axes[1].set_title('Normed actions')
	axes[2].set_title('Cost and horizon cost')
	plt.xlabel("Env steps")
	min_state = -0.03
	max_state = 1.03
	axes[0].set_ylim(min_state, max_state)
	axes[0].grid()
	axes[1].set_ylim(-0.03, 1.03)
	axes[1].grid()
	axes[2].set_xlim(0, num_steps_total)
	axes[2].grid()
	plt.tight_layout()

	states = np.empty((num_steps_total, obs_space.shape[0]))
	actions = np.empty((num_steps_total, action_space.shape[0]))
	costs = np.empty(num_steps_total)
	mean_cost_pred = np.empty_like(costs)
	mean_cost_std_pred = np.empty_like(mean_cost_pred)
	
	min_obs = obs_space.low
	max_obs = obs_space.high
	min_action = action_space.low
	max_action = action_space.high
	if use_constraints:
		if state_min is not None:
			for idx_state in range(obs_space.shape[0]):
				axes[0].axhline(y=state_min[idx_state],
					color=plt.get_cmap('tab20').colors[2 * idx_state],
					linestyle='-.')
		if state_max is not None:
			for idx_state in range(obs_space.shape[0]):
				axes[0].axhline(y=state_max[idx_state],
					color=plt.get_cmap('tab20').colors[2 * idx_state],
					linestyle='-.')
	lines_states = [axes[0].plot([], [], label='state' + str(state_idx),
		color=plt.get_cmap('tab20').colors[2 * state_idx]) for state_idx in range(obs_space.shape[0])]
	lines_actions = [axes[1].step([], [], label='action' + str(action_idx),
		color=plt.get_cmap('tab20').colors[1 + 2 * action_idx]) for action_idx in range(action_space.shape[0])]

	line_cost = axes[2].plot([], [], label='cost', color='k')
	line_costs_mean_pred = axes[2].plot([], [], label='mean predicted cost', color='orange')

	lines_states_pred = [axes[0].plot([], [],
		label='predicted_states' + str(state_idx),
		color=plt.get_cmap('tab20').colors[2 * state_idx], linestyle='dashed')
		for state_idx in range(obs_space.shape[0])]
	lines_actions_pred = [axes[1].step([], [], label='predicted_action' + str(action_idx),
		color=plt.get_cmap('tab20').colors[1 + 2 * action_idx], linestyle='dashed')
		for action_idx in range(action_space.shape[0])]
	line_costs_pred = axes[2].plot([], [], label='predicted cost', color='k', linestyle='dashed')

	axes[0].legend(fontsize=FONTSIZE)
	axes[1].legend(fontsize=FONTSIZE)
	axes[2].legend(fontsize=FONTSIZE)
	fig.canvas.draw()  # draw and show it
	plt.show(block=False)

	num_pts_show = 0
	last_update_time = time.time()
	while True:
		if (time.time() - last_update_time) > time_between_updates:
			recieved_data = False
			last_update_time = time.time()
			while not (queue.empty()):
				msg = queue.get()  # Read from the queue
				obs_norm = (np.array(msg[0]) - min_obs) / (max_obs - min_obs)
				action_norm = (np.array(msg[1]) - min_action) / (max_action - min_action)
				states[num_pts_show] = obs_norm
				actions[num_pts_show] = action_norm
				costs[num_pts_show] = msg[2]

				update_limits = False
				min_state_actual = np.min(states[num_pts_show])
				if min_state_actual < min_state:
					min_state = min_state_actual
					update_limits = True

				max_state_actual = np.max(states[num_pts_show])
				if max_state_actual > max_state:
					max_state = max_state_actual
					update_limits = True

				if update_limits:
					axes[0].set_ylim(min_state, max_state)

				if len(msg) > 3:
					mean_cost_pred[num_pts_show] = np.nan_to_num(msg[3], nan=-1, posinf=-1, neginf=-1)
					mean_cost_std_pred[num_pts_show] = np.nan_to_num(msg[4], copy=True, nan=99, posinf=99, neginf=99)
				else:
					if num_pts_show == 0:
						# Init values for mean cost pred and mean cost std pred at the beginning when the control
						# has not started yet and only random actions has been applied.
						# Thus, the mean future cost and the std of the mean future cost has not been computed yet
						# and have not been provided to the graph object for the update
						mean_cost_pred[num_pts_show] = MEAN_PRED_COST_INIT
						mean_cost_std_pred[num_pts_show] = STD_MEAN_PRED_COST_INIT
					else:
						# For when the information about prediction have not been provided to the graph object
						# for the update, for example at the init period when random actions are applied
						mean_cost_pred[num_pts_show] = mean_cost_pred[num_pts_show - 1]
						mean_cost_std_pred[num_pts_show] = mean_cost_std_pred[num_pts_show - 1]
				num_pts_show += 1
				recieved_data = True
				last_update_time = time.time()

			if recieved_data:
				for idx_axes in range(len(axes)):
					axes[idx_axes].collections.clear()

				idxs = np.arange(0, num_pts_show)
				for idx_axes in range(len(axes)):
					axes[idx_axes].collections.clear()

				for idx_state in range(len(obs_norm)):
					lines_states[idx_state][0].set_data(idxs, states[:num_pts_show, idx_state])

				for idx_action in range(len(action_norm)):
					lines_actions[idx_action][0].set_data(idxs, actions[:num_pts_show, idx_action])

				line_cost[0].set_data(idxs, costs[:num_pts_show])

				line_costs_mean_pred[0].set_data(idxs, mean_cost_pred[:num_pts_show])
				axes[2].fill_between(idxs,
					mean_cost_pred[:num_pts_show] -
					mean_cost_std_pred[:num_pts_show] * mul_std_bounds,
					mean_cost_pred[:num_pts_show] +
					mean_cost_std_pred[:num_pts_show] * mul_std_bounds,
					facecolor='orange', alpha=ALPHA_CONFIDENCE_BOUNDS,
					label='mean predicted ' + str(mul_std_bounds) + ' std cost bounds')
				
				axes[2].set_ylim(0, np.max([np.max(mean_cost_pred[:num_pts_show]),
					np.max(costs[:num_pts_show])]) * 1.1)
				axes[2].set_ylim(0, np.max(line_cost[0].get_ydata()) * 1.1)
				
				# If predictions are not given in the last update, do not show them on the graph
				if len(msg) > 3:
					states_pred = np.nan_to_num(msg[5], nan=-1, posinf=-1, neginf=-1)
					states_std_pred = np.nan_to_num(msg[6], copy=False, nan=99, posinf=99, neginf=99)
					actions_pred = msg[7]
					costs_pred = np.nan_to_num(msg[8], nan=-1, posinf=-1, neginf=-1)
					costs_std_pred = np.nan_to_num(msg[9], nan=99, posinf=99, neginf=0)
					# if num_repeat_action is not 1, the control do not happen at each iteration,
					# we must select the last index where the control happened as the start of the prediction horizon
					idx_prev_control = (idxs[-1] // step_pred) * step_pred
					idxs_future = np.arange(idx_prev_control,
												idx_prev_control + step_pred + len(states_pred) * step_pred,
												step_pred)

					for idx_state in range(len(obs_norm)):
						future_states_show = np.concatenate(([states[idx_prev_control, idx_state]],
															states_pred[:, idx_state]))
						lines_states_pred[idx_state][0].set_data(idxs_future, future_states_show)

						future_states_std_show = np.concatenate(([0], states_std_pred[:, idx_state]))
						axes[0].fill_between(idxs_future,
							future_states_show - future_states_std_show * mul_std_bounds,
							future_states_show + future_states_std_show * mul_std_bounds,
							facecolor=plt.get_cmap('tab20').colors[2 * idx_state], alpha=ALPHA_CONFIDENCE_BOUNDS,
							label='predicted ' + str(mul_std_bounds) + ' std bounds state ' + str(idx_state))

					for idx_action in range(len(action_norm)):
						future_actions_show = \
							np.concatenate(([actions[idx_prev_control, idx_action]], actions_pred[:, idx_action]))
						lines_actions_pred[idx_action][0].set_data(idxs_future, future_actions_show)
	
					future_costs_show = np.concatenate(([costs[idx_prev_control]], costs_pred))
					line_costs_pred[0].set_data(idxs_future, future_costs_show)
	
					future_cost_std_show = np.concatenate(([0], costs_std_pred))
					axes[2].fill_between(idxs_future,
						future_costs_show - future_cost_std_show * mul_std_bounds,
						future_costs_show + future_cost_std_show * mul_std_bounds,
						facecolor='black', alpha=ALPHA_CONFIDENCE_BOUNDS,
						label='predicted ' + str(mul_std_bounds) + ' std cost bounds')

				for ax in axes:
					ax.relim()
					ax.autoscale_view(True, True, True)

				fig.canvas.draw()
				plt.pause(0.01)


class LivePlotParallel:
	def __init__(self, num_steps_total, obs_space, action_space, step_pred, use_constraints=False,
			state_min=None, state_max=None, time_between_updates=0.75, use_thread=False):
		if use_thread:
			self.pqueue = queue.Queue()
			self.graph_p = threading.Thread(target=anim_plots_2d_p,
									args=(self.pqueue, num_steps_total, step_pred, obs_space,
										action_space, use_constraints,
										state_min, state_max, time_between_updates))
		else:
			self.pqueue = multiprocessing.Queue()
			self.graph_p = multiprocessing.Process(target=anim_plots_2d_p,
									args=(self.pqueue, num_steps_total, step_pred, obs_space,
											action_space, use_constraints,
											state_min, state_max, time_between_updates))
			# graph_p.daemon = True
		self.graph_p.start()

	def update(self, obs, action, cost, iter_info:IterationInformation=None):
		if iter_info is None:
			self.pqueue.put([obs, action, cost])
		else:
			self.pqueue.put([obs, action, cost, 
								iter_info.mean_predicted_cost, 
								iter_info.mean_predicted_cost_std,
								iter_info.predicted_states,
								iter_info.predicted_states_std,
								iter_info.predicted_actions,
								iter_info.predicted_costs,
								iter_info.predicted_costs_std])
			
			'''info_dict['mean predicted cost'],
				info_dict['mean predicted cost std'], info_dict['predicted states'],
				info_dict['predicted states std'], info_dict['predicted actions'],
				info_dict['predicted costs'], info_dict['predicted costs std']])'''


class LivePlotSequential:
	def __init__(self, num_steps_total, obs_space, action_space, step_pred, use_constraints=False,
			state_min=None, state_max=None, mul_std_bounds=3, fontsize=6):
		self.fig, self.axes = plt.subplots(nrows=3, figsize=(6, 5), sharex=True)
		self.axes[0].set_title('Normed states and predictions')
		self.axes[1].set_title('Normed actions')
		self.axes[2].set_title('Cost and horizon cost')
		plt.xlabel("Env steps")
		self.min_state = -0.03
		self.max_state = 1.03
		self.axes[0].set_ylim(self.min_state, self.max_state)
		self.axes[1].set_ylim(-0.03, 1.03)
		self.axes[2].set_xlim(0, num_steps_total)
		plt.tight_layout()

		self.step_pred = step_pred
		self.mul_std_bounds = mul_std_bounds

		self.states = np.empty((num_steps_total, obs_space.shape[0]))
		self.actions = np.empty((num_steps_total, action_space.shape[0]))
		self.costs = np.empty(num_steps_total)

		self.mean_costs_pred = np.empty_like(self.costs)
		self.mean_costs_std_pred = np.empty_like(self.costs)

		self.min_obs = obs_space.low
		self.max_obs = obs_space.high
		self.min_action = action_space.low
		self.max_action = action_space.high

		self.num_points_show = 0
		if use_constraints:
			if state_min is not None:
				for state_idx in range(obs_space.shape[0]):
					self.axes[0].axhline(y=state_min[state_idx],
						color=plt.get_cmap('tab20').colors[2 * state_idx],
						linestyle='-.')
			if state_max is not None:
				for state_idx in range(obs_space.shape[0]):
					self.axes[0].axhline(y=state_max[state_idx],
						color=plt.get_cmap('tab20').colors[2 * state_idx],
						linestyle='-.')
		self.lines_states = [self.axes[0].plot([], [], label='state' + str(state_idx),
				color=plt.get_cmap('tab20').colors[2 * state_idx]) for state_idx in range(obs_space.shape[0])]
		self.line_cost = self.axes[2].plot([], [], label='cost', color='k')

		self.lines_actions = [self.axes[1].step([], [], label='action' + str(action_idx),
				color=plt.get_cmap('tab20').colors[1 + 2 * action_idx]) for action_idx in range(action_space.shape[0])]

		self.line_mean_costs_pred = self.axes[2].plot([], [], label='mean predicted cost', color='orange')

		self.lines_states_pred = [self.axes[0].plot([], [],
				label='predicted_states' + str(state_idx),
				color=plt.get_cmap('tab20').colors[2 * state_idx], linestyle='dashed')
			for state_idx in range(obs_space.shape[0])]
		self.lines_actions_pred = [self.axes[1].step([], [], label='predicted_action' + str(action_idx),
				color=plt.get_cmap('tab20').colors[1 + 2 * action_idx], linestyle='dashed')
				for action_idx in range(action_space.shape[0])]
		self.line_costs_pred = self.axes[2].plot([], [], label='predicted cost', color='k', linestyle='dashed')

		self.axes[0].legend(fontsize=fontsize)
		self.axes[0].grid()
		self.axes[1].legend(fontsize=fontsize)
		self.axes[1].grid()
		self.axes[2].legend(fontsize=fontsize)
		self.axes[2].grid()
		plt.show(block=False)

	def update(self, obs, action, cost, iter_info:IterationInformation=None):
		obs_norm = (obs - self.min_obs) / (self.max_obs - self.min_obs)
		action_norm = (action - self.min_action) / (self.max_action - self.min_action)
		self.states[self.num_points_show] = obs_norm
		self.costs[self.num_points_show] = cost

		update_limits = False
		min_state_actual = np.min(obs_norm)
		if min_state_actual < self.min_state:
			self.min_state = min_state_actual
			update_limits = True

		max_state_actual = np.max(obs_norm)
		if max_state_actual > self.max_state:
			self.max_state = max_state_actual
			update_limits = True

		if update_limits:
			self.axes[0].set_ylim(self.min_state, self.max_state)

		idxs = np.arange(0, (self.num_points_show + 1))
		for idx_axes in range(len(self.axes)):
			self.axes[idx_axes].collections.clear()

		for idx_state in range(len(obs_norm)):
			self.lines_states[idx_state][0].set_data(idxs, self.states[:(self.num_points_show + 1), idx_state])

		self.actions[self.num_points_show] = action_norm
		for idx_action in range(len(action_norm)):
			self.lines_actions[idx_action][0].set_data(idxs, self.actions[:(self.num_points_show + 1), idx_action])
			
		self.line_cost[0].set_data(idxs, self.costs[:(self.num_points_show + 1)])

		if iter_info is not None:
			np.nan_to_num(iter_info.mean_costs_pred, copy=False, nan=-1, posinf=-1, neginf=-1)
			np.nan_to_num(iter_info.mean_costs_std_pred, copy=False, nan=99, posinf=99, neginf=99)
			np.nan_to_num(iter_info.states_pred, copy=False, nan=-1, posinf=-1, neginf=-1)
			np.nan_to_num(iter_info.states_std_pred, copy=False, nan=99, posinf=99, neginf=0)
			np.nan_to_num(iter_info.costs_pred, copy=False, nan=-1, posinf=-1, neginf=-1)
			np.nan_to_num(iter_info.costs_std_pred, copy=False, nan=99, posinf=99, neginf=0)
			# if num_repeat_action is not 1, the control do not happen at each iteration,
			# we must select the last index where the control happened as the start of the prediction horizon
			idx_prev_control = (idxs[-1] // self.step_pred) * self.step_pred
			idxs_future = np.arange(idx_prev_control,
										idx_prev_control + self.step_pred + len(iter_info.states_pred) * self.step_pred,
										self.step_pred)

			self.mean_costs_pred[self.num_points_show] = iter_info.mean_costs_pred
			self.mean_costs_std_pred[self.num_points_show] = iter_info.mean_costs_std_pred

			for idx_state in range(len(obs_norm)):
				future_states_show = np.concatenate(([self.states[idx_prev_control, idx_state]], iter_info.states_pred[:, idx_state]))
				self.lines_states_pred[idx_state][0].set_data(idxs_future, future_states_show)
				future_states_std_show = np.concatenate(([0], iter_info.states_std_pred[:, idx_state]))
				self.axes[0].fill_between(idxs_future,
					future_states_show - future_states_std_show * self.mul_std_bounds,
					future_states_show + future_states_std_show * self.mul_std_bounds,
					facecolor=plt.get_cmap('tab20').colors[2 * idx_state], alpha=ALPHA_CONFIDENCE_BOUNDS,
					label='predicted ' + str(self.mul_std_bounds) + ' std bounds state ' + str(idx_state))
			for idx_action in range(len(action_norm)):
				self.lines_actions_pred[idx_action][0].set_data(idxs_future,
					np.concatenate(([self.actions[idx_prev_control, idx_action]], iter_info.actions_pred[:, idx_action])))

			future_costs_show = np.concatenate(([self.costs[idx_prev_control]], iter_info.costs_pred))
			self.line_costs_pred[0].set_data(idxs_future, future_costs_show)

			future_cost_std_show = np.concatenate(([0], iter_info.costs_std_pred))
			self.axes[2].fill_between(idxs_future,
				future_costs_show - future_cost_std_show * self.mul_std_bounds,
				future_costs_show + future_cost_std_show * self.mul_std_bounds,
				facecolor='black', alpha=ALPHA_CONFIDENCE_BOUNDS,
				label='predicted ' + str(self.mul_std_bounds) + ' std cost bounds')
		else:
			if self.num_points_show == 0:
				self.mean_costs_pred[self.num_points_show] = MEAN_PRED_COST_INIT
				self.mean_costs_std_pred[self.num_points_show] = STD_MEAN_PRED_COST_INIT
			else:
				self.mean_costs_pred[self.num_points_show] = self.mean_costs_pred[self.num_points_show - 1]
				self.mean_costs_std_pred[self.num_points_show] = self.mean_costs_std_pred[self.num_points_show - 1]

		self.line_mean_costs_pred[0].set_data(idxs, self.mean_costs_pred[:(self.num_points_show + 1)])
		self.axes[2].set_ylim(0, np.max([np.max(self.mean_costs_pred[:(self.num_points_show + 1)]),
												np.max(self.costs[:(self.num_points_show + 1)])]) * 1.1)
		self.axes[2].fill_between(idxs,
			self.mean_costs_pred[:(self.num_points_show + 1)] -
			self.mean_costs_std_pred[:(self.num_points_show + 1)] * self.mul_std_bounds,
			self.mean_costs_pred[:(self.num_points_show + 1)] +
			self.mean_costs_std_pred[:(self.num_points_show + 1)] * self.mul_std_bounds,
			facecolor='orange', alpha=ALPHA_CONFIDENCE_BOUNDS,
			label='mean predicted ' + str(self.mul_std_bounds) + ' std cost bounds')

		self.fig.canvas.draw()
		plt.pause(0.01)
		self.num_points_show += 1


def save_plot_model_3d_process(inputs, targets, parameters, constraints_gp, folder_save,
		idxs_in_gp_memory=None, prop_extend_domain=1, n_ticks=150, total_col_max=3, fontsize=6):
	with torch.no_grad(), gpytorch.settings.fast_pred_var():
		torch.set_num_threads(1)
		num_input_model = len(inputs[0])
		num_models = len(targets[0])
		idxs_outside_gp_memory = [
			np.delete(np.arange(len(inputs)), idxs_in_gp_memory[idx_model]) for idx_model in range(num_models)]

		models = create_models(inputs, targets, parameters, constraints_gp)
		for idx_model in range(len(models)):
			models[idx_model].eval()
		targets = targets.numpy()

		num_figures = int(num_input_model / total_col_max + 0.5)
		figs = []
		axes_s = []
		for index_figure in range(num_figures):
			fig = plt.figure(figsize=(15, 6))
			axes = []
			columns_active = np.min([num_models - (total_col_max * index_figure), total_col_max])
			for idx_ax in range(columns_active * 2):
				axes.append(fig.add_subplot(2, columns_active, idx_ax + 1, projection='3d'))
			axes_s.append(axes)
			figs.append(fig)
			for idx_subplot in range(columns_active):
				idx_obs_repr = index_figure * columns_active + idx_subplot
				if idx_obs_repr >= (num_models):
					break
				feat_importance = (1 / models[idx_obs_repr].covar_module.base_kernel.lengthscale).numpy()
				feat_importance = feat_importance / np.sum(feat_importance)
				best_features = np.argsort(-feat_importance)[0, :2]
				'''estimator_other_columns = Pipeline(
					steps=[('standardscaler', StandardScaler()),
						('features', PolynomialFeatures(2)),
						('model', LinearRegression())])'''
				estimator_other_columns = Pipeline(
					steps=[('standardscaler', StandardScaler()),
						('features', KNeighborsRegressor(n_neighbors=3, weights='distance'))])

				estimator_other_columns.fit(inputs[:, best_features].numpy(),
					np.delete(inputs.numpy(), best_features, axis=1))

				domain_extension_x = (prop_extend_domain - 1) * (
						inputs[:, best_features[0]].max() - inputs[:, best_features[0]].min())
				domain_extension_y = (prop_extend_domain - 1) * (
						inputs[:, best_features[1]].max() - inputs[:, best_features[1]].min())
				x_grid = np.linspace(inputs[:, best_features[0]].min() - domain_extension_x / 2,
					inputs[:, best_features[0]].max() + domain_extension_x / 2, n_ticks)
				y_grid = np.linspace(inputs[:, best_features[1]].min() - domain_extension_y / 2,
					inputs[:, best_features[1]].max() + domain_extension_y / 2, n_ticks)

				x_grid, y_grid = np.meshgrid(x_grid, y_grid)

				x_grid_1d = np.expand_dims(x_grid.flatten(), axis=-1)
				y_grid_1d = np.expand_dims(y_grid.flatten(), axis=-1)
				xy_grid = np.concatenate((x_grid_1d, y_grid_1d), axis=1)
				pred_other_columns = estimator_other_columns.predict(xy_grid)
				all_col = np.zeros((xy_grid.shape[0], num_input_model))
				all_col[:, best_features] = xy_grid
				all_col[:, np.delete(np.arange(num_input_model), best_features)] = pred_other_columns
				gauss_pred = models[idx_obs_repr].likelihood(models[idx_obs_repr](torch.from_numpy(
																all_col.astype(np.float64))))
				z_grid_1d_mean = gauss_pred.mean.numpy()
				z_grid_1d_std = np.sqrt(gauss_pred.stddev.numpy())

				z_grid_mean = z_grid_1d_mean.reshape(x_grid.shape)
				z_grid_std = z_grid_1d_std.reshape(x_grid.shape)
				surf1 = axes[idx_subplot].contour3D(x_grid, y_grid, z_grid_mean, n_ticks, cmap='cool')
				axes[idx_subplot].set_xlabel('Input ' + str(best_features[0]), fontsize=fontsize, rotation=150)
				axes[idx_subplot].set_ylabel('Input ' + str(best_features[1]), fontsize=fontsize, rotation=150)
				axes[idx_subplot].set_zlabel('Variation state ' + str(idx_obs_repr), fontsize=fontsize,
					rotation=60)
				if targets is not None:
					axes[idx_subplot].scatter(
					inputs[idxs_in_gp_memory[idx_obs_repr], best_features[0]],
					inputs[idxs_in_gp_memory[idx_obs_repr], best_features[1]],
					targets[idxs_in_gp_memory[idx_obs_repr], idx_obs_repr],
					marker='x', c='g')

					axes[idx_subplot].scatter(
					inputs[idxs_outside_gp_memory[idx_obs_repr], best_features[0]],
					inputs[idxs_outside_gp_memory[idx_obs_repr], best_features[1]],
					targets[idxs_outside_gp_memory[idx_obs_repr], idx_obs_repr],
					marker='x', c='k')

					axes[idx_subplot].quiver(inputs[:-1, best_features[0]], inputs[:-1, best_features[1]],
						targets[:-1, idx_obs_repr],
						inputs[1:, best_features[0]] - inputs[:-1, best_features[0]],
						inputs[1:, best_features[1]] - inputs[:-1, best_features[1]],
						targets[1:, idx_obs_repr] - targets[:-1, idx_obs_repr],
						color='k', linestyle="solid", alpha=0.3, arrow_length_ratio=0.001, length=0.9)

				surf2 = axes[idx_subplot + columns_active].contour3D(x_grid, y_grid, z_grid_std, n_ticks, cmap='cool')
				axes[idx_subplot + columns_active].set_xlabel('Input ' + str(best_features[0]), fontsize=fontsize, rotation=150)
				axes[idx_subplot + columns_active].set_ylabel('Input ' + str(best_features[1]), fontsize=fontsize, rotation=150)
				axes[idx_subplot + columns_active].set_zlabel('Uncertainty: std state ' + str(idx_obs_repr),
					fontsize=fontsize, rotation=60)

				if targets is not None:
					pred_outside_memory = models[idx_obs_repr].likelihood(
						models[idx_obs_repr](inputs[idxs_outside_gp_memory[idx_obs_repr]]))
					errors_outside_memory = np.abs(pred_outside_memory.mean.numpy() -
												   targets[idxs_outside_gp_memory[idx_obs_repr], idx_obs_repr])
					errors = np.zeros_like(inputs[:, 0])
					errors[idxs_outside_gp_memory[idx_obs_repr]] = errors_outside_memory
					axes[idx_subplot + columns_active].scatter(
						inputs[idxs_in_gp_memory[idx_obs_repr], best_features[0]],
						inputs[idxs_in_gp_memory[idx_obs_repr], best_features[1]],
						errors[idxs_in_gp_memory[idx_obs_repr]], marker='x', c='g')
					axes[idx_subplot + columns_active].scatter(
						inputs[idxs_outside_gp_memory[idx_obs_repr], best_features[0]],
						inputs[idxs_outside_gp_memory[idx_obs_repr], best_features[1]],
						errors_outside_memory, marker='x', c='k')
					axes[idx_subplot + columns_active].quiver(
						inputs[:-1, best_features[0]], inputs[:-1, best_features[1]],
						errors[:-1],
						inputs[1:, best_features[0]] - inputs[:-1, best_features[0]],
						inputs[1:, best_features[1]] - inputs[:-1, best_features[1]],
						errors[1:] - errors[:-1],
						color='k', linestyle="solid", alpha=0.3, arrow_length_ratio=0.001, length=0.9)

			plt.tight_layout()
			hour = datetime.datetime.now().hour
			minute = datetime.datetime.now().minute
			second = datetime.datetime.now().second
			fig.savefig(os.path.join(folder_save, 'model_3d' + '_h' + str(hour) +
																'_m' + str(minute) +
															  	'_s' + str(second) + '.png'))
			plt.close()