import time
import threading
import queue
import multiprocessing

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import imageio

from rl_gp_mpc.control_objects.controllers.iteration_info_class import IterationInformation

# matplotlib.use('TkAgg')
matplotlib.rc('font', size='6')
ALPHA_CONFIDENCE_BOUNDS = 0.3
FONTSIZE = 6

MUL_STD_BOUNDS = 3
FONTSIZE = 6


class LivePlotParallel:
	def __init__(self, num_steps_total, dim_states, dim_actions, use_constraints=False,
			state_min=None, state_max=None, time_between_updates=0.75, use_thread=False,
			path_save="control_animation.gif", save=False):
		self.use_thread = use_thread

		if use_thread:
			self.pqueue = queue.Queue()
			self.graph_p = threading.Thread(target=anim_plots_2d_p,
									args=(self.pqueue, num_steps_total, dim_states,
										dim_actions, use_constraints,
										state_min, state_max, time_between_updates, path_save, save))
		else:
			self.pqueue = multiprocessing.Queue()
			self.graph_p = multiprocessing.Process(target=anim_plots_2d_p,
									args=(self.pqueue, num_steps_total, dim_states,
											dim_actions, use_constraints,
											state_min, state_max, time_between_updates, path_save, save))
		self.graph_p.start()

	def update(self, state, action, cost, iter_info:IterationInformation=None):
		if iter_info is None:
			self.pqueue.put([state, action, cost])
		else:
			self.pqueue.put([state, action, cost, 
								iter_info.mean_predicted_cost, 
								iter_info.mean_predicted_cost_std,
								iter_info.predicted_states,
								iter_info.predicted_states_std,
								iter_info.predicted_actions,
								iter_info.predicted_costs,
								iter_info.predicted_costs_std,
								iter_info.predicted_idxs])
	def close(self):
		self.pqueue.put(None)
		if self.use_thread:
			self.graph_p.join()
		else:
			self.pqueue.close()
			self.graph_p.join()
			self.graph_p.close()

	def __exit__(self):
		if 'graph_p' in self.__dict__ and self.graph_p.is_alive():
			self.close()


def anim_plots_2d_p(queue, num_steps_total, dim_states, dim_actions, use_constraints=False, state_min=None, state_max=None, time_between_updates=0.5, path_save="control_animation.gif", save=False):
	fig, axes = plt.subplots(nrows=3, figsize=(10, 6), sharex=True)
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

	states = np.empty((num_steps_total, dim_states))
	actions = np.empty((num_steps_total, dim_actions))
	costs = np.empty(num_steps_total)
	mean_cost_pred = np.empty_like(costs)
	mean_cost_std_pred = np.empty_like(mean_cost_pred)
	
	if use_constraints:
		if state_min is not None:
			for idx_state in range(dim_states):
				axes[0].axhline(y=state_min[idx_state],
					color=plt.get_cmap('tab20').colors[2 * idx_state],
					linestyle='-.')
		if state_max is not None:
			for idx_state in range(dim_states):
				axes[0].axhline(y=state_max[idx_state],
					color=plt.get_cmap('tab20').colors[2 * idx_state],
					linestyle='-.')
	lines_states = [axes[0].plot([], [], label='state' + str(state_idx),
		color=plt.get_cmap('tab20').colors[2 * state_idx]) for state_idx in range(dim_states)]
	lines_actions = [axes[1].step([], [], label='action' + str(action_idx),
		color=plt.get_cmap('tab20').colors[1 + 2 * action_idx], where='post') for action_idx in range(dim_actions)]

	line_cost = axes[2].plot([], [], label='cost', color='k')
	line_costs_mean_pred = axes[2].plot([], [], label='mean predicted cost', color='orange')

	lines_predicted_states = [axes[0].plot([], [],
		label='predicted_states' + str(state_idx),
		color=plt.get_cmap('tab20').colors[2 * state_idx], linestyle='dashed')
		for state_idx in range(dim_states)]
	lines_actions_pred = [axes[1].step([], [], label='predicted_action' + str(action_idx),
		color=plt.get_cmap('tab20').colors[1 + 2 * action_idx], linestyle='dashed', where='post')
		for action_idx in range(dim_actions)]
	line_predicted_costs = axes[2].plot([], [], label='predicted cost', color='k', linestyle='dashed')

	axes[0].legend(fontsize=FONTSIZE)
	axes[1].legend(fontsize=FONTSIZE)
	axes[2].legend(fontsize=FONTSIZE)
	fig.canvas.draw()
	plt.show(block=False)

	num_pts_show = 0
	last_update_time = time.time()
	exit_loop = False
	if save:
		writer = imageio.get_writer(path_save, mode='I')
	while True:
		if exit_loop: break
		if (time.time() - last_update_time) > time_between_updates:
			got_data = False
			last_update_time = time.time()
			while not (queue.empty()):
				msg = queue.get()  # Read from the queue
				if msg is None: # Allows to gracefully end the process by send None, then join and close
					exit_loop = True
					continue
				obs_norm = np.array(msg[0])
				action_norm = np.array(msg[1])
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
						mean_cost_pred[num_pts_show] = 0
						mean_cost_std_pred[num_pts_show] = 0
					else:
						# For when the information about prediction have not been provided to the graph object
						# for the update, for example at the init period when random actions are applied
						mean_cost_pred[num_pts_show] = mean_cost_pred[num_pts_show - 1]
						mean_cost_std_pred[num_pts_show] = mean_cost_std_pred[num_pts_show - 1]
				num_pts_show += 1
				got_data = True
				last_update_time = time.time()

			if got_data:
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
					mean_cost_std_pred[:num_pts_show] * MUL_STD_BOUNDS,
					mean_cost_pred[:num_pts_show] +
					mean_cost_std_pred[:num_pts_show] * MUL_STD_BOUNDS,
					facecolor='orange', alpha=ALPHA_CONFIDENCE_BOUNDS,
					label='mean predicted ' + str(MUL_STD_BOUNDS) + ' std cost bounds')
				
				axes[2].set_ylim(0, np.max([np.max(mean_cost_pred[:num_pts_show]),
					np.max(costs[:num_pts_show])]) * 1.1)
				axes[2].set_ylim(0, np.max(line_cost[0].get_ydata()) * 1.1)
				
				# If predictions are not given in the last update, do not show them on the graph
				if len(msg) > 3:
					predicted_states = np.nan_to_num(msg[5], nan=-1, posinf=-1, neginf=-1)
					predicted_states_std = np.nan_to_num(msg[6], copy=False, nan=99, posinf=99, neginf=99)
					actions_pred = msg[7]
					predicted_costs = np.nan_to_num(msg[8], nan=-1, posinf=-1, neginf=-1)
					predicted_costs_std = np.nan_to_num(msg[9], nan=99, posinf=99, neginf=0)
					predicted_idxs = np.array(msg[10])
					step_idx = (predicted_idxs[-1] - predicted_idxs[-2])
					predicted_idxs_states = np.concatenate((predicted_idxs, [predicted_idxs[-1] + step_idx]))

					for idx_state in range(len(obs_norm)):
						future_states_show = predicted_states[:, idx_state]
						lines_predicted_states[idx_state][0].set_data(predicted_idxs_states, future_states_show)

						future_states_std_show = predicted_states_std[:, idx_state]
						axes[0].fill_between(predicted_idxs_states,
							future_states_show - future_states_std_show * MUL_STD_BOUNDS,
							future_states_show + future_states_std_show * MUL_STD_BOUNDS,
							facecolor=plt.get_cmap('tab20').colors[2 * idx_state], alpha=ALPHA_CONFIDENCE_BOUNDS,
							label='predicted ' + str(MUL_STD_BOUNDS) + ' std bounds state ' + str(idx_state))

					for idx_action in range(len(action_norm)):
						future_actions_show = actions_pred[:, idx_action]
						lines_actions_pred[idx_action][0].set_data(predicted_idxs, future_actions_show) # 
	
					future_costs_show = predicted_costs
					line_predicted_costs[0].set_data(predicted_idxs_states, future_costs_show)
	
					future_cost_std_show = predicted_costs_std
					axes[2].fill_between(predicted_idxs_states,
						future_costs_show - future_cost_std_show * MUL_STD_BOUNDS,
						future_costs_show + future_cost_std_show * MUL_STD_BOUNDS,
						facecolor='black', alpha=ALPHA_CONFIDENCE_BOUNDS,
						label='predicted ' + str(MUL_STD_BOUNDS) + ' std cost bounds')

				for ax in axes:
					ax.relim()
					ax.autoscale_view(True, True, True)

				fig.canvas.draw()
				if save:
					image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
					image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
					# Append the image to the animation
					writer.append_data(image)
				plt.pause(0.01)
		time.sleep(0.05)
	if save: writer.close()
	plt.close('all')


class LivePlotSequential:
	'''
	DEPRECATED CLASS, WILL NOT WORK
	'''
	def __init__(self, num_steps_total, obs_space, action_space, step_pred, use_constraints=False, state_min=None, state_max=None):
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

		self.states = np.empty((num_steps_total, obs_space.shape[0]))
		self.actions = np.empty((num_steps_total, action_space.shape[0]))
		self.costs = np.empty(num_steps_total)

		self.mean_predicted_cost = np.empty_like(self.costs)
		self.mean_predicted_cost_std = np.empty_like(self.costs)

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

		self.line_mean_predicted_cost = self.axes[2].plot([], [], label='mean predicted cost', color='orange')

		self.lines_predicted_states = [self.axes[0].plot([], [],
				label='predicted_states' + str(state_idx),
				color=plt.get_cmap('tab20').colors[2 * state_idx], linestyle='dashed')
			for state_idx in range(obs_space.shape[0])]
		self.lines_actions_pred = [self.axes[1].step([], [], label='predicted_action' + str(action_idx),
				color=plt.get_cmap('tab20').colors[1 + 2 * action_idx], linestyle='dashed')
				for action_idx in range(action_space.shape[0])]
		self.line_predicted_costs = self.axes[2].plot([], [], label='predicted cost', color='k', linestyle='dashed')

		self.axes[0].legend(fontsize=FONTSIZE)
		self.axes[0].grid()
		self.axes[1].legend(fontsize=FONTSIZE)
		self.axes[1].grid()
		self.axes[2].legend(fontsize=FONTSIZE)
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
			np.nan_to_num(iter_info.mean_predicted_cost, copy=False, nan=-1, posinf=-1, neginf=-1)
			np.nan_to_num(iter_info.mean_predicted_cost_std, copy=False, nan=99, posinf=99, neginf=99)
			np.nan_to_num(iter_info.predicted_states, copy=False, nan=-1, posinf=-1, neginf=-1)
			np.nan_to_num(iter_info.predicted_states_std, copy=False, nan=99, posinf=99, neginf=0)
			np.nan_to_num(iter_info.predicted_costs, copy=False, nan=-1, posinf=-1, neginf=-1)
			np.nan_to_num(iter_info.predicted_costs_std, copy=False, nan=99, posinf=99, neginf=0)
			# if num_repeat_action is not 1, the control do not happen at each iteration,
			# we must select the last index where the control happened as the start of the prediction horizon
			idx_prev_control = (idxs[-1] // self.step_pred) * self.step_pred
			idxs_future = iter_info.predicted_idxs #np.arange(idx_prev_control,
							#				idx_prev_control + self.step_pred + len(iter_info.predicted_states) * self.step_pred,
							#			self.step_pred)

			self.mean_predicted_cost[self.num_points_show] = iter_info.mean_predicted_cost
			self.mean_predicted_cost_std[self.num_points_show] = iter_info.mean_predicted_cost_std

			for idx_state in range(len(obs_norm)):
				future_states_show = np.concatenate(([self.states[idx_prev_control, idx_state]], iter_info.predicted_states[:, idx_state]))
				self.lines_predicted_states[idx_state][0].set_data(idxs_future, future_states_show)
				future_states_std_show = np.concatenate(([0], iter_info.predicted_states_std[:, idx_state]))
				self.axes[0].fill_between(idxs_future,
					future_states_show - future_states_std_show * MUL_STD_BOUNDS,
					future_states_show + future_states_std_show * MUL_STD_BOUNDS,
					facecolor=plt.get_cmap('tab20').colors[2 * idx_state], alpha=ALPHA_CONFIDENCE_BOUNDS,
					label='predicted ' + str(MUL_STD_BOUNDS) + ' std bounds state ' + str(idx_state))
			for idx_action in range(len(action_norm)):
				self.lines_actions_pred[idx_action][0].set_data(idxs_future, iter_info.actions_pred[:, idx_action])
				#) np.concatenate(([self.actions[idx_prev_control, idx_action]], )

			future_costs_show = np.concatenate(([self.costs[idx_prev_control]], iter_info.predicted_costs))
			self.line_predicted_costs[0].set_data(idxs_future, future_costs_show)

			future_cost_std_show = np.concatenate(([0], iter_info.predicted_costs_std))
			self.axes[2].fill_between(idxs_future,
				future_costs_show - future_cost_std_show * MUL_STD_BOUNDS,
				future_costs_show + future_cost_std_show * MUL_STD_BOUNDS,
				facecolor='black', alpha=ALPHA_CONFIDENCE_BOUNDS,
				label='predicted ' + str(MUL_STD_BOUNDS) + ' std cost bounds')
		else:
			if self.num_points_show == 0:
				self.mean_predicted_cost[self.num_points_show] = MEAN_PRED_COST_INIT
				self.mean_predicted_cost_std[self.num_points_show] = STD_MEAN_PRED_COST_INIT
			else:
				self.mean_predicted_cost[self.num_points_show] = self.mean_predicted_cost[self.num_points_show - 1]
				self.mean_predicted_cost_std[self.num_points_show] = self.mean_predicted_cost_std[self.num_points_show - 1]

		self.line_mean_predicted_cost[0].set_data(idxs, self.mean_predicted_cost[:(self.num_points_show + 1)])
		self.axes[2].set_ylim(0, np.max([np.max(self.mean_predicted_cost[:(self.num_points_show + 1)]),
												np.max(self.costs[:(self.num_points_show + 1)])]) * 1.1)
		self.axes[2].fill_between(idxs,
			self.mean_predicted_cost[:(self.num_points_show + 1)] -
			self.mean_predicted_cost_std[:(self.num_points_show + 1)] * MUL_STD_BOUNDS,
			self.mean_predicted_cost[:(self.num_points_show + 1)] +
			self.mean_predicted_cost_std[:(self.num_points_show + 1)] * MUL_STD_BOUNDS,
			facecolor='orange', alpha=ALPHA_CONFIDENCE_BOUNDS,
			label='mean predicted ' + str(MUL_STD_BOUNDS) + ' std cost bounds')

		self.fig.canvas.draw()
		plt.pause(0.01)
		self.num_points_show += 1
