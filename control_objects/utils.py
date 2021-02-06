import datetime
import os

import torch
import gpytorch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, MultiTaskLasso

from .gp_models import ExactGPModelMonoTask

matplotlib.rc('font', size='6')
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
ALPHA_CONFIDENCE_BOUNDS = 0.3


def draw_history(states, actions, states_next, prediction_info_over_time, num_repeat_actions=1,
		constraints_states=None, num_iter_prediction_ahead_show=3, fig=None, axes=None,
		update_x_limit=True):
	indexes = np.arange(0, len(states) * num_repeat_actions, num_repeat_actions)
	indexes_control = np.array(prediction_info_over_time['iteration']) // num_repeat_actions
	predictions_over_time_show = np.zeros_like(states_next)
	predictions_over_time_show[indexes_control] = \
		[prediction[num_iter_prediction_ahead_show - 1].numpy()
			for prediction in prediction_info_over_time['predicted states']]
	predictions_std_over_time_show = np.ones_like(states_next)
	predictions_std_over_time_show[indexes_control] = \
		[prediction[num_iter_prediction_ahead_show - 1].numpy()
			for prediction in prediction_info_over_time['predicted states std']]
	predictions_over_time_show = \
		np.concatenate([np.zeros((num_iter_prediction_ahead_show, len(predictions_over_time_show[0]))),
			predictions_over_time_show[:-num_iter_prediction_ahead_show]])
	predictions_std_over_time_show = \
		np.concatenate([np.ones((num_iter_prediction_ahead_show, len(predictions_over_time_show[0]))),
			predictions_std_over_time_show[:-num_iter_prediction_ahead_show]])
	mean_cost_traj = np.array([element for element in prediction_info_over_time['mean predicted cost']])
	mean_cost_traj_std = np.array([element for element in prediction_info_over_time['mean predicted cost std']])
	np.nan_to_num(predictions_over_time_show, copy=False, nan=-1, posinf=-1, neginf=-1)
	np.nan_to_num(predictions_std_over_time_show, copy=False, nan=np.max(predictions_std_over_time_show), posinf=99,
		neginf=0)
	np.nan_to_num(mean_cost_traj, copy=False, nan=-1, posinf=-1, neginf=-1)
	np.nan_to_num(predictions_std_over_time_show, copy=False, nan=np.max(mean_cost_traj_std), posinf=99, neginf=0)
	for state_idx in range(len(states[0])):
		axes[0].plot(indexes, np.array(states)[:, state_idx], label='state' + str(state_idx),
			color=plt.get_cmap('tab20').colors[2 * state_idx])
		axes[0].plot(indexes, predictions_over_time_show[:, state_idx],
			label=str(num_iter_prediction_ahead_show * num_repeat_actions) + 'step_' + 'prediction' + str(state_idx),
			color=plt.get_cmap('tab20').colors[2 * state_idx], linestyle='--')
		axes[0].fill_between(indexes,
			predictions_over_time_show[:, state_idx] - predictions_std_over_time_show[:, state_idx],
			predictions_over_time_show[:, state_idx] + predictions_std_over_time_show[:, state_idx],
			label=str(num_iter_prediction_ahead_show * num_repeat_actions) + 'step_' + 'bounds_prediction' + str(
				state_idx), color=plt.get_cmap('tab20').colors[2 * state_idx], alpha=ALPHA_CONFIDENCE_BOUNDS)
		if constraints_states is not None and constraints_states['use_constraints']:
			axes[0].axhline(y=constraints_states['states_min'][state_idx],
				color=plt.get_cmap('tab20').colors[2 * state_idx],
				linestyle='-.')
			axes[0].axhline(y=constraints_states['states_max'][state_idx],
				color=plt.get_cmap('tab20').colors[2 * state_idx],
				linestyle='-.')
	costs = np.array([element for element in prediction_info_over_time['cost']])
	axes[2].plot(list(indexes_control * num_repeat_actions), costs, label='cost', color='k')
	axes[2].plot(list(indexes_control * num_repeat_actions), mean_cost_traj, label='mean predicted cost', color='orange')
	axes[2].fill_between(list(indexes_control * num_repeat_actions), mean_cost_traj - 2 * mean_cost_traj_std,
		mean_cost_traj + 3 * mean_cost_traj_std, label='mean predicted cost 3 std confidence bounds', color='orange',
		alpha=ALPHA_CONFIDENCE_BOUNDS)

	for action_idx in range(len(actions[0])):
		axes[1].step(indexes, np.array(actions)[:, action_idx], label='a_' + str(action_idx),
			color=plt.get_cmap('tab20').colors[1 + 2 * action_idx])

	if update_x_limit is True:
		axes[2].set_xlim(0, np.max(indexes))
	axes[2].set_ylim(0, np.max([np.max(mean_cost_traj), np.max(costs)]) * 1.2)
	return fig, axes


def save_plot_history_process(states, actions, states_next, prediction_info_over_time, folder_save,
		num_repeat_actions=1, constraints_states=None, num_iter_prediction_ahead_show=3):
	with threadpool_limits(limits=1, user_api='blas'), threadpool_limits(limits=1, user_api='openmp'):
		fig, axes = plt.subplots(nrows=3, figsize=(6, 5), sharex=True)
		axes[0].set_title('Normed states and predictions')
		axes[1].set_title('Normed actions')
		axes[2].set_title('Cost and horizon cost')
		plt.xlabel("time")
		axes[0].set_ylim(0, 1.02)
		axes[1].set_ylim(0, 1.02)
		plt.tight_layout()
		fig, axes = draw_history(states, actions, states_next, prediction_info_over_time, num_repeat_actions,
		constraints_states, num_iter_prediction_ahead_show, fig, axes)
		axes[0].legend()
		axes[1].legend()
		axes[2].legend()
		hour = datetime.datetime.now().hour
		minute = datetime.datetime.now().minute
		second = datetime.datetime.now().second
		fig.savefig(os.path.join(folder_save, 'history' + '_h' + str(hour) + '_m' + str(minute)
											  + '_s' + str(second) + '.png'))
		plt.close()


class LivePlotClass:
	def __init__(self, num_steps_total, observation_space, action_space, constraints,
			step_pred, num_std_bounds=2, fontsize=6):
		self.fig, self.axes = plt.subplots(nrows=3, figsize=(6, 5), sharex=True)
		self.axes[0].set_title('Normed states and predictions')
		self.axes[1].set_title('Normed actions')
		self.axes[2].set_title('Cost and horizon cost')
		plt.xlabel("Env steps")
		self.axes[0].set_ylim(-0.03, 1.03)
		self.axes[1].set_ylim(-0.03, 1.03)
		self.axes[2].set_xlim(0, num_steps_total)
		plt.tight_layout()

		num_steps_actions = num_steps_total // step_pred
		self.step_pred = step_pred
		self.num_std_bounds = num_std_bounds

		self.states = np.empty((num_steps_actions, observation_space.shape[0]))
		self.actions = np.empty((num_steps_actions, action_space.shape[0]))
		self.costs = np.empty(num_steps_actions)
		self.mean_predicted_cost = np.empty_like(self.costs)
		self.mean_predicted_cost_std = np.empty_like(self.mean_predicted_cost)

		self.min_observation = observation_space.low
		self.max_observation = observation_space.high
		self.min_action = action_space.low
		self.max_action = action_space.high

		self.num_points_show = 0
		if constraints is not None and constraints['use_constraints']:
			for state_idx in range(observation_space.shape[0]):
				self.axes[0].axhline(y=constraints['states_min'][state_idx],
					color=plt.get_cmap('tab20').colors[2 * state_idx],
					linestyle='-.')
				self.axes[0].axhline(y=constraints['states_max'][state_idx],
					color=plt.get_cmap('tab20').colors[2 * state_idx],
					linestyle='-.')
		self.states_lines = [self.axes[0].plot([], [], label='state' + str(state_idx),
				color=plt.get_cmap('tab20').colors[2 * state_idx]) for state_idx in range(observation_space.shape[0])]
		self.actions_lines = [self.axes[1].step([], [], label='action' + str(action_idx),
				color=plt.get_cmap('tab20').colors[1 + 2 * action_idx]) for action_idx in range(action_space.shape[0])]

		self.cost_line = self.axes[2].plot([], [], label='cost', color='k')
		self.mean_predicted_cost_line = self.axes[2].plot([], [], label='mean predicted cost', color='orange')

		self.predicted_states_lines = [self.axes[0].plot([], [],
				label='predicted_states' + str(state_idx),
				color=plt.get_cmap('tab20').colors[2 * state_idx], linestyle='dashed')
			for state_idx in range(observation_space.shape[0])]
		self.predicted_actions_line = [self.axes[1].step([], [], label='predicted_action' + str(action_idx),
				color=plt.get_cmap('tab20').colors[1 + 2 * action_idx], linestyle='dashed')
				for action_idx in range(action_space.shape[0])]
		self.predicted_cost_line = self.axes[2].plot([], [], label='predicted cost', color='k', linestyle='dashed')

		self.axes[0].legend(fontsize=fontsize)
		self.axes[1].legend(fontsize=fontsize)
		self.axes[2].legend(fontsize=fontsize)
		plt.show(block=False)

	def add_point_update(self, observation, action, add_info_dict=None):
		state = (observation - self.min_observation) / (self.max_observation - self.min_observation)
		action = (action - self.min_action) / (self.max_action - self.min_action)
		self.states[self.num_points_show] = state
		self.actions[self.num_points_show] = action
		x_indexes = torch.arange(0, (self.num_points_show + 1) * self.step_pred, self.step_pred)
		for idx_axes in range(len(self.axes)):
			self.axes[idx_axes].collections.clear()

		for state_idx in range(len(state)):
			self.states_lines[state_idx][0].set_data(x_indexes, self.states[:(self.num_points_show + 1), state_idx])
		for action_idx in range(len(action)):
			self.actions_lines[action_idx][0].set_data(x_indexes, self.actions[:(self.num_points_show + 1), action_idx])

		if add_info_dict is not None:
			cost = add_info_dict['cost']
			mean_predicted_cost = add_info_dict['mean predicted cost']
			mean_predicted_cost_std = add_info_dict['mean predicted cost std']
			future_predicted_states = add_info_dict['predicted states']
			future_predicted_states_std = add_info_dict['predicted states std']
			future_predicted_actions = add_info_dict['predicted actions']
			future_predicted_costs = add_info_dict['predicted costs']
			future_predicted_costs_std = add_info_dict['predicted costs std']
			np.nan_to_num(mean_predicted_cost, copy=False, nan=-1, posinf=-1, neginf=-1)
			np.nan_to_num(mean_predicted_cost_std, copy=False, nan=99, posinf=99, neginf=99)
			np.nan_to_num(future_predicted_states, copy=False, nan=-1, posinf=-1, neginf=-1)
			np.nan_to_num(future_predicted_states_std, copy=False, nan=99, posinf=99, neginf=0)
			np.nan_to_num(future_predicted_costs, copy=False, nan=-1, posinf=-1, neginf=-1)
			np.nan_to_num(future_predicted_costs_std, copy=False, nan=99, posinf=99, neginf=0)
			future_indexes = torch.arange(x_indexes[-1],
				x_indexes[-1] + self.step_pred + len(future_predicted_states) * self.step_pred, self.step_pred)
			self.costs[self.num_points_show] = cost
			self.mean_predicted_cost[self.num_points_show] = mean_predicted_cost
			self.mean_predicted_cost_std[self.num_points_show] = mean_predicted_cost_std

			for state_idx in range(len(state)):
				future_states_show = np.concatenate((state[None, state_idx], future_predicted_states[:, state_idx]))
				self.predicted_states_lines[state_idx][0].set_data(future_indexes, future_states_show)
				future_states_std_show = np.concatenate(([0], future_predicted_states_std[:, state_idx]))
				self.axes[0].fill_between(future_indexes,
					future_states_show - future_states_std_show * self.num_std_bounds,
					future_states_show + future_states_std_show * self.num_std_bounds,
					facecolor=plt.get_cmap('tab20').colors[2 * state_idx], alpha=ALPHA_CONFIDENCE_BOUNDS,
					label='predicted ' + str(self.num_std_bounds) + ' std bounds state ' + str(state_idx))
			for action_idx in range(len(action)):
				self.predicted_actions_line[action_idx][0].set_data(future_indexes,
					np.concatenate((action[None, action_idx], future_predicted_actions[:, action_idx])))

			future_costs_show = np.concatenate(([cost], future_predicted_costs))
			self.predicted_cost_line[0].set_data(future_indexes, future_costs_show)

			future_cost_std_show = np.concatenate(([0], future_predicted_costs_std))
			self.axes[2].fill_between(future_indexes,
				future_costs_show - future_cost_std_show * self.num_std_bounds,
				future_costs_show + future_cost_std_show * self.num_std_bounds,
				facecolor='black', alpha=ALPHA_CONFIDENCE_BOUNDS,
				label='predicted ' + str(self.num_std_bounds) + ' std cost bounds')
		else:
			self.costs[self.num_points_show] = 0
			self.mean_predicted_cost[self.num_points_show] = 0
			self.mean_predicted_cost_std[self.num_points_show] = 10

		self.cost_line[0].set_data(x_indexes, self.costs[:(self.num_points_show + 1)])
		self.mean_predicted_cost_line[0].set_data(x_indexes, self.mean_predicted_cost[:(self.num_points_show + 1)])
		self.axes[2].set_ylim(0, np.max([np.max(self.mean_predicted_cost[:(self.num_points_show + 1)]),
												np.max(self.costs[:(self.num_points_show + 1)])]) * 1.1)
		self.axes[2].fill_between(x_indexes,
			self.mean_predicted_cost[:(self.num_points_show + 1)] -
			self.mean_predicted_cost_std[:(self.num_points_show + 1)] * self.num_std_bounds,
			self.mean_predicted_cost[:(self.num_points_show + 1)] +
			self.mean_predicted_cost_std[:(self.num_points_show + 1)] * self.num_std_bounds,
			facecolor='orange', alpha=ALPHA_CONFIDENCE_BOUNDS,
			label='mean predicted ' + str(self.num_std_bounds) + ' std cost bounds')

		self.fig.canvas.draw()
		self.num_points_show += 1


def save_plot_model_3d_process(input_data, output_data, parameters, constraints, folder_save,
		indexes_points_in_gp_memory=None, prop_extend_domain=1, n_ticks=150, total_col_max=3, fontsize=6):
	with threadpool_limits(limits=1, user_api='blas'), \
			threadpool_limits(limits=1, user_api='openmp'), torch.no_grad(), gpytorch.settings.fast_pred_var():
		torch.set_num_threads(1)
		num_input_model = len(input_data[0])
		num_models = len(output_data[0])
		indexes_points_outside_gp_memory = \
			[np.delete(np.arange(len(input_data)), indexes_points_in_gp_memory[idx_model]) for
														idx_model in range(num_models)]
		models = [ExactGPModelMonoTask(input_data[indexes_points_in_gp_memory[idx_model]],
		output_data[[indexes_points_in_gp_memory[idx_model]], idx_model], num_input_model)
												for idx_model in range(num_models)]
		output_data = output_data.numpy()

		for idx_model in range(num_models):
			# register constraints on parameters
			if "min_std_noise" in constraints.keys():
				models[idx_model].likelihood.noise_covar.register_constraint("raw_noise",
					gpytorch.constraints.Interval(lower_bound=np.power(constraints['min_std_noise'], 2),
						upper_bound=np.power(constraints['max_std_noise'], 2)))
			if "min_outputscale" in constraints.keys():
				models[idx_model].covar_module.register_constraint("raw_outputscale",
					gpytorch.constraints.Interval(
						lower_bound=constraints['min_outputscale'],
						upper_bound=constraints['max_outputscale']))
			if "min_lengthscale" in constraints.keys():
				models[idx_model].covar_module.base_kernel.register_constraint("raw_lengthscale",
					gpytorch.constraints.Interval(
						lower_bound=constraints['min_lengthscale'],
						upper_bound=constraints['max_lengthscale']))
			# load parameters
			models[idx_model].load_state_dict(parameters[idx_model])
			models[idx_model].eval()

		num_figures = int(num_input_model / total_col_max + 0.5)
		figs = []
		axes_s = []
		for index_figure in range(num_figures):
			fig = plt.figure(figsize=(15, 6))
			axes = []
			columns_active = np.min([num_models - (total_col_max * index_figure), total_col_max])
			for index_ax in range(columns_active * 2):
				axes.append(fig.add_subplot(2, columns_active, index_ax + 1, projection='3d'))
			axes_s.append(axes)
			figs.append(fig)
			for subplot_idx in range(columns_active):
				index_observation_represent = index_figure * columns_active + subplot_idx
				if index_observation_represent >= (num_models):
					break
				features_importance = (1 / models[index_observation_represent].covar_module.base_kernel.lengthscale).numpy()
				features_importance = features_importance / np.sum(features_importance)
				best_features = np.argsort(-features_importance)[0, :2]
				'''estimator_other_columns = Pipeline(
					steps=[('standardscaler', StandardScaler()),
						('features', PolynomialFeatures(2)),
						('model', LinearRegression())])'''
				estimator_other_columns = Pipeline(
					steps=[('standardscaler', StandardScaler()),
						('features', KNeighborsRegressor(n_neighbors=3, weights='distance'))])

				estimator_other_columns.fit(input_data[:, best_features].numpy(),
					np.delete(input_data.numpy(), best_features, axis=1))

				domain_extension_x = (prop_extend_domain - 1) * (
						input_data[:, best_features[0]].max() - input_data[:, best_features[0]].min())
				domain_extension_y = (prop_extend_domain - 1) * (
						input_data[:, best_features[1]].max() - input_data[:, best_features[1]].min())
				x_grid = np.linspace(input_data[:, best_features[0]].min() - domain_extension_x / 2,
					input_data[:, best_features[0]].max() + domain_extension_x / 2, n_ticks)
				y_grid = np.linspace(input_data[:, best_features[1]].min() - domain_extension_y / 2,
					input_data[:, best_features[1]].max() + domain_extension_y / 2, n_ticks)

				x_grid, y_grid = np.meshgrid(x_grid, y_grid)

				x_grid_1d = np.expand_dims(x_grid.flatten(), axis=-1)
				y_grid_1d = np.expand_dims(y_grid.flatten(), axis=-1)
				xy_grid = np.concatenate((x_grid_1d, y_grid_1d), axis=1)
				predictions_other_columns = estimator_other_columns.predict(xy_grid)
				all_columns = np.zeros((xy_grid.shape[0], num_input_model))
				all_columns[:, best_features] = xy_grid
				all_columns[:, np.delete(np.arange(num_input_model), best_features)] = predictions_other_columns
				gauss_pred = models[index_observation_represent].likelihood(models[index_observation_represent](
					torch.from_numpy(all_columns.astype(np.float64))))
				z_grid_1d_mean = gauss_pred.mean.numpy()
				z_grid_1d_std = np.sqrt(gauss_pred.stddev.numpy())

				z_grid_mean = z_grid_1d_mean.reshape(x_grid.shape)
				z_grid_std = z_grid_1d_std.reshape(x_grid.shape)
				surf1 = axes[subplot_idx].contour3D(x_grid, y_grid, z_grid_mean, n_ticks, cmap='cool')
				axes[subplot_idx].set_xlabel('Input ' + str(best_features[0]), fontsize=fontsize, rotation=150)
				axes[subplot_idx].set_ylabel('Input ' + str(best_features[1]), fontsize=fontsize, rotation=150)
				axes[subplot_idx].set_zlabel('Variation state ' + str(index_observation_represent), fontsize=fontsize,
					rotation=60)
				if output_data is not None:
					axes[subplot_idx].scatter(
					input_data[indexes_points_in_gp_memory[index_observation_represent], best_features[0]],
					input_data[indexes_points_in_gp_memory[index_observation_represent], best_features[1]],
					output_data[indexes_points_in_gp_memory[index_observation_represent], index_observation_represent],
					marker='x', c='g')

					axes[subplot_idx].scatter(
					input_data[indexes_points_outside_gp_memory[index_observation_represent], best_features[0]],
					input_data[indexes_points_outside_gp_memory[index_observation_represent], best_features[1]],
					output_data[indexes_points_outside_gp_memory[index_observation_represent], index_observation_represent],
					marker='x', c='k')

					axes[subplot_idx].quiver(input_data[:-1, best_features[0]], input_data[:-1, best_features[1]],
						output_data[:-1, index_observation_represent],
						input_data[1:, best_features[0]] - input_data[:-1, best_features[0]],
						input_data[1:, best_features[1]] - input_data[:-1, best_features[1]],
						output_data[1:, index_observation_represent] - output_data[:-1, index_observation_represent],
						color='k', linestyle="solid", alpha=0.3, arrow_length_ratio=0.001, length=0.9)

				surf2 = axes[subplot_idx + columns_active].contour3D(x_grid, y_grid, z_grid_std, n_ticks, cmap='cool')
				axes[subplot_idx + columns_active].set_xlabel('Input ' + str(best_features[0]), fontsize=fontsize, rotation=150)
				axes[subplot_idx + columns_active].set_ylabel('Input ' + str(best_features[1]), fontsize=fontsize, rotation=150)
				axes[subplot_idx + columns_active].set_zlabel('Uncertainty: std state ' + str(index_observation_represent),
					fontsize=fontsize, rotation=60)

				if output_data is not None:
					predictions_data_outside_memory = models[index_observation_represent].likelihood(
						models[index_observation_represent](
							input_data[indexes_points_outside_gp_memory[index_observation_represent]]))
					errors_outside_memory = np.abs(predictions_data_outside_memory.mean.numpy() -
						output_data[indexes_points_outside_gp_memory[index_observation_represent], index_observation_represent])
					errors = np.zeros_like(input_data[:, 0])
					errors[indexes_points_outside_gp_memory[index_observation_represent]] = errors_outside_memory
					axes[subplot_idx + columns_active].scatter(
						input_data[indexes_points_in_gp_memory[index_observation_represent], best_features[0]],
						input_data[indexes_points_in_gp_memory[index_observation_represent], best_features[1]],
						errors[indexes_points_in_gp_memory[index_observation_represent]], marker='x', c='g')
					axes[subplot_idx + columns_active].scatter(
						input_data[indexes_points_outside_gp_memory[index_observation_represent], best_features[0]],
						input_data[indexes_points_outside_gp_memory[index_observation_represent], best_features[1]],
						errors_outside_memory, marker='x', c='k')
					axes[subplot_idx + columns_active].quiver(
						input_data[:-1, best_features[0]], input_data[:-1, best_features[1]],
						errors[:-1],
						input_data[1:, best_features[0]] - input_data[:-1, best_features[0]],
						input_data[1:, best_features[1]] - input_data[:-1, best_features[1]],
						errors[1:] - errors[:-1],
						color='k', linestyle="solid", alpha=0.3, arrow_length_ratio=0.001, length=0.9)

			plt.tight_layout()
			hour = datetime.datetime.now().hour
			minute = datetime.datetime.now().minute
			second = datetime.datetime.now().second
			fig.savefig(os.path.join(folder_save, 'model_3d' + '_h' + str(hour) + '_m' + str(minute)
															  + '_s' + str(second) + '.png'))
			plt.close()


def create_models(train_inputs, train_targets, parameters, constraints):
	num_models = len(train_targets[0])
	models = [ExactGPModelMonoTask(train_inputs, train_targets[:, idx_model], len(train_inputs[0]))
		for idx_model in range(num_models)]

	for idx_model in range(num_models):
		# register constraints on parameters
		if "min_std_noise" in constraints.keys():
			models[idx_model].likelihood.noise_covar.register_constraint("raw_noise",
				gpytorch.constraints.Interval(lower_bound=np.power(constraints['min_std_noise'], 2),
					upper_bound=np.power(constraints['max_std_noise'], 2)))
		if "min_outputscale" in constraints.keys():
			models[idx_model].covar_module.register_constraint("raw_outputscale",
				gpytorch.constraints.Interval(
					lower_bound=constraints['min_outputscale'],
					upper_bound=constraints['max_outputscale']))
		if "min_lengthscale" in constraints.keys():
			models[idx_model].covar_module.base_kernel.register_constraint("raw_lengthscale",
				gpytorch.constraints.Interval(
					lower_bound=constraints['min_lengthscale'],
					upper_bound=constraints['max_lengthscale']))
		# load parameters
		models[idx_model].load_state_dict(parameters[idx_model])
	return models