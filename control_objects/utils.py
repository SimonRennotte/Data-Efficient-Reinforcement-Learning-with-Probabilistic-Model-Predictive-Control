import datetime
import os

import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits

from .gp_models import ExactGPModelMonoTask


def save_plot_history_process(states, actions, states_next, prediction_info_over_time, output_dir, env_to_control=''):
	with threadpool_limits(limits=1, user_api='blas'), threadpool_limits(limits=1, user_api='openmp'):
		indexes = np.arange(len(states))
		fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 6))
		axes[0][0].set_title('Normed states')
		axes[0][1].set_title('Normed actions')
		axes[1][0].set_title('Normed errors and predicted uncertainty')
		axes[1][1].set_title('Cost and horizon cost')
		indexes_control = prediction_info_over_time['iteration']
		predictions_over_time_show = np.zeros_like(states_next)
		predictions_over_time_show[indexes_control] = \
			[prediction[0].numpy() for prediction in prediction_info_over_time['predicted states']]
		predictions_std_over_time_show = np.zeros_like(states_next)
		predictions_std_over_time_show[indexes_control] = \
			[prediction[0].numpy() for prediction in prediction_info_over_time['predicted states std']]
		mean_cost_traj = np.array([element for element in prediction_info_over_time['mean cost trajectory']])
		mean_cost_traj_std = np.array([element for element in prediction_info_over_time['mean cost trajectory std']])

		for state_idx in range(len(states[0])):
			axes[0][0].plot(indexes, np.array(states)[:, state_idx], label='s' + str(state_idx),
				color=plt.get_cmap('tab20').colors[2 * state_idx])
			axes[1][0].plot(indexes, np.abs(np.array(states_next)[:, state_idx] - predictions_over_time_show[:, state_idx]),
				label='ep' + str(state_idx), color=plt.get_cmap('tab20').colors[2 * state_idx])
			axes[1][0].fill_between(indexes, 0, 2 * predictions_std_over_time_show[:, state_idx],
				label='p_2std' + str(state_idx),
				color=plt.get_cmap('tab20').colors[2 * state_idx + 1], alpha=0.6)
		axes[1][1].plot(indexes_control,
			np.array([element for element in prediction_info_over_time['cost']]), label='cost', color='k')
		axes[1][1].plot(indexes_control, mean_cost_traj, label='cost trajectory', color='orange')
		axes[1][1].fill_between(indexes_control, mean_cost_traj - 2 * mean_cost_traj_std,
			mean_cost_traj + 3 * mean_cost_traj_std, label='cost trajectory 3 std', alpha=0.6)

		for action_idx in range(len(actions[0])):
			axes[0][1].plot(indexes, np.array(actions)[:, action_idx], label='a_' + str(action_idx))

		plt.xlabel("time")
		axes[0][0].legend()
		axes[1][0].legend()
		axes[0][1].legend()
		axes[1][0].set_ylim(0, 0.05)
		axes[1][1].legend()
		axes[1][1].set_ylim(0, 1)
		axes[1][1].set_xlim(0, len(indexes))
		plt.tight_layout()
		year = datetime.datetime.now().year
		month = datetime.datetime.now().month
		day = datetime.datetime.now().day
		hour = datetime.datetime.now().hour
		minute = datetime.datetime.now().minute
		second = datetime.datetime.now().second
		folder_save = os.path.join(output_dir, env_to_control, str(year) + '_m' + str(month) + '_d' + str(day))
		if not os.path.exists(folder_save):
			os.makedirs(folder_save)
		fig.savefig(os.path.join(folder_save, 'history' + '_h' + str(hour) + '_m' + str(minute)
											  + '_s' + str(second) + '.png'))
		plt.close()


def save_plot_model_3d_process(input_data, output_data, train_inputs, train_targets, parameters, constraints,
		indexes_points_in_gp_memory=None,
		prop_extend_domain=1, n_ticks=100, total_col_max=3, fontsize=6, output_dir='models_3d', env_to_control=''):
	with threadpool_limits(limits=1, user_api='blas'), \
			threadpool_limits(limits=1, user_api='openmp'), torch.no_grad(), gpytorch.settings.fast_pred_var():
		torch.set_num_threads(1)
		num_input_model = len(input_data[0])
		num_models = len(output_data[0])
		output_data = output_data.numpy()
		models = [ExactGPModelMonoTask(train_inputs, train_targets[idx_model], len(train_inputs[0]))
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
			models[idx_model].eval()

		num_figures = int(num_input_model / total_col_max + 0.5)
		figs = []
		axes_s = []
		for index_figure in range(num_figures):
			fig = plt.figure(figsize=(15, 6))
			axes = []
			for index_ax in range(total_col_max * 2):
				axes.append(fig.add_subplot(2, total_col_max, index_ax + 1, projection='3d'))
			axes_s.append(axes)
			figs.append(fig)
			for subplot_idx in range(total_col_max):
				index_observation_represent = index_figure * total_col_max + subplot_idx
				if index_observation_represent >= (output_data.shape[1]):
					break
				features_importance = (1 / models[index_observation_represent].covar_module.base_kernel.lengthscale).numpy()
				features_importance = features_importance / np.sum(features_importance)
				best_features = np.argsort(-features_importance)[0, :2]
				estimator_other_columns = Pipeline(
					steps=[('standardscaler', StandardScaler()), ('model', LinearRegression())])

				estimator_other_columns.fit(input_data[:, best_features].numpy(), input_data.numpy())

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

				gauss_pred = models[index_observation_represent].likelihood(
					models[index_observation_represent](
						torch.from_numpy(
				estimator_other_columns.predict(
					np.concatenate((x_grid_1d, y_grid_1d), axis=1)).astype(np.float64))))
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
					axes[subplot_idx].scatter(input_data[indexes_points_in_gp_memory, best_features[0]],
						input_data[indexes_points_in_gp_memory, best_features[1]],
						output_data[indexes_points_in_gp_memory, index_observation_represent], marker='x', c='g')

					axes[subplot_idx].scatter(np.delete(input_data[:, best_features[0]], indexes_points_in_gp_memory),
						np.delete(input_data[:, best_features[1]], indexes_points_in_gp_memory),
						np.delete(output_data[:, index_observation_represent], indexes_points_in_gp_memory),
						marker='x', c='k')

					axes[subplot_idx].quiver(input_data[:-1, best_features[0]], input_data[:-1, best_features[1]],
						output_data[:-1, index_observation_represent],
						input_data[1:, best_features[0]] - input_data[:-1, best_features[0]],
						input_data[1:, best_features[1]] - input_data[:-1, best_features[1]],
						output_data[1:, index_observation_represent] - output_data[:-1, index_observation_represent],
						color='k', linestyle="solid", alpha=0.3, arrow_length_ratio=0.001, length=0.9)

				surf2 = axes[subplot_idx + total_col_max].contour3D(x_grid, y_grid, z_grid_std, n_ticks, cmap='cool')
				axes[subplot_idx + total_col_max].set_xlabel('Input ' + str(best_features[0]), fontsize=fontsize, rotation=150)
				axes[subplot_idx + total_col_max].set_ylabel('Input ' + str(best_features[1]), fontsize=fontsize, rotation=150)
				axes[subplot_idx + total_col_max].set_zlabel('Errors and std state ' + str(index_observation_represent),
					fontsize=fontsize, rotation=60)

				if output_data is not None:
					predictions_data = models[index_observation_represent].likelihood(
						models[index_observation_represent](input_data))
					mean_prediction = predictions_data.mean.numpy()
					errors = np.abs(mean_prediction - output_data[:, index_observation_represent])
					axes[subplot_idx + total_col_max].scatter(
						input_data[indexes_points_in_gp_memory, best_features[0]],
						input_data[indexes_points_in_gp_memory, best_features[1]],
						errors[indexes_points_in_gp_memory], marker='x', c='g')
					axes[subplot_idx + total_col_max].scatter(
						np.delete(input_data[:, best_features[0]], indexes_points_in_gp_memory),
						np.delete(input_data[:, best_features[1]], indexes_points_in_gp_memory),
						np.delete(errors, indexes_points_in_gp_memory), marker='x', c='k')
					axes[subplot_idx + total_col_max].quiver(
						input_data[:-1, best_features[0]], input_data[:-1, best_features[1]],
						errors[:-1],
						input_data[1:, best_features[0]] - input_data[:-1, best_features[0]],
						input_data[1:, best_features[1]] - input_data[:-1, best_features[1]],
						errors[1:] - errors[:-1],
						color='k', linestyle="solid", alpha=0.3, arrow_length_ratio=0.001, length=0.9)

			plt.tight_layout()
			year = datetime.datetime.now().year
			month = datetime.datetime.now().month
			day = datetime.datetime.now().day
			hour = datetime.datetime.now().hour
			minute = datetime.datetime.now().minute
			second = datetime.datetime.now().second
			folder_save = os.path.join(output_dir, env_to_control, str(year) + '_m' + str(month) + '_d' + str(day))
			if not os.path.exists(folder_save):
				os.makedirs(folder_save)
			fig.savefig(os.path.join(folder_save, 'model_3d' + '_h' + str(hour) + '_m' + str(minute)
															  + '_s' + str(second) + '.png'))
			plt.close()
