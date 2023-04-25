import datetime
import os

import torch
import gpytorch
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

from rl_gp_mpc.utils.utils import create_models

PROP_EXTEND_DOMAIN = 1
N_TICKS = 150
TOTAL_COL_MAX= 3
FONTSIZE = 6

def save_plot_model_3d_process(inputs, targets, parameters, constraints_gp, folder_save, idxs_in_gp_memory=None):
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

		num_figures = int(num_input_model / TOTAL_COL_MAX + 0.5)
		figs = []
		axes_s = []
		for index_figure in range(num_figures):
			fig = plt.figure(figsize=(15, 6))
			axes = []
			columns_active = np.min([num_models - (TOTAL_COL_MAX * index_figure), TOTAL_COL_MAX])
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

				domain_extension_x = (PROP_EXTEND_DOMAIN - 1) * (
						inputs[:, best_features[0]].max() - inputs[:, best_features[0]].min())
				domain_extension_y = (PROP_EXTEND_DOMAIN - 1) * (
						inputs[:, best_features[1]].max() - inputs[:, best_features[1]].min())
				x_grid = np.linspace(inputs[:, best_features[0]].min() - domain_extension_x / 2,
					inputs[:, best_features[0]].max() + domain_extension_x / 2, N_TICKS)
				y_grid = np.linspace(inputs[:, best_features[1]].min() - domain_extension_y / 2,
					inputs[:, best_features[1]].max() + domain_extension_y / 2, N_TICKS)

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
				surf1 = axes[idx_subplot].contour3D(x_grid, y_grid, z_grid_mean, N_TICKS, cmap='cool')
				axes[idx_subplot].set_xlabel('Input ' + str(best_features[0]), fontsize=FONTSIZE, rotation=150)
				axes[idx_subplot].set_ylabel('Input ' + str(best_features[1]), fontsize=FONTSIZE, rotation=150)
				axes[idx_subplot].set_zlabel('Variation state ' + str(idx_obs_repr), fontsize=FONTSIZE,
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

				surf2 = axes[idx_subplot + columns_active].contour3D(x_grid, y_grid, z_grid_std, N_TICKS, cmap='cool')
				axes[idx_subplot + columns_active].set_xlabel('Input ' + str(best_features[0]), fontsize=FONTSIZE, rotation=150)
				axes[idx_subplot + columns_active].set_ylabel('Input ' + str(best_features[1]), fontsize=FONTSIZE, rotation=150)
				axes[idx_subplot + columns_active].set_zlabel('Uncertainty: std state ' + str(idx_obs_repr),
					fontsize=FONTSIZE, rotation=60)

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