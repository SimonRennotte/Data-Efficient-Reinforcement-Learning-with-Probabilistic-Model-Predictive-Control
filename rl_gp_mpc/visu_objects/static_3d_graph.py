import os
import datetime

import torch
import gpytorch
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

from rl_gp_mpc.control_objects.models.gp_model import create_models, SavedState
from rl_gp_mpc.control_objects.memories.gp_memory import Memory

PROP_EXTEND_DOMAIN = 1
N_TICKS = 75
TOTAL_COL_MAX= 3
FONTSIZE = 6

def save_plot_model_3d(saved_state: SavedState, folder_save, memory:Memory, plot_points_memory=True):
	models = create_models(saved_state.parameters, saved_state.constraints_hyperparams, saved_state.inputs, saved_state.states_change)
	inputs_total, targets_total = memory.get_memory_total()
	active_memory_mask = memory.get_mask_model_inputs()
	inputs_model = inputs_total[active_memory_mask]
	targets_model = targets_total[active_memory_mask]
	with torch.no_grad(), gpytorch.settings.fast_pred_var():
		torch.set_num_threads(1)
		num_input_model = len(inputs_model[0])
		num_models = len(targets_model[0])
		for idx_model in range(len(models)):
			models[idx_model].eval()
		targets_model = targets_model.numpy()

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
				if idx_obs_repr >= num_models:
					break
				feat_importance = (1 / models[idx_obs_repr].covar_module.base_kernel.lengthscale).numpy()
				feat_importance = feat_importance / np.sum(feat_importance)
				best_features = np.argsort(-feat_importance)[0, :2]
				estimator_other_columns = Pipeline(
					steps=[('standardscaler', StandardScaler()),
						('features', KNeighborsRegressor(n_neighbors=3, weights='distance'))])

				estimator_other_columns.fit(inputs_total[:, best_features].numpy(),
					np.delete(inputs_total.numpy(), best_features, axis=1))

				domain_extension_x = (PROP_EXTEND_DOMAIN - 1) * (
						inputs_total[:, best_features[0]].max() - inputs_total[:, best_features[0]].min())
				domain_extension_y = (PROP_EXTEND_DOMAIN - 1) * (
						inputs_total[:, best_features[1]].max() - inputs_total[:, best_features[1]].min())
				x_grid = np.linspace(inputs_total[:, best_features[0]].min() - domain_extension_x / 2,
					inputs_total[:, best_features[0]].max() + domain_extension_x / 2, N_TICKS)
				y_grid = np.linspace(inputs_total[:, best_features[1]].min() - domain_extension_y / 2,
					inputs_total[:, best_features[1]].max() + domain_extension_y / 2, N_TICKS)

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
				
				surf2 = axes[idx_subplot + columns_active].contour3D(x_grid, y_grid, z_grid_std, N_TICKS, cmap='cool')
				axes[idx_subplot + columns_active].set_xlabel('Input ' + str(best_features[0]), fontsize=FONTSIZE, rotation=150)
				axes[idx_subplot + columns_active].set_ylabel('Input ' + str(best_features[1]), fontsize=FONTSIZE, rotation=150)
				axes[idx_subplot + columns_active].set_zlabel('Uncertainty: std state ' + str(idx_obs_repr),
					fontsize=FONTSIZE, rotation=60)

				if plot_points_memory:
					axes[idx_subplot].scatter(
					inputs_total[~active_memory_mask, best_features[0]],
					inputs_total[~active_memory_mask, best_features[1]],
					targets_total[~active_memory_mask, idx_obs_repr],
					marker='x', c='k')

					axes[idx_subplot].scatter(
					inputs_model[:, best_features[0]],
					inputs_model[:, best_features[1]],
					targets_model[:, idx_obs_repr],
					marker='x', c='g')

					axes[idx_subplot].quiver(inputs_total[:-1, best_features[0]], inputs_total[:-1, best_features[1]],
						targets_total[:-1, idx_obs_repr],
						inputs_total[1:, best_features[0]] - inputs_total[:-1, best_features[0]],
						inputs_total[1:, best_features[1]] - inputs_total[:-1, best_features[1]],
						targets_total[1:, idx_obs_repr] - targets_total[:-1, idx_obs_repr],
						color='k', linestyle="solid", alpha=0.3, arrow_length_ratio=0.001, length=0.9)

					preds_total = models[idx_obs_repr].likelihood(models[idx_obs_repr](inputs_total))
					errors_total = np.abs(preds_total.mean.numpy() - targets_total[:, idx_obs_repr].numpy())
					axes[idx_subplot + columns_active].scatter(
						inputs_total[~active_memory_mask, best_features[0]],
						inputs_total[~active_memory_mask, best_features[1]],
						errors_total[~active_memory_mask], marker='x', c='k')
					axes[idx_subplot + columns_active].scatter(
						inputs_total[active_memory_mask, best_features[0]],
						inputs_total[active_memory_mask, best_features[1]],
						errors_total[active_memory_mask], marker='x', c='g')
					axes[idx_subplot + columns_active].quiver(
						inputs_total[:-1, best_features[0]], inputs_total[:-1, best_features[1]],
						errors_total[:-1],
						inputs_total[1:, best_features[0]] - inputs_total[:-1, best_features[0]],
						inputs_total[1:, best_features[1]] - inputs_total[:-1, best_features[1]],
						errors_total[1:] - errors_total[:-1],
						color='k', linestyle="solid", alpha=0.3, arrow_length_ratio=0.001, length=0.9)

			plt.tight_layout()
			fig.savefig(os.path.join(folder_save, f'model_3d_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.png'))
			plt.close()