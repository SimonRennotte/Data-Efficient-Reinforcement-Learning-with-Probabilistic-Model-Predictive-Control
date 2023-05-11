import datetime
import os

import matplotlib.pyplot as plt
import numpy as np


from rl_gp_mpc.control_objects.controllers.iteration_info_class import IterationInformation

ALPHA_CONFIDENCE_BOUNDS = 0.4
UPDATE_X_LIMIT = True
MUL_STD_BOUNDS = 3


def save_plot_2d(states, actions, costs, model_iter_infos: "list[IterationInformation]", folder_save,
				use_constraints=False, state_min_constraint=None, state_max_constraint=None,
				iter_ahead_show=3):
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
							model_iter_infos=model_iter_infos, 
							use_constraints=use_constraints,
							state_min_constraint=state_min_constraint,
							state_max_constraint=state_max_constraint,
							iter_ahead_show=iter_ahead_show, fig=fig, axes=axes)
	axes[0].legend()
	axes[0].grid()
	axes[1].legend()
	axes[1].grid()
	axes[2].legend()
	axes[2].grid()
	fig.savefig(os.path.join(folder_save, f'history_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.png'))
	plt.close()


def draw_plots_2d(states, actions, costs, model_iter_infos: "list[IterationInformation]"=None, 
                use_constraints=False, state_min_constraint=None, state_max_constraint=None, 
                iter_ahead_show=15, fig=None, axes=None,
		):
	obs_idxs = np.arange(0, len(states))
	colors = plt.get_cmap('tab20').colors
	if model_iter_infos is not None:
		pred_idxs = np.array([info.iteration for info in model_iter_infos])
		pred_mean_costs = np.array([info.mean_predicted_cost for info in model_iter_infos])
		pred_mean_cost_std = np.array([info.mean_predicted_cost_std for info in model_iter_infos])

		states_pred_ahead_idxs = np.array([info.predicted_idxs[iter_ahead_show] for info in model_iter_infos])
		states_pred_ahead = np.array([info.predicted_states[iter_ahead_show] for info in model_iter_infos])
		states_pred_std_ahead = np.array([info.predicted_states_std[iter_ahead_show] for info in model_iter_infos])

	for state_idx in range(len(states[0])):
		axes[0].plot(obs_idxs, np.array(states)[:, state_idx], label='state' + str(state_idx),
			color=colors[2 * state_idx])
		if model_iter_infos is not None:
			axes[0].plot(states_pred_ahead_idxs, states_pred_ahead[:, state_idx],
				label=str(iter_ahead_show) + 'step_' + 'prediction' + str(state_idx),
				color=colors[2 * state_idx], linestyle='--')

			axes[0].fill_between(states_pred_ahead_idxs,
				states_pred_ahead[:, state_idx] - MUL_STD_BOUNDS * states_pred_std_ahead[:, state_idx],
				states_pred_ahead[:, state_idx] + MUL_STD_BOUNDS * states_pred_std_ahead[:, state_idx],
				color=colors[2 * state_idx], alpha=ALPHA_CONFIDENCE_BOUNDS)

		if use_constraints and (state_min_constraint is not None) and (state_max_constraint is not None):
			axes[0].axhline(y=state_min_constraint[state_idx],
				color=colors[2 * state_idx], linestyle='-.')
			axes[0].axhline(y=state_max_constraint[state_idx],
				color=colors[2 * state_idx], linestyle='-.')

	axes[2].plot(obs_idxs, costs, label='cost', color='k')

	axes[2].plot(pred_idxs, pred_mean_costs, label='mean predicted cost', color='orange')
	axes[2].fill_between(pred_idxs,
							pred_mean_costs - MUL_STD_BOUNDS * pred_mean_cost_std,
							pred_mean_costs + MUL_STD_BOUNDS * pred_mean_cost_std,
							color='orange', alpha=ALPHA_CONFIDENCE_BOUNDS)

	for idx_action in range(len(actions[0])):
		axes[1].step(obs_idxs, np.array(actions)[:, idx_action], label='a_' + str(idx_action),
			color=colors[1 + 2 * idx_action])

	if UPDATE_X_LIMIT is True:
		axes[2].set_xlim(0, np.max(obs_idxs))
	
	axes[2].set_ylim(0, np.max([np.max(pred_mean_costs), np.max(costs)]) * 1.2)
	plt.xlabel('Env steps')
	return fig, axes