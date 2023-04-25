import datetime
import os

import matplotlib.pyplot as plt
import numpy as np


ALPHA_CONFIDENCE_BOUNDS = 0.4
UPDATE_X_LIMIT = True
MUL_STD_BOUNDS = 3


def draw_plots_2d(observations, actions, costs, pred_info, 
                state_min_constraint=None, state_max_constraint=None, 
                iter_ahead_show=3, fig=None, axes=None,
		):
	idxs_x = np.arange(0, len(observations))
	pred_iters = np.array(pred_info['iteration'])
	idxs_ctrl = pred_iters + random_actions_init
	try:
		states_pred_show = np.zeros((len(idxs_ctrl) - iter_ahead_show, observations.shape[1]))
		states_pred_show[:] = [pred[iter_ahead_show - 1].numpy() for pred in pred_info['predicted states']][:-iter_ahead_show]

		states_pred_std_show = np.ones((len(idxs_ctrl) - iter_ahead_show, observations.shape[1]))
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
	for state_idx in range(len(observations[0])):
		axes[0].plot(idxs_x, np.array(observations)[:, state_idx], label='state' + str(state_idx),
			color=plt.get_cmap('tab20').colors[2 * state_idx])
		if states_pred_show is not None:
			axes[0].plot(idxs_ctrl[iter_ahead_show:], states_pred_show[:, state_idx],
				label=str(iter_ahead_show * num_repeat_actions) + 'step_' + 'prediction' + str(state_idx),
				color=plt.get_cmap('tab20').colors[2 * state_idx], linestyle='--')
		if states_pred_std_show is not None:
			axes[0].fill_between(idxs_ctrl[iter_ahead_show:],
				states_pred_show[:, state_idx] - MUL_STD_BOUNDS * states_pred_std_show[:, state_idx],
				states_pred_show[:, state_idx] + MUL_STD_BOUNDS * states_pred_std_show[:, state_idx],
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
							mean_cost_pred - MUL_STD_BOUNDS * mean_cost_std_pred,
							mean_cost_pred + MUL_STD_BOUNDS * mean_cost_std_pred,
							color='orange', alpha=ALPHA_CONFIDENCE_BOUNDS)

	for idx_action in range(len(actions[0])):
		axes[1].step(idxs_x, np.array(actions)[:, idx_action], label='a_' + str(idx_action),
			color=plt.get_cmap('tab20').colors[1 + 2 * idx_action])

	if UPDATE_X_LIMIT is True:
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
	fig.savefig(os.path.join(folder_save, 'history' + datetime.datetime.now().strftime("%H_%M_%S")))
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