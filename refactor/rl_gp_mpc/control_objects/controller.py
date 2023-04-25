import time
import multiprocessing

import numpy as np
import torch
import gpytorch
from scipy.optimize import minimize
from scipy.optimize import dual_annealing

from rl_gp_mpc.control_objects.utils.abstract_control_obj import BaseControllerObject
from rl_gp_mpc.utils.utils import create_models, get_init_action_change, get_init_action
from rl_gp_mpc.config_classes.config import Config
from rl_gp_mpc.control_objects.utils.iteration_info_class import IterationInformation
from rl_gp_mpc.control_objects.utils.math_utils import calculate_factorizations, compute_cost, compute_cost_terminal, compute_cost_unnormalized
from rl_gp_mpc.control_objects.utils.math_utils import to_normed_action_tensor, to_normed_obs_tensor, to_normed_var_tensor, denorm_action

# precision double tensor necessary for the gaussian processes predictions


class GpMpcController(BaseControllerObject):
	def __init__(self, observation_space, action_space, config:Config):

		self.config = config
		self.folder_save = 'myfoldersave'
		self.observation_state_mapper = None
		self.state_reward_mapper = None
		self.memory = None
		self.model = None
		self.actions_mapper = None

		super().__init__(observation_space, action_space, self.config)
		self.actions_init_optim = None
		
		self.action_idx = 0
		self.iter_info_dict = None

		self.models = create_models(
								train_inputs=None, 
								train_targets=None, 
								params=config.model.gp_init,
								constraints_gp=self.config.model.__dict__, 
								num_models=self.obs_space.shape[0], 
								num_inputs=self.num_inputs
							)

		for idx_model in range(len(self.models)):
			self.models[idx_model].eval()

		self.num_cores_main = multiprocessing.cpu_count()
		self.ctx = multiprocessing.get_context('spawn')
		self.queue_train = self.ctx.Queue()

		self.n_iter_ctrl = 0
		self.n_iter_obs = 0

		self.info_iters = {}
		self.idxs_mem_gp = []

	def compute_action(self, obs_mu, obs_var=None):
		"""
		Get the optimal action given the observation by optimizing
		the actions of the simulated trajectory with the gaussian process models such that the lower confidence bound of
		the mean cost of the trajectory is minimized.
		Only the first action of the prediction window is returned.

		Args:
			obs_mu (numpy.array): unnormalized observation from the gym environment. dim=(Ns)
			obs_var (numpy.array): unnormalized variance of the observation from the gym environment. dim=(Ns, Ns).
									default=None. If it is set to None,
									the observation noise from the json parameters will be used for every iteration.
									Ns is the dimension of states in the gym environment.

		Returns:
			action_denorm (numpy.array): action to use in the gym environment.
										It is denormalized, so it can be used directly.
										dim=(Na), where Ns is the dimension of the action_space
			info_dict (dict): contains all additional information about the iteration.
							Keys:
							- iteration (int): index number of the iteration
							- state (torch.Tensor): current normed state (before applying the action)
							- predicted states (torch.Tensor): mean value of the predicted distribution of the
																normed states in the mpc
							- predicted states std (torch.Tensor): predicted normed standard deviation of the
																	distribution of the states in the mpc
							- predicted actions (torch.Tensor): predicted optimal normed actions that minimize
																the long term cost in the mpc
							cost (float): mean value of the current cost distribution
							cost std (float): standard deviation of the current cost distribution
							predicted costs (torch.Tensor): mean value of the predicted cost distribution in the mpc
							predicted costs std (torch.Tensor): standard deviation of the
																predicted cost distribution in the mpc
							mean predicted cost (float): mean value of the predicted cost distribution in the mpc,
																averaged over future predicted time steps
							mean predicted cost std (float): standard deviation of the predicted cost distribution in the mpc,
																averaged over future predicted time steps
							lower bound mean predicted cost (float): lower bound of the predicted cost distribution
												(cost_mean_future_mean - self.exploration_factor * cost_std_future_mean).
												It is the value minimized by the mpc.

		"""
		# Check for parallel process that are open but not alive at each iteration to retrieve the results and close them
		self.check_and_close_processes()
		torch.set_num_threads(self.num_cores_main)

		with torch.no_grad():
			obs_mu_normed = to_normed_obs_tensor(obs=obs_mu, low=self.obs_space.low, high=self.obs_space.high)
			if obs_var is None:
				obs_var_norm = self.config.observation.obs_var_norm
			else:
				obs_var_norm = to_normed_var_tensor(obs_var=obs_var, low=self.obs_space.low, high=self.obs_space.high)
			# iK and beta are computed outside of the optimization function since it depends only on the points in memory,
			# and not on the input. Otherwise, the optimization time at each iteration would be too high

			self.iK, self.beta = calculate_factorizations(self.x[self.idxs_mem_gp], self.y[self.idxs_mem_gp], self.models)
			# The initial actions_norm values are fixed using the actions_norm predictions of the mpc of the previous iteration,
			# offset by 1, so that the initial values have a correct guess, which allows to get good results
			# by using only 1 to 3 iteration of the action optimizer at each iteration.
			# The value of the last init value of action in the prediction window is set as the same as
			# the last of the prevous iteration.

		# The optimize function from the scipy library.
		# It is used to get the optimal actions_norm in the prediction window
		# that minimizes the lower bound of the predicted cost. The jacobian is used,
		# otherwise the computation times would be 5 to 10x slower (for the tests I used)
		time_start_optim = time.time()
		opt_fun = np.inf
		for idx_restart in range(self.config.controller.restarts_optim):
			if self.config.controller.optimize:
				iter_res = minimize(fun=self.compute_mean_lcb_trajectory,
					x0=self.get_action_optim_init(),
					jac=True,
					args=(obs_mu_normed, obs_var_norm),
					method='L-BFGS-B',
					bounds=self.config.actions.bounds,
					options=self.config.controller.actions_optimizer_params)
				actions = iter_res.x
				func_val = iter_res.fun
			else:
				actions = self.get_action_optim_init()
				func_val, grad_val = self.compute_mean_lcb_trajectory(actions, obs_mu_normed, obs_var_norm)

			if func_val < opt_fun:
				opt_fun = func_val
				actions_optim = actions
		time_end_optim = time.time()
		print("Optimisation time for iteration: %.3f s" % (time_end_optim - time_start_optim))

		actions_norm = actions_optim.reshape(self.config.controller.len_horizon, -1)
		# prepare init values for the next iteration
		self.actions_init_optim = actions_norm.copy()

		with torch.no_grad():
			if self.config.actions.limit_action_change:
				actions_norm[0] += np.array(self.action_previous_iter)
				actions_norm = np.clip(np.cumsum(actions_norm, axis=0), 0, 1)
				action_next = actions_norm[0]
				self.action_previous_iter = torch.Tensor(action_next)
			else:
				action_next = actions_norm[0]

			actions_norm = torch.Tensor(actions_norm)
			action = denorm_action(action_next, low=self.action_space.low, high=self.action_space.high)

			cost, cost_var = self.compute_cost(obs_mu_normed, obs_var_norm, actions_norm[0])
			# states_denorm = self.states_mu_pred[1:] * \
			# (self.observation_space.high - self.observation_space.low) + self.observation_space.low
			# states_std_denorm = states_std_pred * (self.observation_space.high - self.observation_space.low)
			states_std_pred = torch.diagonal(self.states_var_pred, dim1=-2, dim2=-1).sqrt()
			self.iter_info = IterationInformation(
								iteration=self.n_iter_ctrl,
								state=self.mu_states_pred[0],
								predicted_states=self.mu_states_pred[1:],
								predicted_states_std=states_std_pred[1:],
								predicted_actions=actions_norm,
								cost=cost.item(),
								cost_std=cost_var.sqrt().item(),
								predicted_costs=self.costs_trajectory[1:],
								predicted_costs_std=self.costs_traj_var[1:].sqrt(),
								mean_predicted_cost=np.min([self.costs_trajectory[1:].mean().item(), 3]),
								mean_predicted_cost_std=self.costs_traj_var[1:].sqrt().mean().item(),
								lower_bound_mean_predicted_cost=self.cost_traj_mean_lcb.item()
							)
			self.store_iter_info()
			self.n_iter_ctrl += self.config.controller.num_repeat_actions
			return action

	def get_action_optim_init(self):
		if self.actions_init_optim is None or not(self.config.controller.init_from_previous_actions):
			if self.config.actions.limit_action_change:
				self.actions_init_optim = get_init_action_change(
														len_horizon=self.config.controller.len_horizon, 
														max_change_action_norm=self.config.actions.max_change_action_norm
													)
			else:
				self.actions_init_optim = get_init_action(self.config.controller.len_horizon, self.num_actions)
			
			init_actions_optim = self.actions_init_optim
		else:
			init_actions_optim = (np.concatenate((self.actions_init_optim[1:],
									np.expand_dims(self.actions_init_optim[-1], 0)), axis=0))
		# See comment in __init__ above the definition of bounds for more information about limit_action change trick
		# Actions in the minimize function fo scipy must be a 1d vector.
		# If the action is multidimensional, it is resized to a 1d array and passed into the minimize function as
		# a 1d array. The init values and bounds must match the dimension of the passed array.
		# It is reshaped inside the minimize function to get back the true dimensions
		if self.config.actions.limit_action_change:
			init_actions_optim_absolute = np.empty_like(init_actions_optim)
			init_actions_optim_absolute[0] = self.action_previous_iter
			init_actions_optim_absolute += init_actions_optim
			init_actions_optim_absolute = np.cumsum(init_actions_optim_absolute, axis=0)
			if np.logical_or(np.any(init_actions_optim_absolute > 1),
					np.any(init_actions_optim_absolute < 0)):
				for idx_time in range(1, len(init_actions_optim)):
					init_actions_optim_absolute[idx_time] = init_actions_optim_absolute[idx_time - 1] + \
																init_actions_optim[idx_time]
					indexes_above_1 = np.nonzero(init_actions_optim_absolute[idx_time] > 1)[0]
					indexes_under_0 = np.nonzero(init_actions_optim_absolute[idx_time] < 0)[0]
					init_actions_optim[idx_time][indexes_above_1] = \
						1 - init_actions_optim_absolute[idx_time - 1][indexes_above_1]
					init_actions_optim[idx_time][indexes_under_0] = \
						- init_actions_optim_absolute[idx_time - 1][indexes_under_0]
					init_actions_optim_absolute[idx_time][indexes_above_1] = 1
					init_actions_optim_absolute[idx_time][indexes_under_0] = 0

		init_actions_optim = init_actions_optim.flatten()
		return init_actions_optim
	
	def get_iter_info(self):
		return self.iter_info

	def store_iter_info(self):
		iter_info_dict = self.iter_info.__dict__
		for key in iter_info_dict.keys():
			if not key in self.info_iters:
				self.info_iters[key] = [iter_info_dict[key]]
			else:
				self.info_iters[key].append(iter_info_dict[key])

	def compute_mean_lcb_trajectory(self, actions, obs_mu, obs_var):
		"""
		Compute the mean lower bound cost of a trajectory given the actions of the trajectory
		and initial state distribution. The gaussian process models are used to predict the evolution of
		states (mean and variance). Then the cost is computed for each predicted state and the mean is returned.
		The partial derivatives of the mean lower bound cost with respect to the actions are also returned.
		They are computed automatically with autograd from pytorch.
		This function is called multiple times by an optimizer to find the optimal actions.

		Args:
			actions (numpy.array): actions to apply for the simulated trajectory.
									It is a flat 1d array, whatever the dimension of actions
									so that this function can be used by the minimize function of the scipy library.
									It is reshaped and transformed into a tensor inside.
									If self.limit_action_change is true, each element of the array contains the relative
									change with respect to the previous iteration, so that the change can be bounded by
									the optimizer. dim=(Nh x Na,)
									where Nh is the len of the horizon and Na the dimension of actions

			obs_mu (torch.Tensor):	mean value of the inital state distribution.
									dim=(Ns,) where Ns is the dimension of state

			obs_var (torch.Tensor): covariance matrix of the inital state distribution.
									dim=(Ns, Ns) where Ns is the dimension of state

			iK (torch.Tensor): intermediary result for the gp predictions that only depends on the points in memory
								and not on the points to predict.
								It is computed outside the optimization function in self.calculate_factorizations
								for more efficient predictions. Dim=(Ns, Np, Np)
								where Ns is the dimension of state and Np the number of points in gp memory

			beta (torch.Tensor): intermediary result for the gp predictions that only depends on the points in memory
								and not on the points to predict.
								It is computed outside the optimization function in self.calculate_factorizations
								for more efficient predictions. Dim=(Ns, Np)

		Returns:
			mean_cost_traj_lcb.item() (float): lower bound of the mean cost distribution
														of the predicted trajectory.


			gradients_dcost_dactions[:, 0].detach().numpy() (numpy.array):
																Derivative of the lower bound of the mean cost
																distribution with respect to each of the actions in the
																prediction horizon. Dim=(Nh,)
																where Nh is the len of the horizon
		"""
		# reshape actions from flat 1d numpy array into 2d tensor
		actions = np.atleast_2d(actions.reshape(self.config.controller.len_horizon, -1))
		actions = torch.Tensor(actions)
		actions.requires_grad = True
		# If limit_action_change is true, actions are transformed back into absolute values from relative change
		if self.config.actions.limit_action_change:
			actions_input = actions.clone()
			actions_input[0] = self.action_previous_iter + actions_input[0]
			actions_input = torch.clamp(torch.cumsum(actions_input, dim=0), 0, 1)
		else:
			actions_input = actions
		mu_states_pred, s_states_pred, costs_traj, costs_traj_var, costs_traj_lcb = self.predict_trajectory(actions_input, obs_mu, obs_var)
		if self.config.controller.clip_lower_bound_cost_to_0:
			costs_traj_lcb = torch.clamp(costs_traj_lcb, 0, np.inf)
		mean_cost_traj_lcb = costs_traj_lcb.mean()
		gradients_dcost_dactions = torch.autograd.grad(mean_cost_traj_lcb, actions, retain_graph=False)[0]

		self.cost_traj_mean_lcb = mean_cost_traj_lcb.detach()
		self.mu_states_pred = mu_states_pred.detach()
		self.costs_trajectory = costs_traj.detach()
		self.states_var_pred = s_states_pred.detach()
		self.costs_traj_var = costs_traj_var.detach()

		return mean_cost_traj_lcb.item(), gradients_dcost_dactions.flatten().detach().numpy()

	def predict_trajectory(self, actions, obs_mu, obs_var):
		"""
		Compute the future predicted states distribution for the simulated trajectory given the
		current initial state (or observation) distribution (obs_mu and obs_var) and planned actions
		It also returns the costs, the variance of the costs, and the lower confidence bound of the cost
		along the trajectory

		Args:
			actions (torch.Tensor): actions to apply for the simulated trajectory. dim=(Nh, Na)
									where Nh is the len of the horizon and Na the dimension of actions

			obs_mu (torch.Tensor):	mean value of the inital state distribution.
									dim=(Ns,) where Ns is the dimension of state

			obs_var (torch.Tensor): variance matrix of the inital state distribution.
									dim=(Ns, Ns) where Ns is the dimension of state

			iK (torch.Tensor): intermediary result for the gp predictions that only depends on the points in memory
								and not on the points to predict.
								It is computed outside the optimization function in calculate_factorizations
								for more efficient predictions. Dim=(Ns, Np, Np)
								where Ns is the dimension of state and Np the number of points in gp memory

			beta (torch.Tensor): intermediary result for the gp predictions that only depends on the points in memory
								and not on the points to predict.
								It is computed outside the optimization function in calculate_factorizations
								for more efficient predictions. Dim=(Ns, Np)

		Returns:
			states_mu_pred (torch.Tensor): predicted states of the trajectory.
											The first element contains the initial state.
											Dim=(Nh + 1, Ns)

			states_var_pred (torch.Tensor): covariance matrix of the predicted states of the trajectory.
											The first element contains the initial state.
											Dim=(Nh + 1, Ns, Ns)

			costs_traj (torch.Tensor): costs of the predicted trajectory. Dim=(Nh,)

			costs_traj_var (torch.Tensor): variance of the costs of the predicted trajectory. Dim=(Nh,)

			costs_traj_lcb (torch.Tensor): lower confidence bound of the costs of the predicted trajectory.
													Dim=(Nh,)

			where Nh: horizon length, Ns: dimension of states, Na: dimension of actions, Np:number of points in gp memory
		"""
		states_mu_pred = torch.empty((self.config.controller.len_horizon + 1, len(obs_mu)))
		states_var_pred = torch.empty((self.config.controller.len_horizon + 1, self.num_states, self.num_states))
		states_mu_pred[0] = obs_mu
		states_var_pred[0] = obs_var
		state_dim = obs_mu.shape[0]
		# Input of predict_next_state_change is not a state, but the concatenation of state and action
		for idx_time in range(1, self.config.controller.len_horizon + 1):
			input_var = torch.zeros((self.num_inputs, self.num_inputs))
			input_var[:state_dim, :state_dim] = states_var_pred[idx_time - 1]
			input_mean = torch.empty((self.num_inputs,))
			input_mean[:self.num_states] = states_mu_pred[idx_time - 1]
			input_mean[self.num_states:(self.num_states + self.num_actions)] = actions[idx_time - 1]
			if self.config.model.include_time_model:
				input_mean[-1] = self.n_iter_obs + idx_time - 1
			state_change, state_change_var, v = self.predict_next_state_change(input_mean, input_var, self.iK, self.beta)
			# use torch.clamp(states_mu_pred[idx_time], 0, 1) ?
			states_mu_pred[idx_time] = states_mu_pred[idx_time - 1] + state_change
			states_var_pred[idx_time] = state_change_var + states_var_pred[idx_time - 1] + \
									  input_var[:states_var_pred.shape[1]] @ v + \
									  v.t() @ input_var[:states_var_pred.shape[1]].t()

		costs_traj, costs_traj_var = self.compute_cost(states_mu_pred[:-1],
			states_var_pred[:-1], actions)
		cost_traj_final, costs_traj_var_final = self.compute_cost_terminal(states_mu_pred[-1],
			states_var_pred[-1])
		costs_traj = torch.cat((costs_traj, cost_traj_final[None]), 0)
		costs_traj_var = torch.cat((costs_traj_var, costs_traj_var_final[None]), 0)
		costs_traj_lcb = costs_traj - self.config.reward.exploration_factor * torch.sqrt(costs_traj_var)
		return states_mu_pred, states_var_pred, costs_traj, costs_traj_var, costs_traj_lcb

	def predict_next_state_change(self, input_mu, input_var, iK, beta):
		"""
		Approximate GP regression at noisy inputs via moment matching
		IN: mean (m) (row vector) and (s) variance of the state
		OUT: mean (M) (row vector), variance (S) of the action and inv(s)*input-ouputcovariance
		Function inspired from
		https://github.com/nrontsis/PILCO/blob/6a962c8e4172f9e7f29ed6e373c4be2dd4b69cb7/pilco/models/mgpr.py#L81,
		reinterpreted from tensorflow to pytorch
		Args:
			input_mu (torch.Tensor): mean value of the input distribution. Dim=(Ns + Na,)

			input_var (torch.Tensor): covariance matrix of the input distribution. Dim=(Ns + Na, Ns + Na)

		Returns:
			M.t() (torch.Tensor): mean value of the predicted change distribution. Dim=(Ns,)

			S (torch.Tensor): covariance matrix of the predicted change distribution. Dim=(Ns, Ns)

			V.t() (torch.Tensor): Dim=(Ns, Ns + Na)

			where Ns: dimension of state, Na: dimension of action
		"""
		input_var = input_var[None, None, :, :].repeat([self.num_states, self.num_states, 1, 1])
		inp = (self.x[self.idxs_mem_gp[:beta.shape[1]]] - input_mu)[None, :, :].repeat([self.num_states, 1, 1])
		lengthscales = torch.stack([model.covar_module.base_kernel.lengthscale[0] for model in self.models])
		variances = torch.stack([model.covar_module.outputscale for model in self.models])
		# Calculate M and V: mean and inv(s) times input-output covariance
		iL = torch.diag_embed(1 / lengthscales)
		iN = inp @ iL
		B = iL @ input_var[0, ...] @ iL + torch.eye(self.num_inputs)

		# Redefine iN as in^T and t --> t^T
		# B is symmetric, so it is equivalent
		t = torch.transpose(torch.solve(torch.transpose(iN, -1, -2), B).solution, -1, -2)

		lb = torch.exp(-torch.sum(iN * t, -1) / 2) * beta
		tiL = t @ iL
		c = variances / torch.sqrt(torch.det(B))

		M = (torch.sum(lb, -1) * c)[:, None]
		V = torch.matmul(torch.transpose(tiL.conj(), -1, -2), lb[:, :, None])[..., 0] * c[:, None]

		# Calculate S: Predictive Covariance
		R = torch.matmul(input_var, torch.diag_embed(
			1 / torch.square(lengthscales[None, :, :]) +
			1 / torch.square(lengthscales[:, None, :])
		)) + torch.eye(self.num_inputs)

		X = inp[None, :, :, :] / torch.square(lengthscales[:, None, None, :])
		X2 = -inp[:, None, :, :] / torch.square(lengthscales[None, :, None, :])
		Q = torch.solve(input_var, R).solution / 2
		Xs = torch.sum(X @ Q * X, -1)
		X2s = torch.sum(X2 @ Q * X2, -1)
		maha = -2 * torch.matmul(torch.matmul(X, Q), torch.transpose(X2.conj(), -1, -2)) + Xs[:, :, :, None] + X2s[:, :,
		None, :]

		k = torch.log(variances)[:, None] - torch.sum(torch.square(iN), -1) / 2
		L = torch.exp(k[:, None, :, None] + k[None, :, None, :] + maha)
		temp = beta[:, None, None, :].repeat([1, self.num_states, 1, 1]) @ L
		S = (temp @ beta[None, :, :, None].repeat([self.num_states, 1, 1, 1]))[:, :, 0, 0]

		diagL = torch.Tensor.permute(torch.diagonal(torch.Tensor.permute(L, dims=(3, 2, 1, 0)), dim1=-2, dim2=-1),
																			dims=(2, 1, 0))
		S = S - torch.diag_embed(torch.sum(torch.mul(iK, diagL), [1, 2]))
		S = S / torch.sqrt(torch.det(R))
		S = S + torch.diag_embed(variances)
		S = S - M @ torch.transpose(M, -1, -2)

		return M.t(), S, V.t()

	def to_normed_obs_tensor(self, obs):
		"""
		Compute the norm of observation using the min and max of the observation_space of the gym env.

		Args:
			obs  (numpy.array): observation from the gym environment. dim=(Ns,)

		Returns:
			state_mu_norm (torch.Tensor): normed states
		"""
		state_mu_norm = torch.Tensor((obs - self.obs_space.low) / (self.obs_space.high - self.obs_space.low))
		return state_mu_norm

	def to_normed_var_tensor(self, obs_var):
		"""
		Compute the norm of the observation variance matrix using
		the min and max of the observation_space of the gym env.

		Args:
			obs_var  (numpy.array): unnormalized variance of the state. dim=(Ns,)

		Returns:
			obs_var_norm (torch.Tensor): normed variance of the state
		"""
		obs_var_norm = obs_var / (self.obs_space.high - self.obs_space.low)
		obs_var_norm = torch.Tensor(obs_var_norm / (self.obs_space.high - self.obs_space.low).T)
		return obs_var_norm

	def to_normed_action_tensor(self, action):
		"""
		Compute the norm of the action using the min and max of the action_space of the gym env.

		Args:
			action  (numpy.array): un-normalized action. dim=(Na,)
									Na: dimension of action_space
		Returns:
			action_norm (torch.Tensor): normed action

		"""
		action_norm = torch.Tensor((action - self.action_space.low) / (self.action_space.high - self.action_space.low))
		return action_norm

	def denorm_action(self, action_norm):
		"""
		Denormalize the action using the min and max of the action_space of the gym env, so that
		it can be apllied on the gym env

		Args:
			action_norm  (numpy.array or torch.Tensor): normed action. dim=(Na,)
													Na: dimension of action_space
		Returns:
			action (numpy_array or torch.Tensor): un-normalised action. dim=(Na,)
														Na: dimension of action_space
		"""
		action = action_norm * (self.action_space.high - self.action_space.low) + self.action_space.low
		return action

	def compute_cost(self, state_mu, state_var, action):
		"""
		Compute the quadratic cost of one state distribution or a trajectory of states distributions
		given the mean value and variance of states (observations), the weight matrix, and target state.
		The state, state_var and action must be normalized.
		If reading directly from the gym env observation,
		this can be done with the gym env action space and observation space.
		See an example of normalization in the add_points_memory function.
		Args:
			state_mu (torch.Tensor): normed mean value of the state or observation distribution
									(elements between 0 and 1). dim=(Ns) or dim=(Np, Ns)
			state_var (torch.Tensor): normed variance matrix of the state or observation distribution
										(elements between 0 and 1)
										dim=(Ns, Ns) or dim=(Np, Ns, Ns)
			action (torch.Tensor): normed actions. (elements between 0 and 1).
									dim=(Na) or dim=(Np, Na)

			Np: length of the prediction trajectory. (=self.len_horizon)
			Na: dimension of the gym environment actions
			Ns: dimension of the gym environment states

		Returns:
			cost_mu (torch.Tensor): mean value of the cost distribution. dim=(1) or dim=(Np)
			cost_var (torch.Tensor): variance of the cost distribution. dim=(1) or dim=(Np)
		"""

		if state_var.ndim == 3:
			error = torch.cat((state_mu, action), 1) - self.config.reward.target_state_action_norm
			state_action_var = torch.cat((
				torch.cat((state_var, torch.zeros((state_var.shape[0], state_var.shape[1], action.shape[1]))), 2),
				torch.zeros((state_var.shape[0], action.shape[1], state_var.shape[1] + action.shape[1]))), 1)
		else:
			error = torch.cat((state_mu, action), 0) - self.config.reward.target_state_action_norm
			state_action_var = torch.block_diag(state_var, torch.zeros((action.shape[0], action.shape[0])))
		cost_mu = torch.diagonal(torch.matmul(state_action_var, self.config.reward.weight_matrix_cost),
				dim1=-1, dim2=-2).sum(-1) + \
				torch.matmul(torch.matmul(error[..., None].transpose(-1, -2), self.config.reward.weight_matrix_cost),
						error[..., None]).squeeze()
		TS = self.config.reward.weight_matrix_cost @ state_action_var
		cost_var_term1 = torch.diagonal(2 * TS @ TS, dim1=-1, dim2=-2).sum(-1)
		cost_var_term2 = TS @ self.config.reward.weight_matrix_cost
		cost_var_term3 = (4 * error[..., None].transpose(-1, -2) @ cost_var_term2 @ error[..., None]).squeeze()
		cost_var = cost_var_term1 + cost_var_term3
		if self.config.reward.use_constraints:
			if state_mu.ndim == 2:
				state_distribution = [torch.distributions.normal.Normal(state_mu[idx], state_var[idx]) for idx in
					range(state_mu.shape[0])]
				penalty_min_constraint = torch.stack([state_distribution[time_idx].cdf(torch.Tensor(
					self.config.reward.state_min)) * self.config.reward.area_multiplier for
					time_idx in range(state_mu.shape[0])]).diagonal(0, -1, -2).sum(-1)
				penalty_max_constraint = torch.stack([(1 - state_distribution[time_idx].cdf(torch.Tensor(
					self.config.reward.state_max))) * self.config.reward.area_multiplier for
					time_idx in range(state_mu.shape[0])]).diagonal(0, -1, -2).sum(-1)
			else:
				state_distribution = torch.distributions.normal.Normal(state_mu, state_var)
				penalty_min_constraint = ((state_distribution.cdf(torch.Tensor(
					self.config.reward.state_min))) * self.config.reward.area_multiplier).diagonal(0, -1, -2).sum(-1)
				penalty_max_constraint = ((1 - state_distribution.cdf(
					torch.Tensor(self.config.reward.state_max))) * self.config.reward.area_multiplier).diagonal(0, -1, -2).sum(-1)
			cost_mu = cost_mu + penalty_max_constraint + penalty_min_constraint

		return cost_mu, cost_var

	def compute_cost_terminal(self, state_mu, state_var):
		"""
		Compute the terminal cost of the prediction trajectory.
		Args:
			state_mu (torch.Tensor): mean value of the terminal state distribution. dim=(Ns)
			state_var (torch.Tensor): variance matrix of the terminal state distribution. dim=(Ns, Ns)

		Returns:
			cost_mu (torch.Tensor): mean value of the cost distribution. dim=(1)
			cost_var (torch.Tensor): variance of the cost distribution. dim=(1)
		"""
		error = state_mu - self.config.reward.target_state_norm
		cost_mu = torch.trace(torch.matmul(state_var, self.config.reward.weight_matrix_cost_terminal)) + \
				  torch.matmul(torch.matmul(error.t(), self.config.reward.weight_matrix_cost_terminal), error)
		TS = self.config.reward.weight_matrix_cost_terminal @ state_var
		cost_var_term1 = torch.trace(2 * TS @ TS)
		cost_var_term2 = 4 * error.t() @ TS @ self.config.reward.weight_matrix_cost_terminal @ error
		cost_var = cost_var_term1 + cost_var_term2
		return cost_mu, cost_var

	def compute_cost_unnormalized(self, obs, action, obs_var=None):
		"""
		Compute the cost on un-normalized state and actions.
		Takes in numpy array and returns numpy array.
		Meant to be used to compute the cost outside the object.
		Args:
			obs (numpy.array): state (or observation). shape=(Ns,)
			action (numpy.array): action. Shape=(Na,)
			obs_var (numpy.array): state (or observation) variance. Default=None. shape=(Ns, Ns)
									If set to None, the observation constant stored inside the object will be used

		Returns:
			cost_mu (float): Mean of the cost
			cost_var (float): variance of the cost
		"""
		obs_norm = self.to_normed_obs_tensor(obs)
		action_norm = self.to_normed_action_tensor(action)
		if obs_var is None:
			obs_var_norm = self.config.observation.obs_var_norm
		else:
			obs_var_norm = self.to_normed_var_tensor(obs_var)
		cost_mu, cost_var = self.compute_cost(obs_norm, obs_var_norm, action_norm)
		return cost_mu.item(), cost_var.item()

	def add_memory(self, obs, action, obs_new, reward, check_storage=True,
					predicted_state=None, predicted_state_std=None):
		"""
		Add an observation, action and observation after applying the action to the memory that is used
		by the gaussian process models.
		At regular number of points interval (self.training_frequency),
		the training process of the gaussian process models will be launched to optimize the hyper-parameters.

		Args:
			obs (numpy.array): non-normalized observation. Dim=(Ns,)
			action (numpy.array): non-normalized action. Dim=(Ns,)
			obs_new (numpy.array): non-normalized observation obtained after applying the action on the observation.
									Dim=(Ns,)
			reward (float): reward obtained from the gym env. Unused at the moment.
							The cost given state and action is computed instead.
			check_storage (bool): If check_storage is true,
									predicted_state and predicted_state_std will be checked (if not None) to
									know weither to store the point in memory or not.

			predicted_state (numpy.array or torch.Tensor or None):
								if check_storage is True and predicted_state is not None,
								the prediction error for that point will be computed.
								and the point will only be stored in memory if the
								prediction error is larger than self.error_pred_memory. Dim=(Ns,)

			predicted_state_std (numpy.array or torch.Tensor or None):
								If check_storage is true, and predicted_state_std is not None, the point will only be
								stored in memory if it is larger than self.error_pred_memory. Dim=(Ns,)

			where Ns: dimension of states, Na: dimension of actions
		"""
		if obs is None:
			return
		obs_norm = to_normed_obs_tensor(obs, low=self.obs_space.low, high=self.obs_space.high)
		action_norm = to_normed_action_tensor(action=action, low=self.action_space.low, high=self.action_space.high)
		obs_new_norm = to_normed_obs_tensor(obs_new, low=self.obs_space.low, high=self.obs_space.high)

		if len(self.x) < (self.len_mem + 1):
			self.x = torch.cat(self.x, torch.empty(self.config.memory.points_batch_memory, self.x.shape[1]))
			self.y = torch.cat(self.y, torch.empty(self.config.memory.points_batch_memory, self.y.shape[1]))
			self.rewards = torch.cat(self.rewards, torch.empty(self.config.memory.points_batch_memory))

		self.x[self.len_mem, :(obs_norm.shape[0] + action_norm.shape[0])] = \
							torch.cat((obs_norm, action_norm))[None]
		self.y[self.len_mem] = obs_new_norm - obs_norm
		self.rewards[self.len_mem] = reward

		if self.config.model.include_time_model:
			self.x[self.len_mem, -1] = self.n_iter_obs

		store_gp_mem = True
		if check_storage:
			if predicted_state is not None:
				error_prediction = torch.abs(predicted_state - obs_new_norm)
				store_gp_mem = torch.any(error_prediction > self.config.memory.min_error_prediction_state_for_memory)

			if predicted_state_std is not None and store_gp_mem:
				store_gp_mem = torch.any(predicted_state_std > self.config.memory.min_prediction_state_std_for_memory)

		if store_gp_mem:
			self.idxs_mem_gp.append(self.len_mem)

		self.len_mem += 1
		self.n_iter_obs += 1

		if self.len_mem % self.config.training.training_frequency == 0 and \
				not ('p_train' in self.__dict__ and not self.p_train._closed):
			self.p_train = self.ctx.Process(target=self.train, args=(self.queue_train,
			self.x[self.idxs_mem_gp],
			self.y[self.idxs_mem_gp],
			[model.state_dict() for model in self.models],
			self.config.model.__dict__, 
			self.config.training.lr_train, 
			self.config.training.iter_train, 
			self.config.training.clip_grad_value,
			self.config.training.print_train, 
			self.config.training.step_print_train))
			self.p_train.start()
			self.num_cores_main -= 1

	@staticmethod
	def train(queue, train_inputs, train_targets, parameters, constraints_hyperparams, lr_train, num_iter_train,
			clip_grad_value, print_train=False, step_print_train=25):
		"""
		Train the gaussian process models hyper-parameters such that the marginal-log likelihood
		for the predictions of the points in memory is minimized.
		This function is launched in parallel of the main process, which is why a queue is used to tranfer
		information back to the main process and why the gaussian process models are reconstructed
		using the points in memory and hyper-parameters (they cant be sent directly as argument).
		If an error occurs, returns the parameters sent as init values
		(hyper-parameters obtained by the previous training process)
		Args:
			queue (multiprocessing.queues.Queue): queue object used to transfer information to the main process
			train_inputs (torch.Tensor): input data-points of the gaussian process models (concat(obs, actions)). Dim=(Np, Ns + Na)
			train_targets (torch.Tensor): targets data-points of the gaussian process models (obs_new - obs). Dim=(Np, Ns)
			parameters (list of OrderedDict): contains the hyper-parameters of the models used as init values.
												They are obtained by using [model.state_dict() for model in models]
												where models is a list containing gaussian process models of the gpytorch library:
												gpytorch.models.ExactGP
			constraints_hyperparams (dict): Constraints on the hyper-parameters. See parameters.md for more information
			lr_train (float): learning rate of the training
			num_iter_train (int): number of iteration for the training optimizer
			clip_grad_value (float): value at which the gradient are clipped, so that the training is more stable
			print_train (bool): weither to print the information during training. default=False
			step_print_train (int): If print_train is True, only print the information every step_print_train iteration
		"""
		torch.set_num_threads(1)
		start_time = time.time()
		# create models, which is necessary since this function is used in a parallel process
		# that do not share memory with the principal process
		models = create_models(train_inputs, train_targets, parameters, constraints_hyperparams)
		best_outputscales = [model.covar_module.outputscale.detach() for model in models]
		best_noises = [model.likelihood.noise.detach() for model in models]
		best_lengthscales = [model.covar_module.base_kernel.lengthscale.detach() for model in models]
		previous_losses = torch.empty(len(models))

		for model_idx in range(len(models)):
			output = models[model_idx](models[model_idx].train_inputs[0])
			mll = gpytorch.mlls.ExactMarginalLogLikelihood(models[model_idx].likelihood, models[model_idx])
			previous_losses[model_idx] = -mll(output, models[model_idx].train_targets)

		best_losses = previous_losses.detach().clone()
		# Random initialization of the parameters showed better performance than
		# just taking the value from the previous iteration as init values.
		# If parameters found at the end do not better performance than previous iter,
		# return previous parameters
		for model_idx in range(len(models)):
			models[model_idx].covar_module.outputscale = \
				models[model_idx].covar_module.raw_outputscale_constraint.lower_bound + \
				torch.rand(models[model_idx].covar_module.outputscale.shape) * \
				(models[model_idx].covar_module.raw_outputscale_constraint.upper_bound - \
				 models[model_idx].covar_module.raw_outputscale_constraint.lower_bound)

			models[model_idx].covar_module.base_kernel.lengthscale = \
				models[model_idx].covar_module.base_kernel.raw_lengthscale_constraint.lower_bound + \
				torch.rand(models[model_idx].covar_module.base_kernel.lengthscale.shape) * \
				(models[model_idx].covar_module.base_kernel.raw_lengthscale_constraint.upper_bound - \
				 models[model_idx].covar_module.base_kernel.raw_lengthscale_constraint.lower_bound)

			models[model_idx].likelihood.noise = \
				models[model_idx].likelihood.noise_covar.raw_noise_constraint.lower_bound + \
				torch.rand(models[model_idx].likelihood.noise.shape) * \
				(models[model_idx].likelihood.noise_covar.raw_noise_constraint.upper_bound -
				 models[model_idx].likelihood.noise_covar.raw_noise_constraint.lower_bound)
			mll = gpytorch.mlls.ExactMarginalLogLikelihood(models[model_idx].likelihood, models[model_idx])
			models[model_idx].train()
			models[model_idx].likelihood.train()
			optimizer = torch.optim.LBFGS([
				{'params': models[model_idx].parameters()},  # Includes GaussianLikelihood parameters
			], lr=lr_train, line_search_fn='strong_wolfe')
			try:
				for i in range(num_iter_train):
					def closure():
						optimizer.zero_grad()
						# Output from model
						output = models[model_idx](models[model_idx].train_inputs[0])
						# Calc loss and backprop gradients
						loss = -mll(output, models[model_idx].train_targets)
						torch.nn.utils.clip_grad_value_(models[model_idx].parameters(), clip_grad_value)
						loss.backward()
						if print_train:
							if i % step_print_train == 0:
								print(
									'Iter %d/%d - Loss: %.5f   output_scale: %.5f   lengthscale: %s   noise: %.5f' % (
										i + 1, num_iter_train, loss.item(),
										models[model_idx].covar_module.outputscale.item(),
										str(models[
											model_idx].covar_module.base_kernel.lengthscale.detach().numpy()),
										pow(models[model_idx].likelihood.noise.item(), 0.5)
									))
						return loss

					loss = optimizer.step(closure)
					if loss < best_losses[model_idx]:
						best_losses[model_idx] = loss.item()
						best_lengthscales[model_idx] = models[model_idx].covar_module.base_kernel.lengthscale
						best_noises[model_idx] = models[model_idx].likelihood.noise
						best_outputscales[model_idx] = models[model_idx].covar_module.outputscale

			except Exception as e:
				print(e)

			print(
				'training process - model %d - time train %f - output_scale: %s - lengthscales: %s - noise: %s' % (
					model_idx, time.time() - start_time, str(best_outputscales[model_idx].detach().numpy()),
					str(best_lengthscales[model_idx].detach().numpy()),
					str(best_noises[model_idx].detach().numpy())))

		print('training process - previous marginal log likelihood: %s - new marginal log likelihood: %s' %
			  (str(previous_losses.detach().numpy()), str(best_losses.detach().numpy())))
		params_dict_list = []
		for model_idx in range(len(models)):
			params_dict_list.append({
				'covar_module.base_kernel.lengthscale': best_lengthscales[model_idx].detach().numpy(),
				'covar_module.outputscale': best_outputscales[model_idx].detach().numpy(),
				'likelihood.noise': best_noises[model_idx].detach().numpy()})
		queue.put(params_dict_list)

	def check_and_close_processes(self):
		"""
		Check active parallel processes, wait for their resolution, get the parameters and close them
		"""
		if 'p_train' in self.__dict__ and not self.p_train._closed and not (self.p_train.is_alive()):
			params_dict_list = self.queue_train.get()
			self.p_train.join()
			for model_idx in range(len(self.models)):
				self.models[model_idx].initialize(**params_dict_list[model_idx])
			self.p_train.close()
			self.iK, self.beta = calculate_factorizations(self.x[self.idxs_mem_gp], self.y[self.idxs_mem_gp], self.models)
			self.num_cores_main += 1
