import time
import multiprocessing

import numpy as np
from scipy.stats import norm
import torch
import gpytorch
from scipy.optimize import minimize
# from threadpoolctl import threadpool_limits

from control_objects.gp_models import ExactGPModelMonoTask
from control_objects.abstract_class_control_object import BaseControllerObject
from control_objects.utils import save_plot_model_3d_process, save_plot_history_process, create_models

torch.set_default_tensor_type(torch.DoubleTensor)


class ProbabiliticGpMpcController(BaseControllerObject):
	def __init__(self, observation_space, action_space, params_controller, params_train, params_actions_optimizer,
			params_constraints_states, hyperparameters_init,
			target_state, weight_matrix_states_cost_function, weight_matrix_states_cost_function_terminal,
			target_action, weight_matrix_actions_cost_function,
			constraints_hyperparams, env_to_control, folder_save, num_repeat_actions):
		BaseControllerObject.__init__(self, observation_space, action_space)
		self.models = []
		self.weight_matrix_cost_function = \
			torch.block_diag(torch.Tensor(weight_matrix_states_cost_function), 
				torch.Tensor(weight_matrix_actions_cost_function))
		self.weight_matrix_cost_function_terminal = torch.Tensor(weight_matrix_states_cost_function_terminal)
		self.target_state = torch.Tensor(target_state)
		self.target_state_action = torch.cat((self.target_state, torch.Tensor(target_action)))
		for observation_idx in range(observation_space.shape[0]):
			self.models.append(ExactGPModelMonoTask(None, None, self.num_inputs))

		self.len_horizon = params_controller['len_horizon']
		self.exploration_factor = params_controller['exploration_factor']
		self.lr_train = params_train['lr_train']
		self.num_iter_train = params_train['n_iter_train']
		self.clip_grad_value = params_train['clip_grad_value']
		self.train_every_n_points = params_train['train_every_n_points']
		self.print_train = params_train['print_train']
		self.step_print_train = params_train['step_print_train']
		self.compute_factorization_each_iteration = params_controller['compute_factorization_each_iteration']
		self.constraints_hyperparams = constraints_hyperparams
		self.constraints_states = params_constraints_states if params_constraints_states["use_constraints"] else None
		self.params_actions_optimizer = params_actions_optimizer
		self.limit_derivative_actions = params_controller['limit_derivative_actions']
		self.max_derivative_action_norm = np.array(params_controller['max_derivative_actions_norm'])
		self.clip_lower_bound_cost_to_0 = params_controller['clip_lower_bound_cost_to_0']
		self.folder_save = folder_save
		self.env_to_control = env_to_control
		self.num_repeat_actions = num_repeat_actions

		self.prediction_info_over_time = {}
		self.indexes_memory_gp = []

		if self.limit_derivative_actions:
			self.bounds = \
				[(-self.max_derivative_action_norm[idx_action], self.max_derivative_action_norm[idx_action]) \
					for idx_action in range(self.action_space.shape[0])] * self.len_horizon
			self.predicted_actions_previous_iter = \
				np.dot(np.expand_dims(np.random.uniform(low=-1, high=1, size=(self.len_horizon)), 1),
					np.expand_dims(self.max_derivative_action_norm, 0))
		else:
			self.bounds = [(0, 1)] * self.num_actions * self.len_horizon
			self.predicted_actions_previous_iter = np.random.uniform(low=0, high=1,
				size=(self.len_horizon, self.num_actions))

		for model_idx in range(len(self.models)):
			if constraints_hyperparams is not None:
				if "min_std_noise" in constraints_hyperparams.keys():
					self.models[model_idx].likelihood.noise_covar.register_constraint("raw_noise",
						gpytorch.constraints.Interval(lower_bound=np.power(constraints_hyperparams['min_std_noise'], 2),
							upper_bound=np.power(constraints_hyperparams['max_std_noise'], 2)))
				if "min_outputscale" in constraints_hyperparams.keys():
					self.models[model_idx].covar_module.register_constraint("raw_outputscale",
						gpytorch.constraints.Interval(
							lower_bound=constraints_hyperparams['min_outputscale'],
							upper_bound=constraints_hyperparams['max_outputscale']))
				if "min_lengthscale" in constraints_hyperparams.keys():
					self.models[model_idx].covar_module.base_kernel.register_constraint("raw_lengthscale",
						gpytorch.constraints.Interval(
							lower_bound=constraints_hyperparams['min_lengthscale'],
							upper_bound=constraints_hyperparams['max_lengthscale']))

			hypers = {'base_kernel.lengthscale': torch.tensor(hyperparameters_init['lengthscale'][model_idx]),
				'outputscale': torch.tensor(hyperparameters_init['scale'][model_idx])}
			hypers_likelihood = {'noise_covar.noise': torch.tensor(
				np.power(hyperparameters_init['noise_std'][model_idx], 2))}
			self.models[model_idx].likelihood.initialize(**hypers_likelihood)
			self.models[model_idx].covar_module.initialize(**hypers)
			self.models[model_idx].eval()
			self.num_cores_main = multiprocessing.cpu_count()
			self.ctx = multiprocessing.get_context('spawn')
			self.queue_train = self.ctx.Queue()

	def add_point_memory(self, observation, action, new_observation, reward, **kwargs):
		observation = torch.Tensor((observation - self.observation_space.low) /
								   (self.observation_space.high - self.observation_space.low))
		action = torch.Tensor((action - self.action_space.low) / (self.action_space.high - self.action_space.low))
		new_observation = torch.Tensor((new_observation - self.observation_space.low) / (
				self.observation_space.high - self.observation_space.low))
		if len(self.x) < (self.num_points_memory + 1):
			self.x = torch.cat(self.x, torch.empty(self.num_points_add_memory_when_full, self.x.shape[1]))
			self.y = torch.cat(self.y, torch.empty(self.num_points_add_memory_when_full, self.y.shape[1]))
		self.x[self.num_points_memory] = torch.cat((observation, action))
		self.y[self.num_points_memory] = new_observation - observation
		if len(kwargs) != 0 and 'params_memory' in kwargs and 'add_info_dict' in kwargs:
			min_error_states = torch.Tensor(kwargs['params_memory']['min_error_prediction_states_for_storage'])
			min_std = torch.Tensor(kwargs['params_memory']['min_prediction_states_std_for_storage'])
			predicted_state = kwargs['add_info_dict']['predicted states'][0]
			predicted_std = kwargs['add_info_dict']['predicted states std'][0]
			error_prediction = torch.abs(predicted_state - new_observation)
			store_in_memory = torch.any(torch.logical_or(error_prediction > min_error_states, predicted_std > min_std))
		else:
			store_in_memory = True

		if store_in_memory:
			self.indexes_memory_gp.append(self.num_points_memory)

		self.num_points_memory += 1
		self.action_previous_iter = action

		if self.num_points_memory % self.train_every_n_points == 0 and \
				not ('p_train' in self.__dict__ and not self.p_train._closed):
			self.p_train = self.ctx.Process(target=self.train, args=(self.queue_train,
			self.x[self.indexes_memory_gp],
			self.y[self.indexes_memory_gp],
			[model.state_dict() for model in self.models],
			self.constraints_hyperparams, self.lr_train, self.num_iter_train, self.clip_grad_value,
			self.print_train, self.step_print_train))
			self.p_train.start()
			self.num_cores_main -= 1

	def cost_fct_on_uncertain_inputs(self, m_state, s_state, m_action):
		'''This function computes the quadratic cost of a state distribution given the weight matrix, and target state.
		m_state and s_state can be either the full trajectory (m_state.ndim=2 and s_state.ndim=3),
		or a single state (m_state.ndim=1 and s_state.ndim=2)'''
		if s_state.ndim == 3:
			error = torch.cat((m_state, m_action), 1) - self.target_state_action
			s_state_action = torch.cat((
				torch.cat((s_state, torch.zeros((s_state.shape[0], s_state.shape[1], m_action.shape[1]))), 2),
				torch.zeros((s_state.shape[0], m_action.shape[1], s_state.shape[1] + m_action.shape[1]))), 1)
		else:
			error = torch.cat((m_state, m_action), 0) - self.target_state_action
			s_state_action = torch.block_diag(s_state, torch.zeros((m_action.shape[0], m_action.shape[0])))
		expected_cost = torch.diagonal(torch.matmul(s_state_action, self.weight_matrix_cost_function),
				dim1=-1, dim2=-2).sum(-1) + \
				torch.matmul(torch.matmul(error[..., None].transpose(-1, -2), self.weight_matrix_cost_function),
						error[..., None]).squeeze()
		TS = self.weight_matrix_cost_function @ s_state_action
		s_cost_term_1 = torch.diagonal(2 * TS @ TS, dim1=-1, dim2=-2).sum(-1)
		s_cost_term_2 = TS @ self.weight_matrix_cost_function
		s_cost_term_3 = (4 * error[..., None].transpose(-1, -2) @ s_cost_term_2 @ error[..., None]).squeeze()
		s_cost = s_cost_term_1 + s_cost_term_3
		constraint_penalty = 0
		if self.constraints_states is not None and self.constraints_states['use_constraints'] != 0:
			if m_state.ndim == 2:
				distribution_states = [torch.distributions.normal.Normal(m_state[idx], s_state[idx]) for idx in
					range(m_state.shape[0])]
				constraint_penalty_min = torch.stack([distribution_states[time_idx].cdf(torch.Tensor(
					self.constraints_states["states_min"])) * self.constraints_states['area_penalty_multiplier'] for
					time_idx in range(m_state.shape[0])]).diagonal(0, -1, -2).sum(-1)
				constraint_penalty_max = torch.stack([(1 - distribution_states[time_idx].cdf(torch.Tensor(
					self.constraints_states["states_max"]))) * self.constraints_states['area_penalty_multiplier'] for
					time_idx in range(m_state.shape[0])]).diagonal(0, -1, -2).sum(-1)
			else:
				distribution_state = torch.distributions.normal.Normal(m_state, s_state)
				constraint_penalty_min = ((distribution_state.cdf(torch.Tensor(
					self.constraints_states["states_min"]))) * \
									  self.constraints_states['area_penalty_multiplier']).diagonal(0, -1, -2).sum(-1)
				constraint_penalty_max = ((1 - distribution_state.cdf(
					torch.Tensor(self.constraints_states["states_max"]))) * \
									  self.constraints_states['area_penalty_multiplier']).diagonal(0, -1, -2).sum(-1)
			expected_cost = expected_cost + constraint_penalty_max + constraint_penalty_min

		return expected_cost, s_cost

	def terminal_cost(self, m_state, s_state):
		error = m_state - self.target_state
		mean_cost = torch.trace(torch.matmul(s_state, self.weight_matrix_cost_function_terminal)) + \
					torch.matmul(torch.matmul(error.t(), self.weight_matrix_cost_function_terminal), error)
		TS = self.weight_matrix_cost_function_terminal @ s_state
		s_cost_term_1 = torch.trace(2 * TS @ TS)
		s_cost_term_2 = 4 * error.t() @ TS @ self.weight_matrix_cost_function_terminal @ error
		s_cost = s_cost_term_1 + s_cost_term_2
		return mean_cost, s_cost

	def predict_trajectory(self, actions, mu_observation, s_observation, iK, beta):
		''' function that computes the future predicted states mu_states_pred and predicted uncertainty of the states
			via the covariance matrix of the states s_states_pred, given the current state mu_observation,
			current covariance matrix of the state s_observation, and planned actions (actions.shape = (len_horizon, num_actions))
		 	It also computes the losses, the variance of the loss, and the lower cofidence bound of the loss
		 	along the trajectory'''
		mu_states_pred = torch.empty((self.len_horizon + 1, len(mu_observation)))
		s_states_pred = torch.empty((self.len_horizon + 1, self.num_states, self.num_states))
		mu_states_pred[0] = torch.Tensor(mu_observation)
		s_states_pred[0] = torch.Tensor(s_observation)
		for idx_time in range(1, self.len_horizon + 1):
			s_inputs_pred = torch.zeros(
				(s_states_pred.shape[1] + actions.shape[1], s_states_pred.shape[2] + actions.shape[1]))
			s_inputs_pred[:s_states_pred.shape[1], :s_states_pred.shape[2]] = s_states_pred[idx_time - 1]
			d_state, d_s_state, v = self.predict_given_factorizations(
				torch.cat((mu_states_pred[idx_time - 1], actions[idx_time - 1]), axis=0),
				s_inputs_pred, iK, beta)
			mu_states_pred[idx_time] = mu_states_pred[idx_time - 1] + d_state  # torch.clamp(, 0, 1)
			s_states_pred[idx_time] = d_s_state + s_states_pred[idx_time - 1] + \
									  s_inputs_pred[:s_states_pred.shape[1]] @ v + \
									  v.t() @ s_inputs_pred[:s_states_pred.shape[1]].t()

		costs_trajectory, costs_trajectory_variance = self.cost_fct_on_uncertain_inputs(mu_states_pred[:-1],
			s_states_pred[:-1], actions)
		cost_trajectory_final, costs_trajectory_variance_final = self.terminal_cost(mu_states_pred[-1],
			s_states_pred[-1])
		costs_trajectory = torch.cat((costs_trajectory, cost_trajectory_final[None]), 0)
		costs_trajectory_variance = torch.cat((costs_trajectory_variance, costs_trajectory_variance_final[None]), 0)
		costs_trajectory_lcb = costs_trajectory - self.exploration_factor * torch.sqrt(costs_trajectory_variance)
		return mu_states_pred, s_states_pred, costs_trajectory, costs_trajectory_variance, costs_trajectory_lcb

	def compute_cost_trajectory(self, actions, mu_observation, s_observation, iK, beta):
		actions = np.atleast_2d(actions.reshape(self.len_horizon, -1))
		actions = torch.Tensor(actions)
		actions.requires_grad = True
		if self.limit_derivative_actions:
			actions_input = actions.clone()
			actions_input[0] = self.action_previous_iter + actions_input[0]
			actions_input = torch.clamp(torch.cumsum(actions_input, dim=0), 0, 1)
		else:
			actions_input = actions
		mu_states_pred, s_states_pred, costs_trajectory, costs_trajectory_variance, costs_trajectory_lcb = \
			self.predict_trajectory(actions_input, mu_observation, s_observation, iK, beta)
		if self.clip_lower_bound_cost_to_0:
			costs_trajectory_lcb = torch.clamp(costs_trajectory_lcb, 0, np.inf)
		mean_cost_trajectory_lcb = costs_trajectory_lcb.mean()
		gradients_cost = torch.autograd.grad(mean_cost_trajectory_lcb, actions, retain_graph=False)[0]

		self.cost_trajectory_mean_lcb = mean_cost_trajectory_lcb.detach()
		self.mu_states_pred = mu_states_pred.detach()
		self.costs_trajectory = costs_trajectory.detach()
		self.s_states_pred = s_states_pred.detach()
		self.costs_trajectory_variance = costs_trajectory_variance.detach()

		return mean_cost_trajectory_lcb.detach().numpy(), gradients_cost[:, 0].detach().numpy()

	@staticmethod
	def calculate_factorizations(x, y, models):
		""" Function inspired from
		https://github.com/nrontsis/PILCO/blob/6a962c8e4172f9e7f29ed6e373c4be2dd4b69cb7/pilco/models/mgpr.py#L81,
		reimplemented from tensorflow to pytorch """
		K = torch.stack([model.covar_module(x).evaluate() for model in models])
		batched_eye = torch.eye(K.shape[1]).repeat(K.shape[0], 1, 1)
		L = torch.cholesky(K + torch.stack([model.likelihood.noise for model in models])[:, None] * batched_eye)
		iK = torch.cholesky_solve(batched_eye, L)
		Y_ = (y.t())[:, :, None]
		beta = torch.cholesky_solve(Y_, L)[:, :, 0]
		return iK, beta

	def predict_given_factorizations(self, m, s, iK, beta):
		"""
		 Approximate GP regression at noisy inputs via moment matching
		 IN: mean (m) (row vector) and (s) variance of the state
		 OUT: mean (M) (row vector), variance (S) of the action and inv(s)*input-ouputcovariance
		Function inspired from
		https://github.com/nrontsis/PILCO/blob/6a962c8e4172f9e7f29ed6e373c4be2dd4b69cb7/pilco/models/mgpr.py#L81,
		reinterpreted from tensorflow to pytorch
		 """
		s = s[None, None, :, :].repeat([self.num_states, self.num_states, 1, 1])
		inp = (self.x[self.indexes_memory_gp[:beta.shape[1]]] - m)[None, :, :].repeat([self.num_states, 1, 1])
		lengthscales = torch.stack([model.covar_module.base_kernel.lengthscale[0] for model in self.models])
		variances = torch.stack([model.covar_module.outputscale for model in self.models])
		# Calculate M and V: mean and inv(s) times input-output covariance
		iL = torch.diag_embed(1 / lengthscales)
		iN = inp @ iL
		B = iL @ s[0, ...] @ iL + torch.eye(self.num_inputs)

		# Redefine iN as in^T and t --> t^T
		# B is symmetric, so it is equivalent
		t = torch.transpose(torch.solve(torch.transpose(iN, -1, -2), B).solution, -1, -2)

		lb = torch.exp(-torch.sum(iN * t, -1) / 2) * beta
		tiL = t @ iL
		c = variances / torch.sqrt(torch.det(B))

		M = (torch.sum(lb, -1) * c)[:, None]
		V = torch.matmul(torch.transpose(tiL.conj(), -1, -2), lb[:, :, None])[..., 0] * c[:, None]

		# Calculate S: Predictive Covariance
		R = torch.matmul(s, torch.diag_embed(
			1 / torch.square(lengthscales[None, :, :]) +
			1 / torch.square(lengthscales[:, None, :])
		)) + torch.eye(self.num_inputs)

		X = inp[None, :, :, :] / torch.square(lengthscales[:, None, None, :])
		X2 = -inp[:, None, :, :] / torch.square(lengthscales[None, :, None, :])
		Q = torch.solve(s, R).solution / 2
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

	def compute_prediction_action(self, mu_observation, s_observation):
		# Check for parallel process that are open but not alive at each iteration to retrieve the results and close them
		self.check_and_close_processes()
		'''with threadpool_limits(limits=self.num_cores_main, user_api='blas'), \
				threadpool_limits(limits=self.num_cores_main, user_api='openmp'):'''
		torch.set_num_threads(self.num_cores_main)
		with torch.no_grad():
			normed_mu_observation, normed_s_observation = self.norm(mu_observation, s_observation)
			if self.compute_factorization_each_iteration:
				self.iK, self.beta = self.calculate_factorizations(self.x[self.indexes_memory_gp],
																	self.y[self.indexes_memory_gp], self.models)
			else:
				if not 'iK' in self.__dict__.keys() or not 'beta' in self.__dict__.keys():
					self.iK, self.beta = self.calculate_factorizations(self.x[self.indexes_memory_gp],
						self.y[self.indexes_memory_gp], self.models)
			initial_actions_optimizer = (np.concatenate((self.predicted_actions_previous_iter[1:],
			np.expand_dims(self.predicted_actions_previous_iter[-1], 0)), axis=0))
			if self.limit_derivative_actions:
				true_initial_action_optimizer = np.empty_like(initial_actions_optimizer)
				true_initial_action_optimizer[0] = self.action_previous_iter
				true_initial_action_optimizer += initial_actions_optimizer
				true_initial_action_optimizer = np.cumsum(true_initial_action_optimizer, axis=0)
				if np.logical_or(np.any(true_initial_action_optimizer > 1),
						np.any(true_initial_action_optimizer < 0)):
					for idx_time in range(1, len(initial_actions_optimizer)):
						true_initial_action_optimizer[idx_time] = true_initial_action_optimizer[idx_time - 1] + \
																  initial_actions_optimizer[idx_time]
						indexes_above_1 = np.nonzero(true_initial_action_optimizer[idx_time] > 1)[0]
						indexes_under_0 = np.nonzero(true_initial_action_optimizer[idx_time] < 0)[0]
						initial_actions_optimizer[idx_time][indexes_above_1] = \
							1 - true_initial_action_optimizer[idx_time - 1][indexes_above_1]
						initial_actions_optimizer[idx_time][indexes_under_0] = \
							- true_initial_action_optimizer[idx_time - 1][indexes_under_0]
						true_initial_action_optimizer[idx_time][indexes_above_1] = 1
						true_initial_action_optimizer[idx_time][indexes_under_0] = 0

			initial_actions_optimizer = initial_actions_optimizer.flatten()
		res = minimize(fun=self.compute_cost_trajectory,
			x0=initial_actions_optimizer,
			jac=True,
			args=(normed_mu_observation, normed_s_observation, self.iK, self.beta),
			method='L-BFGS-B',
			bounds=self.bounds,
			options=self.params_actions_optimizer)
		actions = res.x.reshape(self.len_horizon, -1)
		self.predicted_actions_previous_iter = actions.copy()

		with torch.no_grad():
			if self.limit_derivative_actions:
				actions[0] += np.array(self.action_previous_iter)
				actions = np.clip(np.cumsum(actions, axis=0), 0, 1)
				next_action = actions[0]
			else:
				next_action = actions[0]
			self.action_previous_iter = next_action
			actions = torch.Tensor(actions)
			cost, cost_var = self.cost_fct_on_uncertain_inputs(normed_mu_observation, normed_s_observation,
				actions[0])
			denorm_act = next_action[0] * (self.action_space.high - self.action_space.low) + self.action_space.low
			# denorm_states = self.mu_states_pred[1:] * \
			# (self.observation_space.high - self.observation_space.low) + self.observation_space.low
			# denorm_std_states = std_states * (self.observation_space.high - self.observation_space.low)
			std_states = torch.diagonal(self.s_states_pred, dim1=-2, dim2=-1).sqrt()
			add_info_dict = {'iteration': self.num_points_memory * self.num_repeat_actions,
				'state': self.mu_states_pred[0],
				'predicted states': self.mu_states_pred[1:],
				'predicted states std': std_states[1:],
				'predicted actions': actions,
				'cost': cost.item(), 'cost std': cost_var.sqrt().item(),
				'predicted costs': self.costs_trajectory[1:],
				'predicted costs std': self.costs_trajectory_variance[1:].sqrt(),
				'mean predicted cost': np.min([self.costs_trajectory[1:].mean().item(), 3]),
				'mean predicted cost std': self.costs_trajectory_variance[1:].sqrt().mean().item(),
				'lower bound mean predicted cost': self.cost_trajectory_mean_lcb.item()}
			for key in add_info_dict.keys():
				if not key in self.prediction_info_over_time:
					self.prediction_info_over_time[key] = [add_info_dict[key]]
				else:
					self.prediction_info_over_time[key].append(add_info_dict[key])
			return denorm_act, add_info_dict

	def norm(self, state, state_s):
		norm_state = (state - self.observation_space.low) / (self.observation_space.high - self.observation_space.low)
		norm_s = state_s / (self.observation_space.high - self.observation_space.low)
		norm_s = norm_s / (self.observation_space.high - self.observation_space.low).T
		return torch.Tensor(norm_state), torch.Tensor(norm_s)

	def save_plot_model_3d(self, prop_extend_domain=1, n_ticks=100, total_col_max=3, fontsize=6):
		if not ('p_save_plot_model_3d' in self.__dict__ and not self.p_save_plot_model_3d._closed):
			self.p_save_plot_model_3d = self.ctx.Process(target=save_plot_model_3d_process,
				args=(self.x[:self.num_points_memory], self.y[:self.num_points_memory],
				[model.state_dict() for model in self.models],
				self.constraints_hyperparams, self.folder_save, [self.indexes_memory_gp] * self.num_states,
				prop_extend_domain, n_ticks, total_col_max, fontsize))
			self.p_save_plot_model_3d.start()
			self.num_cores_main -= 1

	def save_plot_history(self):
		if not ('p_save_plot_history' in self.__dict__ and not self.p_save_plot_history._closed):
			states = self.x[:self.num_points_memory, :self.num_states].numpy()
			actions = self.x[:self.num_points_memory, self.num_states:].numpy()
			states_next = self.y[:self.num_points_memory].numpy() + states
			self.p_save_plot_history = self.ctx.Process(target=save_plot_history_process,
				args=(states, actions, states_next, self.prediction_info_over_time, self.folder_save,
				self.num_repeat_actions, self.constraints_states))
			self.p_save_plot_history.start()
			self.num_cores_main -= 1

	@staticmethod
	def train(queue, train_inputs, train_targets, parameters, constraints_hyperparams, lr_train, num_iter_train,
			clip_grad_value, print_train=0, step_print_train=25):
		# with threadpool_limits(limits=1, user_api='blas'), threadpool_limits(limits=1, user_api='openmp'):
		torch.set_num_threads(1)
		start_time = time.time()
		# create models
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
		for model_idx in range(len(models)):
			models[model_idx].covar_module.outputscale = \
				models[model_idx].covar_module.raw_outputscale_constraint.lower_bound + \
				torch.rand(models[model_idx].covar_module.outputscale.shape) * \
				(models[model_idx].covar_module.raw_outputscale_constraint.upper_bound - \
				 models[model_idx].covar_module.raw_outputscale_constraint.lower_bound)

			models[model_idx].covar_module.base_kernel.lengthscale = \
				models[model_idx].covar_module.base_kernel.raw_lengthscale_constraint.lower_bound + \
				torch.rand(models[model_idx].covar_module.base_kernel.lengthscale.shape) * \
				(torch.min(torch.stack([torch.Tensor(
					[models[model_idx].covar_module.base_kernel.raw_lengthscale_constraint.upper_bound]),
					torch.Tensor([0.5])])) - \
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
		if 'p_train' in self.__dict__ and not self.p_train._closed and not (self.p_train.is_alive()):
			params_dict_list = self.queue_train.get()
			self.p_train.join()
			for model_idx in range(len(self.models)):
				self.models[model_idx].initialize(**params_dict_list[model_idx])
			self.p_train.close()
			self.iK, self.beta = self.calculate_factorizations(self.x[self.indexes_memory_gp],
													self.y[self.indexes_memory_gp], self.models)
			self.num_cores_main += 1

		if 'p_save_plot_history' in self.__dict__ \
				and not self.p_save_plot_history._closed and not (self.p_save_plot_history.is_alive()):
			self.p_save_plot_history.join()
			self.p_save_plot_history.close()
			self.num_cores_main += 1

		if 'p_save_plot_model_3d' in self.__dict__ \
				and not self.p_save_plot_model_3d._closed and not (self.p_save_plot_model_3d.is_alive()):
			self.p_save_plot_model_3d.join()
			self.p_save_plot_model_3d.close()
			self.num_cores_main += 1
