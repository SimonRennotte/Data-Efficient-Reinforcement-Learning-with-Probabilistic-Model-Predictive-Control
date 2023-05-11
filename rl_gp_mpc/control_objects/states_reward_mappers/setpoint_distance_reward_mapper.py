import torch

from rl_gp_mpc.config_classes.reward_config import RewardConfig
from .abstract_state_reward_mapper import AbstractStateRewardMapper


class SetpointStateRewardMapper(AbstractStateRewardMapper):
	def __init__(self, config:RewardConfig):
		super().__init__(config)

	def get_reward(self, state_mu: torch.Tensor, state_var: torch.Tensor, action:torch.Tensor) -> float:
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
		error = torch.cat((state_mu, action), -1) - self.config.target_state_action_norm
		if state_var.ndim == 3:
			#error = torch.cat((state_mu, action), 1) - self.config.target_state_action_norm
			state_action_var = torch.cat((
				torch.cat((state_var, torch.zeros((state_var.shape[0], state_var.shape[1], action.shape[1]))), 2),
				torch.zeros((state_var.shape[0], action.shape[1], state_var.shape[1] + action.shape[1]))), 1)
		else:
			#error = torch.cat((state_mu, action), 0) - self.config.target_state_action_norm
			state_action_var = torch.block_diag(state_var, torch.zeros((action.shape[0], action.shape[0])))

		weight_matrix_cost = self.config.weight_matrix_cost
		cost_mu = torch.diagonal(torch.matmul(state_action_var, weight_matrix_cost),
		dim1=-1, dim2=-2).sum(-1) + \
		torch.matmul(torch.matmul(error[..., None].transpose(-1, -2), weight_matrix_cost),
				error[..., None]).squeeze()
		TS = weight_matrix_cost @ state_action_var
		cost_var_term1 = torch.diagonal(2 * TS @ TS, dim1=-1, dim2=-2).sum(-1)
		cost_var_term2 = TS @ weight_matrix_cost
		cost_var_term3 = (4 * error[..., None].transpose(-1, -2) @ cost_var_term2 @ error[..., None]).squeeze()
		cost_var = cost_var_term1 + cost_var_term3

		if self.config.use_constraints:
			if state_mu.ndim == 2:
				state_distribution = [torch.distributions.normal.Normal(state_mu[idx], state_var[idx]) for idx in
					range(state_mu.shape[0])]
				penalty_min_constraint = torch.stack([state_distribution[time_idx].cdf(torch.Tensor(
					self.config.state_min)) * self.config.area_multiplier for
					time_idx in range(state_mu.shape[0])]).diagonal(0, -1, -2).sum(-1)
				penalty_max_constraint = torch.stack([(1 - state_distribution[time_idx].cdf(torch.Tensor(
					self.config.state_max))) * self.config.area_multiplier for
					time_idx in range(state_mu.shape[0])]).diagonal(0, -1, -2).sum(-1)
			else:
				state_distribution = torch.distributions.normal.Normal(state_mu, state_var)
				penalty_min_constraint = ((state_distribution.cdf(torch.Tensor(
					self.config.state_min))) * self.config.area_multiplier).diagonal(0, -1, -2).sum(-1)
				penalty_max_constraint = ((1 - state_distribution.cdf(
					torch.Tensor(self.config.state_max))) * self.config.area_multiplier).diagonal(0, -1, -2).sum(-1)
			cost_mu = cost_mu + penalty_max_constraint + penalty_min_constraint

		return -cost_mu, cost_var

	def get_rewards(self, state_mu: torch.Tensor, state_var: torch.Tensor, action:torch.Tensor) -> float:
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
			reward_mu (torch.Tensor): mean value of the reward distribution. (=-cost) dim=(1) or dim=(Np)
			cost_var (torch.Tensor): variance of the cost distribution. dim=(1) or dim=(Np)
		"""

		error = torch.cat((state_mu, action), 1) - self.config.target_state_action_norm
		state_action_var = torch.cat((
			torch.cat((state_var, torch.zeros((state_var.shape[0], state_var.shape[1], action.shape[1]))), 2),
			torch.zeros((state_var.shape[0], action.shape[1], state_var.shape[1] + action.shape[1]))), 1)

		cost_mu = torch.diagonal(torch.matmul(state_action_var, self.config.weight_matrix_cost),
				dim1=-1, dim2=-2).sum(-1) + \
				torch.matmul(torch.matmul(error[..., None].transpose(-1, -2), self.config.weight_matrix_cost),
						error[..., None]).squeeze()
		TS = self.config.weight_matrix_cost @ state_action_var
		cost_var_term1 = torch.diagonal(2 * TS @ TS, dim1=-1, dim2=-2).sum(-1)
		cost_var_term2 = TS @ self.config.weight_matrix_cost
		cost_var_term3 = (4 * error[..., None].transpose(-1, -2) @ cost_var_term2 @ error[..., None]).squeeze()

		cost_var = cost_var_term1 + cost_var_term3
		if self.config.use_constraints:
			state_distribution = [torch.distributions.normal.Normal(state_mu[idx], state_var[idx]) for idx in
				range(state_mu.shape[0])]
			penalty_min_constraint = torch.stack([state_distribution[time_idx].cdf(torch.Tensor(
				self.config.state_min)) * self.config.area_multiplier for
				time_idx in range(state_mu.shape[0])]).diagonal(0, -1, -2).sum(-1)
			penalty_max_constraint = torch.stack([(1 - state_distribution[time_idx].cdf(torch.Tensor(
				self.config.state_max))) * self.config.area_multiplier for
				time_idx in range(state_mu.shape[0])]).diagonal(0, -1, -2).sum(-1)
			cost_mu = cost_mu + penalty_max_constraint + penalty_min_constraint

		return -cost_mu, cost_var

	def get_reward_terminal(self, state_mu:torch.Tensor, state_var:torch.Tensor) -> float:
		"""
		Compute the terminal cost of the prediction trajectory.
		Args:
			state_mu (torch.Tensor): mean value of the terminal state distribution. dim=(Ns)
			state_var (torch.Tensor): variance matrix of the terminal state distribution. dim=(Ns, Ns)

		Returns:
			reward_mu (torch.Tensor): mean value of the reward distribution. (=-cost) dim=(1)
			cost_var (torch.Tensor): variance of the cost distribution. dim=(1)
		"""
		error = state_mu - self.config.target_state_norm
		cost_mu = torch.trace(torch.matmul(state_var, self.config.weight_matrix_cost_terminal)) + \
				  torch.matmul(torch.matmul(error.t(), self.config.weight_matrix_cost_terminal), error)
		TS = self.config.weight_matrix_cost_terminal @ state_var
		cost_var_term1 = torch.trace(2 * TS @ TS)
		cost_var_term2 = 4 * error.t() @ TS @ self.config.weight_matrix_cost_terminal @ error
		cost_var = cost_var_term1 + cost_var_term2
		return -cost_mu, cost_var

	def get_rewards_trajectory(self, states_mu:torch.Tensor, states_var:torch.Tensor, actions:torch.Tensor) -> torch.Tensor:
		rewards_traj, rewards_traj_var = self.get_reward(states_mu[:-1], states_var[:-1], actions)
		cost_traj_final, rewards_traj_var_final = self.get_reward_terminal(states_mu[-1], states_var[-1])
		rewards_traj = torch.cat((rewards_traj, cost_traj_final[None]), 0)
		rewards_traj_var = torch.cat((rewards_traj_var, rewards_traj_var_final[None]), 0)
		return rewards_traj, rewards_traj_var


def compute_squared_dist_cost(error, state_action_var, weight_matrix_cost):
	cost_mu = torch.diagonal(torch.matmul(state_action_var, weight_matrix_cost),
			dim1=-1, dim2=-2).sum(-1) + \
			torch.matmul(torch.matmul(error[..., None].transpose(-1, -2), weight_matrix_cost),
					error[..., None]).squeeze()
	TS = weight_matrix_cost @ state_action_var
	cost_var_term1 = torch.diagonal(2 * TS @ TS, dim1=-1, dim2=-2).sum(-1)
	cost_var_term2 = TS @ weight_matrix_cost
	cost_var_term3 = (4 * error[..., None].transpose(-1, -2) @ cost_var_term2 @ error[..., None]).squeeze()
	cost_var = cost_var_term1 + cost_var_term3
	return cost_mu, cost_var