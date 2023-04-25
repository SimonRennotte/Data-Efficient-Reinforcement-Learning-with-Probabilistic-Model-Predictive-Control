from .utils.functions_process_config import convert_config_lists_to_tensor


class ActionsConfig:
	def __init__(
		self, 
		limit_action_change=False,
		max_change_action_norm = [0.05],
		len_horizon=15,
		action_dim=1
	):
		self.limit_action_change = limit_action_change
		self.max_change_action_norm = max_change_action_norm

		self = convert_config_lists_to_tensor(self)

		if self.limit_action_change:
			self.bounds = [(-self.max_change_action_norm[idx_action], self.max_change_action_norm[idx_action]) for idx_action in range(action_dim)] * len_horizon
		else:
			self.bounds = [(0, 1)] * action_dim * len_horizon