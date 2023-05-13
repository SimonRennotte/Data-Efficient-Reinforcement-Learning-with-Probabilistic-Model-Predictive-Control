from .utils.functions_process_config import convert_config_lists_to_tensor


class ActionsConfig:
	def __init__(
		self, 
		limit_action_change:bool=False,
		max_change_action_norm:"list[float]"=[0.05],
	):
		"""
			limit_action_change: if set to true, the actions will stay close to the action of the previous time step
			max_change_action_norm: determines the maximum change of the normalized action if limit_action_change is set to true
		"""
		self.limit_action_change = limit_action_change
		self.max_change_action_norm = max_change_action_norm

		self = convert_config_lists_to_tensor(self)