from .utils.functions_process_config import convert_config_lists_to_tensor


class ActionsConfig:
	def __init__(
		self, 
		limit_action_change=False,
		max_change_action_norm = [0.05],
	):
		self.limit_action_change = limit_action_change
		self.max_change_action_norm = max_change_action_norm

		self = convert_config_lists_to_tensor(self)