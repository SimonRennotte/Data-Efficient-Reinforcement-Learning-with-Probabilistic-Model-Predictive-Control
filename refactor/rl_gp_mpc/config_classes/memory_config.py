from .utils.functions_process_config import convert_config_lists_to_tensor


class MemoryConfig:
	def __init__(self, 	
		min_error_prediction_state_for_memory=[3e-4, 3e-4, 3e-4],
		min_prediction_state_std_for_memory=[3e-3, 3e-3, 3e-3],
		points_batch_memory=1500
	):
		self.min_error_prediction_state_for_memory = min_error_prediction_state_for_memory
		self.min_prediction_state_std_for_memory = min_prediction_state_std_for_memory
		self.points_batch_memory = points_batch_memory

		self = convert_config_lists_to_tensor(self)
