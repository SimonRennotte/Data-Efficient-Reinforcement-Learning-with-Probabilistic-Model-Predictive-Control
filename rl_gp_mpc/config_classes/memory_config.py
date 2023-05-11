from .utils.functions_process_config import convert_config_lists_to_tensor


class MemoryConfig:
	def __init__(self, 	
		check_errors_for_storage:bool=True,
		min_error_prediction_state_for_memory:"list[float]"=[3e-4, 3e-4, 3e-4],
		min_prediction_state_std_for_memory:"list[float]"=[3e-3, 3e-3, 3e-3],
		points_batch_memory:int=1500
	):
		self.check_errors_for_storage = check_errors_for_storage
		self.min_error_prediction_state_for_memory = min_error_prediction_state_for_memory
		self.min_prediction_state_std_for_memory = min_prediction_state_std_for_memory
		self.points_batch_memory = points_batch_memory

		self = convert_config_lists_to_tensor(self)