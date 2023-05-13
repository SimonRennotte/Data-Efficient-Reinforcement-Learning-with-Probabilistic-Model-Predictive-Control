from .utils.functions_process_config import convert_config_lists_to_tensor


class MemoryConfig:
	def __init__(self, 	
		check_errors_for_storage:bool=True,
		min_error_prediction_state_for_memory:"list[float]"=[3e-4, 3e-4, 3e-4],
		min_prediction_state_std_for_memory:"list[float]"=[3e-3, 3e-3, 3e-3],
		points_batch_memory:int=1500
	):
		"""
		check_errors_for_storage: If true, when adding a new point in memory, it will be checked if it is worth adding it to the model memory depending on the prediction by checking the error or uncertainty
		min_error_prediction_state_for_memory: if check_errors_for_storage is true, a point will only be used by the model if the error is above this threshold (any)
		min_prediction_state_std_for_memory: if check_errors_for_storage is true, a point will only be used by the model if the predicted standard deviation is above this threshold (any)
		"""
		self.check_errors_for_storage = check_errors_for_storage
		self.min_error_prediction_state_for_memory = min_error_prediction_state_for_memory
		self.min_prediction_state_std_for_memory = min_prediction_state_std_for_memory
		self.points_batch_memory = points_batch_memory

		self = convert_config_lists_to_tensor(self)
