import torch
from .controller_config import ControllerConfig
from .actions_config import ActionsConfig
from .model_config import ModelConfig
from .training_config import TrainingConfig
from .reward_config import RewardConfig
from .observation_config import ObservationConfig
from .memory_config import MemoryConfig


torch.set_default_tensor_type(torch.DoubleTensor)


class Config:
	def __init__(
		self, 
		observation_config: ObservationConfig=ObservationConfig(),
		reward_config:RewardConfig=RewardConfig(),
		actions_config:ActionsConfig=ActionsConfig(),
		model_config:ModelConfig=ModelConfig(),
		memory_config:MemoryConfig=MemoryConfig(),
		training_config:TrainingConfig=TrainingConfig(),
		controller_config:ControllerConfig=ControllerConfig()
	):
		self.observation = observation_config
		self.reward = reward_config
		self.actions =actions_config
		self.model = model_config
		self.memory = memory_config
		self.training = training_config
		self.controller = controller_config




