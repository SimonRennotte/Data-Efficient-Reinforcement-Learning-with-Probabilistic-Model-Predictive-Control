import torch

from rl_gp_mpc.config_classes.observation_config import ObservationConfig

class AbstractObservationStateMapper:
	def __init__(self, observation_space, config_observation:ObservationConfig):
		self.config=config_observation
		self.obs_low = torch.Tensor(observation_space.low)
		self.obs_high = torch.Tensor(observation_space.high)

	def get_state(self, observation:torch.Tensor) -> torch.Tensor:
		raise(NotImplementedError())

	def get_obs(self):
		raise(NotImplementedError())


class NormalizationObservationStateMapper(AbstractObservationStateMapper):
	def __init__(self, observation_space, config_observation:ObservationConfig):
		super.__init__(observation_space, config_observation)

	def get_state(self, observation: torch.Tensor) -> torch.Tensor:
		return (observation - self.obs_low) / (self.obs_high - self.obs_high)

	def get_obs(self, state):
		return state * (self.obs_high - self.obs_high) + self.obs_low

