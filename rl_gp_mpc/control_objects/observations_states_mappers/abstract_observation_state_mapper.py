from typing import Union

import numpy as np
import torch

from rl_gp_mpc.config_classes.observation_config import ObservationConfig

class AbstractObservationStateMapper:
	def __init__(self, observation_low: Union[np.ndarray, torch.Tensor], observation_high: Union[np.ndarray, torch.Tensor], config:ObservationConfig):
		self.config=config
		self.obs_low = torch.Tensor(observation_low)
		self.obs_high = torch.Tensor(observation_high)
		self.var_norm_factor = torch.pow((self.obs_high - self.obs_low), 2)
		self.dim_observation = len(observation_low)
		self.dim_state = self.dim_observation

	def get_state(self, obs:Union[np.ndarray, torch.Tensor], obs_var:Union[np.ndarray, torch.Tensor], update_internals:bool) -> torch.Tensor:
		raise(NotImplementedError())

	def get_obs(self):
		raise(NotImplementedError())




