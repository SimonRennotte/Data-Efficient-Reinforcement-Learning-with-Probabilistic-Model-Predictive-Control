from typing import Union

import numpy as np
import torch

from rl_gp_mpc.config_classes.observation_config import ObservationConfig
from .abstract_observation_state_mapper import AbstractObservationStateMapper


class NormalizationObservationStateMapper(AbstractObservationStateMapper):
	def __init__(self, observation_low: Union[np.ndarray, torch.Tensor], observation_high: Union[np.ndarray, torch.Tensor], config:ObservationConfig):
		super().__init__(observation_low, observation_high, config)

	def get_state(self, obs: Union[np.ndarray, torch.Tensor], obs_var:Union[np.ndarray, torch.Tensor]=None, update_internals=False) -> torch.Tensor:
		state = (torch.Tensor(obs) - self.obs_low) / (self.obs_high - self.obs_low)

		if obs_var is not None:
			state_var = torch.Tensor(obs_var) / self.var_norm_factor
		else:
			state_var = self.config.obs_var_norm

		return state, state_var




