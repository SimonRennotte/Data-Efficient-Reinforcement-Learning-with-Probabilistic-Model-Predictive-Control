from typing import Union

import numpy as np
import torch

from rl_gp_mpc.config_classes.actions_config import ActionsConfig
from .abstract_action_mapper import AbstractActionMapper


class NormalizationActionMapper(AbstractActionMapper):
	def __init__(self, action_low:Union[np.ndarray, torch.Tensor], action_high:Union[np.ndarray, torch.Tensor], len_horizon: int, config:ActionsConfig):
		super().__init__(action_low, action_high, len_horizon, config)
		self.bounds = [(0, 1)] * self.dim_action * len_horizon

	def transform_action_raw_to_action_model(self, action_raw: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
		return self.norm_action(action_raw)

	def transform_action_model_to_action_raw(self, action_model: Union[np.ndarray, torch.Tensor], update_internals:bool=False) -> torch.Tensor:
		return self.denorm_action(action_model, update_internals=update_internals)

	def transform_action_mpc_to_action_model(self, action_mpc: np.ndarray) -> torch.Tensor:
		actions = torch.atleast_2d(action_mpc.reshape(self.len_horizon, -1))
		return actions