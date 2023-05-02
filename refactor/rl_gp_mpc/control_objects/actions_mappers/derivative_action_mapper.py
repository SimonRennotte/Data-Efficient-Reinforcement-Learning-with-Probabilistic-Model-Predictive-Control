from typing import Union

import numpy as np
import torch

from rl_gp_mpc.config_classes.actions_config import ActionsConfig
from rl_gp_mpc.control_objects.utils.pytorch_utils import Clamp
from .abstract_action_mapper import AbstractActionMapper

class DerivativeActionMapper(AbstractActionMapper):
	def __init__(self, action_low:Union[np.ndarray, torch.Tensor], action_high:Union[np.ndarray, torch.Tensor], len_horizon: int, config:ActionsConfig):
		super().__init__(action_low, action_high, len_horizon, config)
		self.action_model_previous_iter = torch.rand(self.dim_action)
		self.bounds = [(0, 1)] * self.dim_action * len_horizon
		self.clamp_class =  Clamp()

	def transform_action_raw_to_action_model(self, action_raw: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
		action_model = self.norm_action(action_raw)
		return action_model

	def transform_action_model_to_action_raw(self, action_model: Union[np.ndarray, torch.Tensor], update_internals:bool=False) -> torch.Tensor:
		if update_internals:
			self.action_model_previous_iter = action_model[0]

		action_raw = self.denorm_action(action_model, update_internals=update_internals)
		return action_raw

	def transform_action_mpc_to_action_model(self, action_mpc: torch.Tensor) -> torch.Tensor:
		actions_mpc_2d = torch.atleast_2d(action_mpc.reshape(self.len_horizon, -1))
		actions_mpc_2d = actions_mpc_2d * 2 * self.config.max_change_action_norm - self.config.max_change_action_norm
		actions_mpc_2d[0] = actions_mpc_2d[0] + self.action_model_previous_iter
		actions_model = torch.cumsum(actions_mpc_2d, dim=0)
		clamp = self.clamp_class.apply
		actions_model = clamp(actions_model, 0, 1) 
		return actions_model