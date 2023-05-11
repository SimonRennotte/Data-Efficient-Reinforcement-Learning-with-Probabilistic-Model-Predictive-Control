from typing import Union

import numpy as np
import torch

from rl_gp_mpc.config_classes.actions_config import ActionsConfig


class AbstractActionMapper:
	'''
	Class use to map actions to space between 0 and 1.
	The optimiser constraints the embed_action to be between 0 and 1.
	By using this class, the action can be constrainted in interesting ways.
	It is used by DerivativeActionsMapper to limit the rate of change of actions.
	Another application would be to use a autoencoder to allow the mpc controller to work with high dimension actions
	by optimizing in the embed space.
	'''
	def __init__(self, action_low:Union[np.ndarray, torch.Tensor], action_high:Union[np.ndarray, torch.Tensor], len_horizon: int, config:ActionsConfig):
		self.config = config
		self.action_low = torch.Tensor(action_low)
		self.action_high = torch.Tensor(action_high)
		self.dim_action = len(action_low)
		self.len_horizon = len_horizon
		self.n_iter_ctrl = 0

	def transform_action_raw_to_action_model(self, action_raw: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
		raise NotImplementedError

	def transform_action_model_to_action_raw(self, action_model: Union[np.ndarray, torch.Tensor], update_internals:bool=False) -> torch.Tensor:
		raise NotImplementedError

	def transform_action_mpc_to_action_model(self, action_mpc: np.ndarray) -> torch.Tensor:
		raise NotImplementedError

	def transform_action_mpc_to_action_raw(self, action_mpc: Union[np.ndarray, torch.Tensor], update_internals:bool=False) -> torch.Tensor:
		action_model = self.transform_action_mpc_to_action_model(action_mpc)
		action_raw = self.transform_action_model_to_action_raw(action_model, update_internals=update_internals)
		return action_raw

	def norm_action(self, action: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
		return (torch.Tensor(action) - self.action_low) / (self.action_high - self.action_low)

	def denorm_action(self, normed_action: Union[np.ndarray, torch.Tensor], update_internals=False) -> torch.Tensor:
		# Update internals is True when the action will be applied to the environment
		if update_internals:
			self.n_iter_ctrl += 1
		return torch.Tensor(normed_action) * (self.action_high - self.action_low) + self.action_low