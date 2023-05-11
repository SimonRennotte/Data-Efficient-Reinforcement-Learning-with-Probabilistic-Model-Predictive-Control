import torch
import numpy as np

from rl_gp_mpc.config_classes.memory_config import MemoryConfig
from rl_gp_mpc.control_objects.utils.data_utils import form_model_input


class Memory:
    def __init__(self, config:MemoryConfig, dim_input, dim_state, include_time_model=False, step_model=1):
        self.config = config
        self.include_time_model = include_time_model
        self.dim_input = dim_input
        self.dim_state = dim_state
        self.step_model = step_model

        self.inputs = torch.empty(config.points_batch_memory, dim_input)
        self.states_next = torch.empty(config.points_batch_memory, dim_state)
        self.rewards = torch.empty(config.points_batch_memory)
        self.iter_ctrls = torch.empty(config.points_batch_memory)
        self.errors = torch.empty(config.points_batch_memory, dim_state)
        self.stds = torch.empty(config.points_batch_memory, dim_state)

        self.model_inputs = torch.empty(config.points_batch_memory, dim_input)
        self.model_targets = torch.empty(config.points_batch_memory, dim_state)

        self.active_data_mask = np.empty(config.points_batch_memory, dtype=bool)
        self.len_mem = 0
        self.len_mem_last_processed = 0
        self.len_mem_model = 0

    def add(self, state, action_model, state_next, reward, iter_ctrl=0, **kwargs):
        # Add all points to memory but must use the "prepare" function to form the model inputs
        # it is needed if the model predicts multiple steps ahead
        if len(self.inputs) < (self.len_mem + 1):
            self.inputs = torch.cat(self.inputs, torch.empty(self.config.points_batch_memory, self.dim_input))
            self.states_next = torch.cat(self.states_next, torch.empty(self.config.points_batch_memory, self.dim_state))
            self.rewards = torch.cat(self.rewards, torch.empty(self.config.points_batch_memory))
            self.iter_ctrls = torch.cat(self.rewards, torch.empty(self.config.points_batch_memory))
            self.errors = torch.cat(self.errors, torch.empty(self.config.points_batch_memory, self.dim_state))
            self.stds = torch.cat(self.stds, torch.empty(self.config.points_batch_memory, self.dim_state))
        
        x = form_model_input(state=state, action_model=action_model, time_idx=iter_ctrl, include_time_model=self.include_time_model, dim_input=self.dim_input)
        self.inputs[self.len_mem] = x
        self.states_next[self.len_mem] = state_next
        self.rewards[self.len_mem] = reward
        self.iter_ctrls[self.len_mem] = iter_ctrl

        store_gp_mem = True
        if self.config.check_errors_for_storage:
            if 'predicted_state' in kwargs.keys() and kwargs['predicted_state'] is not None:
                error_prediction = torch.abs(kwargs['predicted_state'] - state_next)
                store_gp_mem = torch.any(error_prediction > self.config.min_error_prediction_state_for_memory)
                self.errors[self.len_mem] = error_prediction
            else:
                self.errors[self.len_mem] = np.nan

            if 'predicted_state_std' in kwargs.keys() and kwargs['predicted_state_std'] is not None:
                store_gp_mem = store_gp_mem and (torch.any(kwargs['predicted_state_std'] > self.config.min_prediction_state_std_for_memory))
                self.stds[self.len_mem] = kwargs['predicted_state_std']
            else:
                self.stds[self.len_mem] = np.nan

        self.active_data_mask[self.len_mem] = store_gp_mem
        self.len_mem += 1

    def prepare_for_model(self):
        # form the model inputs from the memory.
        # It is defined in this way to allow to have a model that predicts multiple steps ahead
        if len(self.model_inputs) < (self.len_mem_model + 1):
            self.model_inputs = torch.cat(self.model_inputs, torch.empty(self.config.points_batch_memory, self.model_inputs.shape[1]))
            self.model_targets = torch.cat(self.model_targets, torch.empty(self.config.points_batch_memory, self.model_targets.shape[1]))

        idxs_add_memory = self.get_indexes_to_process()
        mask_add_memory = self.active_data_mask[idxs_add_memory]
        idxs_add_memory = idxs_add_memory[mask_add_memory]
        len_to_add = len(idxs_add_memory)

        new_model_inputs, new_model_targets = self.get_memory_by_index(idxs_add_memory)
        self.model_inputs[self.len_mem_model:self.len_mem_model+len_to_add] = new_model_inputs
        self.model_targets[self.len_mem_model:self.len_mem_model+len_to_add] = new_model_targets

        self.len_mem_model += len_to_add
        self.len_mem_last_processed = self.len_mem

    def get_memory_total(self):
        indexes= self.get_indexes_processed()
        inputs, targets = self.get_memory_by_index(indexes)
        return inputs, targets

    def get_memory_by_index(self, indexes):
        inputs = self.inputs[indexes]
        targets = self.states_next[indexes + self.step_model - 1] - self.inputs[indexes, :self.dim_state]
        return inputs, targets

    def get_indexes_to_process(self):
        return np.arange(self.len_mem_last_processed, self.len_mem, self.step_model)

    def get_indexes_processed(self):
        return np.arange(0, self.len_mem_last_processed, self.step_model)

    def get_mask_model_inputs(self):
        indexes = self.get_indexes_processed()
        return self.active_data_mask[indexes]

    def get(self):
        if self.len_mem_model > 0:
            x_mem = self.model_inputs[:self.len_mem_model]
            y_mem = self.model_targets[:self.len_mem_model]
        else: # If no datapoints, initialize with dummy 0 variable to avoid errors
            x_mem = torch.zeros((1, self.dim_input))
            y_mem = torch.zeros((1, self.dim_state))
        return x_mem, y_mem