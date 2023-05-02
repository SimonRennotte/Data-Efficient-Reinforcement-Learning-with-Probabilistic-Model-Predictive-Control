import torch

from rl_gp_mpc.config_classes.memory_config import MemoryConfig


class GpMemory:
    def __init__(self, config:MemoryConfig, dim_input, dim_state, include_time_model=False):
        self.config = config
        self.include_time_model = include_time_model
        self.x = torch.empty(config.points_batch_memory, dim_input)
        self.y = torch.empty(config.points_batch_memory, dim_state)
        self.rewards = torch.empty(config.points_batch_memory)
        self.idxs_mem_gp = []
        self.len_mem = 0
        self.n_iter_obs = 0

    def add(self, state, action_model, state_next, reward, **kwargs):
        if len(self.x) < (self.len_mem + 1):
            self.x = torch.cat(self.x, torch.empty(self.config.points_batch_memory, self.x.shape[1]))
            self.y = torch.cat(self.y, torch.empty(self.config.points_batch_memory, self.y.shape[1]))
            self.rewards = torch.cat(self.rewards, torch.empty(self.config.points_batch_memory))

        self.x[self.len_mem, :((state.shape[0] + action_model.shape[0]))] = torch.cat((state, action_model))[None]
        self.y[self.len_mem] = state_next - state
        self.rewards[self.len_mem] = reward

        if self.include_time_model:
            self.x[self.len_mem, -1] = self.n_iter_obs

        store_gp_mem = True

        if self.config.check_errors_for_storage:
            if 'predicted_state' in kwargs.keys() and kwargs['predicted_state'] is not None:
                error_prediction = torch.abs(kwargs['predicted_state'] - state_next)
                store_gp_mem = torch.any(error_prediction > self.config.min_error_prediction_state_for_memory)

            if store_gp_mem and 'predicted_state_std' in kwargs.keys() and kwargs['predicted_state_std'] is not None:
                store_gp_mem = torch.any(kwargs['predicted_state_std'] > self.config.min_prediction_state_std_for_memory)

        if store_gp_mem:
            self.idxs_mem_gp.append(self.len_mem)
            self.len_mem += 1

        self.n_iter_obs += 1