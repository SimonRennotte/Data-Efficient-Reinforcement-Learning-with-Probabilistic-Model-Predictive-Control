import torch


def form_model_input(state, action_model, time_idx, include_time_model, dim_input):
    x = torch.empty(dim_input)
    x[:((state.shape[0] + action_model.shape[0]))] = torch.cat((state, action_model))
    if include_time_model:
        x[-1] = time_idx
    return x