import torch


def convert_config_lists_to_tensor(self):
	for attr, value in self.__dict__.items():
		if isinstance(value, list):
			setattr(self, attr, torch.Tensor(value))
	return self


def convert_dict_lists_to_dict_tensor(dict_list):
	for attr, value in dict_list.items():
		if isinstance(value, list):
			dict_list[attr] =  torch.Tensor(value)
	return dict_list


def extend_lengthscale_dim(num_models, num_inputs, lengthscale, lengthscale_time):
	lengthscales = torch.empty((num_models, num_inputs))
	if isinstance(lengthscale, float) or isinstance(lengthscale, int) or lengthscale.dim() != 1:
		lengthscales[:, :-1] = lengthscale
		lengthscales[:, -1] = lengthscale_time
	else:
		lengthscales[:, :-1] = lengthscale[None].t().repeat((1, num_inputs - 1))
		lengthscales[:, -1] = lengthscale_time
	return lengthscales


