import numpy as np
import torch
import gpytorch
from gpytorch.means import Mean


class CustomMean(Mean):
	def __init__(self, index_state_prediction):
		super().__init__()
		self.index_state_prediction = index_state_prediction
		# self.register_parameter(name="constant_multiply", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1), requires_grad=False))
		# self.register_parameter(name="com_optim", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1), requires_grad=False))

	def forward(self, x):
		res = x[:, self.index_state_prediction]  # + self.constant_multiply * x[:, -1]  # - self.com_optim
		return res


class ExactGPModelMultiTask(gpytorch.models.ExactGP):
	def __init__(self, train_x, train_y, priors, dim_input, num_tasks, rank=0):
		self.mean_module = gpytorch.means.ZeroMean()
		if priors is not None:
			likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(noise_prior=
			gpytorch.priors.NormalPrior(priors['noise_mean'], np.power(priors['noise_std'], 2)), num_tasks=num_tasks, rank=rank)
			super(ExactGPModelMultiTask, self).__init__(train_x, train_y, likelihood)
			covariance_matrix = torch.zeros((len(priors['lengthscale_mean_mattern']), len(priors['lengthscale_mean_mattern'])))
			covariance_matrix[range(len(priors['lengthscale_std_mattern'])), range(len(priors['lengthscale_std_mattern']))] = \
				torch.Tensor(np.power(priors['lengthscale_std_mattern'], 2))
			lengthscale_prior_mattern = gpytorch.priors.MultivariateNormalPrior(torch.Tensor(priors['lengthscale_mean_mattern']),
				covariance_matrix=covariance_matrix)
			outputscale_prior_mattern = gpytorch.priors.NormalPrior(priors['outputscale_mean_mattern'],
				np.power(priors['outputscale_std_mattern'], 2))
			variance_prior_lin = gpytorch.priors.NormalPrior(priors['variance_mean_lin'],
				np.power(priors['variance_std_lin'], 2))
			offset_prior_lin = gpytorch.priors.NormalPrior(priors['offset_mean_lin'],
				np.power(priors['offset_std_lin'], 2))
			# mean_prior = gpytorch.priors.NormalPrior(priors['mean_mean'], np.power(priors['mean_std'], 2))

			self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.ScaleKernel(
				gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=dim_input),
				lengthscale_prior=lengthscale_prior_mattern, ard_num_dims=dim_input,
				outputscale_prior=outputscale_prior_mattern) +
				gpytorch.kernels.LinearKernel(offset_prior=offset_prior_lin, variance_prior=variance_prior_lin),
				num_tasks=num_tasks, rank=rank)
		else:
			likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks, rank=rank)
			super(ExactGPModelMultiTask, self).__init__(train_x, train_y, likelihood)
			self.covar_module = gpytorch.kernels.MultitaskKernel(gpytorch.kernels.ScaleKernel(
				gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=dim_input)) + gpytorch.kernels.LinearKernel(),
				num_tasks=num_tasks, rank=rank)

	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class ExactGPModelMonoTask(gpytorch.models.ExactGP):
	def __init__(self, train_x, train_y, dim_input):
		likelihood = gpytorch.likelihoods.GaussianLikelihood()
		super(ExactGPModelMonoTask, self).__init__(train_x, train_y, likelihood)
		self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dim_input))
		self.mean_module = gpytorch.means.ZeroMean()

	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
	def __init__(self, train_x, train_y, priors, dim_input, num_tasks):
		if priors[0] is not None:
			likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks, noise_prior=
			gpytorch.priors.NormalPrior(priors['noise_mean'], np.power(priors['noise_std'], 2)))
			super(BatchIndependentMultitaskGPModel, self).__init__(train_x, train_y, likelihood)
			covariance_matrix = torch.zeros((len(priors['lengthscale_mean_mattern']), len(priors['lengthscale_mean_mattern'])))
			covariance_matrix[range(len(priors['lengthscale_std_mattern'])), range(len(priors['lengthscale_std_mattern']))] = \
				torch.Tensor(np.power(priors['lengthscale_std_mattern'], 2))
			lengthscale_prior_mattern = gpytorch.priors.MultivariateNormalPrior(torch.Tensor(priors['lengthscale_mean_mattern']),
				covariance_matrix=covariance_matrix)
			outputscale_prior_mattern = gpytorch.priors.NormalPrior(priors['outputscale_mean_mattern'],
				np.power(priors['outputscale_std_mattern'], 2))
			variance_prior_lin = gpytorch.priors.NormalPrior(priors['variance_mean_lin'],
				np.power(priors['variance_std_lin'], 2))
			offset_prior_lin = gpytorch.priors.NormalPrior(priors['offset_mean_lin'],
				np.power(priors['offset_std_lin'], 2))
			self.covar_module = gpytorch.kernels.ScaleKernel(
				gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=dim_input, batch_shape=torch.Size([num_tasks]),
					lengthscale_prior=lengthscale_prior_mattern), ard_num_dims=dim_input,
				outputscale_prior=outputscale_prior_mattern) + \
				gpytorch.kernels.LinearKernel(offset_prior=offset_prior_lin, variance_prior=variance_prior_lin,
					batch_shape=torch.Size([num_tasks]))
		else:
			likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
			super(BatchIndependentMultitaskGPModel, self).__init__(train_x, train_y, likelihood)
			self.covar_module = gpytorch.kernels.ScaleKernel(
				gpytorch.kernels.MaternKernel(nu=0.5, ard_num_dims=dim_input, batch_shape=torch.Size([num_tasks])),
												batch_shape=torch.Size([num_tasks])) + \
								gpytorch.kernels.LinearKernel(batch_shape=torch.Size([num_tasks]))

		self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))

	def forward(self, x):
		mean_x = self.mean_module(x)
		covar_x = self.covar_module(x)
		return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
			gpytorch.distributions.MultivariateNormal(mean_x, covar_x))
