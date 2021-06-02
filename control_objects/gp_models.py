import gpytorch


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

