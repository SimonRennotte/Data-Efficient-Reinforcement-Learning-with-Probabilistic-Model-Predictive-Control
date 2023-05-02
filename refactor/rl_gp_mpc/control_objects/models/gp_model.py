import multiprocessing
import time

import torch
import gpytorch
import numpy as np

from rl_gp_mpc.config_classes.model_config import ModelConfig

from .abstract_model import AbstractStateTransitionModel


class SavedState:
    def __init__(self, train_inputs:torch.Tensor, train_targets: torch.Tensor, parameters:"list[dict]", constraints_hyperparams:dict):
        self.train_inputs = train_inputs
        self.train_targets = train_targets
        self.parameters = parameters
        self.constraints_hyperparams = constraints_hyperparams


class GpStateTransitionModel(AbstractStateTransitionModel):
    def __init__(self, config: ModelConfig, dim_state, dim_action):
        super().__init__(config, dim_state, dim_action)

        if self.config.include_time_model:
            self.dim_input += 1

        self.config.extend_dimensions_params(dim_state=self.dim_state, dim_input=self.dim_input)

        self.models = create_models(
                                gp_init_dict=self.config.gp_init,
                                constraints_gp=self.config.__dict__, 
                                train_inputs=None, 
                                train_targets=None, 
                                num_models=self.dim_state, 
                                num_inputs=self.dim_input
                            )

        for idx_model in range(len(self.models)):
            self.models[idx_model].eval()

    def predict_trajectory(self, actions: torch.Tensor, obs_mu: torch.Tensor, obs_var: torch.Tensor, len_horizon: int, current_time_idx: int) -> "tuple[torch.Tensor]":
        """
        Compute the future predicted states distribution for the simulated trajectory given the
        current initial state (or observation) distribution (obs_mu and obs_var) and planned actions
        It also returns the costs, the variance of the costs, and the lower confidence bound of the cost
        along the trajectory

        Args:
            actions: actions to apply for the simulated trajectory. dim=(Nh, Na)
                                    where Nh is the len of the horizon and Na the dimension of actions

            obs_mu:	mean value of the inital state distribution.
                                    dim=(Ns,) where Ns is the dimension of state

            obs_var: variance matrix of the inital state distribution.
                                    dim=(Ns, Ns) where Ns is the dimension of state

        Returns:
            states_mu_pred: predicted states of the trajectory.
                                            The first element contains the initial state.
                                            Dim=(Nh + 1, Ns)

            states_var_pred: covariance matrix of the predicted states of the trajectory.
                                            The first element contains the initial state.
                                            Dim=(Nh + 1, Ns, Ns)
        """
        dim_state = self.dim_state
        dim_input_model = self.dim_input
        dim_action = self.dim_action
        states_mu_pred = torch.empty((len_horizon + 1, len(obs_mu)))
        states_var_pred = torch.empty((len_horizon + 1, dim_state, dim_state))
        states_mu_pred[0] = obs_mu
        states_var_pred[0] = obs_var
        state_dim = obs_mu.shape[0]
        # Input of predict_next_state_change is not a state, but the concatenation of state and action
        for idx_time in range(1, len_horizon + 1):
            input_var = torch.zeros((dim_input_model, dim_input_model))
            input_var[:state_dim, :state_dim] = states_var_pred[idx_time - 1]
            input_mean = torch.empty((dim_input_model,))
            input_mean[:dim_state] = states_mu_pred[idx_time - 1]
            input_mean[dim_state:(dim_state + dim_action)] = actions[idx_time - 1]
            if self.config.include_time_model:
                input_mean[-1] = current_time_idx + idx_time - 1
            state_change, state_change_var, v = self.predict_next_state_change(input_mean, input_var)
            # use torch.clamp(states_mu_pred[idx_time], 0, 1) ?
            states_mu_pred[idx_time] = states_mu_pred[idx_time - 1] + state_change
            states_var_pred[idx_time] = state_change_var + states_var_pred[idx_time - 1] + \
                                      input_var[:states_var_pred.shape[1]] @ v + \
                                      v.t() @ input_var[:states_var_pred.shape[1]].t()

        return states_mu_pred, states_var_pred

    def predict_next_state_change(self, input_mu: torch.Tensor, input_var: torch.Tensor):
        """
        Approximate GP regression at noisy inputs via moment matching
        IN: mean (m) (row vector) and (s) variance of the state
        OUT: mean (M) (row vector), variance (S) of the action and inv(s)*input-ouputcovariance
        Function inspired from
        https://github.com/nrontsis/PILCO/blob/6a962c8e4172f9e7f29ed6e373c4be2dd4b69cb7/pilco/models/mgpr.py#L81,
        reinterpreted from tensorflow to pytorch.
        Must be called after self.prepare_inference
        Args:
            input_mu: mean value of the input distribution. Dim=(Ns + Na,)

            input_var: covariance matrix of the input distribution. Dim=(Ns + Na, Ns + Na)

        Returns:
            M.t() (torch.Tensor): mean value of the predicted change distribution. Dim=(Ns,)

            S (torch.Tensor): covariance matrix of the predicted change distribution. Dim=(Ns, Ns)

            V.t() (torch.Tensor): Dim=(Ns, Ns + Na)

            where Ns: dimension of state, Na: dimension of action
        """
        dim_state = self.dim_state
        dim_input_model = self.dim_input
        input_var = input_var[None, None, :, :].repeat([dim_state, dim_state, 1, 1])
        inp = (self.x_mem - input_mu)[None, :, :].repeat([dim_state, 1, 1])

        iN = inp @ self.iL
        B = self.iL @ input_var[0, ...] @ self.iL + torch.eye(dim_input_model)

        # Redefine iN as in^T and t --> t^T
        # B is symmetric, so it is equivalent
        t = torch.transpose(torch.solve(torch.transpose(iN, -1, -2), B).solution, -1, -2)

        lb = torch.exp(-torch.sum(iN * t, -1) / 2) * self.beta
        tiL = t @ self.iL
        c = self.variances / torch.sqrt(torch.det(B))

        M = (torch.sum(lb, -1) * c)[:, None]
        V = torch.matmul(torch.transpose(tiL.conj(), -1, -2), lb[:, :, None])[..., 0] * c[:, None]

        # Calculate S: Predictive Covariance
        R = torch.matmul(input_var, torch.diag_embed(
            1 / torch.square(self.lengthscales[None, :, :]) +
            1 / torch.square(self.lengthscales[:, None, :])
        )) + torch.eye(dim_input_model)

        X = inp[None, :, :, :] / torch.square(self.lengthscales[:, None, None, :])
        X2 = -inp[:, None, :, :] / torch.square(self.lengthscales[None, :, None, :])
        Q = torch.solve(input_var, R).solution / 2
        Xs = torch.sum(X @ Q * X, -1)
        X2s = torch.sum(X2 @ Q * X2, -1)
        maha = -2 * torch.matmul(torch.matmul(X, Q), torch.transpose(X2.conj(), -1, -2)) + Xs[:, :, :, None] + X2s[:, :, None, :]

        k = torch.log(self.variances)[:, None] - torch.sum(torch.square(iN), -1) / 2
        L = torch.exp(k[:, None, :, None] + k[None, :, None, :] + maha)
        temp = self.beta[:, None, None, :].repeat([1, dim_state, 1, 1]) @ L
        S = (temp @ self.beta[None, :, :, None].repeat([dim_state, 1, 1, 1]))[:, :, 0, 0]

        diagL = torch.Tensor.permute(torch.diagonal(torch.Tensor.permute(L, dims=(3, 2, 1, 0)), dim1=-2, dim2=-1),
                                                                            dims=(2, 1, 0))
        S = S - torch.diag_embed(torch.sum(torch.mul(self.iK, diagL), [1, 2]))
        S = S / torch.sqrt(torch.det(R))
        S = S + torch.diag_embed(self.variances)
        S = S - M @ torch.transpose(M, -1, -2)

        return M.t(), S, V.t()

    def prepare_inference(self, inputs: torch.Tensor, state_changes: torch.Tensor):
        # compute all the parameters needed for inference that only depend on the memory, and not on the input, such that they can be computed in advance.
        # So they are not computed inside the mpc for each iteration
        self.x_mem = inputs
        self.y_mem = state_changes
        self.iK, self.beta = calculate_factorizations(inputs, state_changes, self.models)
        self.x_mem = inputs[:self.beta.shape[1]]
        self.lengthscales = torch.stack([model.covar_module.base_kernel.lengthscale[0] for model in self.models])
        self.variances = torch.stack([model.covar_module.outputscale for model in self.models])
        self.iL = torch.diag_embed(1 / self.lengthscales)

    @staticmethod
    def train(queue:multiprocessing.Queue, saved_state:SavedState, lr_train:float, num_iter_train:int, clip_grad_value:float, print_train:bool=False, step_print_train:int=25):
        """
        Train the gaussian process models hyper-parameters such that the marginal-log likelihood
        for the predictions of the points in memory is minimized.
        This function is launched in parallel of the main process, which is why a queue is used to tranfer
        information back to the main process and why the gaussian process models are reconstructed
        using the points in memory and hyper-parameters (the objects cant be sent directly as argument).
        If an error occurs, returns the parameters sent as init values
        (hyper-parameters obtained by the previous training process)
        Args:
            queue: queue object used to transfer information to the main process
            saved_state: SavedState, contains all the information to reconstruct the models
            lr_train: learning rate of the training
            num_iter_train: number of iteration for the training optimizer
            clip_grad_value: value at which the gradient are clipped, so that the training is more stable
            print_train: weither to print the information during training. default=False
            step_print_train: If print_train is True, only print the information every step_print_train iteration
        """
        torch.set_num_threads(1)
        start_time = time.time()
        # create models, which is necessary since this function is used in a parallel process
        # that do not share memory with the principal process
        models = create_models(saved_state.parameters, saved_state.constraints_hyperparams, saved_state.train_inputs, saved_state.train_targets)
        best_outputscales = [model.covar_module.outputscale.detach() for model in models]
        best_noises = [model.likelihood.noise.detach() for model in models]
        best_lengthscales = [model.covar_module.base_kernel.lengthscale.detach() for model in models]
        previous_losses = torch.empty(len(models))

        for model_idx in range(len(models)):
            output = models[model_idx](models[model_idx].train_inputs[0])
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(models[model_idx].likelihood, models[model_idx])
            previous_losses[model_idx] = -mll(output, models[model_idx].train_targets)

        best_losses = previous_losses.detach().clone()
        # Random initialization of the parameters showed better performance than
        # just taking the value from the previous iteration as init values.
        # If parameters found at the end do not better performance than previous iter,
        # return previous parameters
        for model_idx in range(len(models)):
            models[model_idx].covar_module.outputscale = \
                models[model_idx].covar_module.raw_outputscale_constraint.lower_bound + \
                torch.rand(models[model_idx].covar_module.outputscale.shape) * \
                (models[model_idx].covar_module.raw_outputscale_constraint.upper_bound - \
                 models[model_idx].covar_module.raw_outputscale_constraint.lower_bound)

            models[model_idx].covar_module.base_kernel.lengthscale = \
                models[model_idx].covar_module.base_kernel.raw_lengthscale_constraint.lower_bound + \
                torch.rand(models[model_idx].covar_module.base_kernel.lengthscale.shape) * \
                (models[model_idx].covar_module.base_kernel.raw_lengthscale_constraint.upper_bound - \
                 models[model_idx].covar_module.base_kernel.raw_lengthscale_constraint.lower_bound)

            models[model_idx].likelihood.noise = \
                models[model_idx].likelihood.noise_covar.raw_noise_constraint.lower_bound + \
                torch.rand(models[model_idx].likelihood.noise.shape) * \
                (models[model_idx].likelihood.noise_covar.raw_noise_constraint.upper_bound -
                 models[model_idx].likelihood.noise_covar.raw_noise_constraint.lower_bound)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(models[model_idx].likelihood, models[model_idx])
            models[model_idx].train()
            models[model_idx].likelihood.train()
            optimizer = torch.optim.LBFGS([
                {'params': models[model_idx].parameters()},  # Includes GaussianLikelihood parameters
            ], lr=lr_train, line_search_fn='strong_wolfe')
            try:
                for i in range(num_iter_train):
                    def closure():
                        optimizer.zero_grad()
                        # Output from model
                        output = models[model_idx](models[model_idx].train_inputs[0])
                        # Calc loss and backprop gradients
                        loss = -mll(output, models[model_idx].train_targets)
                        torch.nn.utils.clip_grad_value_(models[model_idx].parameters(), clip_grad_value)
                        loss.backward()
                        if print_train:
                            if i % step_print_train == 0:
                                print(
                                    'Iter %d/%d - Loss: %.5f   output_scale: %.5f   lengthscale: %s   noise: %.5f' % (
                                        i + 1, num_iter_train, loss.item(),
                                        models[model_idx].covar_module.outputscale.item(),
                                        str(models[
                                            model_idx].covar_module.base_kernel.lengthscale.detach().numpy()),
                                        pow(models[model_idx].likelihood.noise.item(), 0.5)
                                    ))
                        return loss

                    loss = optimizer.step(closure)
                    if loss < best_losses[model_idx]:
                        best_losses[model_idx] = loss.item()
                        best_lengthscales[model_idx] = models[model_idx].covar_module.base_kernel.lengthscale
                        best_noises[model_idx] = models[model_idx].likelihood.noise
                        best_outputscales[model_idx] = models[model_idx].covar_module.outputscale

            except Exception as e:
                print(e)

            print(
                'training process - model %d - time train %f - output_scale: %s - lengthscales: %s - noise: %s' % (
                    model_idx, time.time() - start_time, str(best_outputscales[model_idx].detach().numpy()),
                    str(best_lengthscales[model_idx].detach().numpy()),
                    str(best_noises[model_idx].detach().numpy())))

        print('training process - previous marginal log likelihood: %s - new marginal log likelihood: %s' %
              (str(previous_losses.detach().numpy()), str(best_losses.detach().numpy())))
        params_dict_list = []
        for model_idx in range(len(models)):
            params_dict_list.append({
                'covar_module.base_kernel.lengthscale': best_lengthscales[model_idx].detach().numpy(),
                'covar_module.outputscale': best_outputscales[model_idx].detach().numpy(),
                'likelihood.noise': best_noises[model_idx].detach().numpy()})
        queue.put(params_dict_list)

    def save_state(self) -> SavedState:
        saved_state = SavedState(
            train_inputs=self.x_mem, 
            train_targets=self.y_mem, 
            parameters=[model.state_dict() for model in self.models],
            constraints_hyperparams=self.config.__dict__
        )
        return saved_state


def create_models(gp_init_dict, constraints_gp, train_inputs:torch.Tensor=None, train_targets:torch.Tensor=None, num_models=None, num_inputs=None) -> "list[ExactGPModelMonoTask]":
    """
    Define gaussian process models used for predicting state transition,
    using constraints and init values for (outputscale, noise, lengthscale).

    Args:
        train_inputs (torch.Tensor or None): Input values in the memory of the gps
        train_targets (torch.Tensor or None): target values in the memory of the gps.
                                                Represent the change in state values
        gp_init_dict (dict or list of dict): Value of the hyper-parameters of the gaussian processes.
        constraints_gp (dict): See the ReadMe about parameters for information about keys
        num_models (int or None): Must be provided when train_inputs or train_targets are None.
                                    The number of models should be equal to the dimension of state,
                                    so that the transition for each state can be predicted with a different gp.
                                    Default=None
        num_inputs (int or None): Must be provided when train_inputs or train_targets are None.
                                    The number of inputs should be equal to the sum of the dimension of state
                                    and dimension of action. Default=None
        include_time (bool): If True, gp will have one additional input corresponding to the time of the observation.
                                This is usefull if the env change with time,
                                as more recent points will be trusted more than past points
                                (time closer to the point to make inference at).
                                It is to be specified only if

    Returns:
        models (list of gpytorch.models.ExactGP): models containing the parameters, memory,
                                                    constraints of the gps and functions for exact predictions
    """
    if train_inputs is not None and train_targets is not None:
        num_models = len(train_targets[0])
        models = [ExactGPModelMonoTask(train_inputs, train_targets[:, idx_model], len(train_inputs[0]))
            for idx_model in range(num_models)]
    else:
        if num_models is None or num_inputs is None:
            raise(ValueError('If train_inputs or train_targets are None, num_models and num_inputs must be defined'))
        else:
            models = [ExactGPModelMonoTask(None, None, num_inputs) for _ in range(num_models)]

    for idx_model in range(num_models):
        if constraints_gp is not None:
            if "min_std_noise" in constraints_gp.keys() and "max_std_noise" in constraints_gp.keys():
                    min_var_noise_model = torch.pow(constraints_gp['min_std_noise'][idx_model], 2)
                    max_var_noise_model = torch.pow(constraints_gp['max_std_noise'][idx_model], 2)
                    models[idx_model].likelihood.noise_covar.register_constraint("raw_noise",
                        gpytorch.constraints.Interval(lower_bound=min_var_noise_model, upper_bound=max_var_noise_model))

            if "min_outputscale" in constraints_gp.keys():
                    min_outputscale_model = constraints_gp['min_outputscale'][idx_model]
                    max_outputscale_model = constraints_gp['max_outputscale'][idx_model]
                    models[idx_model].covar_module.register_constraint("raw_outputscale",
                        gpytorch.constraints.Interval(lower_bound=min_outputscale_model, upper_bound=max_outputscale_model))

            if "min_lengthscale" in constraints_gp.keys():
                    min_lengthscale_model = constraints_gp['min_lengthscale'][idx_model]
                    max_lengthscale_model = constraints_gp['max_lengthscale'][idx_model]
                    models[idx_model].covar_module.base_kernel.register_constraint("raw_lengthscale",
                    gpytorch.constraints.Interval(lower_bound=min_lengthscale_model, upper_bound=max_lengthscale_model))

        if type(gp_init_dict) == list:
            models[idx_model].load_state_dict(gp_init_dict[idx_model])
        else:
            hypers = {'base_kernel.lengthscale': gp_init_dict['base_kernel.lengthscale'][idx_model],
                'outputscale': gp_init_dict['outputscale'][idx_model]}
            hypers_likelihood = {'noise_covar.noise': gp_init_dict['noise_covar.noise'][idx_model]}
            models[idx_model].likelihood.initialize(**hypers_likelihood)
            models[idx_model].covar_module.initialize(**hypers)
    return models


class GpModel:
    def __init__(self):
        pass

    def __forward__(self):
        pass

    def compute_covar(self, x1, x2):
        pass

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


def calculate_factorizations(x:torch.Tensor, y:torch.Tensor, models: "list[ExactGPModelMonoTask]"):
        """
        Compute iK and beta using the points in memory, which are needed to make predictions with the gaussian processes.
        These two variables only depends on data in memory, and not on input distribution,
        so they separated from other computation such that they can be computed outside the optimisation function,
        which is computed multiple times at each iteration

        Function inspired from
        https://github.com/nrontsis/PILCO/blob/6a962c8e4172f9e7f29ed6e373c4be2dd4b69cb7/pilco/models/mgpr.py#L81,
        reimplemented from tensorflow to pytorch
        Args:
            x (torch.Tensor): matrix containing the states and actions. Dim=(Nm, Ns + Na)
            y (torch.Tensor): matrix containing the states change. Dim=(Nm, Ns)
            models (list of gpytorch.models.ExactGP): list containing the gp models used to predict each state change.
                                                        Len=Ns
            Ns: number of states
            Na: number of actions
            Nm: number of points in memory

        Returns:
            iK (torch.Tensor): needed by the gaussian processes models to compute the predictions
            beta (torch.Tensor): needed by the gaussian processes models to compute the predictions

        """
        K = torch.stack([model.covar_module(x).evaluate() for model in models])
        batched_eye = torch.eye(K.shape[1]).repeat(K.shape[0], 1, 1)
        L = torch.cholesky(K + torch.stack([model.likelihood.noise for model in models])[:, None] * batched_eye)
        iK = torch.cholesky_solve(batched_eye, L)
        Y_ = (y.t())[:, :, None]
        beta = torch.cholesky_solve(Y_, L)[:, :, 0]
        return iK, beta
