# Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control
![control animation](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/gif_total.gif?)

Control agents learning from scratch
## Overview
Unofficial implementation of the paper [Data-Efficient Reinforcement Learning with Probabilistic Model Predictive Control](https://arxiv.org/pdf/1706.06491v1.pdf) with pytorch and gpytorch.

### Abstract of the paper
Trial-and-error based reinforcement learning (RL) has seen rapid advancements in recent times, especially with the advent of deep neural networks. 
However, the majority of autonomous RL algorithms either rely on engineered features or a large number of interactions with the environment. 
Such a large number of interactions may be impractical in many real-world applications. 
For example, robots are subject to wear and tear and, hence, millions of interactions may change or damage the system. 
Moreover, practical systems have limitations in the form of the maximum torque that can be safely applied.  
To reduce the number of system interactions while naturally handling constraints, we propose a model-based RL framework based on Model Predictive Control (MPC). 
In particular, we propose to learn a probabilistic transition model using Gaussian Processes (GPs) to incorporate model uncertainties into long-term predictions, thereby, 
reducing the impact of model errors. We then use MPC to find a control sequence that minimises the expected long-term cost.  
We provide theoretical guarantees for the first-order optimality in the GP-based transition models with deterministic approximate inference for long-term planning. 
The proposed framework demonstrates superior data efficiency and learning rates compared to the current state of the art.

---

## Table of contents
  * [Experiments](##Experiments)
    * [Pendulum-v0](###Pendulum-v0)
    * [MountainCarContinuous-v0](###MountainCarContinuous-v0)
  * [Implementation differences from the paper](##Differences)
  * [Limitations](##Limitations)
  * [Installation](##Installation)
  * [How to run](##Run)
  * [TODO](##TODO)
  * [Ressources](##Ressources)
    * [Summary](###Summary)
    * [Why is this paper important](###Why is this paper important)
    * [Talks/Tutorials](###Talks/Tutorials)
    * [Papers](###Papers)
    * [Textbooks](###Textbooks)
    * [Projects](###Projects)
    
## Experiments
For each experiment, two plots that allow to see the learning progress are saved in the folder "folder_save":

- A time graph that shows how the control variables evolve.
   - The top graph: states along with the predicted states and uncertainty from n time steps prior. The value of n is specified in the legend. 
   - The middle graph: actions
   - The bottom graph: The actual cost, predicted trajectory cost (mean of future predicted cost) and its uncertainty. Note that the uncertainty of the trajectory cost can be        used to identify times where the future is uncertain for the model.

- 3d visualizations that shows the Gaussian processes model and points in memory. 
     In this plot, each of the graphs of the top line represents the variation of states for the next step as a function of the current states and actions. The indices represented in the xy axis name represent either states or actions. For example, the input with index 3 represent the action for the pendulum. Action indices are defined as higher than the state indices.
     The axes of the 3d graph are chosen to represent the two inputs (state or action) with the smallest lengthscales in the gaussian process for the predicted state variation, so that the x-y axes may be different for each graph. The graphs of the bottom line represent the predicted uncertainty, and the points are the prediction errors.
     The points stored in the memory of the Gaussian process model are shown in green, and the points that are not stored in black.
     
During the control, a dynamic graph similar to the time graph allows to see the evolution of the states, action and costs, but also shows the predicted states, actions and costs computed by the model for the MPC.

### Pendulum-v0
The following figure shows the mean cost over 10 runs:

![losses](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/Cost_runs_Pendulum-v0.png?)

We can see that the model allows to control the environment in less than hundred interactions with the environment from scratch.
As a comparison, the state of the art of model free reinforcement learning algorithms in https://github.com/quantumiracle/SOTA-RL-Algorithms solves the problem in more than 15 episodes of 200 interactions with the environment.

The following figures and animation shows an example of control.

![control animation](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/anim_pendulum.gif?)

The following figure shows the time graph for the inverted pendulum that is shown in the animation :

![stories](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/history_pendulum.png?raw=true) 

The gaussian process models along the points in memory are represented in the following figure.

![3d models](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/model_pendulum.png?raw=true)

The following animation shows the dynamic graph updated in real time (for another run).

![animation_real_time](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/control_anim_Pendulum-v0.gif?)

### MountainCarContinuous-v0

The mountain car problem is a little bit different in that the number of time steps to plan in order to control the environment is higher. To avoid this problem, the parameter to repeat the actions has been set to 3. For the shown example, 1 control time step correspond to 3 time steps where the action is maintained. If this trick is not used, the control is not possible, or the computation times become too high.

The mean costs over 10 runs can be seen in the following figure:

![losses](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/Cost_runs_MountainCarContinuous-v0.png?)

As for the pendulum, the optimal control is obtained in very few steps compared to the state of the art of model-free reinforcement agents

The following figures and animation shows an example of control.

![animation](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/anim_mountain_car.gif?)

![histories](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/history_mountain_car.png?raw=True)

![3d_models](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/model_mountain_car.png?raw=True)

The following animation shows the dynamic graph updated in real time (for another run).

![animation_real_time](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/control_anim_MountainCarContinuous-v0.gif?)

## Implementation differences from the paper

Compared to the implementation in the document, the scripts have been designed to perform the control in one go over a long period of time, which means :
- The optimized function in the mpc is the lower confidence bound of the expected long-term cost to reward exploration and avoid getting stuck in a local minimum.
- The environment is not reset, learning is done in one go. Thus, the hyper-parameters training can not be done between trials. The learning of the hyperparameters and the storage of the visualizations are performed in a parallel process at regular time intervals in order to minimize the computation time at each control iteration.
- An option has been added to decide to include a point in the model memory depending on the prediction error at that point and the predicted uncertainty to avoid having too many points in memory. Only points with a predicted uncertainty or a prediction error greater than a threshold are stored in memory.
- The optimizer for actions is LBFGS directly applied on the actions. The particular structure of the problem is not used to speed the computation times. The animations displayed do not show the computation times between each iteration. 

An option has been added to repeat the predicted actions, so that a longer time horizon can be used with the MPC, which is crucial for certain environments such as the mountain car. 

## Limitations

- The cost function must be clearly defined as a squared distance function of the states and actions from a reference point.
- The number of time step of the mpc will greatly impact computation times. In case that a environment need the model to plan too much ahead, the computations time might become too much to solve it in real time. This can also be a problem when the dimensionality of the action space is too high. To have lower computation time, you can reduce the horizon, but it will decrease the performances.
- The dimension of the input and output of the gaussian process must stay low (below 20 approximately), which limits application to cases with low dimensionality of the states and actions. 
- If too much points are stored in the memory of the gaussian process, the computation times might become too high per iteration.
- The current implementation will not work for gym environments with discrete states or actions.
- No guarentee are given for the time per iteration.
- Actions must have an effect on the observation of the next observed step. Delay are not supported in the model. Observation must unequivocally describe the system states.
- Sensible to observation noise: it will impact the memory of the gps and thus future predictions uncertainty.
## Installation
### Dependencies
numpy, gym, pytorch, gpytorch, matplotlib, scikit-learn, ffmpeg
### Install with Anaconda (recommended)
Download [anaconda](https://www.anaconda.com/products/individual)

Open an anaconda prompt window

You can then create and install the environment with:

`conda env create -f environment.yml`

And activate it with:

`conda activate gp_rl_env`

Depending on your platform, you may have to modify the yml file to install pytorch following the instructions [here](https://pytorch.org/get-started/locally/)
## Run

Once your virtual environment is activated, write: python main.py

All parameters are stored in two two json files.
- The parameters of the main script are stored in parameters.json, which specifies which gym environment to use, the parameters relative to visualizations, and the loss over multiple runs must be computed.

- For each gym environment, a json file containing the gym environment name contains all the parameters relative to this environement, and the control used.
The syntax is parameters_"gym_env_name".json

To use the model on a different gym environement, an other json file must be created, which contains the same structure and parameters, but with different values.

The plots and animations will be saved in the folder "folder_save", with the following structure:
folder_save => environment name => time and date of the run

### Json parameters to be specified for each gym environment
- hyperparameters_init: initial values of the hyperparameters of the gaussian processes
    - noise_std: vector representing the standard deviation of the uncertainty of predictions of the gaussian processes at known points (dim=(number of states))
    - lengthscale: matrix representing the lengthscales for each input, and for each gaussian process (dim=(number of states, number of input))
    - scale: vector representing the scale of the gaussian processes (dim=(number of states)))

- params_constraints: constrainsts on the hyperparameters of the gaussian processes 
    - min_std_noise: minimum value of the parameter noise_std (dim=scalar)
    - max_std_noise: maximum value of the parameter noise_std (dim=scalar)
    - min_outputscale: minimum value of the outputscales (dim=scalar)
    - max_outputscale: maximum value of the outputscales (dim=scalar)
    - min_lengthscale: minimum value of the lengthscales (dim=scalar)
    - max_lengthscale: maximum value of the lengthscales (dim=scalar)
     
- params_controller: parameters relative to the cost function and MPC
    - target_state: The value of the states to attain to minimize the cost (dim=(dimension of states))
    - weights_target_state: Weigths of each state dimension in the cost function (dim=(dimension of states))
    - weights_target_state_terminal_cost: Weigths of each state dimension in the terminal cost function, at the end of the prediction horizon (dim=(dimension of states))
    - target_action: The value of the actions to attain to minimize the cost (dim=(dimension of actions))
    = weights_target_action: Weigths of each action dimension in the cost function (dim=(dimension of actions))
    - s_observation: variance of the observations of states, if there is observation noise (dim=(dimension of states))
    - len_horizon: length of the horizon used to find the optinal actions. The total horizon length in timesteps = len_horizon * num_repeat_actions (dim=scalar)
    - num_repeat_actions: number of timesteps to repeat the planned actions
    - exploration_factor: The value to be minimized is the sum of predicted cost - exploration_factor * sum of the predicted cost uncertainty. A higher value will lead to more exploration (dim=scalar) 
    - limit_derivative_actions: if set to 1, the variation of the normalized actions will be limited, from timestep to timestep by the parameter max_derivative_actions_norm (dim=scalar)
    - max_derivative_actions_norm: limitation on the variation of normalized actions from on control step to another if limit_derivative_actions is set to 1. (dim=(dimension of actions))
    - clip_lower_bound_cost_to_0: if set to 1, the optimized cost (with exploration parameter) will be clipped to 0 if negative.
    - compute_factorization_each_iteration: If set to 0, the factorization of the gaussian processes will only computed after each time the hyper parameters of the gaussian processes are trained, which means that new points in the gps memory will not be used until the end of the next training process. This reduce iteration times if there are many points in the gaussian process memory but reduces performances.

- params_constraints_states: 
     - use_constraints: if set to 1, the constraints will be used, if set to 0, it will be ignored
     - states_min: minimum allowed value of the states (dim=(number of states))
     - states_max: maximum allowed value of the states (dim=(number of states))
     - area_penalty_multiplier: At the moment, constraints on the states are added as a penalty in the predicted cost trajectory. The value of this penalty is the area of the        predicted state distribution that violate the constraints. This penalty is multiplied by this parameter 
     
- params_train: parameters used for the training of the gaussian processes hyper parameters, done in a parallel process
    - lr_train learning rate
    - n_iter_train": number of training iteration
    - train_every_n_points: training will occur at constant time interval. This parameter set the frequency of training process in number of time steps
    - clip_grad_value: maximum value of the gradient of the trained parameters, for more stability
    - print_train: if set to 1, the values optimized will be printed during training
    - step_print_train: if print_train is set to 1, this parameter specifies the frequency of printing during training
     
- params_init: 
    - num_random_actions_init: number of inital random actions (one action is multiple time steps if num_repeat_actions is different than 1)
     
- params_actions_optimizer: parameters of the optimizer used to find the optimal actions by minimizing the lower bound of the predicted cost. See the scipy documentation: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html?highlight=minimize Note: the jacobian is used for optimization, so the parameters eps is not used.
     
- params_memory: parameters relative to memory storage. Every point is not stored in memory. Either the error of the prediction must be above a thresold or the standard deviation of the prediction uncertainty must be above a threshold
    - min_error_prediction_prop_for_storage: minimum prediction error for each of the predicted states (dim=(number of states))
    - min_prediction_std_prop_for_storage: minimum predicted standard deviation for each of the predicted states (dim=(number of states))
     
- num_steps_env: total number of steps in the environment
 
## TODO
This github is under active development. If you detect any bugs or potential improvements, please let me know.
The following features are still to be developed :
- Using a multiple shooting method for optimization for faster optimization, and (?) better constraints, from https://hal.inria.fr/inria-00390435/file/Diehl.pdf
- Updating the dynamic graph in a parallel process
- Providing guarentee on the maximum time per iteration
- Gym environment with discrete states and actions

## Ressources

### Summary
The approach uses a model to control the environment. This family of methods are called Model Predictive Control (MPC). At each interaction with the real environment, the optimal action is obtained through an iterative approach. The model is used to evaluate certain actions over a fixed time horizon in a simulation of the environment with the learned model. This simulation is used several times at each interaction with the real world to find the optimal actions in the time horizon window with a optimizer. The first control of the time horizon is then used for the next action in the real world. In traditional control theory, the model is a mathematical model obtained from theory. Here, the model is a Gaussian process that learns from observed data. 

Gaussian processes are used to predict the variation of states as a function of states and input actions. The predictions have the form of a distribution, which also allows the uncertainty of these predictions. Gaussian processes are defined by a mean and covariance function, and store previous points (states(t), actions(t), (states(t+1) - states(t))) in memory. To compute the new predictions, the covariance between the new points and the points stored in memory is calculated, which allows, with a little mathematics, to obtain the predicted distribution. Conceptually, Gaussian processes can be seen as if they were looking at adjacent points in memory to compute predictions at new points. Depending on the distance between the new point and the points stored in memory, the uncertainty will be greater or smaller. In our case, 
for each state, one Gaussian process is used which has n (number of states) + m (number of actions) inputs, and 1 output that is used to predict the variation of that state.

The specificity of the paper lies in the fact that uncertainties propagate during trajectory calculations, which allows to calculate the loss, but also the uncertainty of the loss in the window of the simulation horizon. This makes it possible to explore more efficiently by visiting states where the uncertainty of the loss is high. It can also be used to get a real-time idea of the model's certainty about the future. Uncertainty can also be used to impose security constraints. This can be done by prohibiting visits to states where the uncertainty is too high, by imposing constraints on the lower or upper limit of the state confidence interval. This method is already used for safe Bayesian optimization. For example, it has been used [to optimize UAV controllers to avoid crashes during optimization.](https://www.youtube.com/watch?v=GiqNQdzc5TI)

This approach allows learning fast enough to enable online learning from scratch, which opens up many possibilities for Reinforcement Learning in new applications, with some more research. 
The following results are reported in the original paper for the double inverted pendulum. 

![result paper](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/Article_results.png?raw=true)

Finally, an interesting feature of using this model based reinforcement learning method that was not investigated in the original paper, is the possibility to learn the model from trajectories that were obtained by other controllers, such as humans. It is thus possible to show some examples of good actions in an environment before using this method to show the agent how the environment works and speed the learning even more.

### Why is this paper important?
Currently, real-world applications of model-free reinforcement learning algorithms are limited due to the number of interactions they require with the environment.

There is a debate within the reinforcement learning community about the use of model-based reinforcement learning algorithms to improve sample efficiency, but the extent to which it can improve sample efficiency is unknown.

With all the limitations that this method presents, it shows that for the applications on which it can be used, the same learning as for state-of-the-art model-free algorithms (to the extent of my knowledge) can be done with 10 to 20 times less interaction with the environment for the tests I used.

This increased efficiency can be explained by different reasons, and open the search for algorithms with the same improvement in sample efficiency but without the limitations mentioned above.

For example, the future predicted reward (or loss) is predicted as a distribution. By maximizing the upper confidence limit of rewards, future states with high reward uncertainty are encouraged, allowing for effective exploration.

Maximizing future state uncertainty could also be used to explore environments without rewards.

If future research removes the limitations of this method, this type of data efficiency could be used for real world applications where real-time learning is required and thus open many new applications for reinforcement learning.
### Talks/Tutorials

Gaussian processes: https://www.youtube.com/watch?v=92-98SYOdlY&list=PL93aLKqThq4hbACqDja5QFuFKDcNtkPXz&index=2

Presentation of PILCO by Marc Deisenroth: https://www.youtube.com/watch?v=AVdx2hbcsfI (method that uses the same gaussian process model, but without an MPC controller)

Safe bayesian optimization: https://www.youtube.com/watch?v=sMfPRLtrob4

### Papers

Original paper: https://deepai.org/publication/data-efficient-reinforcement-learning-with-probabilistic-model-predictive-control

PILCO paper that describes the moment matching approximation: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6654139

Marc deisenroth thesis: https://deisenroth.cc/pdf/thesis.pdf

### Textbooks

http://www.gaussianprocess.org/gpml/

### Projects

https://github.com/nrontsis/PILCO

## Contact me
You can contact me on linkedin: https://www.linkedin.com/in/simon-rennotte-96aa04169/ 
or by email: simon.rennotte2@gmail.com

I plan to do my PHD in UMontreal or UCL in the beginning of 2022 to improve this method and extend it to more application cases, with high dimensionality of states and actions, noise, delayed reward, etc. 
If you know someone there or work there yourself, I would like to chat to have more informations. Thank you ! 
