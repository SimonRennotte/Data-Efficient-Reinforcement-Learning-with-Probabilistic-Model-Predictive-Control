# Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control
![control animation](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/gif_total.gif?)

Example of control algorithms learning from scratch
## Overview
Implementation of the paper [Data-Efficient Reinforcement Learning with Probabilistic Model Predictive Control](https://arxiv.org/pdf/1706.06491v1.pdf) with pytorch and gpytorch.

Trial-and-error based reinforcement learning (RL) has seen rapid advancements in recent times, especially with the advent of deep neural networks. 
However, the majority of autonomous RL algorithms either rely on engineered features or a large number of interactions with the environment. 
Such a large number of interactions may be impractical in many real-world applications. 
For example, robots are subject to wear and tear and, hence, millions of interactions may change or damage the system. 
Moreover, practical systems have limitations in the form of the maximumtorque that can be safely applied.  
To reduce the number of system interactions while naturally handling constraints, we propose a model-based RL framework based on Model Predictive Control (MPC). 
In particular, we propose to learn a probabilistic transition model using Gaussian Processes (GPs) to incorporate model uncertainties into long-term predictions, thereby, 
reducing the impact of model errors. We then use MPC to find a control sequence that minimises the expected long-term cost.  
We provide theoretical guarantees for the first-order optimality in the GP-based transition models with deterministic approximate inference for long-term planning. 
The proposed framework demonstrates superior data efficiency and learning rates compared to the current state of the art.

---

Summary for non experts: the approach uses a model to control the environment. This is called Model Predictive Control (MPC) and is commonly used in process control theory. At each interaction with the real environment, the optimal action is obtained with an iterative approach using the model to predict the evolution of states given control signals over a fixed time horizon. This simulation is used multiple times to find the optimal actions in the time horizon window with a gradient descent optimizer. The first control of the time horizon is then used for the current step of the simulation. At each step, the simulation is used multiple times again.
In traditional control theory, the model is a mathematical model obtained from fondamental theory. Here the model is a gaussian process. 

Gaussian processes allow to predict the variation of states given the states and input signals, and the confidence interval of these predictions given its parameters and points stored in memory. The specificity of the paper resides in that the uncertainties are propagated during the trajectory computations, which allow us to compute the loss, but also the uncertainty of the loss in the simulation horizon. This can be used to explore more efficiently by visiting the states where the loss uncertainty is high. It can also be used to get a sense in real time of how sure the model is about the future. Uncertainty can also be used to impose security constraints. This can be done by forbidding to visit states where the uncertainty is too high, by imposing constraints on the lower bound or upper bound of the confidence interval of the states. This is already used for safe bayesian optimization. For example, it has been used [to optimize drone controllers to avoid crashes during the optimization.](https://www.youtube.com/watch?v=GiqNQdzc5TI)

For each states, one gaussian process is used that has n (number of states) + m (number of control signal outputs), and n number of outputs. Conceptually, gaussian processes can be seen as if they look at adjacent points in memory to compute the predictons at new unseen points. Depending on how far the new point is from points that are stored in memory, the uncertainty will be higher or lower.

The approach allows to learn sufficiently fast to allow online learning from scratch, which open many opportunities for RL in new applications. 
The following results are reported for the double inverted pendulum. 

![result paper](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/Article_results.png?raw=true)

## Table of contents
  * [Implementation differences from the paper](##Differences)
  * [Experiments](##Experiments)
    * [Pendulum-v0](###Pendulum-v0)
    * [MountainCarContinuous-v0](###MountainCarContinuous-v0)
  * [Limitations](##Limitations)
  * [Installation](##Installation)
  * [How to run](##Run)
  * [Ressources](##Ressources)
    * [Talks/Tutorials](###Talks/Tutorials)
    * [Papers](###Papers)
    * [Textbooks](###Textbooks)
    * [Projects](###Projects)
    

## Implementation differences from the paper

Compared to the implementation in the paper, the scripts have been designed to perform the control in one trial over a long time, which means:
- The function optimized in the mpc is the lower confidence bound of the long term predicted cost to reward exploration and avoid being stuck in a local minima.
- The environment is not reset, the learning happens in one run.
- Training of the hyper-parameters and storage of vizualisations are performed in a parallel process at regular time interval
- An option has been added to decide to include a point in the memory of the model depending on the prediction error at that point and the predicted uncertainty to avoid having too much points in memory

An option has been added to repeat the predicted actions, so that longer time horizon can be used with the MPC, which is cruciar for some environment like the mountain car.
Finally, the analytical derivatives of the cost function and gaussian processes are not used to find the optimal control. The current processing times are thus higher.

## Experiments
### Pendulum-v0

![control animation](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/anim.gif?)

Two visualisations allow to see the progress of learning:

- An history plot, which plot the relevant informations in function of the number of iteration, which can be seen on the following figure:

We can see that the model allows the control of the environment in about 100 interactions with the environment from scratch.
As a comparison, the state of the art of model free reinforcement learning algorithms, soft actor critic, defined in https://github.com/quantumiracle/SOTA-RL-Algorithms solves the problem in more than 15 episodes of 200 interactions with the environment.

![histories](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/history_example.png?raw=true)
The cost of the trajectory is the mean predicted cost on the horizon following the predicted actions of the mpc

- A 3d visualisation of the model predictions and the points in memory. 

![3d models](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/model_example.png?raw=true)
Each of the graphs represented on the upper row represents the variation of the state in function of the inputs state. 
The axis of the 3d plot are chosen to represent the two inputs with the lowest lengthscales in the model, so the x-y axes are different for the different state predictions.
The graphs represented on the lower row represent the predicted uncertainty, and the points are the prediction errors.
The points stored in the memory of the gaussian process model are represented in green, and the points not stored in black.
Note that the uncertainty represented in the 3d plot do not represent the uncertainty on the points, since the last dimension is not visible and has been set using linear regression fo the visible states input

### MountainCarContinuous-v0

The mountain car problem is a little bit different in that the number of time steps to plan in order to control the environment is higher. To avoid this problem, a parameter has been added to allow to repeat actions during planning, such that the horizon can be longer. For the shown example, 1 control time step correspond to 5 time steps where the action is maintained. If this is not used, the control is not possible.

![animation](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/anim_mountain_car.gif?)

![histories](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/history_mountain_car.png?raw=True)

![3d_models](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/model_mountain_car.png?raw=True)

As for the pendulum, the optimal control is obtained in very few steps compared to the state of the art of model-free reinforcement agents

## Limitations

- The cost function must be clearly defined as a distance function of the states/observations
- The number of time step of the mpc will greatly impact computation times. In case that a environment need the model to plan too much ahead, the computations time might become too much to solve it in real time.
- The dimension of the input and output of the model must stay low (below 20)
- The implementation is not complete yet. The analytical derivates of the fmm and lmm functions have yet to be computed to speed up the actions optimizations, 
which explains why the computation times are so important. The optimization of the hamiltonian has been replaced by the optimization of the predicted long term cost.
To have lower computation time, you can reduce the horizon, but it will decrease the performances.

## Installation
### Dependencies
numpy, gym, pytorch, gpytorch, matplotlib, threadpoolctl, scikit-learn, ffmpeg
### Install with Anaconda (recommended)
Download [anaconda](https://www.anaconda.com/products/individual)

Open an anaconda prompt window

You can then create and install the environment with:

`conda env create -f environment.yml`

And activate it with:

`conda activate gp_rl_env`

Depending on your platform, you may have to modify the yml file to install pytorch following the instructions [here](https://pytorch.org/get-started/locally/)
## Run
To use the script
The parameters of the main script are stored in parameters.json, which specifies which gym environment to use, and the usage of vizualisations.

For each gym environment, a json file containing the gym environment name contains all the parameters relative to this environement, and the control used.
The syntax is parameters_"gym_env_name".json
- Training parameters
- Initialisation of the models
- Constraints of the models
- Actions optimizer
- Memory

To use the model on a different gym environement, an other json file must be created, which contains the same structure and parameters, but with different values.

## Ressources

### Talks/Tutorials

Gaussian processes: https://www.youtube.com/watch?v=92-98SYOdlY&list=PL93aLKqThq4hbACqDja5QFuFKDcNtkPXz&index=2

Presentation of PILCO (method similar to the one used here), by one of the coauthors: https://www.youtube.com/watch?v=AVdx2hbcsfI

Safe bayesian optimization: https://www.youtube.com/watch?v=sMfPRLtrob4

### Papers

PILCO paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6654139

Original paper: https://deepai.org/publication/data-efficient-reinforcement-learning-with-probabilistic-model-predictive-control

Thesis: https://deisenroth.cc/pdf/thesis.pdf

### Textbooks

http://www.gaussianprocess.org/gpml/

### Projects

https://github.com/nrontsis/PILCO
