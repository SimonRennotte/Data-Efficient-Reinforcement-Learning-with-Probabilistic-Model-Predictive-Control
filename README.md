# Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control
![control animation](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/anim.gif?)

Example of control algorithm learning from scratch in less than 100 interactions with the environment
## Overview
Implementation of the paper [Data-Efficient Reinforcement Learning with Probabilistic Model Predictive Control](https://arxiv.org/pdf/1706.06491v1.pdf) with pytorch and gpytorch.

Trial-and-error based reinforcement learning (RL) has seen rapid advancements in recent times, especially with the advent of deep neural networks. 
However, the majority of autonomous RL algorithms either rely on engineered features or a large number of interactions with the environment. 
Such a large number of interactions may be impractical in many real-world applications. 
For example, robots are subject to wear and tear and, hence, millions of interactions may change or damage the system. 
Moreover, practical systems have limitations in the form of the maximumtorque that can be safely applied.  
To reduce the number of system interactionswhile naturally handling constraints, we propose a model-based RL frameworkbased on Model Predictive Control (MPC). 
In particular, we propose to learn aprobabilistic transition model using Gaussian Processes (GPs) to incorporate modeluncertainties into long-term predictions, thereby, 
reducing the impact of modelerrors. We then use MPC to find a control sequence that minimises the expectedlong-term cost.  
We provide theoretical guarantees for the first-order optimality in the GP-based transition models with deterministic approximate inference forlong-term planning. 
The proposed framework demonstrates superior data efficiency and learning rates compared to the current state of the art.

---

Summary for non experts: the approach uses a model a model to predict and control the environment. This is called Model Predictive Control (MPC) and is commonly used in process control theory. At each interaction with the real environment, the optimal control is obtained with an iterative approach using the model to predict the evolution of states given control signals over a time horizon, which is chosen and fixed. The simulation is used multiple times to find the optimal controls in the time horizon with a gradient descent optimizer. The first control of the time horizon is then used for the current step of the simulation. At each step, the approach is used again.
In traditional control theory, the model is a mathematical model obtained from fondamental theory. Here the model is a gaussian process. 

Gaussian process allow to predict the variation of states given the states and input signals, and the confidence interval of these predictions given its parameters and points stored in memory. The specificity of the article resides in that the uncertainties are propagated during the trajectory computations, which allow us to compute the loss, but also the uncertainty of the loss. This can be used to explore more efficiently by visiting the states where the uncertainty is high, and to get a sense in real time of how sure the model is. Uncertainty can also be used for to use security constraint. This is done by forbidding to visit states where the uncertainty is too high (constraints on the lower bound or upper bound of the confidence interval of the states). This is already used for safe bayesian optimization to optimize, for example, ![drone controllers to avoid crashes during the optimization.]{https://www.youtube.com/watch?v=GiqNQdzc5TI}

For each states, one gaussian process is used that has n (number of states) + m (number of control signal outputs), and n number of outputs. Simply put, gaussian processes look at how far we are from points that are stored in memory to make new predictions. Depending on how far we are, the uncertainty will be higher or lower. The gaussian process models obtained for each environmenent are represented below.

The approach allows to learn sufficiently fast to allow online learning from scratch, which open many opportunities for RL in new applications. 
The following results are reported for the double inverted pendulum. 

![result paper](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/Article_results.png?raw=true)

## Table of contents
  * [Experiments](##Experiments)
    * [Pendulum-v0](###Pendulum-v0)
  * [Methodology](##Methodology)
  * [Limitations](##Limitations)
  * [Installation](##Installation)
  * [How to run](##Run)
  * [Ressources](##Ressources)
    * [Talks/Tutorials](###Talks/Tutorials)
    * [Papers](###Papers)
    * [Textbooks](###Textbooks)
    * [Projects](###Projects)
    
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

### Mountain car

The mountain car problem is a little bit different in that the number of time steps to plan in order to control the environment is higher. To avoid this problem, a parameter has been added to allow to repeat actions during planning, such that the horizon can be longer. For the shown example, 1 time step correspond to 5 time steps where the action is maintained. If this is not used, the control is not possible. In this particular example, 75 x 5 random random steps have been used as initialization.

![animation](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/anim_mountain_car.gif?)

![histories](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/history_mountain_car.png?raw=True)

![3d_models](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/model_mountain_car.png?raw=True)

As for the pendulum, the optimal control is obtained in very few steps compared to the state of the art of model-free reinforcement agents
## Methodology

Compared to the implementation in the paper, the scripts have been designed to perform the control in one trial over a long time, which means:
- The environment is not reset
- Training of the hyper-parameters and storage of vizualisations are performed in a parallel process at regular time interval
- An option has been added to decide to include a point in the memory of the model depending on the prediction error at that point and the predicted uncertainty
- The function optimized in the mpc is the lower confidence bound of the long term predicted cost to reward exploration and avoid being stuck in a local minima.

These models are obtained without using any paramterics models. The only prior used is in the initializations values of the hyper-parameters of the gaussian process.
Yet, the control is stabilized most of the time before 100 iterations, with 30 iterations being random actions.

## Limitations

- The cost function must be clearly defined as a distance function of the states/observations
- The number of time step of the mpc will greatly impact computation times. In case that a environment need the model to plan too much ahead, the computations time might become too much to solve it in real time. This can happen if the sampling time between points is too low. An example of that is the mountain car problem.
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

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6654139

https://deepai.org/publication/data-efficient-reinforcement-learning-with-probabilistic-model-predictive-control

https://deisenroth.cc/pdf/thesis.pdf

### Textbooks

http://www.gaussianprocess.org/gpml/

### Projects

https://github.com/nrontsis/PILCO
