# Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control
![control animation](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/gif_total.gif?)

Example of control algorithms learning from scratch
## Overview
Implementation of the paper [Data-Efficient Reinforcement Learning with Probabilistic Model Predictive Control](https://arxiv.org/pdf/1706.06491v1.pdf) with pytorch and gpytorch.

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

### Concept summary
The approach uses a model to control the environment. This method is called Model Predictive Control (MPC) and is commonly used in process control theory. At each interaction with the real environment, the optimal action is obtained through an iterative approach. The model is used to evaluate certain actions over a fixed time horizon using a simulation by predicting the evolution of states with the model, and calculating the corresponding loss. This simulation is used several times at each interaction with the real world to find the optimal actions in the time horizon window with a gradient descent optimizer (for the current implementation). The first control of the time horizon is then used for the next action in the real world. 
In traditional control theory, the model is a mathematical model obtained from theory. Here, the model is a Gaussian process. 

Gaussian processes are used to predict the variation of states as a function of states and input actions. The predictions have the form of a distribution, which also allows the uncertainty of these predictions. Gaussian processes are defined by a mean and covariance function, and store previous points (states(t), actions(t), states(t+1)) in memory. To compute the new predictions, the covariance between the new points and the points stored in memory is calculated, which allows, with a little mathematics, to obtain the predicted distribution. Conceptually, Gaussian processes can be seen as if they were looking at adjacent points in memory to compute predictions at new points. Depending on the distance between the new point and the points stored in memory, the uncertainty will be greater or smaller. In our case, 
for each state, one Gaussian process is used which has n (number of states) + m (number of actions) inputs, and n number of outputs that is used to predict the variation of that state.

The specificity of the paper lies in the fact that uncertainties propagate during trajectory calculations, which allows to calculate the loss, but also the uncertainty of the loss in the window of the simulation horizon. This makes it possible to explore more efficiently by visiting states where the uncertainty of the loss is high. It can also be used to get a real-time idea of the model's certainty about the future. Uncertainty can also be used to impose security constraints. This can be done by prohibiting visits to states where the uncertainty is too high, by imposing constraints on the lower or upper limit of the state confidence interval. This method is already used for safe Bayesian optimization. For example, it has been used [to optimize UAV controllers to avoid crashes during optimization.](https://www.youtube.com/watch?v=GiqNQdzc5TI)

This approach allows learning fast enough to enable online learning from scratch, which opens up many possibilities for LR in new applications. 
The following results are reported for the double inverted pendulum. 

![result paper](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/Article_results.png?raw=true)

## Table of contents
  * [Experiments](##Experiments)
    * [Pendulum-v0](###Pendulum-v0)
    * [MountainCarContinuous-v0](###MountainCarContinuous-v0)
  * [Implementation differences from the paper](##Differences)
  * [Limitations](##Limitations)
  * [Installation](##Installation)
  * [How to run](##Run)
  * [Ressources](##Ressources)
    * [Talks/Tutorials](###Talks/Tutorials)
    * [Papers](###Papers)
    * [Textbooks](###Textbooks)
    * [Projects](###Projects)
    
## Experiments
For each experiment, two plots allow to see the learning progress:

- A time graph, which plots the evolution of states in the top left corner, and actions in the top right corner. Prediction errors and predicted uncertainty are plotted on the bottom left. The loss and cost of the trajectory as well as its uncertainty are plotted on the bottom right. 
The cost of the path is the average cost expected over the horizon as a result of the planned actions of the MPC. Note that the uncertainty of the loss over the horizon can be used to identify states where the future is uncertain for the model.

- A 3d vizualization that allows to visualize the Gaussian process model and points in memory. In this plot, each of the graphs on the top line represents the variation in status as a function of the state of the inputs and actions. By definition, the action indices are higher than the state indices.
The axes of the 3d graph are chosen to represent the two inputs (state or action) with the smallest lengthscales in the gaussian process for the predicted state variation, so that the x-y axes may be different for each graph. The graphs on the bottom line represent the predicted uncertainty, and the points are the prediction errors.
The points stored in the memory of the Gaussian process model are shown in green, and the points that are not stored in black.
Note that the uncertainty represented in the 3d graph does not represent the exact uncertainty on the points, since the last dimension is not visible and has been defined using linear regression with the visible states as input.

### Pendulum-v0

![control animation](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/anim.gif?)

The following figure shows the time graph for the inverted pendulum that is shown in the animation :

![stories](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/history_example.png?raw=true)

We can see that the model allows to control the environment in about a hundred interactions with the environment from scratch.
As a comparison, the state of the art of model free reinforcement learning algorithms that I found in https://github.com/quantumiracle/SOTA-RL-Algorithms solves the problem in more than 15 episodes of 200 interactions with the environment. 

The gaussian process models along the points in memory are represented in the following figure.

![3d models](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/model_example.png?raw=true)

### MountainCarContinuous-v0

The mountain car problem is a little bit different in that the number of time steps to plan in order to control the environment is higher. To avoid this problem, the parameter to repeat the actions has been set to 5. For the shown example, 1 control time step correspond to 5 time steps where the action is maintained. If this trick is not used, the control is not possible, or the computation times become too high.

![animation](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/anim_mountain_car.gif?)

![histories](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/history_mountain_car.png?raw=True)

![3d_models](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/model_mountain_car.png?raw=True)

As for the pendulum, the optimal control is obtained in very few steps compared to the state of the art of model-free reinforcement agents

## Implementation differences from the paper

Compared to the implementation in the document, the scripts have been designed to perform the control in one go over a long period of time, which means :
- The optimized function in the mpc is the lower confidence bound of the expected long-term cost to reward exploration and avoid getting stuck in a local minimum.
- The environment is not reset, learning is done in one go. Thus, the hyper-parameters training can not be done between trials. The learning of the hyperparameters and the storage of the visualizations are performed in a parallel process at regular time intervals in order to minimize the computation time at each control iteration.
- An option has been added to decide to include a point in the model memory depending on the prediction error at that point and the predicted uncertainty to avoid having too many points in memory. Only points with a predicted uncertainty or a prediction error greater than a threshold are stored in memory.

An option has been added to repeat the predicted actions, so that a longer time horizon can be used with the MPC, which is crucial for certain environments such as the mountain car. 

Finally, the implementation is not complete yet. analytical derivatives of the cost function and Gaussian processes are not used to find the optimal control. To be precise, the analytical derivates of the fmm and lmm functions have yet to be computed. These are needed to compute the hamiltonians used for the Pontryagin maximum principle. This explains why the computation times are so important in the current implementation that naively uses gradient descent (LBFGS) of the actions. The optimization of the hamiltonian has been replaced by the optimization of the predicted long term cost. Therefore, current processing times are higher. The animations displayed do not show the computation times between each iteration. To have lower computation time, you can reduce the horizon, but it will decrease the performances.

## Limitations

- The cost function must be clearly defined as a squared distance function of the states/observations
- The number of time step of the mpc will greatly impact computation times. In case that a environment need the model to plan too much ahead, the computations time might become too much to solve it in real time. This can also be a problem when the dimensionality of the action space is too high.
- The dimension of the input and output of the gaussian process must stay low (below 20 approximately). 
- If too much points are stored in the memory of the gaussian process, the computation times might become very high. The computation times scale in n³.
- The current implementation will not not with environement with discrete states

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

## Cite this work
If my implementation has been useful to you, please cite my work in your research. Thank you.

## Contact me
You can contact me on linkedin: https://www.linkedin.com/in/simon-rennotte-96aa04169/

I plan to do my PHD in UMontreal in the fall of 2021 or the beginning of 2022 to improve this method and extend it to more application cases, with high dimensionality, noise, delayed reward, no reward, etc. 
If you know someone there or work there yourself, I would like to chat to have more informations. Thank you ! 
