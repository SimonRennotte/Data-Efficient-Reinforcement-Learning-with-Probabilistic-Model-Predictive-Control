# Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control
![control animation](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/gif_total.gif?)

Example of control agents learning from scratch
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
    * [Talks/Tutorials](###Talks/Tutorials)
    * [Papers](###Papers)
    * [Textbooks](###Textbooks)
    * [Projects](###Projects)
    
## Experiments
For each experiment, two plots allow to see the learning progress:



- A time graph that shows how the control variables evolve.
   - The top graph: states along with the predicted states and uncertainty from n time steps prior. The value of n is specified in the legend. 
   - The middle graph: actions
   - The bottom graph: The actual cost, predicted trajectory cost (mean of future predicted cost) and its uncertainty. Note that the uncertainty of the trajectory cost can be        used to identify times where the future is uncertain for the model.

- 3d visualizations that shows the Gaussian processes model and points in memory. 
     In this plot, each of the graphs on the top line represents the variation of states for the next step as a function of the current states and actions. The indices represented in the xy axis name represent either states or actions. For example, the input with index 3 represent the action for the pendulum. Action indices are defined as higher than the state indices.
     The axes of the 3d graph are chosen to represent the two inputs (state or action) with the smallest lengthscales in the gaussian process for the predicted state variation, so that the x-y axes may be different for each graph. The graphs on the bottom line represent the predicted uncertainty, and the points are the prediction errors.
     The points stored in the memory of the Gaussian process model are shown in green, and the points that are not stored in black.

### Pendulum-v0

![control animation](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/anim_pendulum.gif?)

The following figure shows the time graph for the inverted pendulum that is shown in the animation :

![stories](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/history_pendulum.png?raw=true)

We can see that the model allows to control the environment in about a hundred interactions with the environment from scratch.
As a comparison, the state of the art of model free reinforcement learning algorithms that I found in https://github.com/quantumiracle/SOTA-RL-Algorithms solves the problem in more than 15 episodes of 200 interactions with the environment. 

The gaussian process models along the points in memory are represented in the following figure.

![3d models](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/model_pendulum.png?raw=true)

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
- Constrainsts on states are not available yet
- The optimizer for actions is LBFGS directly applied on the actions. The particular structure of the problem is not used to speed the computation times, which is why the time between iterations is high. The animations displayed do not show the computation times between each iteration. 

An option has been added to repeat the predicted actions, so that a longer time horizon can be used with the MPC, which is crucial for certain environments such as the mountain car. 

## Limitations

- The cost function must be clearly defined as a squared distance function of the states/observations
- The number of time step of the mpc will greatly impact computation times. In case that a environment need the model to plan too much ahead, the computations time might become too much to solve it in real time. This can also be a problem when the dimensionality of the action space is too high. To have lower computation time, you can reduce the horizon, but it will decrease the performances.
- The dimension of the input and output of the gaussian process must stay low (below 20 approximately). 
- If too much points are stored in the memory of the gaussian process, the computation times might become very high. The computation times scale in n³.
- The current implementation will not work for gym environments with discrete states

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
The parameters of the main script are stored in parameters.json, which specifies which gym environment to use, and the parameters relative to visualizations.

For each gym environment, a json file containing the gym environment name contains all the parameters relative to this environement, and the control used.
The syntax is parameters_"gym_env_name".json
- Training parameters
- Initialisation of the models
- Constraints of the models
- Actions optimizer
- Memory

To use the model on a different gym environement, an other json file must be created, which contains the same structure and parameters, but with different values.

The plots and animations will be saved in the folder "folder_save", with the following structure:
folder_save => environment name => time and date of the run

## TODO
This github is under active development. If you detect any bugs or potential improvements, please let me know.
The following features are still to be developed :
- A faster optimizer
- Constraints on the lower and upper limit of states
- Dynamic representation of the model and the predicted trajectory in graphs updated in real time
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

### Talks/Tutorials

Gaussian processes: https://www.youtube.com/watch?v=92-98SYOdlY&list=PL93aLKqThq4hbACqDja5QFuFKDcNtkPXz&index=2

Presentation of PILCO (method similar to the one used here), by one of the coauthors: https://www.youtube.com/watch?v=AVdx2hbcsfI

Safe bayesian optimization: https://www.youtube.com/watch?v=sMfPRLtrob4

### Papers

Original paper: https://deepai.org/publication/data-efficient-reinforcement-learning-with-probabilistic-model-predictive-control

PILCO paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6654139

Marc deisenroth thesis: https://deisenroth.cc/pdf/thesis.pdf

### Textbooks

http://www.gaussianprocess.org/gpml/

### Projects

https://github.com/nrontsis/PILCO

## Cite this work
If my implementation has been useful to you, please cite this github in your research. Thank you.

## Contact me
You can contact me on linkedin: https://www.linkedin.com/in/simon-rennotte-96aa04169/

I plan to do my PHD in UMontreal or UCL in the fall of 2021 or the beginning of 2022 to improve this method and extend it to more application cases, with high dimensionality of states and actions, noise, delayed reward, etc. 
If you know someone there or work there yourself, I would like to chat to have more informations. Thank you ! 
