# Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control
![control animation](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/gif_total.gif?)


<p align="middle">
  <img src="https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/control_anim_Pendulum-v0.gif?" width="80%" />
</p>

<p align="middle">
<img src="https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/model_pendulum.png?raw=true" width="80%" /> 
</p>

## Overview
Unofficial implementation of the paper [Data-Efficient Reinforcement Learning with Probabilistic Model Predictive Control](https://arxiv.org/pdf/1706.06491v1.pdf) with Pytorch and GPyTorch.

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
  * [Results](#results)
    * [Pendulum-v0](#pendulum-v0)
    * [MountainCarContinuous-v0](#mountaincarcontinuous-v0)
  * [Usage](#Usage)
    * [Installation](#installation)
    * [How to run](#run)
  * [Resources](#resources)
    * [Brief explanation of the method](#brief-explanation)
    * [Why is this paper important](#why-is-this-paper-important)
    * [Remarks](#remarks)
        * [Differences from the paper](#differences)
        * [Limitations](#limitations)
    * [Talks/Tutorials](#talks-tutorials)
    * [Papers](#papers)
    * [Textbooks](#textbooks)
    * [Projects](#projects)
 
<a name="results"/>
    
## Results
For each experiment, two plots allow to see and understand the control. 

- 2d plots showing the states, actions and costs during control
   - The top graph shows the states along with the predicted states and uncertainty from n time steps earlier. The value of n is specified in the legend. 
   - The middle graph shows the actions
   - The bottom graph shows the real cost alongside with the predicted trajectory cost, which is the mean of future predicted cost, and its uncertainty.

- 3d plots showing the Gaussian processes models and points in memory. 
     In this plot, each of the graphs of the top line represents the change in states for the next step as a function of the current states and actions. 
     The indices represented in the xy axis name represent either states or actions. 
     For example, the input with index 3 represent the action for the pendulum. Action indices are defined as higher than the state indices.
     As not every input of the gp can be shown on the 3d graph, 
     the axes of the 3d graph are chosen to represent the two inputs (state or action) with the smallest lengthscales.
     So, the x-y axes may be different for each graph. 
     The graphs of the bottom line represent the predicted uncertainty, and the points are the prediction errors.
     The points stored in the memory of the Gaussian process model are shown in green, 
     and the points that are not stored because they were too similar to other points already in memory are represented in black.
     
During the control, a dynamic graph similar to the 2d plot described above allows to see the evolution of the 
states, action and costs, but also shows the predicted states, actions and costs computed by the model for the MPC. 
The predicted future states, actions and loss are represented with dashed lines, along with their confidence interval (2 standard deviation).

<a name="pendulum-v0"/>

### Pendulum-v0
The following figure shows the mean cost over 10 runs:

<p align="middle">
  <img src="https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/Cost_runs_Pendulum-v0.png?" width="80%" />
</p>

We can see that the model allows to control the environment in less than hundred interactions with the environment from scratch.
As a comparison, the state of the art of model free reinforcement learning algorithms in https://github.com/quantumiracle/SOTA-RL-Algorithms solves the problem in more than 15 episodes of 200 interactions with the environment.

The following figures and animation shows an example of control.

<p align="middle">
  <img src="https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/anim_pendulum.gif?" width="80%" />
</p>

The following figure shows the 2d graphs for the inverted pendulum that is shown in the animation.

<p align="middle">
  <img src="https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/history_pendulum.png?raw=true" width="80%" />
</p>

And the gaussian process models and the points in memory:

<p align="middle">
  <img src="https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/model_pendulum.png?raw=true" width="80%" />
</p>

The dynamic graph updated in real-time:

<p align="middle">
  <img src="https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/control_anim_Pendulum-v0.gif?" width="80%" />
</p>

<a name="mountaincarcontinuous-v0"/>

### MountainCarContinuous-v0

The mountain car problem is different in that the number of time steps to plan in order to control the environment is higher. To avoid this problem, the parameter to repeat the actions has been set to 5. For the shown example, 1 control time step corresponds to 5 time steps where the action is held constant. If this trick is not used, the control is not possible, or the computation times become too high.

The mean costs over 10 runs can be seen in the following figure:

<p align="middle">
  <img src="https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/Cost_runs_MountainCarContinuous-v0.png?" width="80%" />
</p>

As for the pendulum, the optimal control is obtained in very few steps compared to the state of the art of model-free reinforcement agents

The following figures and animation shows an example of control.

<p align="middle">
  <img src="https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/anim_mountain_car.gif?" width="80%" />
</p>

<p align="middle">
  <img src="https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/history_mountain_car.png?raw=True" width="80%" />
</p>

<p align="middle">
  <img src="https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/model_mountain_car.png?raw=True" width="80%" />
</p>

<p align="middle">
  <img src="https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/control_anim_MountainCarContinuous-v0.gif?" width="80%" />
</p>

<a name="usage"/>

## Usage

<a name="installation"/>

### Installation
#### Dependencies
numpy, gym, pytorch, gpytorch, matplotlib, scikit-learn, ffmpeg
#### Install with anaconda

Download [anaconda](https://www.anaconda.com/products/individual)
Open an anaconda prompt window:
```
git clone https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control
cd Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control
conda env create -f environment.yml
conda activate gp_rl_env
python main.py
```

<a name="run"/>

### Run for your gym environment

To run the script using your environment, you must first define it as a gym environment, 
then create two json files inside the folder params that contains all the parameters relative to the control.

- The parameters of the main script are stored in main_parameters_env.json, which specifies:
    - Which gym environment to use, 
    - The parameters relative to visualizations. 
    - The number of runs to perform for the computation of mean losses. 
        If it is set to 1, the mean losses will not be computed.

- For each gym environment, a json file containing all the parameters relative to this environment for the control used.
The syntax is parameters_"gym_env_name".json

The plots and animations will be saved in the folder "folder_save", with the following structure:
folder_save => environment name => time and date of the run

For more information about the parameters, see PARAMETERS.md

<a name="resources"/>

## Resources

<a name="brief-explanation"/>

### Brief explanation of the method
The approach uses a model to control the environment. 
This family of methods are called Model Predictive Control (MPC). 
At each interaction with the real environment, the optimal action is obtained through an iterative approach. 
The model is used to evaluate certain actions over a fixed time horizon by simulating the environment. 
This simulation is used several times with different actions at each interaction with the real world to find the optimal actions in the time horizon window. 
The first control of the time horizon is then used for the next action in the real world. 
In traditional control theory, the model is a mathematical model obtained from theory. 
Here, the model is a Gaussian process that learns from observed data. 

Gaussian processes are used to predict the change of states as a function of states and actions. 
The predictions have the form of a distribution, which also allows the uncertainty of these predictions. 
Gaussian processes are defined by a mean and covariance function, and store previous points (states(t), actions(t), (states(t+1) - states(t))) in memory. 
To compute new predictions, the covariance between the new points and the points stored in memory is calculated, 
which allows, with a little mathematics, to get the predicted distribution. 
Conceptually, Gaussian processes can be seen as if they were looking at adjacent points in memory to compute predictions at new points. 
Depending on the distance between the new point and the points stored in memory, the uncertainty will be greater or smaller. 
In our case, for each state, one Gaussian process is used which has n (number of states) + m (number of actions) inputs, 
and 1 output used to predict the variation of that state.

One specificity of the paper is that for this method, 
uncertainties propagate during trajectory calculations which allows to calculate the uncertainty of the loss in the window of the simulation horizon. 
This makes it possible to explore more efficiently by rewarding states where the uncertainty of the loss is high. 
It can also be used to get a real-time idea of the model's certainty about the future. 
Uncertainty can also be used to impose security constraints. 
This can be done by prohibiting visits to states where the uncertainty is too high by imposing constraints on the lower or upper limit of the state confidence interval. 
This method is already used for safe Bayesian optimization. 
For example, it has been used [to optimize UAV controllers to avoid crashes during optimization.](https://www.youtube.com/watch?v=GiqNQdzc5TI)

This approach allows learning fast enough to enable online learning from scratch, 
which opens up many possibilities for Reinforcement Learning in new applications with some more research. 

<a name="why-is-this-paper-important"/>

### Why is this paper important?
Currently, real-world applications of model-free reinforcement learning algorithms are limited due to the number of interactions they require with the environment.

There is a debate within the reinforcement learning community about the use of model-based reinforcement learning algorithms to improve sample efficiency, but the extent to which it can improve sample efficiency is unknown.

With all the limitations that this method presents, it shows that for the applications on which it can be used, the same learning as for state-of-the-art model-free algorithms (to the extent of my knowledge) can be done with 10 to 20 times less interaction with the environment for the tests I used.

This increased efficiency can be explained by different reasons, and open the search for algorithms with the same improvement in sample efficiency but without the limitations mentioned above.

For example, the future predicted reward (or cost) is predicted as a distribution. By maximizing the upper confidence limit of rewards, future states with high reward uncertainty are encouraged, allowing for effective exploration.

Maximizing future state uncertainty could also be used to explore environments without rewards.

If future research removes the limitations of this method, this type of data efficiency could be used for real world applications where real-time learning is required and thus open many new applications for reinforcement learning.

<a name="talks-tutorials"/>

### Remarks

<a name="differences"/>

#### Implementation differences from the paper

Compared to the implementation in the paper, the scripts have been designed to perform the control over a long time without any reset, which means :
- The optimized function in the mpc is the lower confidence bound of the expected long-term cost to reward exploration and avoid getting stuck in a local minimum.
- The environment is not reset, learning is done in one go. Thus, the hyper-parameters training can not be done between trials. The learning of the hyperparameters and the storage of the visualizations are performed in a parallel process at regular time intervals in order to minimize the computation time at each control iteration.
- An option has been added to decide to include a point in the model memory depending on the prediction error at that point and the predicted uncertainty to avoid having too many points in memory. Only points with a predicted uncertainty or a prediction error greater than a threshold are stored in memory.
- The optimizer for actions is LBFGS

<a name="limitations"/>

#### Limitations

- The cost function must be clearly defined as a squared distance function of the states and actions from a reference point.
- The number of time step of the mpc will greatly impact computation times. If an environment needs the model to plan too much ahead, the computations time might become too much to solve it in real-time. This can also be a problem when the dimensionality of the action space is too high. To have lower computation time, you can reduce the horizon length, but it might decrease the performances.
- The dimension of the input and output of the gaussian process must stay low (below 20 approximately), which limits application to cases with low dimensionality of the states and actions.
- If too much points are stored in the memory of the gaussian process, the computation times might become too high per iteration.
- The current implementation will not work for gym environments with discrete states or actions.
- No guarantee is given for the time per iteration.
- Actions must have an effect on the observation of the next observed step. Delays are not supported in the model. Observation must unequivocally describe the system states.
- Sensitivity to observation noise: it will impact the memory of the gps and thus future predictions uncertainty.

### Talks/Tutorials

Gaussian processes: https://www.youtube.com/watch?v=92-98SYOdlY&list=PL93aLKqThq4hbACqDja5QFuFKDcNtkPXz&index=2

Presentation of PILCO by Marc Deisenroth: https://www.youtube.com/watch?v=AVdx2hbcsfI (method that uses the same gaussian process model, but without an MPC controller)

Safe Bayesian optimization: https://www.youtube.com/watch?v=sMfPRLtrob4

<a name="papers"/>

### Papers

Original paper: https://deepai.org/publication/data-efficient-reinforcement-learning-with-probabilistic-model-predictive-control

PILCO paper that describes the moment matching approximation used for states uncertainty propagation: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6654139

Marc Deisenroth thesis: https://deisenroth.cc/pdf/thesis.pdf

<a name="textbooks"/>

### Textbooks

http://www.gaussianprocess.org/gpml/

<a name="projects"/>

### Projects

https://github.com/nrontsis/PILCO

<a name="contact-me"/>

## Contact me
You can contact me on Linkedin: https://www.linkedin.com/in/simon-rennotte-96aa04169/
or by email: simon.rennotte@protonmail.com

I plan to do my PhD at Mila in Montreal in the beginning of 2022 to improve this method and extend it to more application cases, with high dimensionality of states and actions, noise, delayed reward, etc. 
If you know someone there or work there yourself, I would like to chat to have more information. Thank you ! 
