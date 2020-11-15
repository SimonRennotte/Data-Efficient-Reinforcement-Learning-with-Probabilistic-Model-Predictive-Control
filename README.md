# Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control
Implementation of the paper Data-Efficient Reinforcement Learning with Probabilistic Model Predictive Control

![result paper](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/Article_results.png?raw=true)

Abstract:
Trial-and-error based reinforcement learning (RL) has seen rapid advancementsin recent times, especially with the advent of deep neural networks. 
However, themajority of autonomous RL algorithms either rely on engineered features or a largenumber of interactions with the environment. 
Such a large number of interactionsmay be impractical in many real-world applications. 
For example, robots are subjectto wear and tear and, hence, millions of interactions may change or damage thesystem. 
Moreover, practical systems have limitations in the form of the maximumtorque that can be safely applied.  
To reduce the number of system interactionswhile naturally handling constraints, we propose a model-based RL frameworkbased on Model Predictive Control (MPC). 
In particular, we propose to learn aprobabilistic transition model using Gaussian Processes (GPs) to incorporate modeluncertainties into long-term predictions, thereby, 
reducing the impact of modelerrors. We then use MPC to find a control sequence that minimises the expectedlong-term cost.  
We provide theoretical guarantees for the first-order optimalityin the GP-based transition models with deterministic approximate inference forlong-term planning. 
The proposed framework demonstrates superior data efficiency and learning rates compared to the current state of the art.

The papers used for the model are:

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6654139

https://deepai.org/publication/data-efficient-reinforcement-learning-with-probabilistic-model-predictive-control

https://deisenroth.cc/pdf/thesis.pdf

The following videos will allow to understand the concepts more quickly:

https://www.youtube.com/watch?v=92-98SYOdlY&list=PL93aLKqThq4hbACqDja5QFuFKDcNtkPXz&index=2

https://www.youtube.com/watch?v=AVdx2hbcsfI

The implementation is not complete yet. The analytical derivates of the fmm and lmm functions have yet to be computed to speed up the actions optimizations, 
which explains why the computation times are so important. The optimization of the hamiltonian has been replaced by the optimization of the predicted long term cost.
To have lower computation time, you can reduce the horizon, but it will decrease the performances.

Compared to the implementation described in the papers above, the scripts have been designed to perform the control in one trial over a long time, which means:
- The environment are not reset
- Training of the hyper-parameters and vizualisations are recurrently performed in a parallel process
- An option has been added to decide to include a point in the memory of the gps depending on the prediction error at that point and the predicted uncertainty
- The function optimized is the lower confidence bound of the long term predicted cost to reward exploration and avoid being stuck in a local minima.

-The controller in action:

![control animation](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/anim.gif?)

Two visualisations allow to see the progress of learning:
- An history plot, which plot the relevant informations in function of the number of iteration, which can be seen on the following figure:

![histories](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/history_example.png?raw=true)
The cost of the trajectory is the mean predicted cost on the horizon, obtained by the model following the actions of the mpc

- A 3d visualisation of the model predictions and the points in memory. 

![3d models](https://github.com/SimonRennotte/Data-Efficient-Reinforcement-Learning-with-Probabilistic-Model-Predictive-Control/blob/master/images/model_3d_example.png?raw=true)
Each of the graphs represented on the upper row represents the variation of the state in function of the inputs state. 
The axis of the 3d plot are chosen to represent the two inputs with the lowest lengthscales in the model, so the x-y axes are different for the different state predictions.
The graphs represented on the lower row represent the predicted uncertainty, and the points are the prediction errors.
The points stored in the memory of the gaussian process model are represented in green, and the points not stored in black.
Note that the uncertainty represented in the 3d plot do not represent the uncertainty on the points, since the last dimension is not visible and has been set using linear regression fo the visible states input

These models are obtained without using any paramterics models. The only prior used is in the initializations values of the hyper-parameters of the gaussian process.
Yet, the control is stabilized most of the time before 100 iterations, with 30 iterations being random actions.

Dynamic vizualizations of the 3d update model and history plots are coming soon

To use the script
The parameters of the main script are stored in parameters.json, which specifies which gym environment to use, and the usage of vizualisations.

For each gym environment, a json file containing the gym environment name contains all the parameters relative to this environement, and the control used.
The syntax is parameters_"gym_env_name".json
- Training parameters
- Initialisation of the models
- Constraints of the models
- Actions optimizer
- Memory

To use the model on a different gym environement, an other json file must be created, which contains the same structure.


