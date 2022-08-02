## MADDPG
The position and velocity of the ball and racket define the state space of the enviroment, consisting of 8 variables. The racket's move and jumping are the possible actions. The task is episodic, and in order to solve the environment, the two competing agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Then the environment will be considered as solved.

This project uses an off-policy method called *Multi Agent Deep Deterministic Policy Gradient* (MADDPG) which is based Deep Deterministic Policy Gradient (DDPG). It uses *off-policy* data and the Bellman equation to learn the Q-function to learn the policy.

## Loss
The critic loss is computed as the mean squared error (MSE)

## Network architectures
Both Actor and Critic contain 3 hidden layers with 2 ReLU activation functions. At the end of Actor, there is an extra tanh activation function to guarantee that the range of actions is (-1, 1).

## Hyperparameters

### MADDPG
```
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
GAMMA = 0.99            # discount factor
TAU = 8e-3              # for soft update of target parameters
```

### Ornstein-Uhlenbeck
```
mu = 0.0 
theta = 0.15
sigma = 0.1
```

## Performance
The MADDPG algorithm achieves the minimum mean score threshold of 0.5 points reward over at least 100 episodes around episode 29 when trained.

![Plot rewards](https://github.com/d-kleine/Udacity_DRLND/blob/main/Project3_Collaboration-competition/plot_rewards-mean.png)

## Improvements
* Improving network architecture, including normalization and dropout
* Finetuning hyperparameters
* Using the symmetry of the environment that two agents can share experiences 
* other types of Actor-Critic algorithms (e.g. PPO, A3C, and D4PG) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.
* more advanced DDPG algorithms (e.g. Twin Delayed DDPG)
