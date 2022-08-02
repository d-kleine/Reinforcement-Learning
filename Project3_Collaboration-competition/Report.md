## MADDPG
This project uses an off-policy method called *Multi Agent Deep Deterministic Policy Gradient* (MADDPG) which is based on the Deep Deterministic Policy Gradient (DDPG) algorithm. It uses *off-policy* data and the Bellman equation to learn the Q-function to infer a policy.

## Loss
The critic loss is computed as the mean squared error (MSE).

## Network architectures
Both Actor and Critic contain a simple deep neural network with ReLU activation functions. At the end of Actor, there is an extra tanh activation function to guarantee that the range of actions is (-1, 1).

## Hyperparameters

### MADDPG
```
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 8e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
```

### Ornstein-Uhlenbeck
```
mu = 0.0 
theta = 0.15
sigma = 0.1
```

## Performance
The MADDPG algorithm achieves the minimum mean score threshold of 0.5 points reward over at least 100 episodes around episode 362 when trained.

![Plot rewards](https://github.com/d-kleine/Udacity_DRLND/blob/main/Project3_Collaboration-competition/plot_rewards-mean.png)

## Improvements
* Improving network architecture, including normalization and dropout
* Finetuning hyperparameters
* Using the symmetry of the environment that two agents can share experiences 
* Other types of Actor-Critic algorithms (e.g. PPO, A3C, and D4PG) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.
* More advanced DDPG techniques (e.g. Twin Delayed DDPG, DDPG with Prioritized Experience Replay)
