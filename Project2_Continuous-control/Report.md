## DDPG
DDPG is an off-policy Actor-Critic method for learning actions in continuous spaces. In this environment, the state space has 33 dimensions (consisting of arm positions, accelerations, velocities and angles, ...) and outputs a single action out of 4 possible actions (torque that is sent to each joint). The Critic network inputs action and state and outputs a Q-value for the given inputs. Ornstein-Uhlenbeck noise is added to the action to encourage exploration. DDPG also uses experience replay to stabilize the training process.

The actor is implemented as a target network due to target networks decorrelating the process and generally improving performance. Soft updates are used in DDPG, which means that at every update step, the local networks are updated by a certain amount of the weights of the target networks according to a hyperparameter Tau $/tau$.

## Loss
The critic loss is computed as the mean squared error of the targets of the current state (calculated by the target network) and the output of the local network. The actor is updated by backpropagating the error that is evaluated by calculating the average of the Q-values for each state-action pair as output.

## Network architectures
Both networks (Actor and Critic) are simple deep feedforward networks with 1 input- and 1 hidden layer activated by a ReLU function. Batch Normalization has been added to improve computation speed. The Actor uses a *Tangens hyperbolicus* (Tanh) function to output a continuous action in the range of [-1,1] for each of the 4 actions. The Critic outputs a Q-value.

## Hyperparameters

### DDPG
```
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 1e-6     # L2 weight decay
```

### Ornstein-Uhlenbeck
```
mu = 0.0 
theta = 0.15
sigma = 0.1
```

## Performance
For this project, the single agent has been used. DDPG achieves the minimum mean score threshold of 30+ points reward over at least 100 episodes around episode 29 when trained.

![Plot rewards](https://github.com/d-kleine/Udacity_DRLND/blob/main/Project2_Continuous-control/plot_rewards.png)

## Improvements
In order to gain better overall performance, other types of Actor-Critic methods (e.g., GAE, Q-Prop) could be used. Also, recurrent versions of these algorithms could have a positive effect on the overall performance.
