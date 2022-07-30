## DQN Algorithm 

The agent uses a standard DQN consisting of a local network and a target network (replacing traditional Q-learning's lookup table process). Local network and target network are used to avoid high correlations that may occur in observation space and between Q-values and TD-target value. The target network is updated every 4 episodes by copying the weights from the local network. For reproduction, a random state has been initialized. Experience replay is used to prevent high correlations in the training data. It essentially collects a database of past examples to learn from. A replay buffer is used to store a random sample of numbers of previous instances. This prevents high oscillations and reduces the risk of divergence. The epsilon-greedy algorithm is used to explore the action space by selecting the non-optimal action according to the Q-value with a probability of epsilon. This allows the agent to explore strategies that may be better than the current optimal one and to learn about different sequences of actions. The loss function is chosen as the mean squared error (MSE) loss which calculates the loss between *Q_target* (calculated with the target network output) and *Q_expected* (calculated with the local network output). This loss is then used to backpropagate the error and then performs an update step according to the optimizer used.

The used model a simple deep neural network architecture with 3 fully connected layers. ReLU activation will be performed for the input and hidden layer in order to achieve non-linear transformation. The state space is 37 states and the action space is 4 actions given by the project's description. The model is trained with the Adam optimizer in order to find the best parameters to minimize the model's loss. The size of the first two layers (input- and hidden layer) is 64 units each. The last output layer consists of 4 units. 


Hyperparameters were set to:
```python
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
```

The agent successfully learns to receive episodic scores resulting in an average score of 13 (project's minimum score threshold) at episode 489. Compared to the untrained, randomly interacting agent this a good improvement!

![Plot rewars](https://github.com/d-kleine/Udacity_DRLND/blob/main/Project1_Unity-CollectingBananas/plot_rewards.png)

Probably a more complicated neural network would improve the model's performance. In order to gain better performance, other types of DQNs (e.g., Double DQN, Prioritized experience replay, Dueling DQN, Learning from multi-step bootstrap targets, Distributional DQN, Noisy DQN, etc.). The Rainbow algorithm could combine the advantages of these models resulting in a significantly better performance.
