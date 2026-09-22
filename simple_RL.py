"""
A SIMPLE REINFORCEMENT LEARNING GUIDE CODE UNDER 200 LINES ON IMPLEMENTING PPO ALGORITHM WITH PYTORCH
GOAL: TEACH AN AGENT TO MOVE TOWARDS COORDINATE X = 0 FROM RANDOM COORDINATES
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# PYTORCH & CUDA SETUP
print(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ENVIRONMENT SETUP
# GOAL: Teach the model to move to x = 0
class SimpleEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = np.random.uniform(low=-2.0, high=2.0) # Set the agent at random spawn point between -2 to 2. x represents the agent position
        return np.array([self.x], dtype=np.float32) # Return the agent position in 1D array (i.e. [0.4])

    def step(self, action):
        action = np.clip(action, -1.0, 1.0) # Clip the action to prevent explosion
        self.x += action[0] * 0.1 # update the position of the model

        reward = -abs(self.x) # If the agent is too far from the objective(x = 0) (i.e. x = -2 or x = 3), then it gets penalized for being far from the position.
        # We use absolute value to ignore direction, just magnitude
        done = abs(self.x) < 0.01 # Stop when the done mask is less than 0.01

        return np.array([self.x], dtype=np.float32), reward, done

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )

        self.actor_mean = nn.Linear(64, 1)
        self.actor_logstd = nn.Parameter(torch.zeros(1))

        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)
        mean = self.actor_mean(x) # Action mean
        std = torch.exp(self.actor_logstd) # Exploration rate

        value = self.critic(x) # Critic predicted values

        return mean, std, value

class Buffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def clear(self):
        self.__init__()

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lambd=0.95):
    T = len(rewards)

    advantages = torch.zeros(T).to(rewards.device) # Initialize advantage tensor
    gae = 0

    for t in reversed(range(T)):
        if t == T - 1: # If the index is at the last index of the array, we stop the loop and use it as the last value
            next_value = last_value
        else:
            next_value = values[t+1] # Continue the loop, we take the next value and used it in our GAE calculation

        next_nonterminal = 1 - dones[t] # Determine if the value is still in the same episode, if no we mask it by multiplying 0

        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t] # Calculate the temporal differences, which is difference between current and past values

        gae = delta + gamma * lambd * (1 - dones[t]) * gae # Calculate the advantage

        advantages[t] = gae # Store the advantage value in the array
        advantages[t] = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    returns = advantages + values # Calculate the returns
    return advantages, returns

def ppo_update(model, optimizer, states, actions, old_log_probs, returns, advantages, clip_eps=0.2):
    mean, std, values = model(states) # PPO Outputs
    dist = torch.distributions.Normal(mean, std) # Create a policy with a probabilities of values
    new_log_probs = dist.log_prob(actions).sum(-1) # How likely is the action under the new policy

    ratio = torch.exp(new_log_probs - old_log_probs) # Calculate ratio (How different past policy vs new policy, used in KL & PPO Loss)

    # Surrogate objectives
    surr1 = ratio * advantages # Original PPO objective (difference in policy * advantage function)
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages # Clipping of the original objective to prevent explosion

    # Actor Loss
    actor_loss = -torch.min(surr1, surr2).mean() # Minimize to choose the smaller objective to minimize update fluctuations

    # Critic Loss
    value_loss = (returns - values.squeeze()).pow(2).mean() # Similar to MSE loss, where it takes the loss (difference between return & critic prediction) and square it to penalize the error

    # Entropy
    entropy = dist.entropy().mean() # Calculate how exploration rate of the agent in the environment

    loss = actor_loss + 0.5 * value_loss - 0.01 * entropy # Final Loss Function

    optimizer.zero_grad() # Optimizer
    loss.backward() # Backpropagation to find gradient
    optimizer.step() # Adjust the params in the agent policy

    return loss.item()

env = SimpleEnv() # Initialize environment
model = ActorCritic().to(device) # Initialize model
optimizer = optim.Adam(model.parameters(), lr=3e-4) # Initialize optimizer. Adam is chosen for fast convergence
rollout_length = 64
buffer = Buffer()
episode_rewards = []
episode_reward = 0

for iteration in range(2000):
    state = env.reset() # (1,)

    for t in range(rollout_length):
        state_tensor = torch.tensor(state).unsqueeze(0).to(device) # (1, 1)

        with torch.no_grad():
            mean, std, value = model(state_tensor) # Output the PPO model results
            dist = torch.distributions.Normal(mean, std) # Normalize mean and std of the model
            action = dist.sample() # Get the list of actions value based on the distribution of the agent
            log_prob = dist.log_prob(action).sum(-1) # Calculate how likely the action would be under the current distribution

        next_state, reward, done = env.step(action.cpu().numpy()[0]) # Compute the next state after each action is taken, their rewards and whether if the optimal is reached (done)
        episode_reward += reward

        # Append values to respective array
        buffer.states.append(state_tensor.squeeze(0)) # Shape (1, )
        buffer.actions.append(action.squeeze(0)) # Shape (1, )
        buffer.log_probs.append(log_prob.squeeze(0)) # Shape (1, )
        buffer.rewards.append(torch.tensor(reward, dtype=torch.float32).to(device)) # Shape (1, )
        buffer.values.append(value.squeeze(0)) # Shape (1, )
        buffer.dones.append(torch.tensor(float(done), dtype=torch.float32).to(device)) # Shape (1, )

        state = next_state
        if done: # If the agent position < 0.01, the episode will be ended but the rollout continues. I.e. if episode ends at rollout=10, rollout 11-63 will still continue
            state = env.reset()
            episode_rewards.append(episode_reward)
            episode_reward = 0
            state = env.reset()

    # Conversion to tensors
    states = torch.stack(buffer.states)
    actions = torch.stack(buffer.actions)
    log_probs = torch.stack(buffer.log_probs)
    rewards = torch.stack(buffer.rewards)
    values = torch.stack(buffer.values)
    dones = torch.stack(buffer.dones)

    # Bootstrap
    with torch.no_grad():
        last_state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        last_value = model(last_state_tensor)[2].squeeze(0) # Calculate the value after the rollout index has been completed (final state after the action is done)

    advantages, returns = compute_gae(rewards, values, dones, last_value) # Compute advantage and returns

    for epoch in range(4):
        loss = ppo_update(model, optimizer, states, actions, log_probs, returns, advantages) # Compute loss value

    buffer.clear() # Reset the buffer every iteration

    if iteration % 10 == 0:
        print(
            f"Iter {iteration} | "
            f"Loss: {loss:.2f} | "
            f"Reward: {reward:.2f}"
        )


import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(episode_rewards, alpha=0.4, label="Episode Reward")

smoothed = [
    np.mean(episode_rewards[max(0, i-20):i+1])
    for i in range(len(episode_rewards))
]

plt.plot(smoothed, linewidth=2, label="Moving Average")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("PPO Training Reward")
plt.legend()
plt.show()