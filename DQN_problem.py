# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

import numpy as np
import gym
import torch
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tqdm import trange
from DQN_agent import RandomAgent, EpsGreedyAgent
import random
from collections import namedtuple


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()

# Parameters
n_episodes = 300                             # Number of episodes
discount_factor = 0.95                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

# Random agent initialization
# agent = RandomAgent(n_actions)
print(env.observation_space)
print(env.observation_space.high)
print(dim_state)
print(n_actions)
hidden_size = 128

def DQN(hidden_size):
    return nn.Sequential(
        nn.Linear(dim_state, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, n_actions)
    )


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

EPISODES = trange(n_episodes, desc='Episode: ', leave=True)
render_period = 150

loss_fn = torch.nn.MSELoss()

policy_net = DQN(hidden_size)
target_net = DQN(hidden_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

agent = EpsGreedyAgent(n_actions, n_episodes, policy_net)

BATCH_SIZE = 128
BUFFER_SIZE = 20000
TARGET_UPDATE = BUFFER_SIZE // BATCH_SIZE
LEARNING_RATE = 2e-04

memory = ReplayMemory(BUFFER_SIZE)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
t = 0
for episode in EPISODES:
    # Reset enviroment data and initialize variables
    done = False
    state = torch.from_numpy(env.reset())
    total_episode_reward = 0.
    while not done:
        if episode % render_period == 0:
            env.render()

        # Take action
        action = agent.forward(state, episode)

        next_state, reward, done, _ = env.step(action.numpy()[0][0])
        next_state = torch.from_numpy(next_state)

        # Store the transition in memory
        memory.push(state,
            action, next_state, torch.tensor([reward], dtype=torch.float), done)

        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t += 1
        # Filling up buffer
        if len(memory) < BATCH_SIZE*4:
            continue
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.logical_not(torch.tensor(batch.done))
        next_state_batch = torch.cat(batch.next_state).reshape((BATCH_SIZE, -1))
        state_batch = torch.cat(batch.state).reshape((BATCH_SIZE, -1))
        action_batch = torch.cat(batch.action).reshape((BATCH_SIZE, -1))
        reward_batch = torch.cat(batch.reward).reshape((BATCH_SIZE, -1))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values.
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE)
        next_state_pred = target_net(
            next_state_batch).max(1).values
        next_state_values[non_final_mask] = next_state_pred[non_final_mask]
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * discount_factor) + reward_batch.squeeze()

        loss = loss_fn(
            state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()

        loss.backward()
        torch.nn.utils.clip_grad_norm(policy_net.parameters(), 1)

        optimizer.step()

        # Update the target network, copying all weights and biases in DQN
        if t % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            episode, total_episode_reward, t,
            running_average(episode_reward_list, n_ep_running_average)[-1],
            running_average(episode_number_of_steps, n_ep_running_average)[-1]))

    if running_average(episode_reward_list, n_ep_running_average)[-1] > 51:
        print("Good enough! Stopping learning")
        break

torch.save(policy_net, 'neural-network-1.pth')

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, n_episodes+1)],
           episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, n_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, n_episodes+1)],
           episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, n_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()

# (f)

angles = np.arange(-np.pi, np.pi, 0.1)
heights = np.arange(0, 1.5, 0.05)
q_values = np.zeros((len(heights), len(angles)))

for i, angle in enumerate(angles):
    for j, height in enumerate(heights):
        state = torch.tensor((0, height, 0, 0, angle, 0, 0, 0))
        net_input = torch.unsqueeze(state, 0)
        action_value = policy_net(net_input).max(1).values[0]
        q_values[j, i] = action_value

fig = plt.figure()
ax = fig.gca(projection='3d')

X, Y = np.meshgrid(angles, heights)
surf = ax.plot_surface(X, Y, q_values, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()