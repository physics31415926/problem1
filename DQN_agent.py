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
# Last update: 20th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class Agent(nn.Module):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_states, n_actions: int):
        self.n_states = n_states
        self.n_actions = n_actions
        self.last_action = None
        super().__init__()

        # Create input layer with ReLU activation
        self.input_layer = nn.Linear(n_states, 128)
        self.input_layer_activation = nn.ReLU()

        # Create output layer
        self.output_layer = nn.Linear(128, n_actions)

    def forward(self, x):
        ''' Performs a forward computation '''
        # Function used to compute the forward pass

        # Compute first layer
        l1 = self.input_layer(x)
        l1 = self.input_layer_activation(l1)

        # Compute output layer
        out = self.output_layer(l1)
        return out

    def backward(self, buffer, optimizer, targetNet, discount_factor, batchNumber):
        ''' Performs a backward pass on the network '''
        # Sample a batch of 3 elements
        states, actions, rewards, next_states, dones = buffer.sample_batch(batchNumber)

        # Training process, set gradients to 0
        optimizer.zero_grad()

        # Compute output of the network given the states batch
        values = torch.zeros(batchNumber)
        valuesTarget = torch.zeros(batchNumber)
        for i in range(batchNumber):
            values[i] = self(torch.tensor(states[i], requires_grad=True, dtype=torch.float32))[actions[i]]
            if dones[i]:
                valuesTarget[i] = rewards[i]
            else:
                valuesTarget[i] = discount_factor * targetNet(torch.tensor(next_states[i], requires_grad=True, dtype=torch.float32)).max() + rewards[i]
        # Compute loss function
        loss = nn.functional.mse_loss(values,valuesTarget)
        #print(loss)

        # Compute gradient
        loss.backward()

        # Clip gradient norm to 1
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)

        # Perform backward pass (backpropagation)
        optimizer.step()


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_states, n_actions: int):
        super(RandomAgent, self).__init__(n_states, n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action
