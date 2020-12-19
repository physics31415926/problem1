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
import math
import torch

class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action

class EpsGreedyAgent(Agent):

    EPS_START = 0.99
    EPS_END = 0.05
    ''' Agent taking actions with epsilon greedy policy, child of the class Agent'''
    def __init__(self, n_actions: int, n_episodes: int, policy_net):
        super(EpsGreedyAgent, self).__init__(n_actions)
        self.EPS_DECAY = int(n_episodes*0.9)
        self.steps_done = 0
        self.policy_net = policy_net

    def forward(self, state: np.ndarray, episode) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        sample = np.random.random()
        eps_decayed = self.EPS_START - (self.EPS_START-self.EPS_END)*(episode-1) / (self.EPS_DECAY - 1)
        eps_threshold = np.max([self.EPS_END, eps_decayed])
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state.unsqueeze(0)).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[np.random.randint(0, self.n_actions)]], dtype=torch.long)

