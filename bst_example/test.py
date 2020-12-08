import random
import sys
from bst import BinarySearchTree
from collections import Counter
from oracles import MockOracle, Oracle
from state_abstraction import parent_state_ngram_fn, left_right_parent_state_ngram_fn, sequence_ngram_fn

import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import math

from dqn_agent import Agent
agent = Agent(state_size=8, action_size=13, seed=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


epsilon = 0.25
gamma = 1
BATCH_SIZE = 16
MAX_DEPTH = 4
TRIALS = 10000
TARGET_UPDATE = 10
done =0

valids = 0
valid_set = set()

def get_reward(tree):
    global valids
    global valid_set
    if tree.is_bst(): # It is a valid tree
        valids += 1
        if tree.__repr__() not in valid_set:
            valid_set.add(tree.__repr__())
            return 20
        else:
            return 10
    else:
        return -1

class State(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = [0] * capacity
        self.record = []

    def push(self, action_taken):
        prev_state = self.memory.copy()
        self.memory.append(action_taken)
        if (len(self) > self.capacity):
            self.memory = self.memory[1:]
        self.record.append((prev_state, action_taken, self.memory.copy()))

    def collect_history():
        return self.record

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.linear1 = nn.Linear(6, 16)
        self.linear2 = nn.Linear(16, 32)
        self.linear3 = nn.Linear(32, 10)  
        self.relu = nn.ReLU()          


    def forward(self, x):   

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        # x = self.relu(x)
        return x

def state_to_tensor(state):
    # state = [v/10 for v in state]

    a = np.array(state, dtype=np.float32)
    return  torch.from_numpy(a)

def optimize_model(trans):
    y_values = []
    p_values = []

    # Collect the y and p values to calculate the loss
    (state, action, reward, next_state) = trans
    idx_action = action-1 # The index in the list is action - 1


    y = torch.from_numpy(np.array(reward, dtype=np.float32))

    state = state_to_tensor(state)
    p = policy_net(state)[idx_action]
    y_values.append(y)
    p_values.append(p)
    y_values = torch.stack(y_values).detach()
    p_values = torch.stack(p_values)
    loss = F.mse_loss(p_values, y_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


policy_net = DQN().to(device)
# target_net = DQN().to(device)
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())


def select_action(state):
    if random.random() < epsilon:
        a = random.choices([1,2,3,4,5,6,7,8,9,10])
        return torch.from_numpy(np.array(a))

    else:   
        state = state_to_tensor(state.memory)
        output = policy_net(state)
        return torch.argmax(output) + 1



# TODO: When do we use the target network?
def select_exploit_action(model, state):
    if random.random() < epsilon:
        a = random.choice([1,2,3,4,5,6,7,8,9,10])
        return a
    else:
        state = state_to_tensor(state.memory)
        output = model(state)
        # #print("State: ", state, end="")
        print(model.parameters)
        # print(output)
        return int(torch.argmax(output)) + 1


def generate_tree(state, depth=0):
    curState = [0]+state.memory[:]
    value = agent.act(np.array([0]+state.memory), epsilon)
    agent.step(curState, value, 0, [0]+state.memory, done)
    # print(value)
    state.push(value)
    tree = BinarySearchTree(value) 
    if depth < MAX_DEPTH and \
            random.choice([True,False]):
        # state.push(0) # To denote left subtree existence
        tree.left = generate_tree(state, depth+1)
    if depth < MAX_DEPTH and \
            random.choice([True, False]):
        # state.push(-1) # To denote right subtree existence
        tree.right = generate_tree(state, depth+1) 
    return tree 


def fuzz():
    for i in range(TRIALS):
        state = State(7)
        curState = [0]+state.memory[:]
        # print("curState",curState)
        tree = generate_tree(state)
        # action = agent.act(np.array(state.memory), epsilon)
        reward = get_reward(tree)
        agent.step(curState, state.memory[-1], reward, [1]+state.memory, done)

        # state = State(6)
        # tree = generate_tree(state)

        # #print("=========================================")
        # #print(tree)
        # #print("=========================================")
        # reward = get_reward(tree)
        # for (cur_state, action, new_state) in state.record:
        #     new_state = state_to_tensor(new_state)
        #     reward_val = torch.from_numpy(np.array(reward))
        #     transition = (cur_state, action, reward_val, new_state)
        #     optimize_model(transition)

        # if (i%TARGET_UPDATE == 0):
        #     target_net.load_state_dict(policy_net.state_dict())
        # #print("epoch = ", i)
        print("{} trials, {} valids, {} unique valids".format(i, valids, len(valid_set)), end ='\n')





fuzz()
# torch.save(policy_net.state_dict(), "my_model")

