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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


epsilon = 0.25
gamma = 1
BATCH_SIZE = 16
MAX_DEPTH = 4

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
            return 10
        else:
            return 1
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


def generate_tree(oracle, depth=0):
    value = oracle.Select(nodeValues,1)
    tree = BinarySearchTree(value) 

    if depth < MAX_DEPTH and oracle.Select(branchValues,2):
        tree.left = generate_tree(oracle, depth+1)

    if depth < MAX_DEPTH and oracle.Select(branchValues,3):
        tree.right = generate_tree(oracle, depth+1) 
    return tree 

state_size = 10
class Oracle:
    def __init__(self, epsilon=0.25, gamma=1.0, initial_val=0):
        self.state = State(state_size)
        self.learners = {}

    def Select(self, domain, idx):
        if not idx in self.learners:
            self.learners[idx] = Agent(state_size=state_size,action_size=len(domain),seed=0)
        choice = self.learners[idx].select(domain,self.state)
        return choice
    def reward(self, reward):
        for learner in self.learners.values():
            learner.reward(reward)
        self.state = State(state_size)


nodeValues = range(0,11)
# print(nodeValues)
branchValues = [False,True]
# agent = Agent(state_size=8, action_size=13, seed=0)

def fuzz():
    TRIALS = 10000
    oracle = Oracle()
    for i in range(TRIALS):
        
        # curState = [0]+state.memory[:]
        # print("curState",curState)
        tree = generate_tree(oracle)
        # action = agent.act(np.array(state.memory), epsilon)
        reward = get_reward(tree)
        # for agent in oracle.learners.items:
        oracle.reward(reward)
        # agent.step(curState, state.memory[-1], reward, [1]+state.memory, done)

        # state = State(6)
        # tree = generate_tree(state)

        # #print("=========================================")
        # print(tree)
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
        # print("{}".format(valids/(i+1)), end ='\n')






fuzz()
# torch.save(policy_net.state_dict(), "my_model")

