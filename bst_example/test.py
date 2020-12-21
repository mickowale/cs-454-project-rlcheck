from bst import BinarySearchTree
from dqn_agent import Agent
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import math


MAX_DEPTH = 4
state_size = 10
nodeValues = range(0,11)
branchValues = [False,True]

# Initializing the valids count and set
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
            return 0
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

    def collect_history(self):
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


class Oracle:
    def __init__(self):
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

def fuzz():
    TRIALS = 100001
    oracle = Oracle()
    for i in range(TRIALS):
        
        tree = generate_tree(oracle)

        reward = get_reward(tree)
        oracle.reward(reward)

        print("{} trials, {} valids, {} unique valids".format(i, valids, len(valid_set)), end ='\n')


fuzz()
# torch.save(policy_net.state_dict(), "my_model")

