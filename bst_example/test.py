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

# MAX_DEPTH = 4

# def generate_tree(oracle, depth=0):
#     value = oracle.Select(range(0, 11), 1)
#     tree = BinarySearchTree(value) 
#     if depth < MAX_DEPTH and \
#             oracle.Select([True, False], 2):
#         tree.left = generate_tree(oracle, depth+1)
#     if depth < MAX_DEPTH and \
#             oracle.Select([True, False], 3):
#         tree.right = generate_tree(oracle, depth+1) 
#     return tree 

# def is_BST(tree):
#     return tree.is_bst()

# def fuzz(oracle, validity_fn):
#     valids = 0
#  #   print("Starting!", file=sys.stderr)
#     valid_set = set()
#     trials = 100000
#     for i in range(trials):
#        # print("{} trials, {} valids, {} unique valids             ".format(i+1, valids, len(valid_set)), end ='\r', file=sys.stderr)
#         tree = generate_tree(oracle)
#         is_valid = validity_fn(tree)
#         if is_valid:
#             valids += 1
#             if tree.__repr__() not in valid_set:
#                 valid_set.add(tree.__repr__())
#                 oracle.reward(20)
#             else:
#                 oracle.reward(0)
#         else:
#             oracle.reward(-1)
#     sizes = [valid_tree.count("(") for valid_tree in valid_set]
#     print("{} trials, {} valids, {} unique valids".format(trials, valids, len(valid_set)), end ='\n')
#   #  print("\ndone!", file=sys.stderr)
#     print(Counter(sizes))
#     # print("soemthing")
# if __name__ == '__main__':
#     print("====Random====")
#     mo = MockOracle()
#     fuzz(mo, is_BST)
#     print("====Sequence====")
#     oracle_s = Oracle(sequence_ngram_fn(4), epsilon=0.25)
#     fuzz(oracle_s, is_BST)
#     print("====Tree====")
#     oracle_t = Oracle(parent_state_ngram_fn(4, MAX_DEPTH), epsilon=0.25)
#     fuzz(oracle_t, is_BST)
#     print("====Tree L/R====")
#     oracle_lrt = Oracle(left_right_parent_state_ngram_fn(4, MAX_DEPTH), epsilon=0.25)
#     fuzz(oracle_lrt, is_BST)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_epochs = 2000
epsilon = 0.25
gamma = 1
BATCH_SIZE = 64


def reward(coor, done):
    """
    Task 6 (optional) - design your own reward function
    """

    x, y = coor
    if coor == goal:
        return 100
    elif done:
        return -100
    elif coor in [(6,5), (5,6)]:
        return
   
    return 0
    # return -5 


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition):

        """
        Task 3 - 
        push input: "transition" into replay meory
        """
        self.memory.insert(0, transition)
        if (len(self) > self.capacity):
            self.memory = self.memory[:self.capacity]

    def sample(self, batch_size):
        """
        Task 3 - 
        give a batch size, pull out batch_sized samples from the memory
        """
        if batch_size > len(self):
            return self.memory.copy()
        else:
            return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        """
        Task 1 -
        generate your own deep neural network
        """
        self.linear1 = nn.Linear(6, 16)
        self.linear2 = nn.Linear(16, 32)
        self.linear3 = nn.Linear(32, 10)  
        self.relu = nn.ReLU()          



    def forward(self, x):
        """
        Task 1 - 
        generate your own deep neural network
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

def state_to_tensor(state):
    a = np.array(state, dtype=np.float32)
    return  torch.from_numpy(a)

# transition = (cur_state, idx_action, reward_val, new_state)
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    """
    Task 4: optimize model
    """

    y_values = []
    p_values = []
    for trans in transitions:   
        (state, idx_action, reward, next_state) = trans
        # if done:
        #     y = torch.from_numpy(np.array(reward, dtype=np.float32))
        # else:
        #     y = reward + gamma * torch.max(target_net(next_state)) 
        y = torch.from_numpy(np.array(reward, dtype=np.float32))


        p = policy_net(state)[idx_action]
        y_values.append(y.unsqueeze(0))
        p_values.append(p)
    y_values = torch.stack(y_values).detach()
    p_values = torch.stack(p_values)
    loss = F.mse_loss(p_values, y_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(1000)


# transition = (cur_state, idx_action, reward_val, new_state)
def select_action(state):
    """
    Task 2: select action
    """
    if random.random() < epsilon:
        # Fixme: Why does index action become 10 sometimes with this code
        # a = random.choices([1,2,3,4,5,6,7,8,9,10])
        a = random.choices([0,1,2,3,4,5,6,7,8,9])
        # print(a)
        return torch.from_numpy(np.array(a))

    else:   

        # FixMe: Use an abstracted form of the state
        state = [8,2,0,-1,2,3]
        a = np.array(state, dtype=np.float32)   
        t = torch.from_numpy(a)
        output = policy_net(t)
        return torch.argmax(output.detach()).unsqueeze(0)


def select_exploit_action(model, state):
    # a = state_astraction(state)
    state = [8,2,0,-1,2,3]
    a = np.array(state, dtype=np.float32)
    t = torch.from_numpy(a)
    output = model(t)
    return torch.argmax(output)


state = ReplayMemory(10)
memory = ReplayMemory(100)
state_action_record = []

MAX_DEPTH = 10
def generate_tree(depth=0):
    global state
    global state_action_record
    if (depth == 0): # Clear the state when generating a new tree
        state = ReplayMemory(10)
        state_action_record = []
    value = select_exploit_action(policy_net, state)
    state.push(value)
    state.memory = [8,2,0,-1,2,3]
    state_action_record.append((state.memory.copy(), value))
    tree = BinarySearchTree(value) 
    if depth < MAX_DEPTH and \
            random.choice([True, False]):
        state.push(0) # To denote left subtree existence
        tree.left = generate_tree(depth+1)
    if depth < MAX_DEPTH and \
            random.choice([True, False]):
        state.push(-1) # To denote right subtree existence
        tree.right = generate_tree(depth+1) 
    return tree 


def fuzz():

    valids = 0
    valid_set = set()
    trials = 100000
    TARGET_UPDATE = 5
    for i in range(trials):
        # print(i)
        # while not done:
        tree = generate_tree()
        print(tree)
        # reward = get_reward(tree)
        reward = 10

        for state_action in state_action_record:
            # cur_state = [x, y, o_i] + sensor 
            (state, action) = state_action
            a = np.array(state, dtype=np.float32)
            cur_state = torch.from_numpy(a)


            action_t = select_action(cur_state).long()
            # idx_action = action_t.item()
            # action = actions[idx_action]
            # (new_x, new_y), new_ori, new_sensor, done = action()
            # new_o_i = orientation.index(new_ori)

            # reward_val = reward((new_x,new_y),done)
            reward_val = reward


            new_state = state.copy()















            a = np.array(new_state, dtype=np.float32)
            new_state = torch.from_numpy(a) 
            reward_val = torch.from_numpy(np.array(reward_val))
            # transition = (cur_state, action_t, reward_val, new_state)

            transition = (cur_state, action_t, reward_val, new_state)

            memory.push(transition)
            # (x, y), ori, sensor = (new_x, new_y), new_ori, new_sensor
            optimize_model()

        if (i%TARGET_UPDATE == 0):
            target_net.load_state_dict(policy_net.state_dict())
        # if (i%(num_epochs//10) == 0):
        #     epsilon *= 0.85
        print("epoch = ", i)




fuzz()
"""
Task 5 - save your policy net
"""
torch.save(policy_net.state_dict(), "my_model")

