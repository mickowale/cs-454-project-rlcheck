import random
import sys
from bst import BinarySearchTree
from collections import Counter
from oracles import MockOracle, Oracle
from state_abstraction import parent_state_ngram_fn, left_right_parent_state_ngram_fn, sequence_ngram_fn
import csv
import time
import os,psutil
MAX_DEPTH = 4

def generate_tree(oracle, depth=0):
    value = oracle.Select(range(0, 10), 1)
    tree = BinarySearchTree(value) 
    if depth < MAX_DEPTH and \
            oracle.Select([True, False], 2):
        tree.left = generate_tree(oracle, depth+1)
    if depth < MAX_DEPTH and \
            oracle.Select([True, False], 3):
        tree.right = generate_tree(oracle, depth+1) 
    return tree 

def is_BST(tree):
    return tree.is_bst()

def fuzz(oracle, validity_fn,filename):
    filename += "_range_10.csv"
    with open(filename,'w+') as file:
        valids = 0
     #   print("Starting!", file=sys.stderr)
        valid_set = set()
        trials = 100000
        for i in range(trials):
    #        print("{} trials, {} valids, {} unique valids             ".format(i+1, valids, len(valid_set)), end ='\r', file=sys.stderr)
            tree = generate_tree(oracle)
            is_valid = validity_fn(tree)
            if is_valid:
                valids += 1
                if tree.__repr__() not in valid_set:
                    valid_set.add(tree.__repr__())
                    oracle.reward(20)
                else:
                    oracle.reward(0)
            else:
                oracle.reward(-1)
            file.write(str(i) + "," + str(valids) + "," + str(len(valid_set)))
            file.write("\n")
        sizes = [valid_tree.count("(") for valid_tree in valid_set]
        print("{} trials, {} valids, {} unique valids".format(trials, valids, len(valid_set)), end ='\n')

      #  print("\ndone!", file=sys.stderr)
        file.close()
        print(Counter(sizes))

if __name__ == '__main__':
    pid = os.getpid( )
    ps = psutil.Process(pid)
    # print("====Random====")
    # mo = MockOracle()
    # fuzz(mo, is_BST,"random")
    # memoryuse = ps.memory_info()
    # print(memoryuse)
    # print(memoryuse.rss/1024**2)
    # print("====Sequence====")
    # oracle_s = Oracle(sequence_ngram_fn(4), epsilon=0.25)
    # fuzz(oracle_s, is_BST,"sequence")
    # memoryuse = ps.memory_info()
    # print(memoryuse)
    # print(memoryuse.rss/1024**2)
    print("====Tree====")
    oracle_t = Oracle(parent_state_ngram_fn(4, MAX_DEPTH), epsilon=0.25)
    fuzz(oracle_t, is_BST,"Tree")
    memoryuse = ps.memory_info()
    print(memoryuse)
    print(memoryuse.rss/1024**2)
    # print("====Tree L/R====")
    # oracle_lrt = Oracle(left_right_parent_state_ngram_fn(4, MAX_DEPTH), epsilon=0.25)
    # fuzz(oracle_lrt, is_BST,"Tree_LR")
    # memoryuse = ps.memory_info()
    # print(memoryuse)
    # print(memoryuse.rss/1024**2)
