#!/usr/bin/env python
import random
import math
import hashlib
import argparse
import pandas as pd
import trainer_cifar
import argparse
from utils import Namespace
from model import HeterogenousNetworkCIFAR, HeterogenousNetworkImageNet
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import latency_profiler

"""
Original version can be seen on https://github.com/ajax98/catan/blob/master/mcts.py
A quick Monte Carlo Tree Search implementation.  
For more details on MCTS see See http://pubs.doc.ic.ac.uk/survey-mcts-methods/survey-mcts-methods.pdf
The State is just a game where you have NUM_TURNS and at turn i you can make
a choice from set of possible moves.  The goal is for the accumulated value to be as close to 0 as possible.
For our purpose, we change the reward function is defined as (1 - accumulated value)
Accumulated value is latency * accuracy profile. 
We definied our best reward model candidates as 
(1) model with tail latency 99 closest to the latency strata
(2) model with highest accuracy
The game choice is not very that important but it allows one to study MCTS.  Some features 
of the example by design are that moves do not commute and early mistakes are more costly.  
In particular there are two models of best child that one can use 
"""

# MCTS scalar.  Larger scalar will increase exploitation,
# smaller will increase exploration.
SCALAR = 1 / math.sqrt(2.0)
INPUT_BATCH = 1
INPUT_CHANNEL = 3
CIFAR_INPUT_SIZE = 32
CIFAR_CLASSES = 10
IMAGENET_INPUT_SIZE = 224
IMAGENET_CLASSES = 1000


def gaussian(X, mu, cov):
    n = X.shape[1]
    diff = (X - mu).T
    return np.diagonal(
        1
        / ((2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5)
        * np.exp(-0.5 * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff))
    ).reshape(-1, 1)


class State:
    MOVES = pd.read_csv("generated_micro_cuda:0_cpu_center.csv")["genotype"].to_list()
    MOVES.append("none")
    num_moves = len(MOVES)

    def __init__(
        self,
        moves,
        selected_med_idx,
        turn,
        n_family,
        target_latency,
        max_layers,
        dataset_name,
        config,
    ):
        # print("MOVES {}".format(self.MOVES))
        self.turn = turn
        self.moves = []  # selected moves represent selected layers
        self.selected_med_idx = []  # selected medioid layers
        self.n_family = n_family  #
        self.acc = 0  # current state acc in %
        self.lat = 1000  # current state lat 99th in ms
        self.target_latency = target_latency  # array of target lat in ms
        self.max_layers = max_layers
        self.dataset_name = dataset_name  # cifar10 or imagenet
        self.config = config
        # print("__init called __ {}".format(self.moves))

    def next_state(self):
        nextmove = []
        med_idx = []
        ln_valid_choice = len(self.MOVES) - 1
        for i in range(self.turn):
            rand_idx = random.randint(0, ln_valid_choice)
            med_idx.append(rand_idx)
            nextmove.append(self.MOVES[rand_idx])

        self.moves = self.moves + nextmove
        self.selected_med_idx = self.selected_med_idx + med_idx

        return State(
            self.moves,
            self.selected_med_idx,
            self.turn - 1,
            self.n_family,
            self.target_latency,
            self.max_layers,
            self.dataset_name,
            self.config,
        )

    def terminal(self):
        if self.turn == 0:
            return True
        return False

    def get_acc_latency(self):
        latencies = []
        if self.dataset_name == "cifar10":
            model = HeterogenousNetworkCIFAR(
                self.config["architecture"]["init_channels"],
                self.config["architecture"]["num_classes"],
                self.config["architecture"]["layers"],
                self.config["architecture"]["auxiliary"],
                self.moves,
            )
            model.drop_path_prob = self.config["architecture"]["drop_path_prob"]
            model.to(self.config["device"])
            dummy_input = torch.zeros(
                INPUT_BATCH, INPUT_CHANNEL, CIFAR_INPUT_SIZE, CIFAR_INPUT_SIZE
            ).to(self.config["device"])
            mean_lat, latencies = latency_profiler.test_latency(
                model, dummy_input, self.config["device"]
            )
        elif self.dataset_name == "imagenet":
            model = HeterogenousNetworkImageNet(
                self.config["architecture"]["init_channels"],
                self.config["architecture"]["num_classes"],
                self.config["architecture"]["layers"],
                self.config["architecture"]["auxiliary"],
                self.moves,
            )
            model.drop_path_prob = self.config["architecture"]["drop_path_prob"]
            model.to(self.config["device"])
            dummy_input = torch.zeros(
                INPUT_BATCH, INPUT_CHANNEL, IMAGENET_INPUT_SIZE, IMAGENET_INPUT_SIZE
            ).to(self.config["device"])
            mean_lat, latencies = latency_profiler.test_latency(
                model, dummy_input, self.config["device"]
            )
        train_result = {"acc": 0, "lat": latencies[98]}
        print("train_result ", train_result)
        return train_result

    def reward(self, num_layers):
        w = 0.5
        train_result = self.get_acc_latency()
        self.acc = train_result["acc"]
        self.lat = train_result["lat"]
        l_stratas = self.target_latency
        lat_part = 1
        for strata in l_stratas:
            if self.lat < strata:
                lat_part = lat_part * abs((1 - (self.lat / strata))) ** (1 - w)
        acc_part = abs((1 - (self.acc / 100))) ** w
        r = 1 - (acc_part * lat_part) + (1 / (1 + math.exp(-(num_layers - (self.max_layers - 10) ))))
        return r


class Node:
    def __init__(self, state, parent=None):
        self.visits = 1
        self.reward = 0.0
        self.state = state
        self.children = []
        self.parent = parent

    def add_child(self, child_state):
        child = Node(child_state, self)
        self.children.append(child)

    def fully_expanded(self):
        if len(self.children) == self.state.num_moves:
            return True
        return False

    def __repr__(self):
        s = "Node; children: %d; visits: %d; reward: %f" % (
            len(self.children),
            self.visits,
            self.reward,
        )
        return s


bestchildren = []


def UCTSEARCH(budget, root):
    global bestchildren
    bestchildren = []

    for iter in range(int(budget)):
        if (iter % 10000) == 9999:
            print("simulation: %d" % iter)
            print(root)
        front = TREEPOLICY(root)
        reward, train_result = DEFAULTPOLICY(front.state)
        print("train_result[lat] ", train_result["lat"])
        BACKUP(front, reward, train_result["lat"], train_result["acc"])
    return BESTCHILDREN(root, 0, bestchildren)


def TREEPOLICY(node):
    # a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
    while node.state.terminal() == False:
        if len(node.state.moves) <= node.state.max_layers:
            if len(node.children) == 0:
                return EXPAND(node)
            elif random.uniform(0, 1) < 0.5:
                node = BESTCHILD(node, SCALAR)
            else:
                if node.fully_expanded() == False:
                    return EXPAND(node)
                else:
                    node = BESTCHILD(node, SCALAR)
        else:
            print(
                "EXCEED MAX LAYERS WITH CURR LENGTH: {} and LIMIT {}".format(
                    len(node.state.moves), node.state.max_layers
                )
            )
            node.state.moves = node.state.moves[: node.state.max_layers]
            return BESTCHILD(node, SCALAR)
    return node


def EXPAND(node):
    # print("EXPAND")
    tried_children = [c.state for c in node.children]
    new_state = node.state.next_state()
    while new_state in tried_children:
        new_state = node.state.next_state()
        # print("EXPAND to new_state ", new_state)
    node.add_child(new_state)
    # print("EXPAND to node.add_child ", node)
    return node.children[-1]


# This algorithm uses the vanilla MCTS formula
def BESTCHILD(node, scalar):
    bestscore = 0.0
    global bestchildren
    children_lat = [c.state.lat for c in node.children]
    children_reward = [c.reward for c in node.children]
    print("children_lat ", children_lat)
    print("children_reward ", children_reward)
    target_latency = node.state.target_latency

    if not node.children:
        return node
    else:
        total_children = len(node.children)
        for c in node.children:
            exploit = c.reward / c.visits
            explore = math.sqrt(2.0 * math.log(node.visits) / float(c.visits))
            score = exploit + scalar * explore

            if score == bestscore:
                bestchildren.append(c)
            if score > bestscore:
                bestchildren = [c]
                bestscore = score
        if len(bestchildren) == 0:
            print("OOPS: no best child found, probably fatal")
        print("bestchildren: ", bestchildren, ", type: ", type(bestchildren))
        top_one = sorted(bestchildren, key=lambda x: x.reward, reverse=True)[0]
        return top_one


def BESTCHILDREN(node, scalar, bestchildren):
    bestscore = 0.0
    children_lat = [c.state.lat for c in node.children]
    children_reward = [c.reward for c in node.children]
    print("children_lat ", children_lat)
    print("children_reward ", children_reward)
    target_latency = node.state.target_latency

    if not node.children:
        return node
    else:
        for c in node.children:
            exploit = c.reward / c.visits
            explore = math.sqrt(2.0 * math.log(node.visits) / float(c.visits))
            score = exploit + scalar * explore

            if score > bestscore:
                bestchildren.append(c)
                bestscore = score
        if len(bestchildren) == 0:
            print("OOPS: no best child found, probably fatal")
        print("bestchildren: ", bestchildren, ", type: ", type(bestchildren))
        top_five = sorted(bestchildren, key=lambda x: x.reward, reverse=True)[:5]
        return top_five


def DEFAULTPOLICY(state):
    tree_length = len(state.moves)
    while state.terminal() == False:
        state = state.next_state()
    return state.reward(tree_length), state.get_acc_latency()


def BACKUP(node, reward, latest_latency, latest_acc):
    while node != None:
        node.reward += reward
        node.state.acc = latest_acc
        node.state.lat = latest_latency
        node.visits += 1
        node = node.parent
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCTS research code")
    parser.add_argument("--num_sims", action="store", required=True, type=int)
    parser.add_argument(
        "--levels",
        action="store",
        required=True,
        type=int,
        choices=range(State.NUM_TURNS),
    )
    args = parser.parse_args()

    current_node = Node(State())
    for l in range(args.levels):
        current_node = UCTSEARCH(args.num_sims / (l + 1), current_node)
        print("level %d" % l)
        print("Num Children: %d" % len(current_node.children))
        for i, c in enumerate(current_node.children):
            print(i, c)
        print("Best Child: %s" % current_node.state)
        print("--------------------------------")
