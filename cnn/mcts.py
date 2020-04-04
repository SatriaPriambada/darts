#!/usr/bin/env python
import random
import math
import hashlib
import argparse
import pandas as pd

"""
Original version can be seen on https://github.com/ajax98/catan/blob/master/mcts.py
A quick Monte Carlo Tree Search implementation.  For more details on MCTS see See http://pubs.doc.ic.ac.uk/survey-mcts-methods/survey-mcts-methods.pdf
The State is just a game where you have NUM_TURNS and at turn i you can make
a choice from [-2,2,3,-3]*i and this to to an accumulated value.  The goal is for the accumulated value to be as close to 0 as possible.
The game is not very interesting but it allows one to study MCTS which is.  Some features 
of the example by design are that moves do not commute and early mistakes are more costly.  
In particular there are two models of best child that one can use 
"""

#MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration. 
NUM_CHILDREN = 4
SCALAR=1/math.sqrt(2.0)

class State():
	NUM_TURNS = 10	
	GOAL = 10
	MAX_VALUE= (5.0*(NUM_TURNS-1)*NUM_TURNS) / 2
	MOVES = pd.read_csv('generated_micro_cpu_center.csv')["genotype"].to_list()
	num_moves = len(MOVES)
	
	def __init__(self, value, moves, turn):
		#print("MOVES {}".format(self.MOVES))
		self.value = value
		self.turn = turn
		self.moves = []
		self.target_latency = [7, 12, 25] #ms
		# print("__init called __ {}".format(self.moves))

	def next_state(self):
		nextmove = []
		for i in range(self.turn):
			nextmove.append(random.choice(self.MOVES))
		print("Current nextmove: {} {}".format(nextmove, type(nextmove)))
		
		self.moves += nextmove
		next = State(self.value , self.moves, self.turn - 1)
		return next

	def terminal(self):
		if self.turn == 0:
			return True
		return False

	def get_accuracy(self):
		return 100 #person

	def get_latency(self):
		return 10 #ms

	def reward(self):
		w = 0.5
		acc = self.get_accuracy()
		l_t = self.target_latency
		lat = self.get_latency()
		lat_part = 1
		for strata in l_t:
			lat_part = lat_part * abs((1 - (lat / strata))) ** (1 - w)
		acc_part = abs((1 - (acc/100))) ** w
		r = 1 - (acc_part * lat_part)
		print("r: ", r, ", type: ", type(r))
		return r

class Node():
	def __init__(self, state, parent=None):
		self.visits = 1
		self.reward = 0.0	
		self.latency = 0
		self.state = state
		self.children = []
		self.parent = parent

	def add_child(self,child_state):
		child = Node(child_state,self)
		self.children.append(child)

	def update(self,reward, latest_latency):
		self.reward += reward
		self.latency = latest_latency
		self.visits += 1 

	def fully_expanded(self):
		if len(self.children) == self.state.num_moves:
			return True
		return False

	def __repr__(self):
		s="Node; children: %d; visits: %d; reward: %f"%(len(self.children),self.visits,self.reward)
		return s
		


def UCTSEARCH(budget,root):
	for iter in range(int(budget)):
		if (iter % 10000) == 9999:
			print("simulation: %d"%iter)
			print(root)
		front = TREEPOLICY(root)
		reward, latest_latency = DEFAULTPOLICY(front.state)
		BACKUP(front, reward, latest_latency)
	return BESTCHILD(root, 0)

def TREEPOLICY(node):
	#a hack to force 'exploitation' in a game where there are many options, and you may never/not want to fully expand first
	while node.state.terminal() == False:
		if len(node.children) == 0:
			return EXPAND(node)
		elif random.uniform(0,1) < .5:
			node = BESTCHILD(node,SCALAR)
		else:
			if node.fully_expanded() == False:	
				return EXPAND(node)
			else:
				node = BESTCHILD(node,SCALAR)
	return node

def EXPAND(node):
	tried_children = [c.state for c in node.children]
	new_state = node.state.next_state()
	while new_state in tried_children:
		new_state = node.state.next_state()
	node.add_child(new_state)
	return node.children[-1]

#current this uses the most vanilla MCTS formula it is worth experimenting with THRESHOLD ASCENT (TAGS)
def BESTCHILD(node,scalar):
	bestscore = 0.0
	children = []
	target = node.state.target_latency
	for c in node.children:
		exploit = c.reward / c.visits
		explore = math.sqrt(2.0*math.log(node.visits)/float(c.visits))	
		score = exploit+scalar*explore
		if score == bestscore:
			bestchildren.append(c)
		if score > bestscore:
			bestchildren = [c]
			bestscore = score
	if len(bestchildren) == 0:
		logger.warn("OOPS: no best child found, probably fatal")
	print("bestchildren: ", bestchildren, ", type: ", type(bestchildren))
	top_one = sorted(bestchildren, key=lambda x: x.reward, reverse=True)[0]
	return top_one

def DEFAULTPOLICY(state):
	while state.terminal() == False:
		state = state.next_state()
	return state.reward(), state.get_latency()

def BACKUP(node, reward, latest_latency):
	while node != None:
		node.reward += reward
		node.latency = latest_latency
		node.visits += 1 
		node = node.parent
	return

if __name__=="__main__":
	print("CALL main")
	parser = argparse.ArgumentParser(description='MCTS research code')
	parser.add_argument('--num_sims', action="store", required=True, type=int)
	parser.add_argument('--levels', action="store", required=True, type=int, choices=range(State.NUM_TURNS))
	args=parser.parse_args()
	
	current_node=Node(State())
	for l in range(args.levels):
		current_node = UCTSEARCH(args.num_sims/(l+1),current_node)
		print("level %d"%l)
		print("Num Children: %d"%len(current_node.children))
		for i,c in enumerate(current_node.children):
			print(i,c)
		print("Best Child: %s"%current_node.state)
		print("--------------------------------")