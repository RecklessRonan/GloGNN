#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Graph utilities."""

from collections import defaultdict, Iterable
from io import open
from itertools import permutations
from time import time

from six import iterkeys
from six.moves import range, zip_longest
from torch_geometric.utils import is_undirected, to_networkx


class Graph(defaultdict):
	"""Efficient basic implementation of nx `Graph' â€“ Undirected graphs with self loops"""

	def __init__(self):
		super(Graph, self).__init__(list)

	def nodes(self):
		return self.keys()

	def adjacency_iter(self):
		return self.items()

	def subgraph(self, nodes={}):
		subgraph = Graph()

		for n in nodes:
			if n in self:
				subgraph[n] = [x for x in self[n] if x in nodes]

		return subgraph

	def make_undirected(self):

		t0 = time()

		for v in self.keys():
			for other in self[v]:
				if v != other:
					self[other].append(v)

		t1 = time()
		# logger.info('make_directed: added missing edges {}s'.format(t1-t0))

		self.make_consistent()
		return self

	def make_consistent(self):
		t0 = time()
		for k in iterkeys(self):
			self[k] = list(sorted(set(self[k])))

		t1 = time()
		# logger.info('make_consistent: made consistent in {}s'.format(t1-t0))

		# self.remove_self_loops()

		return self

	def remove_self_loops(self):

		removed = 0
		t0 = time()

		for x in self:
			if x in self[x]:
				self[x].remove(x)
				removed += 1

		t1 = time()

		# logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
		return self

	def check_self_loops(self):
		for x in self:
			for y in self[x]:
				if x == y:
					return True

		return False

	def has_edge(self, v1, v2):
		if v2 in self[v1] or v1 in self[v2]:
			return True
		return False

	def degree(self, nodes=None):
		if isinstance(nodes, Iterable):
			return {v: len(self[v]) for v in nodes}
		else:
			return len(self[nodes])

	def order(self):
		"Returns the number of nodes in the graph"
		return len(self)

	def number_of_edges(self):
		"Returns the number of nodes in the graph"
		return sum([self.degree(x) for x in self.keys()]) / 2

	def number_of_nodes(self):
		"Returns the number of nodes in the graph"
		return self.order()

	def gToDict(self):
		d = {}
		for k, v in self.items():
			d[k] = v
		return d

	def printAdjList(self):
		for key, value in self.items():
			print(key, ":", value)


def clique(size):
	return from_adjlist(permutations(range(1, size + 1)))


# http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
def grouper(n, iterable, padvalue=None):
	"grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
	return zip_longest(*[iter(iterable)] * n, fillvalue=padvalue)


def parse_adjacencylist(f):
	adjlist = []
	for l in f:
		if l and l[0] != "#":
			introw = [int(x) for x in l.strip().split()]
			row = [introw[0]]
			row.extend(set(sorted(introw[1:])))
			adjlist.extend([row])

	return adjlist


def parse_adjacencylist_unchecked(f):
	adjlist = []
	for l in f:
		if l and l[0] != "#":
			adjlist.extend([[int(x) for x in l.strip().split()]])
	return adjlist

def from_pyg(pyg_data):
	edge_index = pyg_data.edge_index
	tmp = to_networkx(pyg_data)
	n_nodes = pyg_data.x.shape[0]
	return from_networkx(tmp, undirected=is_undirected(edge_index))

def load_edgelist(file_, undirected=True):
	G = Graph()
	with open(file_) as f:
		for l in f:
			if "node" in l:
				# first line in file i.e. has header so ignore
				continue
			if (len(l.strip().split()[:2]) > 1):
				x, y = l.strip().split()[:2]
				x = int(x)
				y = int(y)
				G[x].append(y)
				if undirected:
					G[y].append(x)
			else:
				x = l.strip().split()[:2]
				x = int(x[0])
				G[x] = []

	G.make_consistent()
	return G

def from_networkx(G_input, undirected=True):
	G = Graph()

	for idx, x in enumerate(G_input.nodes()):
		for y in iterkeys(G_input[x]):
			G[x].append(y)

	if undirected:
		G.make_undirected()

	return G

def from_adjlist(adjlist):
	G = Graph()

	for row in adjlist:
		node = row[0]
		neighbors = row[1:]
		G[node] = list(sorted(set(neighbors)))

	return G


def from_adjlist_unchecked(adjlist):
	G = Graph()

	for row in adjlist:
		node = row[0]
		neighbors = row[1:]
		G[node] = neighbors

	return G


def from_dict(d):
	G = Graph()
	for k, v in d.iteritems():
		G[k] = v

	return G
