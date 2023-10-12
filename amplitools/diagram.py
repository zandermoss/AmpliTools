import itertools
import networkx as nx
from .mrational import MRational
from .mring import MRing
from sympy import *
from tqdm import tqdm
from hashable_containers import hmap, hlist, HMultiGraph


class Diagram(object):
	"""A Feynman diagram

	* ``Diagram`` contains the underlying graph as well as expressions
	  for propagators and interaction vertices (Feynman rules).
	* Includes addition and multiplication methods implementing diagram union
	  and combinatorial product, respectively.
	* Provides a method to contruct an MRational expression for the diagram
	  using the Feynman rules.

	"""
	def __init__(*_arg):
		self = _arg[0]
		arg = _arg[1:]
		if len(arg) == 1 and type(arg[0]) == type(self):
			other = arg[0]
			self.G = graph_copy(other.G)
			self.propagators = propagator_copy(other.propagators)
			self.valent_nodes = {}
			for key, val in other.valent_nodes.items():
				self.valent_nodes[key] = list(val)
			self.connected_valent_nodes = {}
			for key, val in other.connected_valent_nodes.items():
				self.connected_valent_nodes[key] = list(val)
			self.count = other.count
		elif (len(arg) == 2
			  and isinstance(arg[0], HMultiGraph)
			  and isinstance(arg[1], dict)):
			self.G = graph_copy(arg[0])
			self.propagators = propagator_copy(arg[1])
			self.valent_nodes = {}
			self.connected_valent_nodes = {}
			self.collect_valent_nodes()
			self.count = 1
		else:
			assert False, "Bad Argument to Diagram.__init__()"


	def collect_valent_nodes(self):
		"""Collect labels of nodes with unbound operators
		and index them by particle type.

		"""
		nodes = self.G.nodes
		self.valent_nodes = {}
		for node in nodes:
			for ptype, opcount in nodes[node]['unbound_opmap'].items():
				if opcount > 0:
					self.valent_nodes.setdefault(ptype, [])
					self.valent_nodes[ptype].append(node)


	def __add__(self, other):
		assert type(self) == type(other)
		propagators = dict(list(self.propagators.items())
						   + list(other.propagators.items()))
		if len(set(self.G.nodes).intersection(set(other.G.nodes))) == 0:
			return Diagram(HMultiGraph(nx.union(self.G, other.G)), propagators)
		else:
			# Assume integer labels and stabilize self.G labels.
			G = graph_copy(self.G)
			H = nx.convert_node_labels_to_integers(graph_copy(other.G),
												   first_label=max(G.nodes)+1)
			return Diagram(HMultiGraph(nx.union(G, H)), propagators)


	def __mul__(self, other):
		assert type(self) == type(other)
		# Assume integer labels and stabilize self.G labels.
		# G = nx.convert_node_labels_to_integers(self.G)
		G = graph_copy(self.G)
		GD = Diagram(G, self.propagators)
		H = nx.convert_node_labels_to_integers(
			graph_copy(other.G), first_label=max(G.nodes)+1)
		HD = Diagram(H, other.propagators)
		products = []
		ptypes = (set(GD.valent_nodes.keys()).intersection(
				 set(HD.valent_nodes.keys())))
		for ptype in ptypes:
			pairs = itertools.product(GD.valent_nodes[ptype],
									  HD.valent_nodes[ptype])
			for pair in pairs:
				D = GD + HD
				D.G.add_edge(pair[0], pair[1], ptype=ptype)
				for node in pair:
					D.G.nodes[node]['unbound_opmap'][ptype] -= 1
					if D.G.nodes[node]['unbound_opmap'][ptype] == 0:
						D.valent_nodes[ptype].remove(node)
				products.append(D)
		return products


	def __str__(self):
		nodes = self.G.nodes
		string = "Nodes:\n\n"
		for node in nodes:
			string += "Label: "+node.__str__()+'\n'
			string += "Name: "+nodes[node]['name'].__str__()+'\n'
			string += ("Multiplicity: "+nodes[node]['multiplicity'].__str__()
					   + '\n')
			string += "Oplist: "+nodes[node]['oplist'].__str__()+'\n'
			string += "Opmap: "+nodes[node]['opmap'].__str__()+'\n'
			string += ("Unbound Opmap: "+nodes[node]['unbound_opmap'].__str__()
					   + '\n')
			string += ("External Label: "+nodes[node]['ext_label'].__str__()
					   + '\n')
			string += "MRational: \n"+nodes[node]['mrational'].__str__()
			string += "\n\n"
		edges = self.G.edges
		string += "Edges:\n"
		for edge in edges:
			string += "Label: "+edge.__str__()+'\n'
			string += "Ptype: "+edges[edge]['ptype'].__str__()
			string += "\n\n"
		string += "\n"
		return string


	def compute_mrational(self, root_node, delta_head="ID_"):
		"""Construct an MRational expression from the diagram.

		Perform a depth-first traversal of the diagram, using the Feynman rules
		associated to each vertex (interaction) and edge (propagator) to assemble
		the expression.

		"""
		# Get first rational.
		nodes = self.G.nodes
		neighbors = list(self.G[root_node].keys())
		assert len(neighbors) == 1
		neighbor = neighbors[0]
		rat, plist, label_counter = self.burrow(neighbor,
												root_node,
												len(nodes))
		rat = rat.contract_deltas(delta_head)
		# rat = rat.CleanDummyIndices()
		# rat = rat.canonize_indices()
		return rat


	def burrow(self, node, parent, label_counter):
		"""Depth-first recursion for constructing
		an MRational expression from a diagram.

		"""
		nodes = self.G.nodes
		edges = self.G.edges

		# FIXME: initialize label_counter at initial call of burrow
		# FIXME: assign *symmetrized* vertex expressions to the vertices.
		rat = MRational(nodes[node]['mrational'])
		multiplicity = nodes[node]['multiplicity']
		oplist = hlist(nodes[node]['oplist'])

		# Relabel the vertex rational.
		index_map = {}
		for i in range(1, multiplicity+1):
			label_counter += 1
			index_map[label_counter] = oplist[i-1]
			rat = rat.kinematic_replacement({i: [[1, label_counter], ]})
			rat = rat.kinematic_replacement({-i: [[1, -label_counter], ]})
			rat = rat.tensor_index_replacement(i, label_counter)

		# Isolate the parent index.
		keys = list(index_map.keys())
		values = list(index_map.values())
		#Hack: need to generalize for multigraphs
		parent_index = keys[values.index(edges[node, parent,0]['ptype'])]
		del index_map[parent_index]

		# Downstream work (inductive step).
		downstream_labels = []
		downstream_neighbors = list(self.G[node].keys())
		downstream_neighbors.remove(parent)
		for neighbor in downstream_neighbors:
			keys = list(index_map.keys())
			values = list(index_map.values())
			#Hack: need to generalize for multigraphs
			first_index = keys[values.index(edges[node, neighbor,0]['ptype'])]
			del index_map[first_index]

			neighbor_label = nodes[neighbor]['ext_label']
			# Leaf Case
			if neighbor_label >= 0:
				rat = rat.kinematic_replacement({first_index: [[1,
												neighbor_label], ]})
				rat = rat.kinematic_replacement({-first_index: [[1,
												-neighbor_label], ]})
				rat = rat.tensor_index_replacement(first_index, neighbor_label)
				downstream_labels += [neighbor_label, ]
			# Generic Case
			else:
				r, labels, label_counter = self.burrow(neighbor,
													   node,
													   label_counter)
				downstream_labels += labels
				rat = rat.kinematic_replacement({-first_index: [[1, -label]
												for label in labels]})
				rat = rat.tensor_index_replacement(first_index, label_counter)
				rat *= r
				rat = rat.link([[first_index, label_counter], ])

		# Upstream work
		parent_label = nodes[parent]['ext_label']

		# Root Case
		if parent_label >= 0:
			rat = rat.kinematic_replacement({parent_index: [[1,
											parent_label], ]})
			rat = rat.tensor_index_replacement(parent_index, parent_label)
			# Don't eliminate p_n yet. Need it for permutation.
			# elimination by way of momentum conservation will happen
			# *after* orbit in the onshell code.
			# rat = rat.kinematic_replacement({-parent_index:[[-1,-label]
			# for label in downstream_labels]})
			rat = rat.kinematic_replacement({-parent_index: [[1,
											-parent_label], ]})
		# Generic Case
		else:
			# Copy and relabel the upstream propagator.
			#Hack: need to generalize for multigraphs
			upstream_propagator = self.propagators[edges[node,parent,0]['ptype']]
			label_counter += 1
			propagator_index_1 = label_counter
			upstream_propagator = upstream_propagator.kinematic_replacement(
										{-1: [[1, -propagator_index_1], ]})
			upstream_propagator = upstream_propagator.kinematic_replacement(
										{1: [[1, propagator_index_1], ]})
			upstream_propagator = upstream_propagator.tensor_index_replacement(
										1, propagator_index_1)

			label_counter += 1
			propagator_index_2 = label_counter
			# Note: the propagator should not depend on momentum label 2,
			# only on 1. Just in case there is dependence on both labels,
			# we'll do a second replacement with the opposite sign.
			upstream_propagator = upstream_propagator.kinematic_replacement(
										{-2: [[1, -propagator_index_2], ]})
			upstream_propagator = upstream_propagator.kinematic_replacement(
										{2: [[1, propagator_index_2], ]})
			upstream_propagator = upstream_propagator.tensor_index_replacement(
										2, propagator_index_2)

			# Product
			rat *= upstream_propagator
			# link indices.
			rat = rat.link([[parent_index, propagator_index_1], ])
			rat = rat.tensor_index_replacement(parent_index, propagator_index_1)
			# Replace momenta with appropriate signs.
			rat = rat.kinematic_replacement({-parent_index: [[-1, -label]
											for label in downstream_labels]})
			rat = rat.kinematic_replacement({-propagator_index_1: [[1, -label]
											for label in downstream_labels]})
			rat = rat.kinematic_replacement({-propagator_index_2: [[-1, -label]
											for label in downstream_labels]})

		return rat, downstream_labels, label_counter


def graph_copy(graph):
	"""Graph copy utility for ``Diagram``"""
	G = HMultiGraph()
	for node in graph.nodes:
		G.add_node(
				   node,
				   multiplicity=graph.nodes[node]['multiplicity'],
				   oplist=hlist(graph.nodes[node]['oplist']),
				   opmap=hmap(graph.nodes[node]['opmap']),
				   unbound_opmap=hmap(graph.nodes[node]['unbound_opmap']),
				   ext_label=graph.nodes[node]['ext_label'],
				   mrational=MRational(graph.nodes[node]['mrational']),
				   name=graph.nodes[node]['name']
				   )
	for edge in graph.edges:
		G.add_edge(*edge, ptype=graph.edges[edge]['ptype'])
	return G


def propagator_copy(propagator):
	"""Propagator copy utility for ``Diagram``"""
	prop = {}
	for ptype, mrational in propagator.items():
		prop[ptype] = MRational(mrational)
	return prop
