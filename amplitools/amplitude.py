import itertools
import networkx as nx
from .diagram import Diagram
from .interface import draw_graphs
from .mrational import MRational
from .mring import MRing
from sympy import *
from tqdm import tqdm
from hashable_containers import hmap, hlist, HMultiGraph
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics.named_groups import SymmetricGroup
import nautypy as nty
import matplotlib.pyplot as plt

"""High level functions for computing amplitudes"""

def isomorphism_classification(diagrams):
	"""Reduce diagrams by isomorphism class.

	Args:
		diagrams (list): A list of all diagrams resulting from the Wick contraction of a particular set of interaction vertices and external leg vertices. Generally contains many isomorphs of a small number of isomorphism class representatives (canonically labeled graphs).

	Returns:
		classes (hashable_containers.hmap): An hmap keyed by canonically labeled graphs and valued by hmaps, which themselves are keyed by external label permutations ``s_g_ext`` and valued by integer counts for the corresponding isomorph.

	"""
	# Initialize the class hmap.
	classes = hmap()
	# Step through the diagrams, computing canonical isomorphs and corresponding isomorphisms.
	for d in diagrams:
		# Remember that the external legs are labeled by g.nodes[label]['ext_label'],
		# not by 'label' itself, so we need to convert the canonical mapping of the
		# *node labels* to a canonical mapping of the *external labels*.
		# For convenience, we will replace the existing node labels with sequential
		# integers (indices) starting at zero.
		ext_nodes = [node for node in d.G.nodes if d.G.nodes[node]['ext_label']>=0]
		n_ext = len(ext_nodes)
		int_nodes = [node for node in d.G.nodes if d.G.nodes[node]['ext_label']<0]
		index_to_node = {i:node for i,node in enumerate(sorted(ext_nodes))}
		index_to_node.update({(i+n_ext):node for i,node in enumerate(sorted(int_nodes))})
		node_to_index = {val:key for key,val in index_to_node.items()}
		# Homogenize external vertex labels.
		g = HMultiGraph(nx.relabel_nodes(d.G,node_to_index,copy=True))
		node_to_ext = {node:g.nodes[node]['ext_label'] for node in range(n_ext)}
		ext_to_node = {ext:node for node,ext in node_to_ext.items()}
		for node in g.nodes:
			if g.nodes[node]['ext_label'] >= 0:
				g.nodes[node]['ext_label'] = 1
		# Call nautypy to canonize ``g``.
		# Nautypy returns the canonical isomorph ``g_canonical``,
		# a generating set for the automorphism group of ``g``,
		# and a vertex relabeling map ``canonical_map`` which sends ``g_canonical``
		# to ``g``.
		# We sort output color labels by their corresponding 'ext_label' attributes,
		# guaranteeing that the canonical map will not mix the labels of external
		# and internal vertices.
		g_canonical, autgens, canonical_map = nty.canonize_multigraph(g,
												color_sort_conditions=[('ext_label',1)])
		# Restore external vertex labels.
		for node in g_canonical.nodes:
			if g_canonical.nodes[node]['ext_label'] >= 0:
				g_canonical.nodes[node]['ext_label'] = node_to_ext[node]
		# It is possible for two distinct canonical maps to produce the same isomorph
		# from a given canonically labeled graph. This occurs whenever the graph admits
		# a nontrivial automorphism group Aut(g). To maximally simplify ``classes``, we reduce
		# canonical maps by the automorphism group, sending any two canonical maps c1,c2
		# in the same coset of Aut(g) to a canonical coset representative c*.
		# Sympy has convenient functions for this purpose which use the Schreier-Sims algorithm
		# under the hood.
		Gp = SymmetricGroup(n_ext)
		restr_auts = [Permutation([aut[node] for node in range(n_ext)]) for aut in autgens]
		Aut = PermutationGroup(restr_auts)
		restr_canmap = [canonical_map[node] for node in range(n_ext)]
		canonical_perm = Permutation(restr_canmap)
		canonical_perm_mod_aut = ~(Gp._coset_representative(~canonical_perm,Aut))
		# Convert the canonical node label map to a permutation map acting on the external labels.
		ext_canonical_map_mod_aut = hmap({ext:node_to_ext[canonical_perm_mod_aut(node)] for ext,node in ext_to_node.items()})
		#Initialize and increment the entry for [g_canonical][ext_canonical_map] in classes.
		classes.setdefault(g_canonical,hmap())
		classes[g_canonical].setdefault(ext_canonical_map_mod_aut,0)
		classes[g_canonical][ext_canonical_map_mod_aut] += 1
	return classes


def supergraph_dfs(diagram, node):
	"""Generate all connected trees resulting from Wick contraction of a given
	collection of interaction vertices and external-leg vertices.

	Given a set of operators (in this case containing both single external-leg operators and
	groups of operators corresponding to interaction vertices), Wick contraction is the
	process of generating all possible pairs of operators, such that each operator appears
	in at most one pair.

	Each such pairing (or "contraction") can be represented by a graph, or "Feynman diagram" `d`.
	The vertices of `d` correspond to sets of field operators. "External" vertices
	map to singleton sets containing only the corresponding external field operator, while
	"internal/interaction" vertices map to sets containing all the field operators
	involved in the interaction. Contracted pairs of operators are represented by edges
	between the vertices containing those operators. External vertices are incident to a single
	edge, while interaction vertices are incident to three or more edges.

	Each Feynman diagram can then be combined with a set of algebraic rules (the "Feynman rules")
	to construct the corresponding contribution to the scattering amplitude involving the
	specified external states.

	We focus only on *connected* *tree* diagrams (though it is straightforward to generalize
	this approach to loop diagrams and disconnected diagrams). We generate all contractions
	using a backtacking algorithm. In particular, we perform a depth-first search of an implicit
	tree containing all possible partial Wick contraction diagrams as nodes.
	This tree (the "supergraph") is rooted at a completely disconnected diagram (no edges)
	containing the interaction and external vertices, with a designated root vertex to include
	in the first contractions.

	If it is possible to follow a chain of contractions beginning with an operator of some
	vertex `v` in diagram `d` all the way back to the root diagram, then `v` is
	said to be *connected*. Otherwise it is *disconnected*.

	If at least one operator of a vertex `v` has not yet been contracted with another operator,
	`v` is said to be *valent*.

	Given a supergraph node diagram `d` and a *connected*, *valent* vertex `v` in `d`, calling
	``supergraph_dfs(d,v)`` first generates the child nodes of `d` (if they exist) in two steps:

	1. For each particle type `t`, select one uncontracted operator of type `t` belonging
	   to `v` (if it exists) and another uncontracted operator of type `t` belonging to
	   a *valent*, *disconnected* vertex `w` (if it exists); Create a copy `d*` of the diagram
	   `d` and contract these two operators, adding the edge (`v`,`w`) to `d*`.
	2. Add each such `d*` as a node of the supergraph, and add the pair (`d`, `d*`) as an edge.

	Next, ``supergraph_dfs`` determines what to do with each child diagram `d*` in turn:

	* If there are valent vertices in `d*`, but none of them are *connected*, the only way to
	  proceed would be to begin contractions in a *disconnected component* of the diagram. We
	  are only interested in connected diagrams, so we prune the search tree and discard `d*`.
	* If there are valent vertices in `d*`, but none of them are *disconnected*, the only way
	  to proceed would be to add an edge between two tree vertices, creating a loop diagram.
	  For the time being, we are focusing on tree diagrams, and so we prune the search tree
	  and discard `d*` whenever this happens. 
	* If no valent vertices remain in `d*`, it must be a connected Feynman diagram
	  representing a complete Wick contraction, and is therefore a leaf of the search tree
	  (supergraph). In this case, we append `d*` to the list of complete, connected
	  tree diagrams (``diagram_list``).
	* If `d*` contains both connected and disconnected valent nodes, we select a
	  connected, valent vertex `v*` in `d*`; call ``supergraph_dfs(d*,v*)``, which
	  returns a list of all leaf diagrams belonging to the subtree rooted at `d*`;
	  and concatenate this list to ``diagram_list``.

	Once all children `d*` have been processed, ``diagram_list`` contains all leaf diagrams
	belonging to the subtree rooted at `d`. Finally, this list is returned from
	``supergraph_dfs(`d`,`v`)``. It follows immediately that calling ``supergraph_dfs`` on
	a completely disconnected diagram containing an even number of operators (which we always
	assume) grouped into internal and external vertices with no edges between them will return
	a complete list of connected, tree diagrams involving that particular set of interactions
	and external states.

	Args:
		diagram (amplitools.Diagram): A partially contracted diagram serving as the root node for the search tree.
		node (some hashable object): A valent, connected vertex in ``diagram''. Edges will be added to ``diagram`` connecting ``node`` to other valent vertices in ``diagram``, if possible, to generate child diagrams of ``diagram``.

	Returns:
		diagram_list (list): A list of all connected tree diagrams resulting from Wick contraction.

	"""
	unbound_opmap = diagram.G.nodes[node]['unbound_opmap']
	ptypes = unbound_opmap.keys()
	targets = []
	# valent_ptypes=[]
	for ptype in ptypes:
		if node in diagram.valent_nodes[ptype]:
			diagram.valent_nodes[ptype].remove(node)
		if ptype in diagram.connected_valent_nodes.keys():
			if node in diagram.connected_valent_nodes[ptype]:
				diagram.connected_valent_nodes[ptype].remove(node)
	for ptype in ptypes:
		targets.append(itertools.combinations(diagram.valent_nodes[ptype],
											  unbound_opmap[ptype]))
	diagram_list = []
	for target in itertools.product(*targets):
		next_diagram = Diagram(diagram)
		valent_neighbors = []
		for ptype, neighbors in zip(ptypes, target):
			del next_diagram.G.nodes[node]['unbound_opmap'][ptype]
			for neighbor in neighbors:
				nextkeys = (next_diagram.G.nodes[neighbor]
							['unbound_opmap'].keys())
				for neighbor_ptype in nextkeys:
					next_diagram.connected_valent_nodes.setdefault(
															neighbor_ptype, [])
					if (neighbor not in next_diagram.connected_valent_nodes
							[neighbor_ptype]):
						next_diagram.connected_valent_nodes[
							neighbor_ptype].append(neighbor)
				next_diagram.G.add_edge(node, neighbor, ptype=ptype)
				next_diagram.G.nodes[neighbor]['unbound_opmap'][ptype] -= 1
				if next_diagram.G.nodes[neighbor]['unbound_opmap'][ptype] == 0:
					del next_diagram.G.nodes[neighbor]['unbound_opmap'][ptype]
					next_diagram.valent_nodes[ptype].remove(neighbor)
					next_diagram.connected_valent_nodes[ptype].remove(neighbor)
					if len(next_diagram.valent_nodes[ptype]) == 0:
						del next_diagram.valent_nodes[ptype]
			if len(next_diagram.connected_valent_nodes[ptype]) == 0:
				del next_diagram.connected_valent_nodes[ptype]
		# Leaf (complete Wick contraction)
		if (len(next_diagram.valent_nodes.keys()) == 0
				and len(next_diagram.connected_valent_nodes) == 0):
			diagram_list += [next_diagram, ]
		# Prune (disconnected diagram)
		elif (len(next_diagram.valent_nodes.keys()) != 0
				and len(next_diagram.connected_valent_nodes) == 0):
			continue
		# Recurse
		else:
			diagram_list += supergraph_dfs(
					next_diagram,
					list(next_diagram.connected_valent_nodes.values())[0][0])
	return diagram_list


def compute_contractions(vertices, propagators, external_legs, tensor_symmetries):
	"""Generate all diagrams resulting from Wick contraction of a set of interaction vertices ``vertices``.

	The calculation proceeds in four stages,

	1. Generate all (connected) (multi)graphs resulting from adding contraction edges
	   to a collection of interaction vertices `vertices` and external-leg vertices
	   `external_legs`.
	2. Store these graphs in a dictionary keyed by canonically labeled graphs, and valued
	   with lists of isomorphisms (node relabeling maps) and the number of each isomorph
	   appearing in the full set of contraction multigraphs.
	3. Compute an ``MRational`` expression for each canonically labeled graph using
	   the Feynman rules.
	4. Generate the full amplitude by applying each isomorphism to the canonical MRational
	   and summing over the resulting expressions.

	Generally, symbolic algebra is *much* slower than graph isomorphism or permutation group calculations, so the algorithm outlined above attempts to minimize the number of actual Feynman-rule constructions by building expressions for canonical isomorphs only.

	Note:
		For each diagram's graph ``g``, there exists a canonical labeling ``g_canon``
		which is related to ``g`` by a vertex label permutation ``s_g``.
		The vertices of any such ``g`` are divided into *external leg* vertices, and
		*internal* (interaction) vertices.

		Once the graph ``g`` is combined with the Feynman rules to compute an ``MRational``
		term ``rat(g)`` contributing to the amplitude, all internal leg momenta are replaced
		by linear combinations of external leg momenta (or dummy loop momenta), and all internal
		flavor indices are summed over, so the only labels remaining are the external leg labels.
		The canonical permutation ``s_g`` thus acts on ``rat(g_canon)`` by a restricted permuation
		of the external leg labels ``s_g_ext``.

		We could compute ``rat(g)`` for every graph spat out by the ``supergraph_dfs`` algorithm,
		but we would waste time doing unnecessary rational algebra. Instead, we will "bin" the 
		graphs `g` returned from ``supergraph_dfs`` by their canonical isomorphs ``g_canon`` and
		the associated external label permutation ``s_g_ext`` into a nested hmap structure
		``classes``. 

		To construct the amplitude, we step through the keys of ``classes`` and call
		``Diagram.compute_mrational`` *only once* per isomorphism class. For each ``s_g_ext``
		in ``classes[g_canon]``, we compute ``s_g_ext(rat(g_canon))``, multiply it by the
		number of appearances the associated diagram ``(classes[g_canon][s_g_ext])`` makes in
		``diagrams``, and add the result to the amplitude.

		The number of contractions grows rapidly with the number of external legs,
		but the number of isomorphism classes grows much more slowly. In the worst
		case, we work with an EFT containing an infinite tower of interaction operators
		of unbounded dimension. Wick contractions of such a tower should yield a set
		of labeled trees of cardinality O(n^n), where n is the number of external legs.
		The number of isomorphism classes, on the other hand, is roughly O(3^n)
		[doi:10.2307/1969046]. 

		An additional improvement to this approach would see canonization used
		as a pruning technique in the ``supergraph_dfs`` algorithm to reduce the number
		of contractions constructed by binning "as we go". Empirically, constructing
		and manipulating rational expressions appears to take much longer than 
		generating graphs from contractions, so we'll punt that optimization.

	Args:
		vertices (list): The interaction vertices appearing in each diagram.
		propagators (dict): A dict, keyed by particle type, containing free propagators (from ``interface.FeynmanRules``).
		external_legs (list): A list of particle type strings corresponding to the external fields.
		tensor_symmetries (dict): A dict of tensor symmetry generating sets keyed by tensor head string (from ``TensorSymmetries``).
	
	Returns:
		amp_term (MRational): The amplitude term corresponding to all diagrams containing the collection of interaction vertices ``vertices``.


	"""
	# Define the identity polynomials.
	p_one = hlist(propagators.values())[0].nd_list[0][0].poly_one()
	p_zero = hlist(propagators.values())[0].nd_list[0][0].poly_zero()
	# Define the multiplicative identity in the MRational field.
	rat_one = MRational(hlist([hlist([MRing(hmap({((0, 0),): p_one})),
						hmap({MRing(hmap({((0, 0),): p_one})): 1})]), ]))
	# Define the additive identity in the MRational field.
	rat_zero = MRational(hlist([hlist([MRing(hmap({((0, 0),): p_zero})),
						 hmap({MRing(hmap({((0, 0),): p_one})): 1})]), ]))
	# Generate internal vertex list.
	internal_list = []
	label_counter = 1
	for multiplicity, oplist, r, name in vertices:
		G = HMultiGraph()
		opmap = hmap()
		for ptype in oplist:
			opmap.setdefault(ptype, 0)
			opmap[ptype] += 1
		G.add_node(label_counter,
				   multiplicity=multiplicity,
				   oplist=hlist(oplist),
				   opmap=opmap,
				   unbound_opmap=hmap(opmap),
				   ext_label=(-1),
				   mrational=MRational(r),
				   name=name)
		internal_list.append(G)
		label_counter += 1
	# Generate external vertex list.
	external_list = []
	root_vertex = None
	ext_label = 1
	symbolblocks = []
	for ptype in external_legs:
		G = HMultiGraph()
		G.add_node(ext_label,
				   multiplicity=1,
				   oplist=hlist([ptype, ]),
				   opmap=hmap({ptype: 1}),
				   unbound_opmap=hmap({ptype: 1}),
				   ext_label=ext_label,
				   mrational=rat_one,
				   name='')
		external_list.append(G)
		symbolblocks.append([-ext_label, ext_label])
		ext_label += 1
	# Compute the list of all complete, connected contractions of internal
	# vertices.
	seed_diagram = Diagram(internal_list[0], propagators)
	for i in range(1, len(internal_list)):
		seed_diagram += Diagram(internal_list[i], propagators)
	for i in range(0, len(external_list)):
		seed_diagram += Diagram(external_list[i], propagators)
	#for i in tqdm([1,],desc="Contractions"):
	diagrams = supergraph_dfs(seed_diagram,
					list(seed_diagram.G.nodes.keys())[0])
	if len(diagrams) == 0:
		return rat_zero
	#Classify diagrams by isomorphism.
	classes = isomorphism_classification(diagrams)
	#Draw the isomorphs
	draw_graphs(list(classes.keys()))
	amp_term = MRational(rat_zero)
	for g_canon, isomaps in classes.items():
		d_canon = Diagram(g_canon, propagators)
		root_label = 0
		# Construct an MRational from each canonical isomorph using the Feynman rules.
		#for i in tqdm([1, ], desc="Construct MRational Expression"):
		rat_canon = d_canon.compute_mrational(root_label)
		# Find a canonical labeling of dummy tensor indices.
		#for i in tqdm([1, ], desc="Canonize Tensor Indices"):
		rat_canon = rat_canon.canonize_indices(len(external_legs))
		# Find a canonical index tuple for each tensor w/r/t its symmetry group.
		#for i in tqdm([1, ], desc="Sort Symmetric Tensor Indices"):
		rat_canon = rat_canon.sort_indices(tensor_symmetries)
		# Collect MRational terms.
		#for i in tqdm([1, ], desc="Collect Terms"):
		rat_canon = rat_canon.collect()
		# Generate and sum the isomorphism orbit of the canonical diagram.
		#for isomap,count in tqdm(isomaps.items(), desc="Isomorphism Orbit"):
		for isomap,count in isomaps.items():
			rat_isomorph = rat_canon.block_replacement(isomap, symbolblocks)
			amp_term += rat_isomorph*count
	return amp_term


def evolution_expansion(symbol_list, g_order):
	""" Expand the time evolution operator U to order ``g_order`` in the
	perturbative parameter ``g``.

	Args:
		symbol_list (list): A list of operator coefficients, ordered so that ``symbol_list[i]`` corresponds to the coefficient of ``g^i``.

	Returns:
		U_series (sympy.series): An expansion of ``U`` to order ``g_order``. 

	"""
	g = symbol_list[0]
	f = poly(0, symbol_list, domain='QQ_I')
	for i in range(1, g_order+1):
		f += I*symbol_list[i]*g**i
	U_series = series(exp(f.as_expr()), g, n=g_order+1)
	return U_series


def compute_tree_amplitude(interactions,
						 propagators,
						 external_legs,
						 tensor_symmetries):
	"""Compute a tree amplitude from interaction operators ``interactions``,
	free propagators ``propagators``, external-leg operators ``external_legs``,
	and a description of the symmetries of any flavor tensors appearing
	in these operators.

	The calculation proceeds in four stages,

	1. Expand the time-evolution operator in powers of g.
	2. Extract the tree contribution.
	3. Strip off its operator-valued coefficient.
	4. Compute all possible Wick contractions of the various interaction operators
	   appearing in the coefficient.

	Args:
		interactions (list): A list of interaction operators (from ``interface.FeynmanRules``).
		propagators (dict): A dict, keyed by particle type, containing free propagators (from ``interface.FeynmanRules``).
		external_legs (list): A list of particle type strings corresponding to the external fields.
		tensor_symmetries (dict): A dict of tensor symmetry generating sets keyed by tensor head string (from ``TensorSymmetries``).
	
	Returns:
		amplitude (MRational): The tree amplitude.

	"""
	# Grab the identity polynomials from an interaction
	# MRational.
	p_one = interactions[0][3].nd_list[0][0].poly_one()
	p_zero = interactions[0][3].nd_list[0][0].poly_zero()
	# Define the additive identity in the MRational field.
	rat_zero = MRational(hlist([hlist([MRing(hmap({((0, 0),): p_zero})),
						 hmap({MRing(hmap({((0, 0),): p_one})): 1})]), ]))
	# Sort the interaction operators by the associated power of g,
	# the perturbative parameter.
	g_ordered_interactions = {}
	for interaction in interactions:
		int_g_order = interaction[0]
		g_ordered_interactions.setdefault(int_g_order, [])
		g_ordered_interactions[int_g_order].append(interaction[1:])
	# Determine the total number of external legs.
	n_external = len(external_legs)
	# Determine total g_order for tree diagrams with n_external legs.
	# Recall L = 1/2(g_order - n_external)+1, where L is the number
	# of loops involved in a given diagram (zero, in our case).
	g_order = n_external-2
	# Compute the taylor expansion of the time-evolution operator U to g_order.
	# This expansion will consist of powers of g with operator valued
	# coefficients. Of course, there will generally be multiple interaction
	# operators with varying particle content at a given g_order. So, we will
	# define symbols O_i to stand in for the sum of all interaction operators
	# at g-order i. Later on, we'll substitute in the actual operators, as
	# sorted in g_ordered_interactions above.
	g = symbols('g')
	symbol_list = [g, ]+[symbols("O_"+str(i)) for i in range(1, g_order+1)]
	U_series = evolution_expansion(symbol_list, g_order)
	# Extract the coefficient of the g^g_order term.
	tree_term = U_series.coeff(g**g_order).as_poly(domain='QQ_I')
	# Replace the O_i symbols with the corresponding operators
	# from g_ordered_interactions. We'll convert tree_term to sympy's
	# polynomial dictionary representation. tree_term is a polynomial in g
	# (g^0) and the various O_i.
	# We'll end up with a dictionary of the form (e.g.)
	# { ( 0,   1,   0,   1,   ...) : coefficient }
	#	 g	O_1  O_2  O_3  ...
	# It is then trivial to replace the O_i with powers of the corresponding
	# sets of interactions.
	for monomial_tuple, coefficient in tree_term.as_dict().items():
		interaction_list_factors = []
		missingterm = False
		for i, power in enumerate(monomial_tuple):
			operator_g_order = i+1
			if operator_g_order in g_ordered_interactions.keys():
				for n in range(power):
					interaction_list_factors.append(
									g_ordered_interactions[operator_g_order])
			elif power != 0:
				missingterm = True
		if missingterm:
			continue
		# Take the set product of these interaction lists. The result is
		# a list of lists of interactions, each of which represents the list
		# of vertices appearing, after contractions, in various feynman
		# diagrams.
		vertex_lists = list(itertools.product(*interaction_list_factors))
		# Compute and sum all diagram MRationals.
		amplitude = rat_zero
		for vertices in vertex_lists:
			# Compute the diagram terms generated from each combination
			# of interaction vertices.
			#for i in tqdm([1, ], desc="compute_contractions"):
			diagram = compute_contractions(vertices,
										   propagators,
										   external_legs,
										   tensor_symmetries)
			diagram *= coefficient
			amplitude += diagram
		# Sort tensor indices into standard order.
		#for i in tqdm([1, ], desc="sort_indices"):
		amplitude = amplitude.sort_indices(tensor_symmetries)
		# Collect terms.
		#for i in tqdm([1, ], desc="collect"):
		amplitude = amplitude.collect()
	# Voila! An amplitude is born.
	return amplitude
