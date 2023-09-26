import itertools
import networkx as nx
from networkx.algorithms import isomorphism
from MRational import MRational
from MRing import MRing
from sympy import *
from tqdm import tqdm
from Interface import DrawNXGraphList


def GraphCopy(graph):
    G = nx.Graph()
    for node in graph.nodes:
        G.add_node(
                   node,
                   multiplicity=graph.nodes[node]['multiplicity'],
                   oplist=graph.nodes[node]['oplist'],
                   opmap=dict(graph.nodes[node]['opmap']),
                   unbound_opmap=dict(graph.nodes[node]['unbound_opmap']),
                   ext_label=graph.nodes[node]['ext_label'],
                   mrational=MRational(graph.nodes[node]['mrational']),
                   name=graph.nodes[node]['name']
                   )
    for edge in graph.edges:
        G.add_edge(*edge, ptype=graph.edges[edge]['ptype'])
    return G


def PropCopy(propagator):
    prop = {}
    for ptype, mrational in propagator.items():
        prop[ptype] = MRational(mrational)
    return prop


class Diagram(object):

    def __init__(*_arg):
        self = _arg[0]
        arg = _arg[1:]
        if len(arg) == 1 and type(arg[0]) == type(self):
            other = arg[0]
            self.G = GraphCopy(other.G)
            self.propagators = PropCopy(other.propagators)
            self.valent_nodes = {}
            for key, val in other.valent_nodes.items():
                self.valent_nodes[key] = list(val)
            self.connected_valent_nodes = {}
            for key, val in other.connected_valent_nodes.items():
                self.connected_valent_nodes[key] = list(val)
            self.count = other.count
        elif (len(arg) == 2
              # and type(arg[0]) == type(nx.Graph())
              and isinstance(arg[0], nx.Graph)
              # and type(arg[1]) == type(dict())):
              and isinstance(arg[1], dict)):
            self.G = GraphCopy(arg[0])
            self.propagators = PropCopy(arg[1])
            self.valent_nodes = {}
            self.connected_valent_nodes = {}
            self.CollectValentNodes()
            self.count = 1
        else:
            assert False, "Bad Argument to Diagram.__init__()"

    def CollectValentNodes(self):
        """ Collect labels of nodes with unbound operators
            and index them by particle type. """
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
            return Diagram(nx.union(self.G, other.G), propagators)
        else:
            # Assume integer labels and stabilize self.G labels.
            G = GraphCopy(self.G)
            H = nx.convert_node_labels_to_integers(GraphCopy(other.G),
                                                   first_label=max(G.nodes)+1)
            return Diagram(nx.union(G, H), propagators)

    def __mul__(self, other):
        assert type(self) == type(other)
        # Assume integer labels and stabilize self.G labels.
        # G = nx.convert_node_labels_to_integers(self.G)
        G = GraphCopy(self.G)
        GD = Diagram(G, self.propagators)
        H = nx.convert_node_labels_to_integers(
            GraphCopy(other.G), first_label=max(G.nodes)+1)
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

    def ComputeMRational(self, root_node, delta_head="ID_"):
        # Get first rational.
        nodes = self.G.nodes
        neighbors = list(self.G[root_node].keys())
        assert len(neighbors) == 1
        neighbor = neighbors[0]
        rat, plist, label_counter = self.Burrow(neighbor,
                                                root_node,
                                                len(nodes))
        rat = rat.DeltaContract(delta_head)
        # rat = rat.CleanDummyIndices()
        # rat = rat.CanonicalizeIndices()
        return rat

    def Burrow(self, node, parent, label_counter):
        """ Depth-first recursion for constructing
            an MRational expression from a diagram. """
        nodes = self.G.nodes
        edges = self.G.edges

        # FIXME: initialize label_counter at initial call of Burrow
        # FIXME: assign *symmetrized* vertex expressions to the vertices.
        rat = MRational(nodes[node]['mrational'])
        multiplicity = nodes[node]['multiplicity']
        oplist = nodes[node]['oplist']

        # Relabel the vertex rational.
        index_map = {}
        for i in range(1, multiplicity+1):
            label_counter += 1
            index_map[label_counter] = oplist[i-1]
            rat = rat.KinematicReplacement({i: [[1, label_counter], ]})
            rat = rat.KinematicReplacement({-i: [[1, -label_counter], ]})
            rat = rat.TensorIndexReplacement(i, label_counter)

        # Isolate the parent index.
        keys = list(index_map.keys())
        values = list(index_map.values())
        parent_index = keys[values.index(edges[node, parent]['ptype'])]
        del index_map[parent_index]

        # Downstream work (inductive step).
        downstream_labels = []
        downstream_neighbors = list(self.G[node].keys())
        downstream_neighbors.remove(parent)
        for neighbor in downstream_neighbors:
            keys = list(index_map.keys())
            values = list(index_map.values())
            first_index = keys[values.index(edges[node, neighbor]['ptype'])]
            del index_map[first_index]

            neighbor_label = nodes[neighbor]['ext_label']
            # Leaf Case
            if neighbor_label is not None:
                rat = rat.KinematicReplacement({first_index: [[1,
                                                neighbor_label], ]})
                rat = rat.KinematicReplacement({-first_index: [[1,
                                                -neighbor_label], ]})
                rat = rat.TensorIndexReplacement(first_index, neighbor_label)
                downstream_labels += [neighbor_label, ]
            # Generic Case
            else:
                r, labels, label_counter = self.Burrow(neighbor,
                                                       node,
                                                       label_counter)
                downstream_labels += labels
                rat = rat.KinematicReplacement({-first_index: [[1, -label]
                                                for label in labels]})
                rat = rat.TensorIndexReplacement(first_index, label_counter)
                rat *= r
                rat = rat.Link([[first_index, label_counter], ])

        # Upstream work
        parent_label = nodes[parent]['ext_label']

        # Root Case
        if parent_label is not None:
            rat = rat.KinematicReplacement({parent_index: [[1,
                                            parent_label], ]})
            rat = rat.TensorIndexReplacement(parent_index, parent_label)
            # Don't eliminate p_n yet. Need it for permutation.
            # elimination by way of momentum conservation will happen
            # *after* orbit in the onshell code.
            # rat = rat.KinematicReplacement({-parent_index:[[-1,-label]
            # for label in downstream_labels]})
            rat = rat.KinematicReplacement({-parent_index: [[1,
                                            -parent_label], ]})
        # Generic Case
        else:
            # Copy and relabel the upstream propagator.
            upstream_propagator = self.propagators[edges[node,
                                                   parent]['ptype']]
            label_counter += 1
            propagator_index_1 = label_counter
            upstream_propagator = upstream_propagator.KinematicReplacement(
                                        {-1: [[1, -propagator_index_1], ]})
            upstream_propagator = upstream_propagator.KinematicReplacement(
                                        {1: [[1, propagator_index_1], ]})
            upstream_propagator = upstream_propagator.TensorIndexReplacement(
                                        1, propagator_index_1)

            label_counter += 1
            propagator_index_2 = label_counter
            # Note: the propagator should not depend on momentum label 2,
            # only on 1. Just in case there is dependence on both labels,
            # we'll do a second replacement with the opposite sign.
            upstream_propagator = upstream_propagator.KinematicReplacement(
                                        {-2: [[1, -propagator_index_2], ]})
            upstream_propagator = upstream_propagator.KinematicReplacement(
                                        {2: [[1, propagator_index_2], ]})
            upstream_propagator = upstream_propagator.TensorIndexReplacement(
                                        2, propagator_index_2)

            # Product
            rat *= upstream_propagator
            # Link indices.
            rat = rat.Link([[parent_index, propagator_index_1], ])
            rat = rat.TensorIndexReplacement(parent_index, propagator_index_1)
            # Replace momenta with appropriate signs.
            rat = rat.KinematicReplacement({-parent_index: [[-1, -label]
                                            for label in downstream_labels]})
            rat = rat.KinematicReplacement({-propagator_index_1: [[1, -label]
                                            for label in downstream_labels]})
            rat = rat.KinematicReplacement({-propagator_index_2: [[-1, -label]
                                            for label in downstream_labels]})

        return rat, downstream_labels, label_counter


class DRing(object):
    """Put a ring on it!"""

    def __init__(self, diagrams):
        self.diagrams = diagrams

    def ComputeMRational(self, root_node, delta_head="ID_"):
        # Get first rational.
        rat = self.diagrams[0].GetMRational(root_node, delta_head)
        # Compute the rest and sum up.
        for i in tqdm(range(1, len(self.diagrams)),
                      desc="Computing Rationals"):
            rat += self.diagrams[i].GetMRational(root_node, delta_head)
        return rat

    def __add__(self, other):
        # Copy all diagrams and return sum list.
        return DRing([Diagram(dia) for dia in (self.diagrams
                                               + other.diagrams)])

    def __mul__(self, other):
        if len(self.diagrams) != 0 and len(other.diagrams) != 0:
            product_lists = ([pair[0]*pair[1] for pair in
                             list(itertools.product(self.diagrams,
                              other.diagrams))])
            return DRing(reduce(lambda x, y: x+y, product_lists))
        else:
            return DRing([])

    def __str__(self):
        string = ""
        for dia in self.diagrams:
            string += dia.__str__()+'\n\n'
        return string


class Isomorphism(object):
    def node_match_with_extlabel(self, node1, node2):
        """Match nodes, including ext_label attributes. [node1] and [node2]
        are node attribute dictionaries corresponding to the pair of nodes
        being matched."""
        # Compare solely on the basis of MRational content.
        return ((node1['mrational'] == node2['mrational'])
                and (node1['ext_label'] == node2['ext_label']))

    def node_match_without_extlabel(self, node1, node2):
        """Match nodes, excluding ext_label attributes. [node1] and [node2]
        are node attribute dictionaries corresponding to the pair of nodes
        being matched."""
        # Compare solely on the basis of MRational content.
        return node1['mrational'] == node2['mrational']

    def edge_match_ptype(self, edge1, edge2):
        """Match edges on the basis of particle type. [edge1] and [edge2]
        are edge attribute dictionaries corresponding to the pair of edges
        being matched."""
        return edge1['ptype'] == edge2['ptype']

    def BinInternalIsomorphism(self, dring):
        nx_graphs = [dia.G for dia in dring.diagrams]
        # Initialize and seed the internal isometry dictionary.
        internal_iso_dict = {}
        nx_graph = nx_graphs[0]
        internal_iso_dict.setdefault(nx_graph, 1)
        nx_graphs.remove(nx_graph)

        # Bin the remaining nxgraphs by isometry relative to a class
        # representative serving as the key for each class bucket in
        # internal_iso_dict.
        while len(nx_graphs) > 0:
            nx_graph = nx_graphs[0]
            class_exists = False
            for key in internal_iso_dict.keys():
                GM = isomorphism.GraphMatcher(
                                    key,
                                    nx_graph,
                                    edge_match=self.edge_match_ptype,
                                    node_match=self.node_match_with_extlabel)
                if GM.is_isomorphic():
                    internal_iso_dict[key] += 1
                    class_exists = True
                    break
            if not class_exists:
                internal_iso_dict.setdefault(nx_graph, 1)
            nx_graphs.remove(nx_graph)
        return internal_iso_dict

    def InternalIsomorphismReduce(self, diagrams):
        my_diagrams = list(diagrams)

        # Initialize and seed the isomorphism-reduced diagram list.
        new_diagrams = []
        dia = my_diagrams[0]
        new_diagrams.append(Diagram(dia))
        my_diagrams.remove(dia)

        # Bin the remaining nxgraphs by isometry relative to a class
        # representative serving as the key for each class bucket.
        while len(my_diagrams) > 0:
            dia = my_diagrams[0]
            class_exists = False
            for n, diagram in enumerate(new_diagrams):
                GM = isomorphism.GraphMatcher(
                                    dia.G,
                                    diagram.G,
                                    edge_match=self.edge_match_ptype,
                                    node_match=self.node_match_with_extlabel)
                if GM.is_isomorphic():
                    new_diagrams[n].count += dia.count
                    class_exists = True
                    break
            if not class_exists:
                new_diagrams.append(Diagram(dia))
            my_diagrams.remove(dia)
        return new_diagrams

    def IsoMapToExtMap(self, source_graph, target_graph, isomap):
        extmap = {}
        for source, target in isomap.items():
            source_extlabel = source_graph.nodes[source]['ext_label']
            target_extlabel = target_graph.nodes[target]['ext_label']
            if source_extlabel is None:
                assert target_extlabel is None
            if target_extlabel is None:
                assert source_extlabel is None
            if source_extlabel is not None:
                extmap[source_extlabel] = target_extlabel
        return extmap

    def IdentityExtMap(self, nx_graph):
        extmap = {}
        for node in nx_graph.nodes:
            extlabel = nx_graph.nodes[node]['ext_label']
            if extlabel is not None:
                extmap[extlabel] = extlabel
        return extmap

    """
    def ClassifyFullIsomorphism(self,iso_dict):
        #Initialize and seed the full isometry dictionary.
        my_iso_dict = {}
        for key,val in iso_dict.items():
            my_iso_dict[key] = int(val)
        full_iso_dict = {}
        nx_graph = my_iso_dict.keys()[0]
        count = my_iso_dict[nx_graph]
        idmap = self.IdentityExtMap(nx_graph)
        full_iso_dict.setdefault(nx_graph,(count,[idmap,]))
        del my_iso_dict[nx_graph]
        #Categorize the remaining nxgraphs (and their counts) by isometry
        #relative to a class representative serving as the key for each
        #class bucket in internal_iso_dict.
        while len(my_iso_dict)>0:
            nx_graph = my_iso_dict.keys()[0]
            count = my_iso_dict[nx_graph]
            class_exists=False
            for key in full_iso_dict.keys():
                GM = isomorphism.GraphMatcher(
                                key,
                                nx_graph,
                                edge_match=self.edge_match_ptype,
                                node_match=self.node_match_without_extlabel)
                if GM.is_isomorphic():
                    assert full_iso_dict[key][0]==my_iso_dict[nx_graph]
                    extmap = self.IsoMapToExtMap(key,nx_graph,GM.mapping)
                    full_iso_dict[key][1].append(extmap)
                    class_exists=True
                    break
            if not class_exists:
                full_iso_dict.setdefault(nx_graph,(count,[idmap,]))
            del my_iso_dict[nx_graph]
        return full_iso_dict
    """

    def ClassifyFullIsomorphism(self, diagrams):
        # Initialize and seed the full isometry dictionary.
        my_iso_dict = {}
        for diagram in diagrams:
            my_iso_dict[diagram.G] = int(diagram.count)
        full_iso_dict = {}
        nx_graph = list(my_iso_dict.keys())[0]
        count = my_iso_dict[nx_graph]
        idmap = self.IdentityExtMap(nx_graph)
        full_iso_dict.setdefault(nx_graph, (count, [idmap, ]))
        del my_iso_dict[nx_graph]
        # Categorize the remaining nxgraphs (and their counts) by isometry
        # relative to a class representative serving as the key for each
        # class bucket in diagrams.
        while len(my_iso_dict) > 0:
            nx_graph = list(my_iso_dict.keys())[0]
            count = my_iso_dict[nx_graph]
            class_exists = False
            for key in full_iso_dict.keys():
                GM = isomorphism.GraphMatcher(
                                key,
                                nx_graph,
                                edge_match=self.edge_match_ptype,
                                node_match=self.node_match_without_extlabel)
                if GM.is_isomorphic():
                    assert full_iso_dict[key][0] == my_iso_dict[nx_graph]
                    extmap = self.IsoMapToExtMap(key, nx_graph, GM.mapping)
                    full_iso_dict[key][1].append(extmap)
                    class_exists = True
                    break
            if not class_exists:
                full_iso_dict.setdefault(nx_graph, (count, [idmap, ]))
            del my_iso_dict[nx_graph]
        return full_iso_dict


def SuperGraphBFS(diagram, node):
    # FIXME: need a sensible docstring
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
    # 	if len(diagram.valent_nodes[ptype])>0:
    # 		valent_ptypes.append(ptype)
    # assert len(valent_ptypes)>0, "No valent ptypes!"
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
# 		for neighbors in target:
# 			for neighbor in neighbors:
# 				if len(next_diagram.G.nodes[neighbor]['unbound_opmap'])>0:
# 					if neighbor not in valent_neighbors:
# 						valent_neighbors.append(neighbor)
# 		if (len(valent_neighbors)==0
#                and len(next_diagram.valent_nodes.keys())>0):
#           print "TERMINATE"
# 			print "NODE: ",node
# 			print next_diagram.valent_nodes
# 			DrawNXGraphList([next_diagram.G,],waittime=0)
# 			continue
        if (len(next_diagram.valent_nodes.keys()) == 0
                and len(next_diagram.connected_valent_nodes) == 0):
            # print "LEAF"
            # DrawNXGraphList([next_diagram.G,],waittime=0)
            diagram_list += [next_diagram, ]
        elif (len(next_diagram.valent_nodes.keys()) != 0
                and len(next_diagram.connected_valent_nodes) == 0):
            # print "TERMINATE"
            # DrawNXGraphList([next_diagram.G,],waittime=0)
            continue
        else:
            # print "RECURSE"
            # DrawNXGraphList([next_diagram.G,],waittime=0)
            diagram_list += SuperGraphBFS(
                    next_diagram,
                    list(next_diagram.connected_valent_nodes.values())[0][0])
    return diagram_list


def ComputeContractions(vertices,
                        propagators,
                        external_legs,
                        tensor_symmetries):
    # FIXME: need a sensible docstring
    # Grab the basis object and identity polynomials from a propagator
    # MRational.
    basis = list(propagators.values())[0].basis
    # p_one = propagators.values()[0].nd_list[0][0].Mdict.values()[0].one
    p_one = list(propagators.values())[0].nd_list[0][0].PolyOne()
    # p_zero = propagators.values()[0].nd_list[0][0].Mdict.values()[0].zero
    p_zero = list(propagators.values())[0].nd_list[0][0].PolyZero()
    # Define the multiplicative identity in the MRational field.
    rat_one = MRational([[MRing({((0, 0),): p_one}),
                        {MRing({((0, 0),): p_one}): 1}], ], basis)
    # Define the additive identity in the MRational field.
    rat_zero = MRational([[MRing({((0, 0),): p_zero}),
                         {MRing({((0, 0),): p_one}): 1}], ], basis)

    # Generate internal vertex DRing list.
    internal_list = []
    label_counter = 1
    for multiplicity, oplist, r, name in vertices:
        G = nx.Graph()
        opmap = {}
        for ptype in oplist:
            opmap.setdefault(ptype, 0)
            opmap[ptype] += 1
        G.add_node(label_counter,
                   multiplicity=multiplicity,
                   oplist=oplist,
                   opmap=opmap,
                   unbound_opmap=dict(opmap),
                   ext_label=None,
                   mrational=MRational(r),
                   name=name)
        internal_list.append(G)
        label_counter += 1

    # Generate external vertex DRing list.
    external_list = []
    root_vertex = None
    ext_label = 1
    # FIXME: this is hardcoded for scalars, need to tweak to admit higher spins
    symbolblocks = []
    for ptype in external_legs:
        G = nx.Graph()
        G.add_node(ext_label,
                   multiplicity=1,
                   oplist=[ptype, ],
                   opmap={ptype: 1},
                   unbound_opmap={ptype: 1},
                   ext_label=ext_label,
                   mrational=rat_one,
                   name=None)
        external_list.append(G)
        # FIXME: spin1-specific!
        symbolblocks.append([-ext_label, ext_label])
        ext_label += 1
    # Use the largest ext_label as the root node label
    root_label = ext_label-1
    # Compute the list of all complete, connected contractions of internal
    # vertex DRings.

    seed_diagram = Diagram(internal_list[0], propagators)
    for i in range(1, len(internal_list)):
        seed_diagram += Diagram(internal_list[i], propagators)
    for i in range(0, len(external_list)):
        seed_diagram += Diagram(external_list[i], propagators)
    # for i in tqdm([1,],desc="Contractions"):
    diagrams = SuperGraphBFS(seed_diagram,
                             list(seed_diagram.G.nodes.keys())[0])

    if len(diagrams) == 0:
        # print "ZERORETURN"
        return rat_zero

    iso = Isomorphism()
    for i in tqdm([1, ], desc="Internal Isomorphism"):
        diagrams = iso.InternalIsomorphismReduce(diagrams)

    DR = DRing(diagrams)

    # Now, compute the list of all contractions of these internally
    # contracted vertices with the external vertices (these
    # represent all external leg permutations).
    # for D_ext in tqdm(external_list,desc="External Contractions"):
    # 	DR = D_ext*DR
    # Finally, compute the MRational corresponding to each of these
    # contractions and sum them.

    # If the internal contraction yields the wrong set of external legs,
    # return zero.

    # for i in tqdm([1,],desc="Internal Isomorphism"):
    # 	int_iso_dict = iso.BinInternalIsomorphism(DR)
    # print "INTISODICT"
    # for key,val in int_iso_dict.items():
    # 	DrawNXGraphList([key,],waittime=2)
    # 	print Diagram(key,propagators)
    # print "LEN INTISODICT: ",len(int_iso_dict)
    for i in tqdm([1, ], desc="Full Isomorphism"):
        ext_iso_dict = iso.ClassifyFullIsomorphism(DR.diagrams)

    contraction_sum = MRational(rat_zero)
    # for key,value in tqdm(ext_iso_dict.items(),desc="Computing MRationals"):
    for key, value in ext_iso_dict.items():
        DrawNXGraphList([key, ], waittime=2)
        key_diagram = Diagram(key, propagators)
        # print key_diagram
        for i in tqdm([1, ], desc="Construct MRational Expression"):
            key_rat = key_diagram.ComputeMRational(root_label)
        for i in tqdm([1, ], desc="Canonicalize Tensor Indices"):
            key_rat = key_rat.CanonicalizeIndices(len(external_legs))
        #for i in tqdm([1, ], desc="Sort Symmetric Tensor Indices"):
        key_rat = key_rat.SortSymmetricIndices(tensor_symmetries)
        key_rat = key_rat.Collect()
        coefficient = value[0]
        #for blockmap in tqdm(value[1], desc="BlockPermute"):
        for blockmap in value[1]:
            perm_key_rat = key_rat.BlockReplacement(blockmap, symbolblocks)
            contraction_sum += perm_key_rat*coefficient

    return contraction_sum


def UTaylorSeries(symbol_list, g_order):
    # Compute the taylor series of the time evolution operator U
    # to order [g_order] in the perturbative parameter g, as well
    # as a set of operator coefficients, all of which are contained
    # in [symbol_list].
    g = symbol_list[0]
    f = poly(0, symbol_list, domain='QQ_I')
    for i in range(1, g_order+1):
        f += I*symbol_list[i]*g**i
    U_series = series(exp(f.as_expr()), g, n=g_order+1)
    return U_series


def ComputeTreeAmplitude(interactions,
                         propagators,
                         external_legs,
                         tensor_symmetries):
    """ Given a lagrangian [interactions], [propagators], and [external_legs],
    compute the corresponding tree amplitude.
    [interactions] = [(g_order, MRational, N_operators, op_map),...]
    where g_order is the perturbative order of the interaction,
    MRational is the algebraic representation of the interaction,
    N_operators is the number operators comprising the interaction,
    and op_map is a dictionary mapping the particle type strings of the
    operators involved in the interaction to the number of times each particle
    type appears in the interaction.

    This function uses the canonical quantization approach of taylor expanding
    the time-evolution operator in powers of g, extracting the term
    corresponding to tree-level contribution (at the given number of external
    legs), stripping off the operator-valued coefficient of this g-monomial,
    and then computing all possible contractions of the various operators in
    this coefficient.
    """

    # Grab the basis object and identity polynomials from an interaction
    # MRational.
    basis = interactions[0][3].basis
    # p_one = interactions[0][3].nd_list[0][0].Mdict.values()[0].one
    p_one = interactions[0][3].nd_list[0][0].PolyOne()
    # p_zero = interactions[0][3].nd_list[0][0].Mdict.values()[0].zero
    p_zero = interactions[0][3].nd_list[0][0].PolyZero()
    # Define the additive identity in the MRational field.
    rat_zero = MRational([[MRing({((0, 0),): p_zero}),
                         {MRing({((0, 0),): p_one}): 1}], ], basis)

    # First, sort the interaction operators by the associated power of g,
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
    U_series = UTaylorSeries(symbol_list, g_order)

    # Extract the coefficient of the g^g_order term.
    tree_term = U_series.coeff(g**g_order).as_poly(domain='QQ_I')

    #print("TREETERM")
    #print(tree_term)

    # Now it's time to replace the O_i symbols with the corresponding operators
    # from g_ordered_interactions. We'll convert tree_term to sympy's
    # polynomial dictionary representation. tree_term is a polynomial in g
    # (g^0) and the various O_i.
    # We'll end up with a dictionary of the form (e.g.)
    # { ( 0,   1,   0,   1,   ...) : coefficient }
    #     g    O_1  O_2  O_3  ...
    # We can then easily replace the O_i with powers of the corresponding
    # sets of interactions.
    amplitude = rat_zero

    for monomial_tuple, coefficient in tree_term.as_dict().items():
        #print("________________________________TREETERM"
        #      + "_____________________________________")
        #print(monomial_tuple)
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

        # We take the set product of these interaction lists. The result is
        # a list of lists of interactions, each of which represents the list
        # of vertices appearing, after contractions, in various feynman
        # diagrams.

        vertex_lists = list(itertools.product(*interaction_list_factors))

        # FIXME: we can collect terms in these set product to save time.
        # That's an optimization for later.
        # Finally, we compute the sum of all contractions of each vertex list,
        # multiply by the coefficient given by tree_term, and sum them into
        # amplitude.

        # For the time being, we'll do the isomorphism reductions in
        # ComputeContractions.

        for vertices in vertex_lists:
            # print "----------__VERTEXSET__---------"
            # print vertices
            diagram = ComputeContractions(vertices,
                                          propagators,
                                          external_legs,
                                          tensor_symmetries)
            diagram *= coefficient
            amplitude += diagram
        #for i in tqdm([1, ], desc="SORTING"):
        amplitude = amplitude.SortSymmetricIndices(tensor_symmetries)
        #for i in tqdm([1, ], desc="AMPCOLLECT"):
        amplitude = amplitude.Collect()
        print("\n\n\n")

# 	print "DIAGRAM"
# 	print amplitude
# 	print "12swap-----------------------"
# 	symbolblocks = [[-1,],[-2,],[-3,],[-4,]]
# 	swapped = amplitude.BlockReplacement({1:2,2:1,3:3,4:4},symbolblocks)
# 	delta = swapped-amplitude
# 	delta = delta.OnShell()
# 	delta = delta.EjectMasses()
# 	delta = delta.ZeroMasses((symbols('m_4'),))
# 	delta = delta.GroupMasses({(symbols('m_1'),):symbols('m_{i1}')})
# 	delta = delta.GroupMasses({(symbols('m_2'),):symbols('m_{i2}')})
# 	delta = delta.GroupMasses({(symbols('m_3'),):symbols('m_{i3}')})
# 	delta = delta.Collect()
# 	print delta
# 	print "-------------------------------"
# 	print

    # And voila! An amplitude is born.

    return amplitude
