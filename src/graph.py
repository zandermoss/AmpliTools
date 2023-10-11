import itertools
import networkx as nx
from networkx.algorithms import isomorphism
from mrational import MRational
from mring import MRing
from sympy import *
from tqdm import tqdm
from interface import draw_graphs
from hashable_containers import hmap, hlist, HMultiGraph
from sympy.combinatorics import Permutation, PermutationGroup
from sympy.combinatorics.named_groups import SymmetricGroup
import nautypy as nty
import matplotlib.pyplot as plt


def graph_copy(graph):
    #G = nx.Graph()
    G = HMultiGraph()
    for node in graph.nodes:
        G.add_node(
                   node,
                   multiplicity=graph.nodes[node]['multiplicity'],
                   #oplist=graph.nodes[node]['oplist'],
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
              # and type(arg[0]) == type(nx.Graph())
              #and isinstance(arg[0], nx.Graph)
              and isinstance(arg[0], HMultiGraph)
              # and type(arg[1]) == type(dict())):
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
        """ collect labels of nodes with unbound operators
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
        """ Depth-first recursion for constructing
            an MRational expression from a diagram. """
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


class DRing(object):
    """Put a ring on it!"""

    def __init__(self, diagrams):
        self.diagrams = diagrams

    def compute_mrational(self, root_node, delta_head="ID_"):
        # Get first rational.
        rat = self.diagrams[0].get_mrational(root_node, delta_head)
        # Compute the rest and sum up.
        for i in tqdm(range(1, len(self.diagrams)),
                      desc="Computing Rationals"):
            rat += self.diagrams[i].get_mrational(root_node, delta_head)
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


# class Isomorphism(object):
#     def node_match_with_extlabel(self, node1, node2):
#         """Match nodes, including ext_label attributes. [node1] and [node2]
#         are node attribute dictionaries corresponding to the pair of nodes
#         being matched."""
#         # Compare solely on the basis of MRational content.
#         return ((node1['mrational'] == node2['mrational'])
#                 and (node1['ext_label'] == node2['ext_label']))
#
#     def node_match_without_extlabel(self, node1, node2):
#         """Match nodes, excluding ext_label attributes. [node1] and [node2]
#         are node attribute dictionaries corresponding to the pair of nodes
#         being matched."""
#         # Compare solely on the basis of MRational content.
#         return node1['mrational'] == node2['mrational']
#
#     def edge_match_ptype(self, edge1, edge2):
#         """Match edges on the basis of particle type. [edge1] and [edge2]
#         are edge attribute dictionaries corresponding to the pair of edges
#         being matched."""
#         return edge1['ptype'] == edge2['ptype']
#
#
#     def InternalIsomorphismReduce(self, diagrams):
#         my_diagrams = list(diagrams)
#
#         # Initialize and seed the isomorphism-reduced diagram list.
#         new_diagrams = []
#         dia = my_diagrams[0]
#         new_diagrams.append(Diagram(dia))
#         my_diagrams.remove(dia)
#
#         # Bin the remaining nxgraphs by isometry relative to a class
#         # representative serving as the key for each class bucket.
#         while len(my_diagrams) > 0:
#             dia = my_diagrams[0]
#             class_exists = False
#             for n, diagram in enumerate(new_diagrams):
#                 GM = isomorphism.GraphMatcher(
#                                     dia.G,
#                                     diagram.G,
#                                     edge_match=self.edge_match_ptype,
#                                     node_match=self.node_match_with_extlabel)
#                 if GM.is_isomorphic():
#                     new_diagrams[n].count += dia.count
#                     class_exists = True
#                     break
#             if not class_exists:
#                 new_diagrams.append(Diagram(dia))
#             my_diagrams.remove(dia)
#         return new_diagrams
#
#     def IsoMapToExtMap(self, source_graph, target_graph, isomap):
#         extmap = {}
#         for source, target in isomap.items():
#             source_extlabel = source_graph.nodes[source]['ext_label']
#             target_extlabel = target_graph.nodes[target]['ext_label']
#             if source_extlabel is None:
#                 assert target_extlabel is None
#             if target_extlabel is None:
#                 assert source_extlabel is None
#             if source_extlabel is not None:
#                 extmap[source_extlabel] = target_extlabel
#         return extmap
#
#     def IdentityExtMap(self, nx_graph):
#         extmap = {}
#         for node in nx_graph.nodes:
#             extlabel = nx_graph.nodes[node]['ext_label']
#             if extlabel is not None:
#                 extmap[extlabel] = extlabel
#         return extmap


    # def ClassifyFullIsomorphism(self, diagrams):
    #     # Initialize and seed the full isometry dictionary.
    #     my_iso_dict = {}
    #     for diagram in diagrams:
    #         my_iso_dict[diagram.G] = int(diagram.count)
    #     full_iso_dict = {}
    #     nx_graph = list(my_iso_dict.keys())[0]
    #     count = my_iso_dict[nx_graph]
    #     idmap = self.IdentityExtMap(nx_graph)
    #     full_iso_dict.setdefault(nx_graph, (count, [idmap, ]))
    #     del my_iso_dict[nx_graph]
    #
    #     # Categorize the remaining nxgraphs (and their counts) by isometry
    #     # relative to a class representative serving as the key for each
    #     # class bucket in diagrams.
    #     while len(my_iso_dict) > 0:
    #         nx_graph = list(my_iso_dict.keys())[0]
    #         count = my_iso_dict[nx_graph]
    #         class_exists = False
    #         for key in full_iso_dict.keys():
    #             GM = isomorphism.GraphMatcher(
    #                             key,
    #                             nx_graph,
    #                             edge_match=self.edge_match_ptype,
    #                             node_match=self.node_match_without_extlabel)
    #             if GM.is_isomorphic():
    #                 assert full_iso_dict[key][0] == my_iso_dict[nx_graph]
    #                 extmap = self.IsoMapToExtMap(key, nx_graph, GM.mapping)
    #                 full_iso_dict[key][1].append(extmap)
    #                 class_exists = True
    #                 break
    #         if not class_exists:
    #             full_iso_dict.setdefault(nx_graph, (count, [idmap, ]))
    #         del my_iso_dict[nx_graph]
    #     return full_iso_dict


def labels_to_ext(g):
    return hmap({l:g.nodes[l]['ext_label'] for l in g.nodes if g.nodes[l]['ext_label'] >= 0})

def isomap_to_ext(isomap,labels_to_ext):
    return hmap({labels_to_ext[l]:labels_to_ext[isomap[l]] for l in labels_to_ext})

def isomorphism_classification(diagrams):
    """
    For each diagram's graph (g), there exists a canonical labeling (g_canon)
    which is related to (g) by a node label permutation (s_g).
    The nodes of any such (g) are divided into *external leg* nodes, and *internal* nodes.
    Once the graph (g) is combined with the Feynman rules to compute an MRational
    term (rat(g)) contributing to the amplitude, all internal leg momenta are replaced
    by linear combinations of external leg momenta (for tree diagrams), and all internal
    flavor indices are summed over, so the only labels remaining are the external leg labels.
    The canonical permutation (s_g) thus acts on rat(g_canon) by a restricted permuation
    of the external leg labels (s_g_ext).

    We could compute rat(g) for every graph spat out by the supergraph_dfs algorithm, but 
    we would waste time doing unnecessary rational algebra. Instead, we will 'bin' the 
    graphs (g) from supergraph_dfs by their canonizations (g_canon) and the associated
    external label permutation (s_g_ext) into a nested hmap structure (classes). 

    To construct the amplitude, we step through the keys of (classes) and call
    Diagram.compute_mrational *only once* per isomorphism class. For each (s_g_ext)
    in classes[g_canon], we compute s_g_ext(rat(g_canon)), multiply it by the number of
    times the associated diagram appears (classes[g_canon][s_g_ext]), and add the
    result to the amplitude.

    The number of contractions grows rapidly with the number of external legs,
    but the number of isomorphism classes grows much more slowly. In the worst
    case, we work with an EFT containing an infinite tower of interaction operators
    of unbounded dimension. Wick contractions of such a tower should yield a set
    of labeled trees of cardinality O(n^n), where n is the number of external legs.
    The number of isomorphism classes, on the other hand, is roughly O(3^n)
    [doi:10.2307/1969046]. 

    An additional improvement to this approach would see canonization used
    as a pruning technique in the supergraph_dfs algorithm to reduce the number
    of contractions constructed by binning "online". Empirically, constructing
    and manipulating rational expressions appears to take much longer than 
    generating graphs from contractions, so we'll punt that optimization.

    isomorphism_classification constructs an hmap (classes) keyed by canonically
    labeled graphs and valued by hmaps, which themselves are keyed by external label
    permutations (s_g_ext) and valued by integer counts for the corresponding isomorph.
    """
    print("IN IC0")
    classes = hmap()
    nodelabels_to_extlabels = hmap()

    print("IN IC")

    for d in diagrams:
        #Call NautyPy to canonicalize d.G.
        #NautyPy returns g_canon, generators of its automorphisms,
        #and a node label permutation map from g_canon to d.G.

        #Remember that the external legs are labeled by g.nodes[label]['ext_label'],
        #not by 'label' itself, so we need to conjugate the permutation isomap_from_canon
        #by the 'label' to 'ext_label' permutation isomap_from_canon.
        #We will apply the conjugated map to the MRational associated to the canonical
        #diagram to construct its isomorphs.

        #Canonization
        # print(f'type(d.G): {type(d.G)}')
        # print(repr(d.G))
        # print(f'd.G.nodes: {d.G.nodes}')
        # for node in d.G.nodes:
        #     print('============================================================')
        #     print(f'type(node): {type(node)}')
        #     print(f'node: {node}')
        #
        #     print(f'type(multiplicity): {type(d.G.nodes[node]["multiplicity"])}')
        #     print(f'multiplicity: {d.G.nodes[node]["multiplicity"]}')
        #
        #     print(f'type(oplist): {type(d.G.nodes[node]["oplist"])}')
        #     print(f'oplist: {d.G.nodes[node]["oplist"]}')
        #
        #     print(f'type(opmap): {type(d.G.nodes[node]["opmap"])}')
        #     print(f'opmap: {d.G.nodes[node]["opmap"]}')
        #
        #     print(f'type(unbound_opmap): {type(d.G.nodes[node]["unbound_opmap"])}')
        #     print(f'unbound_opmap: {d.G.nodes[node]["unbound_opmap"]}')
        #
        #     print(f'type(ext_label): {type(d.G.nodes[node]["ext_label"])}')
        #     print(f'ext_label: {d.G.nodes[node]["ext_label"]}')
        #
        #     print(f'type(mrational): {type(d.G.nodes[node]["mrational"])}')
        #     print(f'mrational: {d.G.nodes[node]["mrational"]}')
        #
        #     print(f'type(name): {type(d.G.nodes[node]["name"])}')
        #     print(f'name: {d.G.nodes[node]["name"]}')
       

        # Cache node label -> external label mapping.
        #nodes_to_ext = labels_to_ext(d.G)

       
        nty.gprint(d.G)
        ext_nodes = [node for node in d.G.nodes if d.G.nodes[node]['ext_label']>=0]
        n_ext = len(ext_nodes)
        int_nodes = [node for node in d.G.nodes if d.G.nodes[node]['ext_label']<0]
        index_to_node = {i:node for i,node in enumerate(sorted(ext_nodes))}
        index_to_node.update({(i+n_ext):node for i,node in enumerate(sorted(int_nodes))})
        node_to_index = {val:key for key,val in index_to_node.items()}
        print(f"index_to_node:    {index_to_node}")

        # Homogenize external vertex labels.
        g = HMultiGraph(nx.relabel_nodes(d.G,node_to_index,copy=True))
        node_to_ext = {node:g.nodes[node]['ext_label'] for node in range(n_ext)}
        ext_to_node = {ext:node for node,ext in node_to_ext.items()}
        for node in g.nodes:
            if g.nodes[node]['ext_label'] >= 0:
                g.nodes[node]['ext_label'] = 1
        print(f"node_to_ext:    {node_to_ext}")

        # # Canonize graph.
        # # Don't mix external and internal node labels
        # hostgraphs = dict()
        g_canonical, autgens, canonical_map = nty.canonize_multigraph(g,
                                                color_sort_conditions=[('ext_label',1)])
                                                # hostgraphs = hostgraphs)
        # g_canonical, autgens, canonical_map = nty.canonize_multigraph(g,
        #                                         color_sort_conditions=[])
        # mg_perm_host = hostgraphs['host']
        # mg_perm_host_canonical = hostgraphs['host_canonical']
        # #Draw graphs.
        # def draw_diagram(_g,title=''):
        #     #g = _g.copy()
        #     g = nx.relabel_nodes(_g,{node: f"{node}:{_g.nodes[node].get('ext_label','')}" for node in _g.nodes}, copy=True)
        #
        #     g.graph['node']={'shape':'circle'}
        #     for node in g.nodes:
        #         if g.nodes[node].get("ext", False) == True:
        #             g.nodes[node]["shape"] = "triangle"
        #         if g.nodes[node].get("type", "vertex") == "edge":
        #             g.nodes[node]["shape"] = "square"
        #     nty.gdraw(g,title=title)
        #     return


        # Restore external vertex labels.
        for node in g_canonical.nodes:
            if g_canonical.nodes[node]['ext_label'] >= 0:
                g_canonical.nodes[node]['ext_label'] = node_to_ext[node]
    
        # Mod out auts
        Gp = SymmetricGroup(n_ext)
        restr_auts = [Permutation([aut[node] for node in range(n_ext)]) for aut in autgens]
        Aut = PermutationGroup(restr_auts)
        restr_canmap = [canonical_map[node] for node in range(n_ext)]
        canonical_perm = Permutation(restr_canmap)
        canonical_perm_mod_aut = ~(Gp._coset_representative(~canonical_perm,Aut))


        # Convert canonical node map to external label map.
        ext_canonical_map = hmap({ext:node_to_ext[canonical_perm(node)] for ext,node in ext_to_node.items()})
        ext_canonical_map_mod_aut = hmap({ext:node_to_ext[canonical_perm_mod_aut(node)] for ext,node in ext_to_node.items()})

        print("nty_autgens")
        print(autgens)
        print("Aut")
        print(Aut)
        print()
        print(f"ext_canonical_map:            {ext_canonical_map}")
        print(f"ext_canonical_map_mod_aut:    {ext_canonical_map_mod_aut}")
        #nty.gprint(g)

        # ax1 = plt.subplot(221)
        # draw_diagram(g,title="MultiGraph")
        #
        # ax2 = plt.subplot(222)
        # draw_diagram(mg_perm_host,title="Host Graph")
        #
        # ax3 = plt.subplot(223)
        # draw_diagram(mg_perm_host_canonical,title="Canonical Host Graph")
        #
        # ax4 = plt.subplot(224)
        # draw_diagram(g_canonical,title="Canonical MultiGraph")
        #
        # plt.show()

        # ax1 = plt.subplot(211)
        # draw_diagram(hostgraphs['mg_z'],title="mg_z")
        #
        # ax2 = plt.subplot(212)
        # draw_diagram(hostgraphs['g_z'],title="g_z")
        # plt.show()
    
        # print("canonical")
        # nty.gprint(g_canonical)
        # print("canonical")
        # print("nty_autgens")
        # print(autgens)
        # print("Aut")
        # print(Aut)
        # draw_graphs([g_canonical,])
        #
        # print("rawgraph")
        # nty.gprint(g)
        # print("rawgraph")
        # print("nty_autgens")
        # print(autgens)
        # print("Aut")
        # print(Aut)
        # print()
        # print(f"ext_canonical_map:            {ext_canonical_map}")
        # print(f"ext_canonical_map_mod_aut:    {ext_canonical_map_mod_aut}")
        # Restore external vertex labels.
        # for node in g.nodes:
        #     if g.nodes[node]['ext_label'] >= 0:
        #         g.nodes[node]['ext_label'] = node_to_ext[node]
        # draw_graphs([g,])


        #Initialize and increment the entry for [g_canonical][ext_canonical_map] in classes.
        classes.setdefault(g_canonical,hmap())
        classes[g_canonical].setdefault(ext_canonical_map_mod_aut,0)
        classes[g_canonical][ext_canonical_map_mod_aut] += 1

    return classes


def supergraph_dfs(diagram, node):
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

        # Leaf
        if (len(next_diagram.valent_nodes.keys()) == 0
                and len(next_diagram.connected_valent_nodes) == 0):
            diagram_list += [next_diagram, ]

        # Terminate
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
    print("IN CC")
    # FIXME: need a sensible docstring
    # Grab the identity polynomials from a propagator
    # MRational.
    # p_one = propagators.values()[0].nd_list[0][0].mdict.values()[0].one
    p_one = hlist(propagators.values())[0].nd_list[0][0].poly_one()
    # p_zero = propagators.values()[0].nd_list[0][0].mdict.values()[0].zero
    p_zero = hlist(propagators.values())[0].nd_list[0][0].poly_zero()
    # Define the multiplicative identity in the MRational field.
    rat_one = MRational(hlist([hlist([MRing(hmap({((0, 0),): p_one})),
                        hmap({MRing(hmap({((0, 0),): p_one})): 1})]), ]))
    # Define the additive identity in the MRational field.
    rat_zero = MRational(hlist([hlist([MRing(hmap({((0, 0),): p_zero})),
                         hmap({MRing(hmap({((0, 0),): p_one})): 1})]), ]))

    # Generate internal vertex DRing list.
    internal_list = []
    label_counter = 1
    for multiplicity, oplist, r, name in vertices:
        #G = nx.Graph()
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
    print("Done int DRING")
    print("===============VERTICES===================")
    for G in internal_list:
        for node in G.nodes:
            print(f'node: {node}    att: {G.nodes[node]}')
        for edge in G.edges:
            print(f'edge: {edge}    att: {G.edges[edge]}')
        print()
        #nty.gdraw(G)
        #draw_graphs([G, ], waittime=2)
    print("======================================")
    # Generate external vertex DRing list.
    external_list = []
    root_vertex = None
    ext_label = 1
    symbolblocks = []
    for ptype in external_legs:
        #G = nx.Graph()
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
    # Use the largest ext_label as the root node label
    #root_label = ext_label-1
    print("===============EXTVERTICES===================")
    for G in external_list:
        for node in G.nodes:
            print(f'node: {node}    att: {G.nodes[node]}')
        for edge in G.edges:
            print(f'edge: {edge}    att: {G.edges[edge]}')
        print()
        #nty.gdraw(G)
        #draw_graphs([G, ], waittime=2)
    print("======================================")

    print("Done ext DRING")

    # Compute the list of all complete, connected contractions of internal
    # vertex DRing elements.
    seed_diagram = Diagram(internal_list[0], propagators)
    for i in range(1, len(internal_list)):
        seed_diagram += Diagram(internal_list[i], propagators)
    for i in range(0, len(external_list)):
        seed_diagram += Diagram(external_list[i], propagators)
    #for i in tqdm([1,],desc="Contractions"):
    print("Before SGDFS")
    diagrams = supergraph_dfs(seed_diagram,
                    list(seed_diagram.G.nodes.keys())[0])
    print("After SGDFS")

    if len(diagrams) == 0:
        # print "ZERORETURN"
        return rat_zero
    # print("======================================")
    # for d in diagrams:
    #     draw_graphs([d.G, ], waittime=2)
    # print("======================================")

    # iso = Isomorphism()
    # for i in tqdm([1, ], desc="Internal Isomorphism"):
    #     diagrams = iso.InternalIsomorphismReduce(diagrams)
    #
    # DR = DRing(diagrams)

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
    # 	draw_graphs([key,],waittime=2)
    # 	print Diagram(key,propagators)
    # print "LEN INTISODICT: ",len(int_iso_dict)
    #for i in tqdm([1, ], desc="Isomorphism Classification"):
    print ("Before IC")
    classes = isomorphism_classification(diagrams)
    print ("After IC")
    #    classes = ClassifyFullIsomorphism(diagrams)

    #   ext_iso_dict = iso.ClassifyFullIsomorphism(DR.diagrams)

    contraction_sum = MRational(rat_zero)
    # for key,value in tqdm(ext_iso_dict.items(),desc="Computing MRationals"):

    print("======================================")
    for g_canon, isomaps in classes.items():
        draw_graphs([g_canon, ], waittime=2)
        print(f"isomaps:   {isomaps}")
    print("======================================")
    for g_canon, isomaps in classes.items():
        #draw_graphs([g_canon, ], waittime=2)
        d_canon = Diagram(g_canon, propagators)
        # print key_diagram
        root_label = 0
        for i in tqdm([1, ], desc="Construct MRational Expression"):
            rat_canon = d_canon.compute_mrational(root_label)
        for i in tqdm([1, ], desc="Canonicalize Tensor Indices"):
            rat_canon = rat_canon.canonize_indices(len(external_legs))
        for i in tqdm([1, ], desc="Sort Symmetric Tensor Indices"):
            rat_canon = rat_canon.sort_indices(tensor_symmetries)
        for i in tqdm([1, ], desc="collect Terms"):
            rat_canon = rat_canon.collect()

        #coefficient = value[0]
        #for blockmap in tqdm(value[1], desc="External Label Permutations"):
        # for blockmap in value[1]:
        #     perm_key_rat = key_rat.block_replacement(blockmap, symbolblocks)
        #     contraction_sum += perm_key_rat*coefficient
        for isomap,count in isomaps.items():
            rat_isomorph = rat_canon.block_replacement(isomap, symbolblocks)
            contraction_sum += rat_isomorph*count

    return contraction_sum


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
    print("IN CTA")
    # Grab the identity polynomials from an interaction
    # MRational.
    # p_one = interactions[0][3].nd_list[0][0].mdict.values()[0].one
    p_one = interactions[0][3].nd_list[0][0].poly_one()
    # p_zero = interactions[0][3].nd_list[0][0].mdict.values()[0].zero
    p_zero = interactions[0][3].nd_list[0][0].poly_zero()
    # Define the additive identity in the MRational field.
    rat_zero = MRational(hlist([hlist([MRing(hmap({((0, 0),): p_zero})),
                         hmap({MRing(hmap({((0, 0),): p_one})): 1})]), ]))

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
    U_series = evolution_expansion(symbol_list, g_order)
    print("USERIES")
    print(U_series)


    # Extract the coefficient of the g^g_order term.
    tree_term = U_series.coeff(g**g_order).as_poly(domain='QQ_I')
    print("TREETERM")
    print(tree_term)

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

        print(f"monomial_tuple: {monomial_tuple}    coefficient: {coefficient}")
        print(f"g_ordered_interactions:    {g_ordered_interactions}")
        print(f"interaction_list_factors:    {interaction_list_factors}")
        print(f"vertex_lists: {vertex_lists}")


        # FIXME: we can collect terms in these set product to save time.
        # That's an optimization for later.
        # Finally, we compute the sum of all contractions of each vertex list,
        # multiply by the coefficient given by tree_term, and sum them into
        # amplitude.

        # For the time being, we'll do the isomorphism reductions in
        # compute_contractions.
        print("BEFORE compute_contractions")
        for vertices in vertex_lists:
            # print "----------__VERTEXSET__---------"
            # print vertices
            diagram = compute_contractions(vertices,
                                          propagators,
                                          external_legs,
                                          tensor_symmetries)

            diagram *= coefficient
            amplitude += diagram
        print("done compute_contractions")
        #for i in tqdm([1, ], desc="SORTING"):
        amplitude = amplitude.sort_indices(tensor_symmetries)
        #for i in tqdm([1, ], desc="AMPCOLLECT"):
        amplitude = amplitude.collect()
        print("\n\n\n")

# 	print "DIAGRAM"
# 	print amplitude
# 	print "12swap-----------------------"
# 	symbolblocks = [[-1,],[-2,],[-3,],[-4,]]
# 	swapped = amplitude.block_replacement({1:2,2:1,3:3,4:4},symbolblocks)
# 	delta = swapped-amplitude
# 	delta = delta.OnShell()
# 	delta = delta.EjectMasses()
# 	delta = delta.zero_masses((symbols('m_4'),))
# 	delta = delta.GroupMasses({(symbols('m_1'),):symbols('m_{i1}')})
# 	delta = delta.GroupMasses({(symbols('m_2'),):symbols('m_{i2}')})
# 	delta = delta.GroupMasses({(symbols('m_3'),):symbols('m_{i3}')})
# 	delta = delta.collect()
# 	print delta
# 	print "-------------------------------"
# 	print

    # And voila! An amplitude is born.

    return amplitude
