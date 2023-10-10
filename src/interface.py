from mring import MRing
from mrational import MRational
from sympy import *
import networkx as nx
import matplotlib.pyplot as plt
from IPython.display import Markdown, display, clear_output, Math
from functools import cmp_to_key
import time
import math
from permutation_tools import symmetric_partition_permutations
from signed_permutations import SignedPermutationGroup, SignedPermutation
import re


class Interface(object):
    def __init__(self,momentum_symbols,polarization_symbols,coefficient_symbols,
                 coefficient_display_map):
        #Sort symbols
        self.coefficient_display_map = coefficient_display_map
        self.momentum_symbols = tuple(sorted(momentum_symbols,
                                             key=cmp_to_key(lambda x,y:x.compare(y))))
        self.polarization_symbols = tuple(sorted(polarization_symbols,
                                                 key=cmp_to_key(lambda x,y:x.compare(y))))
        self.coefficient_symbols = tuple(sorted(coefficient_symbols,
                                                key=cmp_to_key(lambda x,y:x.compare(y))))

    def index_gt_zero(self,mylist):
        for n in range(len(mylist)):
            if mylist[n]>0:
                return n


    def expr_to_mring(self,expr):
        #HACK!!!
        if (type(expr)==type(1) or str(expr)=="1" or str(expr)=="0" or str(expr)=="-1"):
            mypoly = Poly(expr,symbols('q'),domain="QQ_I")
        else:
            mypoly = Poly(expr,domain="QQ_I")
        generators = mypoly.gens
        momentum_symbols = list(filter(lambda s: re.search("^p",str(s)) != None,generators))
        polarization_symbols = list(filter(lambda s: re.search("^e",str(s)) != None,generators))
        coefficient_symbols = list((set(generators) - set(momentum_symbols)) - set(polarization_symbols))
        generators = momentum_symbols + polarization_symbols + coefficient_symbols
        mypoly = Poly(expr,generators,domain="QQ_I")
        poly_dict = mypoly.as_dict()
        r = MRing({})
        for key in poly_dict.keys():
            #print("Key: {}".format(key))
            momentum_polarization_key = list(key[0:len(momentum_symbols+polarization_symbols)])
            momentum_polarization_order = sum(momentum_polarization_key)
            #print("MP_Key: {}    MP_Order: {}\n".format(momentum_polarization_key,momentum_polarization_order))
            assert (momentum_polarization_order==0 or momentum_polarization_order==2)
            mp_pair = [0,0]
            if momentum_polarization_order == 2:
                for i in range(2):
                    index_gtzero = self.index_gt_zero(momentum_polarization_key)
                    if index_gtzero>=len(momentum_symbols):
                        mp_pair[i] = (index_gtzero-len(momentum_symbols))+1
                    else:
                        mp_pair[i] = -(index_gtzero+1)
                    momentum_polarization_key[index_gtzero]-=1
            mp_pair = tuple(mp_pair)
            coefficient_key = key[len(momentum_symbols+polarization_symbols):]
            #coefficient_poly = Poly({coefficient_key:poly_dict[key]},coefficient_symbols+[symbols('x'),],domain="QQ_I")
            if len(coefficient_key)==0:
                coefficient_poly = Poly(poly_dict[key],symbols('q'),domain="QQ_I")
            else:
                    coefficient_poly = Poly({coefficient_key:poly_dict[key]},coefficient_symbols,domain="QQ_I")
            r += MRing({(mp_pair,):coefficient_poly})
        return r

    def expr_to_mrational(self,num,den):
        mrat = MRational([[num,den],])
        return mrat.Collect()

    def prettyprint_mring(self,obj):
        sorted_keys = sorted(obj.Mdict.keys())
        string = ''
        for key in sorted_keys:
            pairstring = ''
            for pair in key:
                if pair[0]==0:
                    pairstring = ''
                    #pairstring = 'R'
                elif pair[0]<0 and pair[1]>0:
                    pairstring+='(p_{} \cdot e_{})'.format(abs(pair[0]),abs(pair[1]))
                elif pair[0]<0 and pair[1]<0:
                    pairstring+='(p_{} \cdot p_{})'.format(abs(pair[1]),abs(pair[0]))
                elif pair[0]>0 and pair[1]>0:
                    pairstring+='(e_{} \cdot e_{})'.format(abs(pair[0]),abs(pair[1]))
                else:
                    pass

            if obj.Mdict[key].length()==1:
                if obj.Mdict[key].as_expr().__str__()=="1":
                    polystring = ''
                else:
                    polystring = obj.Mdict[key].as_expr().__str__()
            else:
                polystring = '('+obj.Mdict[key].as_expr().__str__()+')'
            polystring = polystring.replace("**","^")
            #Format indices properly
            syms = list(obj.Mdict[key].free_symbols)
            symstrings = [sym.__str__() for sym in syms]
            fancystrings = []
            for mystring in symstrings:
                if '{' in mystring:
                    indexblock = list(mystring.split('{')[1].split('}')[0])
                    prefix = mystring.split('{')[0]
                    newstring = prefix+'{'

                    indexstring=""
                    lastprefix=indexblock.pop(0)
                    while True:
                        char = indexblock.pop(0)
                        #if (char in ['f','b','x','y','z','w','a','c','d','g','k','h']):
                        if (char.isalpha()):
                            index = int(indexstring)
                            newstring+=lastprefix+'_{'+str(index)+'}'
                            lastprefix=char
                            indexstring=""
                        elif len(indexblock)==0:
                            indexstring+=char
                            index = int(indexstring)
                            newstring+=lastprefix+'_{'+str(index)+'}'
                            break
                        else:
                            indexstring+=char
                    fancystrings.append(newstring+'}')
                else:
                    fancystrings.append(mystring)
            for symstring,fstring in zip(symstrings,fancystrings):
                polystring = polystring.replace(symstring,fstring)

            string += polystring+pairstring+"+"
        string=string.strip("+")
        return string

    def display(self,obj):
        dummy = symbols('dummy')
        if type(obj)==type(poly(dummy,dummy,domain='QQ_I')):
            string = (obj.as_expr()).__str__()
            string = string.replace("**","^")
            #for source,target in zip(self.coefficient_display_map.keys(),self.coefficient_display_map.values()):
            #    bracketed_target = '{'+target+'}'
            #    string = string.replace(source,bracketed_target)
            display(Math('$$'+string+'$$'))
        elif obj.mathtype=="MRing":
            display(Math('$$'+self.prettyprint_mring(obj)+'$$'))
        elif obj.mathtype=="MRational":
            fracstring = ''
            for pair in obj.nd_list:
                nstring = self.prettyprint_mring(pair[0])
                #display(Math('$$'+self.prettyprint_mring(pair[0])+'$$'))
                dstring = ''
                for key in pair[1].keys():
                    if str(pair[1][key])!="1":
                        dstring+='['+self.prettyprint_mring(key)+']'
                        dstring+='^'+str(pair[1][key])
                    else:
                        dstring += self.prettyprint_mring(key)
                #string+='\n\n'
                nstring=nstring.replace("I","i")
                nstring=nstring.replace("*","")
                dstring=dstring.replace("I","i")
                dstring=dstring.replace("*","")
                if nstring=='':
                    nstring = '1'
                if dstring=='':
                    fracstring += nstring
                    #display(Math(nstring))
                else:
                    fracstring += '\\frac{'+nstring+'}{'+dstring+'}'
                fracstring += '+'
            fracstring = fracstring.strip('+')
            display(Math(fracstring))
        else:
            assert False


def draw_graphs(nx_graph_list,waittime=0.2,ext=True):
    #First, determine the number of external legs and
    #Compute an appropriate circular spring layout for them.
    mygraph = nx_graph_list[0]
    if ext:
        ext_labels = []
        for node in mygraph.nodes:
            mylabel = mygraph.nodes[node]['ext_label']
            if mylabel!=None:
                ext_labels.append(mylabel)
        n_ext = len(ext_labels)
        slice_angle = 2.0*math.pi/float(n_ext)
        posmap = {}
        for i in range(n_ext):
            angle = i*slice_angle+3.0*math.pi/4.0
            x = 0.5*(1+math.sqrt(2.0)*math.cos(angle))
            y = 0.5*(1+math.sqrt(2.0)*math.sin(angle))
            posmap[i+1]=(x,y)
    #Now we can plot!
    fig = plt.figure(dpi=100)
    a = fig.add_subplot()
    a.set_facecolor('white')
    for nx_graph in nx_graph_list:
        if ext:
            labeldict = {}
            spring_pos_dict = {}
            spring_fixed = []
            for node in nx_graph.nodes:
                label = nx_graph.nodes[node]['ext_label']
                if label==None:
                    labeldict[node]=nx_graph.nodes[node]['name']
                    #labeldict[node]=node
                    spring_pos_dict[node] = (0,0)
                else:
                    labeldict[node]=label
                    #labeldict[node]=node
                    spring_pos_dict[node] = posmap[label]
                    spring_fixed.append(node)
        else:
            labeldict = {}
            for node in nx_graph.nodes:
                label = nx_graph.nodes[node]['name']
                if label==None:
                    labeldict[node]=''
                else:
                    labeldict[node]=label
        #if ext:
            #pos_dict = nx.spring_layout(nx_graph,pos=spring_pos_dict,fixed=spring_fixed,iterations=100)
            #pos_dict = nx.circular_layout(nx_graph)
        #else:
            #pos_dict = nx.spring_layout(nx_graph)
            #pos_dict = nx.circular_layout(nx_graph)

        #pos_dict = nx.nx_pydot.graphviz_layout(nx.to_agraph(nx_graph), prog="dot")
#        pos_dict = nx.planar_layout(nx_graph)
        pos_dict = nx.nx_agraph.graphviz_layout(nx_graph,prog='dot')

        a.cla()
        epi = [(u, v) for (u, v, d) in nx_graph.edges(data=True) if d['ptype'] == 'pi']
        ehiggs = [(u, v) for (u, v, d) in nx_graph.edges(data=True) if d['ptype'] == 's']
        evector = [(u, v) for (u, v, d) in nx_graph.edges(data=True) if d['ptype'] == 'Z']
        ephi = [(u, v) for (u, v, d) in nx_graph.edges(data=True) if d['ptype'] == 'phi']
        if ext:
            #nx.draw_networkx_nodes(nx_graph,ax=a,labels=labeldict,font_weight='bold',pos=pos_dict)
            #nx.draw_networkx_nodes(nx_graph,ax=a,pos=pos_dict,node_color='#FF6C0C')
            nx.draw_networkx_nodes(nx_graph,ax=a,pos = pos_dict,node_color='#FF6C0C')
        else:
            #nx.draw_networkx_nodes(nx_graph,ax=a,font_weight='bold',pos=pos_dict)
            nx.draw_networkx_nodes(nx_graph,ax=a,pos = pos_dict,node_color='#FF6C0C')
        nx.draw_networkx_edges(nx_graph,ax=a,pos=pos_dict,edgelist=epi)
        nx.draw_networkx_edges(nx_graph,ax=a,pos=pos_dict,edgelist=ephi)
        nx.draw_networkx_edges(nx_graph,ax=a,pos=pos_dict,edgelist=ehiggs,style='dashed')
        nx.draw_networkx_edges(nx_graph,ax=a,pos=pos_dict,edgelist=evector,style='dotted')
        nx.draw_networkx_labels(nx_graph, labels=labeldict,pos=pos_dict, font_family='sans-serif')
        plt.axis('off')
        plt.show()
        #display(fig)
        #clear_output(wait=True)
        time.sleep(waittime)

class FeynmanRules():
    def __init__(self,tensor_symmetries,io):
        self.tensor_symmetries = tensor_symmetries
        self.io = io
        self.interactions = []
        self.propagators = {}

    def register_interaction(self,expr,fields,max_spin,perturbative_order,name):
        assert (max_spin==0 or max_spin==1), "Higher spin not yet handled!"
        #print("EXPR")
        #print(expr)
        #print("______________________________________")
        numerator = expr
        denominator = {self.io.expr_to_mring(1):1}
        #print("VERTEX")
        pspace_vertex = self.io.expr_to_mrational(numerator,denominator)
        #print(pspace_vertex)
        #print("______________________________________")
        fieldblocks = {}
        for i,field in enumerate(fields):
            fieldblocks.setdefault(field,[])
            fieldblocks[field].append(i+1)
        partitions = [tuple(position) for position in fieldblocks.values()]
        perms = symmetric_partition_permutations(partitions)
        if max_spin==0:
            symbolblocks = [[-i,] for i in range(1,len(fields)+1)]
        elif max_spin==1:
            symbolblocks = [[-i,i] for i in range(1,len(fields)+1)]
        #print("ORBIT")
        feynrule = pspace_vertex.Orbit(perms,symbolblocks)
        #print(feynrule)
        #print("______________________________________")
        #print("SORT")
        feynrule = feynrule.SortSymmetricIndices(self.tensor_symmetries)
        #print(feynrule)
        #print("______________________________________")
        self.interactions.append([perturbative_order,len(fields),fields,feynrule,name])

    def register_propagator(self,expr,particle_id):
        self.propagators[particle_id]=expr


class TensorSymmetries():
    def __init__(self):
        self.tensor_symmetries = {}

    def register_symmetry(self,head,generator_sign_pairs):
        signed_perms = [SignedPermutation(*pair) for pair in generator_sign_pairs]
        self.tensor_symmetries[head] = SignedPermutationGroup(signed_perms)

    def __str__(self):
        string = ""
        for head,group in self.tensor_symmetries.items():
            string+=head+"\n"
            string+=group.__str__()
            string+= "\n\n"
        return string
