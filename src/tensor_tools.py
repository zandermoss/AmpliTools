from copy import deepcopy
import sys
from sympy import symbols, poly, Rational, simplify, factor, Poly
from sympy.polys.polytools import div,compose
from functools import reduce
from tqdm import tqdm
import sympy

def split_tensor_symbol(symbol):
    symstring = symbol.__str__()
    assert "_{" in symstring, "Symbol does not represent a tensor!"
    indexblock = list(symstring.split('{')[1].split('}')[0])
    head = symstring.split('{')[0]
    prefixlist = []
    indexlist = []
    indexstring=""
    lastprefix=indexblock.pop(0)
    while True:
        char = indexblock.pop(0)
        if (char.isalpha()):
            prefixlist.append(lastprefix)
            indexlist.append(int(indexstring))
            lastprefix=char
            indexstring=""
        elif len(indexblock)==0:
            indexstring+=char
            prefixlist.append(lastprefix)
            indexlist.append(int(indexstring))
            break
        else:
            indexstring+=char
    return head,prefixlist,indexlist

def join_tensor_symbol(head,indexblock):
    symstring=head+"{"+reduce(lambda x,y:x+y, [str(a)+str(b) for a,b in indexblock])+"}"
    return symbols(symstring)

def symbol_group_sort(symbol,symgroup):
    head,prefixlist,indexlist = split_tensor_symbol(symbol)
    sorted_indexblock, sorting_permutations = symgroup.group_sort(tuple(zip(prefixlist,indexlist)))
    sorted_symbol = join_tensor_symbol(head,sorted_indexblock)

    if type(sorting_permutations)==type([]):
        first_sign = sorting_permutations[0].sign
        samesign = all([perm.sign==first_sign for perm in sorting_permutations])
        if samesign:
            return sorted_symbol,first_sign
        else:
            return 0,1
    else:
        return sorted_symbol,sorting_permutations.sign

def tensor_index_replacement(q,source,target):
    symbolmap = {}
    for symbol in list(q.free_symbols):
        if "_{" not in symbol.__str__():
            continue
        head,prefixlist,indexlist = split_tensor_symbol(symbol)
        pairs = zip(prefixlist,indexlist)
        if source in pairs:
            addresses = [i for i,x in enumerate(pairs) if x==source]
            for a in addresses:
                pairs[a] = target
        symbolmap[symbol] = join_tensor_symbol(head,pairs)
    for source,target in symbolmap.items():
        q = q.subs(source,target)
    return q

def index_match(pattern, index, match):
    if match=='prefix':
        return pattern[0]==index[0]
    elif match=='suffix':
        return pattern[1]==index[1]
    elif match=='index':
        return pattern==index
    else:
        assert False, "[match] argument must be one of prefix,suffix,index"

def first_matching_index(poly,pattern,match):
    #Need to trust that the expansion has been done properly in bound expansion.
    #For free expansion, all terms must have the same free indices, so
    #we only need to look at free_symbols for the whole poly.
    #assert len(poly.as_dict())==1, "Not a single term!"
    tensor_symbols = set(filter(lambda s: "_{" in s.__str__(),poly.free_symbols))
    for tensym in tensor_symbols:
        heads,prefixes,suffixes = split_tensor_symbol(tensym)
        for index in zip(prefixes,suffixes):
            if index_match(pattern,index,match):
                return index
    return False

def matching_indices(poly,pattern,match):
    assert len(poly.as_dict())==1, "Not a single term!"
    tensor_symbols = set(filter(lambda s: "_{" in s.__str__(),term.free_symbols))
    matching_indices =[]
    for tensym in tensor_symbols:
        head,prefix,suffix = split_tensor_symbol(tensym)
        index = (prefix,suffix)
        if index_match(pattern,index,match):
            matching_indices.append(index)
    return matching_indices

def target_index(source_index,target,match):
    if match=='prefix':
        if type(target)==type(str()):
            target_index = (target,source_index[1])
        else:
            target_index = target
    elif match=='suffix':
        if type(target)==type(int()):
            target_index = (source_index[0],target)
        else:
            target_index = target
    elif match=='index':
        target_index = target
    else:
        assert False, "[match] must be one of prefix,suffix,index"
    return target_index
