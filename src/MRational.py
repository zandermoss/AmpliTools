from MRing import MRing
from PermutationTools import PermuteBlocks,GetSign
from tqdm import tqdm
from sympy import symbols, poly, Rational, factor, simplify, Poly
from math import factorial
from itertools import product
from functools import reduce
from TensorTools import SplitTensorSymbol, TargetIndex, FirstMatchingIndex

#Some functions for comparison of denominator dictionaries (keyed by MRing objects).

def IsDictContained(_x,_y):
    x = dict(_x)
    y = dict(_y)
    for xkey in x.keys():
        if xkey in y.keys():
            if x[xkey] != y[xkey]:
                return False
            else:
                del y[xkey]
        else:
            return False
    return True

def IsDictEqual(x,y):
    return IsDictContained(x,y) and IsDictContained(y,x)

def IsDictInList(x,l):
    if len(l)==0:
        return -1
    for i,y in enumerate(l):
        if IsDictEqual(x,y):
            return i
    return -1

class MRational(object):

    def __init__(*_arg):
        self=_arg[0]
        arg=_arg[1:]
        #Define mathtype
        self.mathtype = "MRational"
        if len(arg)==1 and type(arg[0])==type(self):
            #Copy numerator and denominator to the local nd_list.
            #Numerator should be an MRing object, and denom should be dict of MRings.
            self.nd_list = list()
            for pair in arg[0].nd_list:
                denom_dict = {}
                for key in pair[1].keys():
                    mykey = MRing(key)
                    denom_dict[mykey] = pair[1][key]
                self.nd_list.append([MRing(pair[0]),denom_dict])
        elif len(arg)==1 and hasattr(arg[0], '__iter__'):
            self.nd_list = list()
            for pair in arg[0]:
                denom_dict = {}
                for key in pair[1].keys():
                    mykey = MRing(key)
                    denom_dict[mykey] = pair[1][key]
                self.nd_list.append([MRing(pair[0]),denom_dict])
        else:
            print(arg)
            assert False, "Bad argument to Operator constructor"


    def ApplyToNumerators(self,function):
        nd_list = []
        for pair in self.nd_list:
            target_numerator = function(pair[0])
            target_denominator = {}
            for key in pair[1].keys():
                target_key = MRing(key)
                assert (target_key.IsEmpty())==False
                target_denominator.setdefault(target_key,0)
                target_denominator[target_key]+=pair[1][key]
            if not target_numerator.IsEmpty():
                nd_list.append([target_numerator,target_denominator])
        return MRational(nd_list)

    def ApplyToDenominators(self,function):
        nd_list = []
        for pair in self.nd_list:
            target_numerator = MRing(pair[0])
            target_denominator = {}
            for key in pair[1].keys():
                target_key = function(key)
                assert (target_key.IsEmpty())==False
                target_denominator.setdefault(target_key,0)
                target_denominator[target_key]+=pair[1][key]
            if not target_numerator.IsEmpty():
                nd_list.append([target_numerator,target_denominator])
        return MRational(nd_list)

    def IndicesFromSymbol(self,symbol):
        symstring = symbol.__str__()
        if '_{' not in symstring:
            return []
        indexblock = symstring.split('{')[1].split('}')[0]
        indices = indexblock.split('f')[1:]
        indices = [int(index) for index in indices]
        return indices

    def GetPTerms(self,p,domain='QQ_I'):
        pterms = []
        for term in p.terms():
            coeff = term[1]
            expr = coeff
            for symbol,power in zip(p.gens,term[0]):
                expr*=symbol**power
            pterm = poly(expr,p.gens,domain=domain)
            pterms.append(pterm)
        return pterms

    def CanonicalizeBoundIndices(self,bound_prefixes):
        nd_list = []
        for pair in self.nd_list:
            denominator_index_counts = {}
            denominator_index_map = {}
            index_counter = 1
            #Extract free symbols from denominator factors.
            for factor,power in pair[1].items():
                for mkey,poly in factor.Mdict.items():
                    for monom in poly.monoms():
                        for symbol,count in zip(poly.gens,monom):
                            head,prefixes,indices = SplitTensorSymbol(symbol)
                            for prefix,index in zip(prefixes,indices):
                                if prefix in bound_prefixes:
                                    denominator_index_counts.setdefault((prefix,index),0)
                                    denominator_index_counts[(prefix,index)]+=count*power


            #Iterate over numerator terms and detect index multiplets.
            #Add multiplet indices to the index_map.
            for mkey,poly in pair[0].Mdict.items():
                for monom in poly.monoms():
                    my_index_counts = dict(denominator_index_counts)
                    for symbol,count in zip(poly.gens,monom):
                        head,prefixes,indices = SplitTensorSymbol(symbol)
                        for prefix,index in zip(prefixes,indices):
                            if (prefix,index) in denominator_index_counts.keys():
                                my_index_counts[(prefix,index)]+=count
                    for ipair,count in my_index_counts.items():
                        if count>1 and (ipair not in denominator_index_map.keys()):
                            denominator_index_map[ipair]=index_counter
                            index_counter+=1

            numerator_terms = []
            #Numerator-only contractions.
            for mkey,poly in pair[0].Mdict.items():
                for pterm in self.GetPTerms(poly):
                    my_index_counts = {}
                    my_index_counter = index_counter
                    my_index_map = {}
                    s = MRing({mkey:pterm})
                    monom = pterm.monoms()[0]
                    for symbol,count in zip(pterm.gens,monom):
                        head,prefixes,indices = SplitTensorSymbol(symbol)
                        for prefix,index in zip(prefixes,indices):
                            if (prefix,index) not in denominator_index_counts.keys():
                                if prefix in bound_prefixes:
                                    my_index_counts.setdefault((prefix,index),0)
                                    my_index_counts[(prefix,index)]+=count
                    for ipair,count in my_index_counts.items():
                        if count>1 and (ipair not in my_index_map.keys()):
                            s = s.TensorIndexReplacement(ipair[1],my_index_counter,source_prefix=ipair[0],target_prefix=ipair[0])
                            my_index_counter+=1
                    if not s.IsEmpty():
                        numerator_terms.append(s)
            numerator = reduce(lambda x,y: x+y, numerator_terms)

            #Make the replacments laid out in index_map.
            q = MRational([[numerator,pair[1]],])
            for ipair,target in denominator_index_map.items():
                    q = q.TensorIndexReplacement(ipair[1],target,source_prefix=ipair[0],target_prefix=ipair[0])

            nd_list+=q.nd_list
        rat = MRational(nd_list)
        rat = rat.Collect()

        return rat

    def CanonicalizeIndices(self,max_ext_label):
        """
        Older method. Now trying to explicitly denote bound indices at point
        of user entry with one of bound_prefixes prefixes.
        """
        nd_list = []
        for pair in self.nd_list:
            denominator_index_counts = {}
            denominator_index_map = {}
            index_counter = 1
            #Extract free symbols from denominator factors.
            for factor,power in pair[1].items():
                for mkey,poly in factor.Mdict.items():
                    for monom in poly.monoms():
                        for symbol,count in zip(poly.gens,monom):
                            for index in self.IndicesFromSymbol(symbol):
                                denominator_index_counts.setdefault(index,0)
                                denominator_index_counts[index]+=count*power


            #Iterate over numerator terms and detect index multiplets.
            #Add multiplet indices to the index_map.
            for mkey,poly in pair[0].Mdict.items():
                for monom in poly.monoms():
                    my_index_counts = dict(denominator_index_counts)
                    for symbol,count in zip(poly.gens,monom):
                        for index in self.IndicesFromSymbol(symbol):
                            if index in denominator_index_counts.keys():
                                my_index_counts[index]+=count
                    for index,count in my_index_counts.items():
                        if count>1 and (index not in denominator_index_map.keys()) and (index>max_ext_label):
                            denominator_index_map[index]=index_counter
                            index_counter+=1

            numerator_terms = []
            #Numerator-only contractions.
            for mkey,poly in pair[0].Mdict.items():
                for pterm in self.GetPTerms(poly):
                    my_index_counts = {}
                    my_index_counter = index_counter
                    my_index_map = {}
                    s = MRing({mkey:pterm})
                    monom = pterm.monoms()[0]
                    for symbol,count in zip(pterm.gens,monom):
                        for index in self.IndicesFromSymbol(symbol):
                            if index not in denominator_index_counts.keys():
                                my_index_counts.setdefault(index,0)
                                my_index_counts[index]+=count
                    for index,count in my_index_counts.items():
                        if count>1 and (index not in my_index_map.keys()) and (index>max_ext_label):
                            s = s.TensorIndexReplacement(index,my_index_counter,target_prefix='b')
                            my_index_counter+=1
                    if not s.IsEmpty():
                        numerator_terms.append(s)
            numerator = reduce(lambda x,y: x+y, numerator_terms)

            #Make the replacments laid out in index_map.
            q = MRational([[numerator,pair[1]],])
            for source,target in denominator_index_map.items():
                    q = q.TensorIndexReplacement(source,target,target_prefix='b')

            nd_list+=q.nd_list
        rat = MRational(nd_list)
        rat = rat.Collect()

        return rat

    def ProductReplacementCleanIndices(self):
        target_rat = MRational([])
        #for pair in tqdm(self.nd_list,desc="PRCI"):
        for pair in self.nd_list:
            #First, tabulate all indices.
            index_dict_list = [pair[0].GetIndexDict()]
            for r in pair[1].keys():
                index_dict_list.append(r.GetIndexDict())
            #Now, unify index_dict_list
            index_dict = {}
            for idict in index_dict_list:
                for key,val in idict.items():
                    index_dict.setdefault(key,set())
                    index_dict[key] = index_dict[key] | idict[key]
            #Replace d,g with z,w respectively, avoiding index collisions within
            #the nd_pair.
            rat_term = MRational([[pair[0],pair[1]],])
            pairmap = {'d':'z','g':'w','k':'b'}
            for source,target in pairmap.items():
                if source in index_dict.keys():
                    if target in index_dict.keys():
                        start = max(index_dict[target]) + 1
                    else:
                        start = 1
                    for n,index in enumerate(index_dict[source]):
                        rat_term = rat_term.TensorIndexReplacement(index,start+n,source_prefix=source,target_prefix=target)
            target_rat+=rat_term
        return target_rat


    # def NDPairTensorGetPrefixIndex(self,pair,prefix):
    #     bound_index = pair[0].TensorGetPrefixIndex(prefix)
    #     if bound_index != False:
    #         return bound_index
    #     for key in pair[1].keys():
    #         bound_index = key.TensorGetPrefixIndex(prefix)
    #         if bound_index != False:
    #             return bound_index
    #     return False
    #
    # def SplitRational(self,rat):
    #     target_rat = MRational([],rat.basis)
    #     for pair in rat.nd_list:
    #         bound_index = self.NDPairTensorGetPrefixIndex(pair,'b')
    #         if bound_index==False:
    #             target_rat+=MRational([pair,],rat.basis)
    #         else:
    #             rat1 = MRational([pair,],rat.basis)
    #             rat2 = MRational([pair,],rat.basis)
    #             rat1 = rat1.TensorIndexReplacement(bound_index,bound_index,source_prefix='b',target_prefix='z')
    #             rat2 = rat2.TensorIndexReplacement(bound_index,bound_index,source_prefix='b',target_prefix='w')
    #             target_rat+=self.SplitRational(rat1+rat2)
    #     return target_rat
    #
    #
    # def BranchPrefix(self,source_prefix,target_prefixes):
    #     #FALSE: need to turn pairs into MRational objects!
    #     target_rat = MRational([],self.basis)
    #     for pair in self.nd_list:
    #         for key,poly in pair[0].Mdict.values():
    #             term_ring = MRing({key:poly})
    #             term_rat = MRational([])
    #         working_rat = MRational([],self.basis)
    #         source_index = self.NDPairTensorGetPrefixIndex(pair, source_prefix)
    #         if source_index == False:
    #             target_rat += MRational([pair,],self.basis)
    #         else:
    #             for target in target_prefixes:
    #                 working_rat += MRational([pair,],self.basis).TensorIndexReplacement(source_index,source_index,source_prefix=source_prefix,target_prefix=target)
    #             target_rat += working_rat.BranchPrefix(source_prefix,target_prefixes)
    #     return target_rat
    #
    #
    # def BranchIndex(self,source_prefix,target_prefix,target_indices):
    #     target_rat = MRational([],self.basis)
    #     for pair in self.nd_list:
    #         working_rat = MRational([],self.basis)
    #         source_index = self.NDPairTensorGetPrefixIndex(pair, source_prefix)
    #         if source_index == False:
    #             target_rat += MRational([pair,],self.basis)
    #         else:
    #             for target in target_indices:
    #                 working_rat += MRational(pair,self.basis).TensorIndexReplacement(source_index,target,source_prefix=source_prefix,target_prefix=target_prefix)
    #             target_rat += working_rat.BranchPrefix(source_prefix,target_prefix,target_indices)
    #     return target_rat
    #
    # def BranchFreeIndex(self,source_prefix,target_prefix,target_indices):
    #     #We need to assume that all pairs have the same set of free indices.
    #     #Otherwise, the tensor constraints are inconsistent.
    #     exprs = []
    #     source_index = self.NDPairTensorGetPrefixIndex(self.nd_list[0],
    #                                                    source_prefix)
    #     if source_index == False:
    #         exprs.append(self)
    #     else:
    #         for target in target_indices:
    #             working_rat = self.TensorIndexReplacement(source_index,
    #                             target,source_prefix=source_prefix,
    #                             target_prefix=target_prefix)
    #             exprs += working_rat.BranchFreeIndex(source_prefix,
    #                                                target_prefix,target_indices)
    #     return exprs


    def FirstMatchingPairIndex(self,pair,pattern,match):
        #First, check the numerator.
        for poly in pair[0].Mdict.values():
            index = FirstMatchingIndex(poly,pattern,match)
            if index!=False:
                return index
        #Check each MRing in the denominator.
        for r in pair[1].keys():
            for poly in r.Mdict.values():
                index = FirstMatchingIndex(poly,pattern,match)
                if index!=False:
                    return index
        return False


    def ExpandFreeIndex(self,pattern,targets,match):
        if match=="prefix":
            assert (type(targets[0])==type(str())
                or type(targets[0])==type(tuple())),"Prefix match needs string or tuple targets."
            assert type(pattern)==type(str()),"Prefix match needs string source."
            if type(targets[0])==type(str()):
                assert pattern not in targets,"Pattern in targets, recursion will never terminate!"
            else:
                assert pattern not in [target[0] for target in targets],"Pattern in targets, recursion will never terminate!"
        elif match=="suffix":
            assert (type(targets[0])==type(int())
                or type(targets[0])==type(tuple())),"Suffix match needs int or tuple targets."
            assert type(pattern)==type(int()),"Suffix match needs int source."
            if type(targets[0])==type(int()):
                assert pattern not in targets,"Pattern in targets, recursion will never terminate!"
            else:
                assert pattern not in [target[1] for target in targets],"Pattern in targets, recursion will never terminate!"
        elif match =="index":
            assert type(targets[0])==type(tuple()),"Index match needs tuple targets."
            assert type(pattern[0])==type(tuple()),"Index match needs tuple source."
            assert pattern not in targets,"Pattern in targets, recursion will never terminate!"
        else:
            assert False, "[match] must be one of prefix,suffix,index"

        return self.BranchFreeIndex(pattern,targets,match)


    def BranchFreeIndex(self,pattern,targets,match):
        target_rats = []
        #Every term must have the same free indices, by definition.
        #Therefore, we only need to search for matches in the first term.
        pattern_index = self.FirstMatchingPairIndex(self.nd_list[0],pattern,
                                                    match)
        if pattern_index == False:
            target_rats.append(self)
        else:
            for target in targets:
                target_index = TargetIndex(pattern_index,target,match)
                working_rat = self.TensorIndexReplacement(
                                        pattern_index[1],target_index[1],
                                        source_prefix=pattern_index[0],
                                        target_prefix=target_index[0])
                target_rats += working_rat.BranchFreeIndex(pattern,targets,
                                                           match)
        return target_rats


    def ExpandBoundIndex(self,pattern,targets,match):
        if match=="prefix":
            assert (type(targets[0])==type(str())
                or type(targets[0])==type(tuple())),"Prefix match needs string or tuple targets."
            assert type(pattern)==type(str()),"Prefix match needs string source."
            if type(targets[0])==type(str()):
                assert pattern not in targets,"Pattern in targets, recursion will never terminate!"
            else:
                assert pattern not in [target[0] for target in targets],"Pattern in targets, recursion will never terminate!"
        elif match=="suffix":
            assert (type(targets[0])==type(int())
                or type(targets[0])==type(tuple())),"Suffix match needs int or tuple targets."
            assert type(pattern)==type(int()),"Suffix match needs int source."
            if type(targets[0])==type(int()):
                assert pattern not in targets,"Pattern in targets, recursion will never terminate!"
            else:
                assert pattern not in [target[1] for target in targets],"Pattern in targets, recursion will never terminate!"
        elif match =="index":
            assert type(targets[0])==type(tuple()),"Index match needs tuple targets."
            assert type(pattern[0])==type(tuple()),"Index match needs tuple source."
            assert pattern not in targets,"Pattern in targets, recursion will never terminate!"
        else:
            assert False, "[match] must be one of prefix,suffix,index"

        #Need to expand over all numerator poly terms.
        expanded_rat = self.Expand()
        target_rat = expanded_rat.BranchBoundIndex(pattern,targets,match)
        #We've suppressed collection and zero-culling throughout the recursion.
        #We'll apply these simplifications once to the final product.
        #__add__ performs these automatically, so we simply add "target+0".
        return target_rat + MRational([])

    def BranchBoundIndex(self,pattern,targets,match):
        target_rat = MRational([])
        for pair in self.nd_list:
            pattern_index = self.FirstMatchingPairIndex(pair,pattern,match)
            if pattern_index==False:
                target_rat.nd_list.append(pair)
            else:
                working_rat = MRational([])
                for target in targets:
                    target_index = TargetIndex(pattern_index,target,match)
                    term_rat = MRational([pair,]).TensorIndexReplacement(
                                            pattern_index[1],target_index[1],
                                            source_prefix=pattern_index[0],
                                            target_prefix=target_index[0])
                    working_rat = working_rat.Add(term_rat,collect=False)
                target_rat = target_rat.Add(working_rat.BranchBoundIndex(
                                           pattern,targets,match),collect=False)
        return target_rat


    def ApplyToMRing(self,function):
        r = self.ApplyToNumerators(function)
        r = r.ApplyToDenominators(function)
        return r

    def OnShell(self,basis):
        return self.ApplyToMRing(basis.OnShellRestriction)

    def EvaluateDelta(self):
        return self.ApplyToMRing(lambda r: MRing.EvaluateDelta(r))

    def EjectMasses(self,basis):
        return self.ApplyToMRing(basis.EjectMasses)

    def ZeroMasses(self,masses,basis):
        return self.ApplyToMRing(lambda r: basis.ZeroMasses(r,masses))

    def GroupMasses(self,massmap,basis):
        return self.ApplyToMRing(lambda r: basis.GroupMasses(r,massmap))

    def ZeroIndex(self,index):
        return self.ApplyToMRing(lambda r: MRing.ZeroIndex(r,index))

    def EvaluatePoly(self,symbol,value):
        return self.ApplyToMRing(lambda r: MRing.EvaluatePoly(r,symbol,value))

    def SetDomain(self,domain='QQ_I'):
        return self.ApplyToMRing(lambda r: MRing.SetDomain(r,domain))

    def SortGenerators(self):
        return self.ApplyToMRing(lambda r: MRing.SortGenerators(r))

    def KinematicReplacement(self,rmap):
        return self.ApplyToMRing(lambda r: MRing.Replacement(r,rmap))

    def Link(self,linkpairs):
        return self.ApplyToMRing(lambda r: MRing.Link(r,linkpairs))

    def MonomialReplacement(self,monomial,target_poly):
        return self.ApplyToMRing(lambda r: MRing.MonomialReplacement(r,monomial,target_poly))

    def TensorIndexReplacement(self,source,target,source_prefix='f',target_prefix='f'):
        return self.ApplyToMRing(lambda r: MRing.TensorIndexReplacement(r,source,target,source_prefix,target_prefix))

#    def CanonicalizeBoundIndices(self):
#        #FIXME: only run after merging nd_sums!
#        return self.ApplyToNumerators(lambda r: MRing.CanonicalizeBoundIndices(r))

    def TensorPrefixReplacement(self,source_prefix,target_prefix):
        return self.ApplyToMRing(lambda r: MRing.TensorPrefixReplacement(r,source_prefix,target_prefix))

    def TensorProductReplacement(self,source,target,max_depth=False):
        #FIXME: can't apply this when the bound indices show up in the denominator too!
        #return self.ApplyToMRing(lambda r: MRing.TensorProductReplacement(r,source,target))
        #Assuming no color tensors (except masses) in denominators.
        #max_depth=1
        count=0
        last_rat = MRational(self)
        while True:
            next_rat = last_rat.ApplyToNumerators(lambda r: MRing.TensorProductReplacement(r,source,target))
            next_rat = next_rat.ProductReplacementCleanIndices()
            count+=1
            if next_rat==last_rat:
                return next_rat
            if max_depth!=False and count==max_depth:
                return next_rat
            last_rat = MRational(next_rat)


    def BlockReplacement(self,blockmap,symbolblocks,source_prefix='f',target_prefix='f'):
        return self.ApplyToMRing(lambda r: MRing.BlockReplacement(r,blockmap,symbolblocks,source_prefix=source_prefix,target_prefix=target_prefix))

    def DeltaContract(self,deltahead):
        return self.ApplyToMRing(lambda r: MRing.DeltaContract(r,deltahead))

    def FactorPolynomials(self):
        return self.ApplyToNumerators(lambda r: MRing.FactorPolynomials(r))

    def SimplifyPolynomials(self):
        return self.ApplyToNumerators(lambda r: MRing.SimplifyPolynomials(r))

    def DressMomentumPairs(self,symbol,longitudinal_modes):
        return self.ApplyToMRing(lambda r: MRing.DressMomentumPairs(r,symbol,longitudinal_modes))

    def DressMomentum(self,symbol,label):
        return self.ApplyToMRing(lambda r: MRing.DressMomentum(r,symbol,label))

    def SortSymmetricIndices(self,tensor_symmetries):
        #print("In SSI wrapper")
        return self.ApplyToMRing(lambda r: MRing.SortSymmetricIndices(r,tensor_symmetries))

    def PermuteBlocks(self,perm,symbolblocks,signed=False):
        return self.ApplyToMRing(lambda r: PermuteBlocks(r,perm,symbolblocks,signed))

    def __mul__(self,other):
        dummy = symbols('dummy')
        ptype = type(poly(dummy,dummy,domain='QQ_I'))
        mrtype = type(MRing({}))
        domain_types = [ptype,mrtype,int]
        #Doesn't handle imaginary numbers properly...
        #is_sympy_number = ("sympy.core.numbers" in str(type(other)))
        is_sympy_number = ("sympy.core" in str(type(other)))
        assert (type(other) in domain_types) or is_sympy_number or (type(other) == type(self))
        if ((type(other) in domain_types) or is_sympy_number):
            r = self.ApplyToNumerators(lambda r: MRing.__mul__(r,other))
            #r,upoly = r.UnifyGenerators()
            return r
        else:
            prodterms = list(product(self.nd_list,other.nd_list))
            prod_nd_list=[]
            for term in prodterms:
                #multiply numerator mrings
                numerator = term[0][0]*term[1][0]
                #combine the denominators
                denominator = {}
                for key in term[0][1]:
                    denominator.setdefault(key,0)
                    denominator[key]+=term[0][1][key]
                for key in term[1][1]:
                    denominator.setdefault(key,0)
                    denominator[key]+=term[1][1][key]
                prod_nd_list.append([numerator,denominator])
            r = MRational(prod_nd_list)
            #r,upoly = r.UnifyGenerators()
            return r.Collect()

    def Orbit(self,perms,symbolblocks,signed=False,prefix='f'):
        orbit = MRational([])

        #for perm in tqdm(perms,desc="Permutations"):
        for perm in perms:
            #for i in tqdm([1,],desc="ApplyPerms"):
            #for i in tqdm([1,],desc="ApplyPerms"):
            permrat = self.ApplyToMRing(lambda r: PermuteBlocks(r,perm,symbolblocks,signed=False,source_prefix=prefix,target_prefix=prefix))
            if signed:
                permrat=permrat*GetSign(perm)
            #orbit = orbit.Add(permrat,collect=False)
            #for i in tqdm([1,],desc="AddInPlace"):
            orbit.AddInPlace(permrat)

        #for i in tqdm([1,],desc="Collection"):
        orbit = orbit.Collect()
        orbit = orbit.CullZeros()
        return orbit

    def TensorProductRatReplacement(self,source,target,max_depth=False):
        #NOTE: this is a whole lot of bullshit. The problem is equivalent
        #to subgraph isomorphism, which has much quicker implementations.
        #need to overhaul this code to interpret a fast isomorphism algorithm.
        #FIXME: can't apply this when the bound indices show up in the denominator too!
        #return self.ApplyToMRing(lambda r: MRing.TensorProductReplacement(r,source,target))
        #Assuming no color tensors (except masses) in denominators.
        #max_depth=1

        assert len(source.as_dict())==1, "Source is not monomial."
        source_term = source.as_dict().keys()[0]

        mring = MRing(self.nd_list[0][0])
        source = mring.PolyTensorPrefixReplacement(source,'x','a')
        source = mring.PolyTensorPrefixReplacement(source,'y','c')
        source = mring.PolyTensorPrefixReplacement(source,'z','d')
        source = mring.PolyTensorPrefixReplacement(source,'w','g')
        source = mring.PolyTensorPrefixReplacement(source,'f','h')
        source = mring.PolyTensorPrefixReplacement(source,'b','k')
        target = target.TensorPrefixReplacement('x','a')
        target = target.TensorPrefixReplacement('y','c')
        target = target.TensorPrefixReplacement('z','d')
        target = target.TensorPrefixReplacement('w','g')
        target = target.TensorPrefixReplacement('f','h')
        target = target.TensorPrefixReplacement('b','k')

        last_rat = MRational(self)
        next_rat = MRational([])
        count=0
        while True:
            loopmatch=False
            for num,den in last_rat.nd_list:
                for key,p in num.Mdict.items():
                    for term,coefficient in p.as_dict().items():
                        result = num.MatchTerm(term,p.gens,source_term,source.gens,0,{})

                        if result==False:
                            target_term = Poly({tuple(term):coefficient},p.gens,domain="QQ_I")
                            ring_term = MRing({key:target_term})
                            rat_term = MRational([[ring_term,den],])
                            next_rat += rat_term
                        else:
                            loopmatch=True
                            stripped_term,xmap = result
                            target_term = Poly({tuple(stripped_term):coefficient},p.gens,domain="QQ_I")
                            ring_target = MRing({key:target_term})
                            rat_target = MRational([[ring_target,den],])
                            rat_target*=target
                            for source_pair,target_pair in xmap.items():
                                rat_target = rat_target.TensorIndexReplacement(source_pair[1],target_pair[1],source_prefix = source_pair[0], target_prefix = target_pair[0])
                            next_rat += rat_target
            next_rat = next_rat.ProductReplacementCleanIndices()
            count+=1

            if not loopmatch:
                #BAD NEWS BEARS
                #last_rat = last_rat.TensorPrefixReplacement('a','x')
                #last_rat = last_rat.TensorPrefixReplacement('c','y')
                #last_rat = last_rat.TensorPrefixReplacement('d','z')
                #last_rat = last_rat.TensorPrefixReplacement('g','w')
                #last_rat = last_rat.TensorPrefixReplacement('g','f')
                #last_rat = last_rat.TensorPrefixReplacement('k','b')
                return last_rat
            if max_depth!=False and count==max_depth:
                assert False
            last_rat = MRational(next_rat)

    def DressMasses(self,mass_symbols):
        q = MRational(self.nd_list)
        z = symbols('z')
        for symbol in mass_symbols:
            target = poly(symbol*z,symbol,z,domain='QQ_I')
            q = q.MonomialReplacement(symbol,target)
        return q,z

    def DressAllMasses(self,symbol):
        target_rat = MRational([])
        mass_symbols = []
        #for pair in tqdm(self.nd_list,desc="PRCI"):
        for pair in self.nd_list:
            #First, tabulate all indices.
            for key,polynomial in pair[0].Mdict.items():
                for gen in polynomial.gens:
                    if 'm_{' in gen.__str__():
                        if 'm'==gen.__str__().split('_{'):
                            mass_symbols.append(gen)
            for r in pair[1].keys():
                for key,polynomial in r.Mdict.items():
                    for gen in polynomial.gens:
                        if 'm_{' in gen.__str__():
                            if 'm'==gen.__str__().split('_{')[0]:
                                mass_symbols.append(gen)
        mass_symbols = list(set(mass_symbols))

        q = MRational(self.nd_list)
        for sym in mass_symbols:
            target = poly(sym*symbol,sym,symbol,domain='QQ_I')
            q = q.MonomialReplacement(sym,target)
        return q

    def PartialDerivative(self,symbol):
        nd_list = []
        for pair in self.nd_list:
            N = pair[0].PartialDerivative(symbol)
            D = {}
            for key,val in zip(pair[1].keys(),pair[1].values()):
                mykey = MRing(key)
                D[mykey] = val
            if not N.IsEmpty():
                nd_list.append([N,D])
            for key,val in zip(pair[1].keys(),pair[1].values()):
                N = MRing(pair[0])
                N *= key.PartialDerivative(symbol)
                #N *= (key**(val-1))*val
                #N *= (-1)
                N *= (-1)*val
                D = {}
                for newkey,newval in zip(pair[1].keys(),pair[1].values()):
                    mykey = MRing(newkey)
                    D[mykey] = newval
                D[key]+=1
                if not N.IsEmpty():
                    nd_list.append([N,D])
        return MRational(nd_list)

    def MaclaurinCoefficient(self,symbol,order):
        q = MRational(self.nd_list)
        for i in range(order):
            q = q.PartialDerivative(symbol)
        q = q.EvaluatePoly(symbol,0)
        q *= Rational(1,factorial(order))
        return q

    #---------Below, we are computing laurent coefficients around a complex-infinite pole----------#

    def ComputeWRational(self,denominator,z,w):
        w_denominator = {}
        m = 0
        for mring,power in denominator.items():
            w_mring = MRing({})
            degrees = []
            for polynomial in mring.Mdict.values():
                #newsymbols =  set(list(polynomial.free_symbols)+[z,])
                newsymbols =  set(list(polynomial.gens)+[z,])
                newpoly = poly(polynomial.as_expr(),newsymbols,domain='QQ_I')
                degrees.append(newpoly.degree(gen=z))
            my_m = max(degrees)

            m += my_m*power
            for mbasis,polynomial in mring.Mdict.items():
                w_polynomial = polynomial.zero
                #new_symbols = set(list(polynomial.free_symbols)+[z,])
                new_symbols = set(list(polynomial.gens)+[z,])
                new_poly = poly(polynomial.as_expr(),new_symbols,domain='QQ_I')
                degree = new_poly.degree(gen=z)
                for p in range(0,degree+1):
                    coeff = Rational(1,factorial(p))*(new_poly.diff((z,p))).eval(z,0)
                    coeff = poly(coeff,new_symbols,domain='QQ_I')
                    w_power = my_m-p
                    #new_symbols = set(list(coeff.free_symbols)+[w,])
                    new_symbols = set(list(coeff.gens)+[w,])
                    newterm = poly(coeff.as_expr()*w**w_power,new_symbols,domain='QQ_I')
                    w_polynomial += newterm
                w_mring += MRing({mbasis:w_polynomial})
            w_denominator[w_mring] = power
        w_rational = MRational([[w_mring.One(),w_denominator],])
        return w_rational,m

    def ComputeDiffMap(self,numerator,z,w,w_rational,zeta,m):
        degrees = []
        for polynomial in numerator.Mdict.values():
            #newsymbols =  set(list(polynomial.free_symbols)+[z,])
            newsymbols =  set(list(polynomial.gens)+[z,])
            newpoly = poly(polynomial.as_expr(),newsymbols,domain='QQ_I')
            degrees.append(newpoly.degree(gen=z))
        max_z_power = max(degrees)
        diff_map = {}
        for i in range(0,max_z_power+1):
            index = i-zeta-m
            if index<0:
                continue
            w_partial = MRational(w_rational)
            for p in range(index):
                w_partial = w_partial.PartialDerivative(w)
            diff_map[index] = w_partial.EvaluatePoly(w,0)
        return diff_map


    def IsSymbolInPolys(self,symbol):
        """
        Check whether [symbol] is contained in any of the gens sets
        of the polynomials contained within this MRational object. Return True
        if so, False if not.
        """
        for pair in self.nd_list:
            for polynomial in pair[0].Mdict.values():
                #if symbol in polynomial.free_symbols:
                if symbol in polynomial.gens:
                    return True
            for mring in pair[1].keys():
                for polynomial in mring.Mdict.values():
                    #if symbol in polynomial.free_symbols:
                    if symbol in polynomial.gens:
                        return True
        return False

    def UVLaurentCoefficient(self,z,zeta):
        """
        Compute the MRational Laurent coefficient of self at order [zeta]
        in terms of the complex variable [z].
        """
        # w = 1/z, so the UV pole is mapped to a pole at the origin in
        # the w-plane.
        w = symbols('w')

        assert not self.IsSymbolInPolys(w), "w symbol is already in use!"
        laurent_coeff = MRational([])
        for pair in tqdm(self.nd_list,desc="Computing UV Laurent Coefficient"):
            numerator = pair[0]
            denominator = pair[1]

            # Compute the inverse denominator, multiplied by w**m, so that
            # it is regular in the neighborhood of the origin.
            w_rational, m = self.ComputeWRational(denominator,z,w)

            # Compute various partial derivatives of w_rational, evaluated
            # at w=0. These are precomputed in the interest of efficiency,
            # and will be accessed repeatedly in the loops below.
            diff_map = self.ComputeDiffMap(numerator,z,w,w_rational,zeta,m)

            for mbasis,polynomial in numerator.Mdict.items():
                #new_symbols = set(list(polynomial.free_symbols)+[z,])
                new_symbols = set(list(polynomial.gens)+[z,])
                new_poly = poly(polynomial.as_expr(),new_symbols,domain='QQ_I')
                for p in range(0,new_poly.degree(gen=z)+1):
                    index = p-zeta-m
                    if index<0:
                        continue
                    coeff = Rational(1,factorial(p))*(new_poly.diff((z,p))).eval(z,0)
                    term = MRational(diff_map[index])
                    term *= Rational(1,factorial(index))
                    #term *= MRing({mbasis:numerator.Mdict.values()[0].one})
                    term *= MRing({mbasis:numerator.PolyOne()})
                    term *= coeff
                    laurent_coeff += term
        laurent_coeff = laurent_coeff.Collect()
        return laurent_coeff

    #---------Below, we are computing laurent coefficients around the origin----------#

    def ComputeZRational(self,denominator,z):
        z_denominator = {}
        m = 0
        for mring,power in denominator.items():
            z_mring = MRing({})
            degrees = []
            for polynomial in mring.Mdict.values():
                newsymbols =  set(list(polynomial.gens)+[z,])
                newpoly = poly(polynomial.as_expr(),newsymbols,domain='QQ_I')
                degrees.append(newpoly.degree(gen=z))
            my_m = min(degrees)

            m += my_m*power
            for mbasis,polynomial in mring.Mdict.items():
                z_polynomial = mring.PolyZero()
                new_symbols = set(list(polynomial.gens)+[z,])
                new_poly = poly(polynomial.as_expr(),new_symbols,domain='QQ_I')
                degree = new_poly.degree(gen=z)
                for p in range(0,degree+1):
                    coeff = Rational(1,factorial(p))*(new_poly.diff((z,p))).eval(z,0)
                    co_poly = poly(coeff,new_symbols,domain='QQ_I')
                    z_power = p-my_m
                    newterm = poly(co_poly.as_expr()*z**z_power,new_symbols,domain='QQ_I')
                    z_polynomial += newterm
                z_mring += MRing({mbasis:z_polynomial})
            z_denominator[z_mring] = power
        z_rational = MRational([[z_mring.One(),z_denominator],],self.basis)
        return z_rational,m

    def ComputeIRDiffMap(self,numerator,z,z_rational,zeta,m):
        degrees = []
        for polynomial in numerator.Mdict.values():
            newsymbols =  set(list(polynomial.gens)+[z,])
            newpoly = poly(polynomial.as_expr(),newsymbols,domain='QQ_I')
            degrees.append(newpoly.degree(gen=z))
        max_z_power = max(degrees)
        diff_map = {}
        for i in range(0,max_z_power+1):
            index = zeta+m-i
            if index<0:
                continue
            z_partial = MRational(z_rational)
            for p in range(index):
                z_partial = z_partial.PartialDerivative(z)
            diff_map[index] = z_partial.EvaluatePoly(z,0)
        return diff_map

    def IRLaurentCoefficient(self,z,zeta):
        """
        Compute the MRational Laurent coefficient of self at order [zeta]
        in terms of the complex variable [z].
        """
        laurent_coeff = MRational([])
        for pair in tqdm(self.nd_list,desc="Computing IR Laurent Coefficient"):
            numerator = pair[0]
            denominator = pair[1]

            # Compute the inverse denominator, divided by z**m, so that
            # it is regular in the neighborhood of the origin.
            z_rational, m = self.ComputeZRational(denominator,z)

            # Compute various partial derivatives of z_rational, evaluated
            # at z=0. These are precomputed in the interest of efficiency,
            # and will be accessed repeatedly in the loops below.
            diff_map = self.ComputeIRDiffMap(numerator,z,z_rational,zeta,m)

            for mbasis,polynomial in numerator.Mdict.items():
                new_symbols = set(list(polynomial.gens)+[z,])
                new_poly = poly(polynomial.as_expr(),new_symbols,domain='QQ_I')
                for p in range(0,new_poly.degree(gen=z)+1):
                    index = zeta+m-p
                    if index<0:
                        continue
                    coeff = Rational(1,factorial(p))*(new_poly.diff((z,p))).eval(z,0)
                    term = MRational(diff_map[index])
                    term *= Rational(1,factorial(index))
                    term *= MRing({mbasis:numerator.PolyOne()})
                    term *= coeff
                    laurent_coeff += term
        laurent_coeff = laurent_coeff.Collect()
        return laurent_coeff

    def __str__(self):
            string = ''
            for pair in self.nd_list:
                string+="NUM\n"
                string+=pair[0].__str__()
                string+="DEN\n"
                for key in pair[1].keys():
                    string+=key.__str__()
                    string+=pair[1][key].__str__()+'\n'
                string+='\n'
            return string

    def FullStr(self):
            string = ''
            for pair in self.nd_list:
                string+="NUM\n"
                string+=pair[0].FullStr()
                string+="DEN\n"
                for key in pair[1].keys():
                    string+=key.FullStr()
                    string+=pair[1][key].__str__()+'\n'
                string+='\n'
            return string

    def UnifyGenerators(self):
        if len(self.nd_list)==0:
            return MRational(self),None
        r,upoly = self.nd_list[0][0].UnifyGenerators()
        for pair in self.nd_list:
            r,upoly = pair[0].UnifyGenerators(upoly)
            for key in pair[1].keys():
                r,upoly = key.UnifyGenerators(upoly)

        new_nd_list = []
        for pair in self.nd_list:
            numerator,mypoly = pair[0].UnifyGenerators(upoly)
            denominator = {}
            for key,val in pair[1].items():
                denkey,mypoly = key.UnifyGenerators(upoly)
                denominator[denkey] = val
            new_nd_list.append([numerator,denominator])
        return MRational(new_nd_list),upoly

    def CullZeros(self):
        clean_nd_list = []
        for pair in self.nd_list:
            if not pair[0].IsEmpty():
                clean_nd_list.append(pair)
        return MRational(clean_nd_list)

    def Add(self,other,collect=True):
        nd_list = self.nd_list + other.nd_list
        r = MRational(nd_list)
        #r,upoly = r.UnifyGenerators()
        if collect:
            r = r.Collect()
            r = r.CullZeros()
#            clean_nd_list = []
#            #Cull Zeros
#            for pair in r.nd_list:
#                if not pair[0].IsEmpty():
#                    clean_nd_list.append(pair)
#            r =  MRational(clean_nd_list,self.basis)
        
        return r

    def AddInPlace(self,other):
        self.nd_list += other.nd_list

    def __add__(self,other):
        return self.Add(other,collect=True)

    def __sub__(self,other):
        return self+(other*-1)

#    def __eq__(self,other):
#        assert type(self)==type(other)
#        self_collected = self.Collect()
#        other_collected = other.Collect()
#        if not (len(self_collected.nd_list) == len(other_collected.nd_list)):
#            return False
#        self_indices = range(len(self_collected.nd_list))
#        self_numerators = [nd[0] for nd in self_collected.nd_list]
#        self_denominators = [nd[1] for nd in self_collected.nd_list]
#        other_indices = range(len(other_collected.nd_list))
#        other_numerators = [nd[0] for nd in other_collected.nd_list]
#        other_denominators = [nd[1] for nd in other_collected.nd_list]
#
#        while len(self_indices)>0:
#            self_index = self_indices[0]
#            other_index = IsDictInList(self_denominators[self_index], other_denominators)
#            if other_index<0:
#                return False
#            if self_numerators[self_index]!=other_numerators[other_index]:
#                return False
#
#            self_indices.remove(self_index)
#            other_indices.remove(other_index)
#
#        return True

    def __eq__(self,other):
        assert type(self)==type(other)
        diff = self-other
        #Set domain to QQ_I
        diff = diff.SetDomain()
        diff = diff.SortGenerators()
        diff = diff.Collect()
        if len(diff.nd_list)==0:
            return True
        else:
            return False

    def __ne__(self,other):
        return not self.__eq__(other)

    def __hash__(self):
        #Implemented sorting to eliminate ordering ambiguities.
        def hash_pair(pair):
            numerator_hash = pair[0].__hash__()
            denominator_list = [(mr.__hash__(),pwr.__hash__()) for mr,pwr in pair[1].items()]
            denominator_list = sorted(denominator_list,key=lambda mp: mp[0])
            denominator_hash = tuple(denominator_list).__hash__()
            return (numerator_hash,denominator_hash).__hash__()

        hashes = sorted([hash_pair(pair) for pair in self.nd_list])
        hash = tuple(hashes).__hash__()
        return hash

    def Collect(self):
        r = MRational(self)
        #Simplify ones in the denominator and clear zeros in the numerator.
        new_nd_list = []
        #for n,d in tqdm(r.nd_list,desc="Simplify ones and zeros"):
        for n,d in r.nd_list:
            new_denom = {}
            for key in d.keys():
                assert not key.IsEmpty()
                if len(key.Mdict)==1 and list(key.Mdict.keys())[0]==((0,0),) and list(key.Mdict.values())[0] == list(key.Mdict.values())[0].one:
                    if len(d.keys())==1:
                        new_denom[key]=1
                    else:
                        pass
                else:
                    new_denom[key] = d[key]
            if not n.IsEmpty():
                new_nd_list.append((n,new_denom))
        r.nd_list = new_nd_list

        #Collect numerators with common denominators.
        nd_keys = []
        nd_vals = []
        for pair in r.nd_list:
            nd_keys.append(pair[1])
            nd_vals.append(pair[0])

        key_index=0
        #for i in tqdm([1,],desc="Collect common denom"):
        while True:
            while True:
                if key_index>=len(nd_keys)-1:
                    break
                #for i in tqdm([1,],desc="IsDictInList"):
                inret = IsDictInList(nd_keys[key_index],nd_keys[key_index+1:])
                if inret<0:
                    break
                second_index = inret+key_index+1

                #for i in tqdm([1,],desc="Adding"):
                #nd_vals[key_index]+=nd_vals[second_index]
                nd_vals[key_index].AddInPlace(nd_vals[second_index])

                #for i in tqdm([1,],desc="DoubleDel"):
                del nd_keys[second_index]
                del nd_vals[second_index]
            key_index+=1
            if key_index>=len(nd_keys)-1:
                break

        #for i in tqdm([1,],desc="Instantiate MRational"):
        r = MRational(zip(nd_vals,nd_keys))

        #Clear zeros in the numerator.
        #for i in tqdm([1,],desc="Clear num zeros"):
        new_nd_list = []
        for n,d in r.nd_list:
            nr = MRing(n)
            nr.CullZeros()
            if not nr.IsEmpty():
                new_nd_list.append((nr,d))
        r = MRational(new_nd_list)


        #for i in tqdm([1,],desc="SignClear"):
        r = self.SignClear(r)

        return r

        #return self.SignClear(r)


    def SignClear(self,r):
        rat = MRational(r)
        for n,pair in enumerate(r.nd_list):
                for ring,power in pair[1].items():
                    firstkey = sorted(ring.Mdict.keys())[0]
                    poly_RO = ring.Mdict[firstkey].reorder(*sorted(ring.Mdict[firstkey].gens,key=str))
                    if poly_RO.coeffs()[0]<0:
                        rat.nd_list[n][0]=rat.nd_list[n][0]*(-1)**power
                        del rat.nd_list[n][1][ring]
                        rat.nd_list[n][1][ring*(-1)]=power
        return rat


    def Expand(self):
        target_rat = MRational([])
        for pair in self.nd_list:
            for key,poly in pair[0].Mdict.items():
                for pterm in pair[0].GetPTerms(poly):
                    term_ring = MRing({key:pterm})
                    term_rat = MRational([[term_ring,pair[1]],])
                    target_rat = target_rat.Add(term_rat,collect=False)
        return target_rat







#        class hashable_dict(object):
#            def __init__(self,mydict):
#                self.mydict = mydict
#            def HashCompare(self,x,y):
#                xhash = x.__hash__()
#                yhash = y.__hash__()
#                if xhash>yhash:
#                    return 1
#                elif xhash==yhash:
#                    return 0
#                else:
#                    return -1
#            def GetCompTuple(self):
#                comp_list = []
#                sorted_keys = sorted(self.mydict.keys(),cmp = lambda x,y: self.HashCompare(x,y))
#                for key in sorted_keys:
#                    comp_list.append((key.__hash__(),self.mydict[key].__hash__()))
#                return tuple(comp_list)
#            def __hash__(self):
#                return hash(self.GetCompTuple())
#            def __eq__(self,other):
#                return self.GetCompTuple()==other.GetCompTuple()
#
#        meta_denom_dict = {}
#        #for pair in tqdm(self.nd_list,desc="Collecting Terms"):
#        for pair in self.nd_list:
#            hashdict = hashable_dict(pair[1])
#            meta_denom_dict.setdefault(hashdict,MRing({}))
#            meta_denom_dict[hashdict] += pair[0]
#        nd_list = []
#        for key,val in zip(meta_denom_dict.keys(),meta_denom_dict.values()):
#            cleanval = MRing(val)
#            cleanval.CullZeros()
#            nd_list.append([cleanval,key.mydict])
#        return MRational(nd_list,self.basis)

    def MergeDenominators(self):
        #This is a refined function which deals directly with pole factors instead
        #of expanded MRings. This way, we can multiply by the polynomial LCD to
        #merge denominators, cutting down on the ultimate size of the numerator
        #MRing, which balloons exponentially in the number of poles merged.

        #First, we must compute the LCD. We will iterate through all poles in each
        #denominator in the nd_list, and record the maximum power of each given
        #pole. The dict containing all distinct poles and their maximum powers is
        #the polynomial LCD.

        LCD = {}
        #for pair in tqdm(self.nd_list,desc="Finding LCD"):
        for pair in self.nd_list:
            for pole,multiplicity in zip(pair[1].keys(),pair[1].values()):
                LCD.setdefault(pole,multiplicity)
                if LCD[pole]<multiplicity:
                    LCD[pole] = multiplicity

        # Now we iterate through the numerators and multiply them by
        # the LCD mod the factors in the corresponding denominator.
        numerator = MRing({})
        #for pair in tqdm(self.nd_list,desc="Merging"):
        for pair in self.nd_list:
            factordict = dict(LCD)
            for key in pair[1].keys():
                factordict[key]-=pair[1][key]
            numerator_term = MRing(pair[0])
            for pole,multiplicity in zip(factordict.keys(),
                                         factordict.values()):
                for i in range(multiplicity):
                    numerator_term*=pole
            numerator+=numerator_term
        return MRational([[numerator,LCD],])


    def GetFreeSymbols(self):
        """
        Returns a `set` of all free symbols appearing in all polynomials
        in both the numerator and the denominator of each pair. Invokes
        `MRing.GetFreeSymbols()`.
        """
        free_symbols = set()
        for pair in self.nd_list:
            # Start with the numerator
            free_symbols |= pair[0].GetFreeSymbols()
            # Now the denominator
            free_symbols |= reduce(lambda s,t: s|t, [r.GetFreeSymbols()
                                   for r in pair[1].keys()])
        return free_symbols

    def GetTensorSymbols(self):
        """
        Returns a `set` of all tensor symbols appearing in all polynomials
        in both the numerator and the denominator of each pair.
        First computes set of all free symbols using
        `MRations.GetTensorSymbols()`, then filters out all symbols without
        "_{" characters, indicative of a tensor index block.
        """
        return set(filter(lambda s: "_{" in s.__str__(),self.GetFreeSymbols()))

    def FindPrefix(self,prefix_key):
        """
        Searches all numerator and denominator polynomials for tensors
        containing `prefix`. Returns a list of all indices carrying this
        prefix.
        """
        tensor_symbols = self.GetTensorSymbols()
        keyed_indices = set()
        for tensym in tensor_symbols:
            head,prefix,index = SplitTensorSymbol(tensym)
            keyed_indices |= {index[i] for i,v in enumerate(prefix)
                             if v==prefix_key}
        keyed_indices = sorted(list(keyed_indices))
        return keyed_indices

    # def BoundIndicesToComponents(self,
    #                              first,
    #                              last,
    #                              bound_prefix='b',
    #                              target_prefix='x'):
    #     """
    #     Expands bound indices into a range of components `range(first,last+1)`.
    #     Components are new indices with 'x' prefix (by default). All indices
    #     with bound_prefix (assumed 'b') are considered bound. Returns target
    #     `MRational` object.
    #     """
    #     tensor_symbols = self.GetTensorSymbols()
    #     bound_indices = set()
    #     for tensym in tensor_symbols:
    #         head,prefix,index = SplitTensorSymbol(tensym)
    #         bound_indices |= {index[i] for i,v in enumerate(prefix)
    #                           if v==bound_prefix}
    #
    #     source_rat = self
    #     for ib in bound_indices:
    #         target_rat = self*0
    #         #source_symbols = source_rat.GetTensorSymbols()
    #         #bound_symbols = set(filter(lambda s: bound_prefix+str(ib)
    #         #                           in s.__str__(),source_symbols))
    #         for target_index in range(first, last+1):
    #             target_rat += source_rat.TensorIndexReplacement(
    #                                             ib,
    #                                             target_index,
    #                                             source_prefix=bound_prefix,
    #                                             target_prefix = target_prefix)
    #         source_rat = target_rat
    #     return target_rat

    def BoundIndicesToComponents(self,
                                 first,
                                 last,
                                 bound_prefix='b',
                                 target_prefix='x'):
        """
        Expands bound indices into sums over component values, between `first`
        and `last` (inclusive). Only applies when no bound indicies appear in
        denominators.
        """
        target_nd_list = []
        for numerator,denominator in self.nd_list:
            target_numerator = numerator.BoundIndicesToComponents(first,last,bound_prefix,target_prefix)
            target_denominator = {}
            for r in denominator.keys():
                assert len(r.FindPrefix(bound_prefix))==0, "Bound indices found in a denominator."
            target_nd_list.append([target_numerator,denominator])
        return MRational(target_nd_list)


    def FreeIndicesToComponents(self,
                                index_map,
                                free_prefix='f',
                                target_prefix='x'):
        """
        Maps free indices to components as specificed by `index_map`.
        Assumes 'f' prefixes free components. Component indices are
        prefixed with 'x' by default. Returns target `MRational` object.
        """
        source_rat = self
        for source_index, target_index in index_map.items():
            source_rat = source_rat.TensorIndexReplacement(
                                                        source_index,
                                                        target_index,
                                                        source_prefix='f',
                                                        target_prefix='x')
        return source_rat

    def GetFreeTensorElements(self,
                              first,
                              last,
                              free_prefix='f',
                              target_prefix='x'):
        """
        Produces a dict of free indices ('f' prefix by default)
        evaluated at all elements of range(first,last+1)^{Nf}, where
        the exponent indicates the cartesian fold, and Nf is the number of
        free indices. Returns dict(index tuple key, MRational value).
        """
        # First enumerate free indices.
        # tensor_symbols = self.GetTensorSymbols()
        # free_indices = set()
        # for tensym in tensor_symbols:
        #     head,prefix,index = SplitTensorSymbol(tensym)
        #     free_indices |= {index[i] for i,v in enumerate(prefix)
        #                      if v==free_prefix}
        # free_indices = sorted(list(free_indices))
        free_indices = self.FindPrefix(free_prefix)


        # Next, compute a dictionary mapping all possible combinations of
        # external index components to `MRational` objects evaluated at those
        # components using `MRational.FreeIndicesToComponents`.
        target_dict = {}
        #for tup in tqdm(list(product(range(first,last+1),repeat=len(free_indices))),desc="Expanding over free index tuples (tuples)"):
        for tup in list(product(range(first,last+1),repeat=len(free_indices))):
            target_dict[tup] = self.FreeIndicesToComponents(dict(zip(
                                                                free_indices,
                                                                tup)))

        return target_dict

    def GetNumeratorPolys(self):
        """
        Returns a list of numerator polynomials.
        """
        polys = []
        for pair in self.nd_list:
            polys += list(pair[0].Mdict.values())
        return polys

    # def RemoveDuplicates(ring_it):
    #     """
    #     The canonical set(list()) trick doesn't seem to be using the overloaded
    #     comparison operator defined within MRational, so this is just a manual
    #     implementation that will reduce using equality in the sense of the
    #     MRational objects. Takes an iterable, returns a dupe-free list.
    #     Edit: Not sure why this wasn't working. Re-tested using list(set)
    #     without issue...
    #     """
    #     source_ring_list = list(ring_it)
    #     target_ring_list = []
    #     while len(source_ring_list) > 0:
    #         x = source_ring_list.pop(0)
    #         source_ring_list = [y for y in source_ring_list if y!=x]
    #         target_ring_list.append(x)
    #     return target_ring_list
