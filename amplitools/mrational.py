from .mring import MRing
from .permutation_tools import permute_blocks, get_sign
from .tensor_tools import split_tensor_symbol, target_index, first_matching_index
from tqdm import tqdm
from sympy import symbols, poly, Rational, factor, simplify, Poly
from math import factorial
from itertools import product
from functools import reduce
from hashable_containers import hmap,hlist


class MRational(object):
	"""The fraction field over ``MRing``

	AmpliTools represents tree amplitudes as MRationals. In the same spirit as MRing,
	MRational provides methods for reduction and manipulation of flavor and kinematics,
	while overloading the expected field operators, so MRationals are "transparently rational".

	"""

	def __init__(*_arg):
		self=_arg[0]
		arg=_arg[1:]
		#Define mathtype
		self.mathtype = "MRational"
		if len(arg)==1 and type(arg[0])==type(self):
			#Copy numerator and denominator to the local nd_list.
			#Numerator should be an MRing object, and denom should be dict of MRings.
			self.nd_list = hlist()
			for pair in arg[0].nd_list:
				denom_dict = hmap()
				for key in pair[1].keys():
					mykey = MRing(key)
					denom_dict[mykey] = pair[1][key]
				self.nd_list.append(hlist([MRing(pair[0]),denom_dict]))
		elif len(arg)==1 and hasattr(arg[0], '__iter__'):
			self.nd_list = hlist()
			for pair in arg[0]:
				denom_dict = hmap()
				for key in pair[1].keys():
					mykey = MRing(key)
					denom_dict[mykey] = pair[1][key]
				self.nd_list.append(hlist([MRing(pair[0]),denom_dict]))
		else:
			print(arg)
			assert False, "Bad argument to Operator constructor"


	def apply_to_numerators(self,function):
		nd_list = hlist()
		for pair in self.nd_list:
			target_numerator = function(pair[0])
			target_denominator = hmap()
			for key in pair[1].keys():
				target_key = MRing(key)
				assert (target_key.is_empty())==False
				target_denominator.setdefault(target_key,0)
				target_denominator[target_key]+=pair[1][key]
			if not target_numerator.is_empty():
				nd_list.append(hlist([target_numerator,target_denominator]))
		return MRational(nd_list)


	def apply_to_denominators(self,function):
		nd_list = hlist()
		for pair in self.nd_list:
			target_numerator = MRing(pair[0])
			target_denominator = hmap()
			for key in pair[1].keys():
				target_key = function(key)
				assert (target_key.is_empty())==False
				target_denominator.setdefault(target_key,0)
				target_denominator[target_key]+=pair[1][key]
			if not target_numerator.is_empty():
				nd_list.append(hlist([target_numerator,target_denominator]))
		return MRational(nd_list)


	def indices_from_symbol(self,symbol):
		symstring = symbol.__str__()
		if '_{' not in symstring:
			return []
		indexblock = symstring.split('{')[1].split('}')[0]
		indices = indexblock.split('f')[1:]
		indices = [int(index) for index in indices]
		return indices


	def p_terms(self,p,domain='QQ_I'):
		pterms = []
		for term in p.terms():
			coeff = term[1]
			expr = coeff
			for symbol,power in zip(p.gens,term[0]):
				expr*=symbol**power
			pterm = poly(expr,p.gens,domain=domain)
			pterms.append(pterm)
		return pterms


	def canonize_bound_indices(self,bound_prefixes):
		nd_list = hlist()
		for pair in self.nd_list:
			denominator_index_counts = {}
			denominator_index_map = {}
			index_counter = 1
			#Extract free symbols from denominator factors.
			for factor,power in pair[1].items():
				for mkey,poly in factor.mdict.items():
					for monom in poly.monoms():
						for symbol,count in zip(poly.gens,monom):
							head,prefixes,indices = split_tensor_symbol(symbol)
							for prefix,index in zip(prefixes,indices):
								if prefix in bound_prefixes:
									denominator_index_counts.setdefault((prefix,index),0)
									denominator_index_counts[(prefix,index)]+=count*power

			#Iterate over numerator terms and detect index multiplets.
			#add multiplet indices to the index_map.
			for mkey,poly in pair[0].mdict.items():
				for monom in poly.monoms():
					my_index_counts = dict(denominator_index_counts)
					for symbol,count in zip(poly.gens,monom):
						head,prefixes,indices = split_tensor_symbol(symbol)
						for prefix,index in zip(prefixes,indices):
							if (prefix,index) in denominator_index_counts.keys():
								my_index_counts[(prefix,index)]+=count
					for ipair,count in my_index_counts.items():
						if count>1 and (ipair not in denominator_index_map.keys()):
							denominator_index_map[ipair]=index_counter
							index_counter+=1
			numerator_terms = []
			#Numerator-only contractions.
			for mkey,poly in pair[0].mdict.items():
				for pterm in self.p_terms(poly):
					my_index_counts = {}
					my_index_counter = index_counter
					my_index_map = {}
					s = MRing(hmap({mkey:pterm}))
					monom = pterm.monoms()[0]
					for symbol,count in zip(pterm.gens,monom):
						head,prefixes,indices = split_tensor_symbol(symbol)
						for prefix,index in zip(prefixes,indices):
							if (prefix,index) not in denominator_index_counts.keys():
								if prefix in bound_prefixes:
									my_index_counts.setdefault((prefix,index),0)
									my_index_counts[(prefix,index)]+=count
					for ipair,count in my_index_counts.items():
						if count>1 and (ipair not in my_index_map.keys()):
							s = s.tensor_index_replacement(ipair[1],my_index_counter,
								source_prefix=ipair[0],target_prefix=ipair[0])
							my_index_counter+=1
					if not s.is_empty():
						numerator_terms.append(s)
			numerator = reduce(lambda x,y: x+y, numerator_terms)
			#Make the replacments laid out in index_map.
			q = MRational(hlist([hlist([numerator,pair[1]]),]))
			for ipair,target in denominator_index_map.items():
					q = q.tensor_index_replacement(ipair[1],target,source_prefix=ipair[0],
												   target_prefix=ipair[0])
			nd_list+=q.nd_list
		rat = MRational(nd_list)
		rat = rat.collect()
		return rat


	def canonize_indices(self,max_ext_label):
		"""
		Older method. Now trying to explicitly denote bound indices at point
		of user entry with one of bound_prefixes prefixes.
		"""
		nd_list = hlist()
		for pair in self.nd_list:
			denominator_index_counts = {}
			denominator_index_map = {}
			index_counter = 1
			#Extract free symbols from denominator factors.
			for factor,power in pair[1].items():
				for mkey,poly in factor.mdict.items():
					for monom in poly.monoms():
						for symbol,count in zip(poly.gens,monom):
							for index in self.indices_from_symbol(symbol):
								denominator_index_counts.setdefault(index,0)
								denominator_index_counts[index]+=count*power
			#Iterate over numerator terms and detect index multiplets.
			#add multiplet indices to the index_map.
			for mkey,poly in pair[0].mdict.items():
				for monom in poly.monoms():
					my_index_counts = dict(denominator_index_counts)
					for symbol,count in zip(poly.gens,monom):
						for index in self.indices_from_symbol(symbol):
							if index in denominator_index_counts.keys():
								my_index_counts[index]+=count
					for index,count in my_index_counts.items():
						if count>1 and (index not in denominator_index_map.keys()) and (index>max_ext_label):
							denominator_index_map[index]=index_counter
							index_counter+=1
			numerator_terms = []
			#Numerator-only contractions.
			for mkey,poly in pair[0].mdict.items():
				for pterm in self.p_terms(poly):
					my_index_counts = {}
					my_index_counter = index_counter
					my_index_map = {}
					s = MRing(hmap({mkey:pterm}))
					monom = pterm.monoms()[0]
					for symbol,count in zip(pterm.gens,monom):
						for index in self.indices_from_symbol(symbol):
							if index not in denominator_index_counts.keys():
								my_index_counts.setdefault(index,0)
								my_index_counts[index]+=count
					for index,count in my_index_counts.items():
						if count>1 and (index not in my_index_map.keys()) and (index>max_ext_label):
							s = s.tensor_index_replacement(index,my_index_counter,target_prefix='b')
							my_index_counter+=1
					if not s.is_empty():
						numerator_terms.append(s)
			numerator = reduce(lambda x,y: x+y, numerator_terms)
			#Make the replacments laid out in index_map.
			q = MRational(hlist([hlist([numerator,pair[1]]),]))
			for source,target in denominator_index_map.items():
					q = q.tensor_index_replacement(source,target,target_prefix='b')
			nd_list+=q.nd_list
		rat = MRational(nd_list)
		rat = rat.collect()
		return rat


	def product_replacement_clean_indices(self):
		target_rat = MRational(hlist())
		#for pair in tqdm(self.nd_list,desc="PRCI"):
		for pair in self.nd_list:
			#First, tabulate all indices.
			index_dict_list = [pair[0].index_dict()]
			for r in pair[1].keys():
				index_dict_list.append(r.index_dict())
			#Now, unify index_dict_list
			index_dict = {}
			for idict in index_dict_list:
				for key,val in idict.items():
					index_dict.setdefault(key,set())
					index_dict[key] = index_dict[key] | idict[key]
			#Replace d,g with z,w respectively, avoiding index collisions within
			#the nd_pair.
			rat_term = MRational(hlist([hlist([pair[0],pair[1]]),]))
			pairmap = {'d':'z','g':'w','k':'b'}
			for source,target in pairmap.items():
				if source in index_dict.keys():
					if target in index_dict.keys():
						start = max(index_dict[target]) + 1
					else:
						start = 1
					for n,index in enumerate(index_dict[source]):
						rat_term = rat_term.tensor_index_replacement(index,start+n,
							source_prefix=source,target_prefix=target)
			target_rat+=rat_term
		return target_rat


	def first_matching_pair_index(self,pair,pattern,match):
		#First, check the numerator.
		for poly in pair[0].mdict.values():
			index = first_matching_index(poly,pattern,match)
			if index!=False:
				return index
		#Check each MRing in the denominator.
		for r in pair[1].keys():
			for poly in r.mdict.values():
				index = first_matching_index(poly,pattern,match)
				if index!=False:
					return index
		return False


	def expand_free_index(self,pattern,targets,match):
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

		return self.branch_free_index(pattern,targets,match)


	def branch_free_index(self,pattern,targets,match):
		target_rats = []
		#Every term must have the same free indices, by definition.
		#Therefore, we only need to search for matches in the first term.
		pattern_index = self.first_matching_pair_index(self.nd_list[0],pattern,
													match)
		if pattern_index == False:
			target_rats.append(self)
		else:
			for target in targets:
				target_index = target_index(pattern_index,target,match)
				working_rat = self.tensor_index_replacement(
										pattern_index[1],target_index[1],
										source_prefix=pattern_index[0],
										target_prefix=target_index[0])
				target_rats += working_rat.branch_free_index(pattern,targets,
														   match)
		return target_rats


	def expand_bound_index(self,pattern,targets,match):
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
		expanded_rat = self.expand()
		target_rat = expanded_rat.branch_bound_index(pattern,targets,match)
		#We've suppressed collection and zero-culling throughout the recursion.
		#We'll apply these simplifications once to the final product.
		#__add__ performs these automatically, so we simply add "target+0".
		return target_rat + MRational(hlist())


	def branch_bound_index(self,pattern,targets,match):
		target_rat = MRational(hlist())
		for pair in self.nd_list:
			pattern_index = self.first_matching_pair_index(pair,pattern,match)
			if pattern_index==False:
				target_rat.nd_list.append(pair)
			else:
				working_rat = MRational(hlist())
				for target in targets:
					target_index = target_index(pattern_index,target,match)
					term_rat = MRational(hlist([hlist(pair),])).tensor_index_replacement(
											pattern_index[1],target_index[1],
											source_prefix=pattern_index[0],
											target_prefix=target_index[0])
					working_rat = working_rat.add(term_rat,collect=False)
				target_rat = target_rat.add(working_rat.branch_bound_index(
										   pattern,targets,match),collect=False)
		return target_rat


	def apply_to_mring(self,function):
		r = self.apply_to_numerators(function)
		r = r.apply_to_denominators(function)
		return r


	def onshell(self,basis):
		return self.apply_to_mring(basis.onshell_restriction)


	def evaluate_deltas(self):
		return self.apply_to_mring(lambda r: MRing.evaluate_deltas(r))


	def eject_masses(self,basis):
		return self.apply_to_mring(basis.eject_masses)


	def zero_masses(self,masses,basis):
		return self.apply_to_mring(lambda r: basis.zero_masses(r,masses))


	def group_masses(self,massmap,basis):
		return self.apply_to_mring(lambda r: basis.group_masses(r,massmap))


	def zero_index(self,index):
		return self.apply_to_mring(lambda r: MRing.zero_index(r,index))


	def evaluate_poly(self,symbol,value):
		return self.apply_to_mring(lambda r: MRing.evaluate_poly(r,symbol,value))


	def set_domain(self,domain='QQ_I'):
		return self.apply_to_mring(lambda r: MRing.set_domain(r,domain))


	def sort_generators(self):
		return self.apply_to_mring(lambda r: MRing.sort_generators(r))


	def kinematic_replacement(self,rmap):
		return self.apply_to_mring(lambda r: MRing.replacement(r,rmap))


	def link(self,linkpairs):
		return self.apply_to_mring(lambda r: MRing.link(r,linkpairs))


	def replace_monomial(self,monomial,target_poly):
		return self.apply_to_mring(lambda r: MRing.replace_monomial(r,monomial,target_poly))


	def tensor_index_replacement(self,source,target,source_prefix='f',target_prefix='f'):
		return self.apply_to_mring(lambda r: MRing.tensor_index_replacement(r,source,
												target,source_prefix,target_prefix))


	def tensor_prefix_replacement(self,source_prefix,target_prefix):
		return self.apply_to_mring(lambda r: MRing.tensor_prefix_replacement(r,
												source_prefix,target_prefix))


	def tensor_product_replacement(self,source,target,max_depth=False):
		#can't apply this when the bound indices show up in the denominator too!
		#return self.apply_to_mring(lambda r: MRing.tensor_product_replacement(r,source,target))
		#Assuming no color tensors (except masses) in denominators.
		#max_depth=1
		count=0
		last_rat = MRational(self)
		while True:
			next_rat = last_rat.apply_to_numerators(lambda r: MRing.tensor_product_replacement(r,source,target))
			next_rat = next_rat.product_replacement_clean_indices()
			count+=1
			if next_rat==last_rat:
				return next_rat
			if max_depth!=False and count==max_depth:
				return next_rat
			last_rat = MRational(next_rat)


	def block_replacement(self,blockmap,symbolblocks,source_prefix='f',target_prefix='f'):
		return self.apply_to_mring(lambda r: MRing.block_replacement(r,blockmap,
												symbolblocks,source_prefix=source_prefix,
												target_prefix=target_prefix))


	def contract_deltas(self,deltahead):
		return self.apply_to_mring(lambda r: MRing.contract_deltas(r,deltahead))


	def factor_polynomials(self):
		return self.apply_to_numerators(lambda r: MRing.factor_polynomials(r))


	def simplify_polynomials(self):
		return self.apply_to_numerators(lambda r: MRing.simplify_polynomials(r))


	def dress_momentum_pairs(self,symbol,longitudinal_modes):
		return self.apply_to_mring(lambda r: MRing.dress_momentum_pairs(r,symbol,
												longitudinal_modes))


	def dress_momentum(self,symbol,label):
		return self.apply_to_mring(lambda r: MRing.dress_momentum(r,symbol,label))


	def sort_indices(self,tensor_symmetries):
		return self.apply_to_mring(lambda r: MRing.sort_indices(r,tensor_symmetries))


	def permute_blocks(self,perm,symbolblocks,signed=False):
		return self.apply_to_mring(lambda r: permute_blocks(r,perm,symbolblocks,signed))


	def __mul__(self,other):
		#Convolution?
		dummy = symbols('dummy')
		ptype = type(poly(dummy,dummy,domain='QQ_I'))
		mrtype = type(MRing(hmap()))
		domain_types = [ptype,mrtype,int]
		is_sympy_number = ("sympy.core" in str(type(other)))
		assert (type(other) in domain_types) or is_sympy_number or (type(other) == type(self))
		if ((type(other) in domain_types) or is_sympy_number):
			r = self.apply_to_numerators(lambda r: MRing.__mul__(r,other))
			return r
		else:
			prodterms = hlist(product(self.nd_list,other.nd_list))
			prod_nd_list=hlist()
			for term in prodterms:
				#multiply numerator mrings
				numerator = term[0][0]*term[1][0]
				#combine the denominators
				denominator = hmap()
				for key in term[0][1]:
					denominator.setdefault(key,0)
					denominator[key]+=term[0][1][key]
				for key in term[1][1]:
					denominator.setdefault(key,0)
					denominator[key]+=term[1][1][key]
				prod_nd_list.append(hlist([numerator,denominator]))
			r = MRational(prod_nd_list)
			return r.collect()


	def orbit(self,perms,symbolblocks,signed=False,prefix='f'):
		orbit = MRational(hlist())
		#for perm in tqdm(perms,desc="Permutations"):
		for perm in perms:
			#for i in tqdm([1,],desc="ApplyPerms"):
			#for i in tqdm([1,],desc="ApplyPerms"):
			permrat = self.apply_to_mring(lambda r: permute_blocks(r,perm,symbolblocks,
				signed=False,source_prefix=prefix,target_prefix=prefix))
			if signed:
				permrat=permrat*get_sign(perm)
			#orbit = orbit.add(permrat,collect=False)
			#for i in tqdm([1,],desc="add_in_place"):
			orbit.add_in_place(permrat)
		#for i in tqdm([1,],desc="collection"):
		orbit = orbit.collect()
		orbit = orbit.cull_zeros()
		return orbit


	def tensor_product_rational_replacement(self,source,target,max_depth=False):
		#The problem is equivalent to subgraph isomorphism, which has much
		#quicker implementations. need to overhaul this code to interpret a
		#fast isomorphism algorithm.
		#can't apply this when the bound indices show up in the denominator too!
		#return self.apply_to_mring(lambda r: MRing.tensor_product_replacement(r,source,target))
		#Assuming no color tensors (except masses) in denominators.
		#max_depth=1
		assert len(source.as_dict())==1, "Source is not monomial."
		source_term = source.as_dict().keys()[0]
		mring = MRing(self.nd_list[0][0])
		source = mring.poly_tensor_prefix_replacement(source,'x','a')
		source = mring.poly_tensor_prefix_replacement(source,'y','c')
		source = mring.poly_tensor_prefix_replacement(source,'z','d')
		source = mring.poly_tensor_prefix_replacement(source,'w','g')
		source = mring.poly_tensor_prefix_replacement(source,'f','h')
		source = mring.poly_tensor_prefix_replacement(source,'b','k')
		target = target.tensor_prefix_replacement('x','a')
		target = target.tensor_prefix_replacement('y','c')
		target = target.tensor_prefix_replacement('z','d')
		target = target.tensor_prefix_replacement('w','g')
		target = target.tensor_prefix_replacement('f','h')
		target = target.tensor_prefix_replacement('b','k')
		last_rat = MRational(self)
		next_rat = MRational(hlist())
		count=0
		while True:
			loopmatch=False
			for num,den in last_rat.nd_list:
				for key,p in num.mdict.items():
					for term,coefficient in p.as_dict().items():
						result = num.match_term(term,p.gens,source_term,source.gens,0,{})
						if result==False:
							target_term = Poly({tuple(term):coefficient},p.gens,domain="QQ_I")
							ring_term = MRing(hmap({key:target_term}))
							rat_term = MRational(hlist([hlist([ring_term,den]),]))
							next_rat += rat_term
						else:
							loopmatch=True
							stripped_term,xmap = result
							target_term = Poly({tuple(stripped_term):coefficient},p.gens,
											   domain="QQ_I")
							ring_target = MRing(hmap({key:target_term}))
							rat_target = MRational(hlist([hlist([ring_target,den]),]))
							rat_target*=target
							for source_pair,target_pair in xmap.items():
								rat_target = rat_target.tensor_index_replacement(source_pair[1],
												target_pair[1],source_prefix = source_pair[0],
												target_prefix = target_pair[0])
							next_rat += rat_target
			next_rat = next_rat.product_replacement_clean_indices()
			count+=1
			if not loopmatch:
				return last_rat
			if max_depth!=False and count==max_depth:
				assert False
			last_rat = MRational(next_rat)


	def dress_masses(self,mass_symbols):
		q = MRational(self.nd_list)
		z = symbols('z')
		for symbol in mass_symbols:
			target = poly(symbol*z,symbol,z,domain='QQ_I')
			q = q.replace_monomial(symbol,target)
		return q,z


	def dress_all_masses(self,symbol):
		target_rat = MRational(hlist())
		mass_symbols = []
		#for pair in tqdm(self.nd_list,desc="PRCI"):
		for pair in self.nd_list:
			#First, tabulate all indices.
			for key,polynomial in pair[0].mdict.items():
				for gen in polynomial.gens:
					if 'm_{' in gen.__str__():
						if 'm'==gen.__str__().split('_{'):
							mass_symbols.append(gen)
			for r in pair[1].keys():
				for key,polynomial in r.mdict.items():
					for gen in polynomial.gens:
						if 'm_{' in gen.__str__():
							if 'm'==gen.__str__().split('_{')[0]:
								mass_symbols.append(gen)
		mass_symbols = list(set(mass_symbols))
		q = MRational(self.nd_list)
		for sym in mass_symbols:
			target = poly(sym*symbol,sym,symbol,domain='QQ_I')
			q = q.replace_monomial(sym,target)
		return q


	def partial_derivative(self,symbol):
		nd_list = hlist()
		for pair in self.nd_list:
			N = pair[0].partial_derivative(symbol)
			D = hmap()
			for key,val in zip(pair[1].keys(),pair[1].values()):
				mykey = MRing(key)
				D[mykey] = val
			if not N.is_empty():
				nd_list.append(hlist([N,D]))
			for key,val in zip(pair[1].keys(),pair[1].values()):
				N = MRing(pair[0])
				N *= key.partial_derivative(symbol)
				N *= (-1)*val
				D = hmap()
				for newkey,newval in zip(pair[1].keys(),pair[1].values()):
					mykey = MRing(newkey)
					D[mykey] = newval
				D[key]+=1
				if not N.is_empty():
					nd_list.append(hlist([N,D]))
		return MRational(nd_list)


	def maclaurin_coefficient(self,symbol,order):
		q = MRational(self.nd_list)
		for i in range(order):
			q = q.partial_derivative(symbol)
		q = q.evaluate_poly(symbol,0)
		q *= Rational(1,factorial(order))
		return q

	#---------Below, we are computing laurent coefficients around a complex-infinite pole----------#

	def compute_w_rational(self,denominator,z,w):
		w_denominator = hmap()
		m = 0
		for mring,power in denominator.items():
			w_mring = MRing(hmap())
			degrees = []
			for polynomial in mring.mdict.values():
				#newsymbols =  set(list(polynomial.free_symbols)+[z,])
				newsymbols =  set(list(polynomial.gens)+[z,])
				newpoly = poly(polynomial.as_expr(),newsymbols,domain='QQ_I')
				degrees.append(newpoly.degree(gen=z))
			my_m = max(degrees)
			m += my_m*power
			for mbasis,polynomial in mring.mdict.items():
				w_polynomial = polynomial.zero
				new_symbols = set(list(polynomial.gens)+[z,])
				new_poly = poly(polynomial.as_expr(),new_symbols,domain='QQ_I')
				degree = new_poly.degree(gen=z)
				for p in range(0,degree+1):
					coeff = Rational(1,factorial(p))*(new_poly.diff((z,p))).eval(z,0)
					coeff = poly(coeff,new_symbols,domain='QQ_I')
					w_power = my_m-p
					new_symbols = set(list(coeff.gens)+[w,])
					newterm = poly(coeff.as_expr()*w**w_power,new_symbols,domain='QQ_I')
					w_polynomial += newterm
				w_mring += MRing(hmap({mbasis:w_polynomial}))
			w_denominator[w_mring] = power
		w_rational = MRational(hlist([hlist([w_mring.one(),w_denominator]),]))
		return w_rational,m


	def compute_diff_map(self,numerator,z,w,w_rational,zeta,m):
		degrees = []
		for polynomial in numerator.mdict.values():
			newsymbols =  set(list(polynomial.gens)+[z,])
			newpoly = poly(polynomial.as_expr(),newsymbols,domain='QQ_I')
			degrees.append(newpoly.degree(gen=z))
		max_z_power = max(degrees)
		diff_map = hmap()
		for i in range(0,max_z_power+1):
			index = i-zeta-m
			if index<0:
				continue
			w_partial = MRational(w_rational)
			for p in range(index):
				w_partial = w_partial.partial_derivative(w)
			diff_map[index] = w_partial.evaluate_poly(w,0)
		return diff_map


	def is_symbol_in_polys(self,symbol):
		"""
		Check whether [symbol] is contained in any of the gens sets
		of the polynomials contained within this MRational object. Return True
		if so, False if not.
		"""
		for pair in self.nd_list:
			for polynomial in pair[0].mdict.values():
				#if symbol in polynomial.free_symbols:
				if symbol in polynomial.gens:
					return True
			for mring in pair[1].keys():
				for polynomial in mring.mdict.values():
					#if symbol in polynomial.free_symbols:
					if symbol in polynomial.gens:
						return True
		return False


	def uv_laurent_coefficient(self,z,zeta):
		"""
		Compute the MRational Laurent coefficient of self at order [zeta]
		in terms of the complex variable [z].
		"""
		# w = 1/z, so the UV pole is mapped to a pole at the origin in
		# the w-plane.
		w = symbols('w')
		assert not self.is_symbol_in_polys(w), "w symbol is already in use!"
		laurent_coeff = MRational(hlist())
		for pair in tqdm(self.nd_list,desc="Computing UV Laurent Coefficient"):
			numerator = pair[0]
			denominator = pair[1]
			# Compute the inverse denominator, multiplied by w**m, so that
			# it is regular in the neighborhood of the origin.
			w_rational, m = self.compute_w_rational(denominator,z,w)
			# Compute various partial derivatives of w_rational, evaluated
			# at w=0. These are precomputed in the interest of efficiency,
			# and will be accessed repeatedly in the loops below.
			diff_map = self.compute_diff_map(numerator,z,w,w_rational,zeta,m)
			for mbasis,polynomial in numerator.mdict.items():
				new_symbols = set(list(polynomial.gens)+[z,])
				new_poly = poly(polynomial.as_expr(),new_symbols,domain='QQ_I')
				for p in range(0,new_poly.degree(gen=z)+1):
					index = p-zeta-m
					if index<0:
						continue
					coeff = Rational(1,factorial(p))*(new_poly.diff((z,p))).eval(z,0)
					term = MRational(diff_map[index])
					term *= Rational(1,factorial(index))
					term *= MRing(hmap({mbasis:numerator.poly_one()}))
					term *= coeff
					laurent_coeff += term
		laurent_coeff = laurent_coeff.collect()
		return laurent_coeff

	#---------Below, we are computing laurent coefficients around the origin----------#

	def compute_z_rational(self,denominator,z):
		z_denominator = hmap()
		m = 0
		for mring,power in denominator.items():
			z_mring = MRing(hmap())
			degrees = []
			for polynomial in mring.mdict.values():
				newsymbols =  set(list(polynomial.gens)+[z,])
				newpoly = poly(polynomial.as_expr(),newsymbols,domain='QQ_I')
				degrees.append(newpoly.degree(gen=z))
			my_m = min(degrees)
			m += my_m*power
			for mbasis,polynomial in mring.mdict.items():
				z_polynomial = mring.poly_zero()
				new_symbols = set(list(polynomial.gens)+[z,])
				new_poly = poly(polynomial.as_expr(),new_symbols,domain='QQ_I')
				degree = new_poly.degree(gen=z)
				for p in range(0,degree+1):
					coeff = Rational(1,factorial(p))*(new_poly.diff((z,p))).eval(z,0)
					co_poly = poly(coeff,new_symbols,domain='QQ_I')
					z_power = p-my_m
					newterm = poly(co_poly.as_expr()*z**z_power,new_symbols,domain='QQ_I')
					z_polynomial += newterm
				z_mring += MRing(hmap({mbasis:z_polynomial}))
			z_denominator[z_mring] = power
		z_rational = MRational(hlist([hlist([z_mring.one(),z_denominator]),]))
		return z_rational,m


	def compute_ir_diff_map(self,numerator,z,z_rational,zeta,m):
		degrees = []
		for polynomial in numerator.mdict.values():
			newsymbols =  set(list(polynomial.gens)+[z,])
			newpoly = poly(polynomial.as_expr(),newsymbols,domain='QQ_I')
			degrees.append(newpoly.degree(gen=z))
		max_z_power = max(degrees)
		diff_map = hmap()
		for i in range(0,max_z_power+1):
			index = zeta+m-i
			if index<0:
				continue
			z_partial = MRational(z_rational)
			for p in range(index):
				z_partial = z_partial.partial_derivative(z)
			diff_map[index] = z_partial.evaluate_poly(z,0)
		return diff_map


	def ir_laurent_coefficient(self,z,zeta):
		"""
		Compute the MRational Laurent coefficient of self at order [zeta]
		in terms of the complex variable [z].
		"""
		laurent_coeff = MRational(hlist())
		for pair in tqdm(self.nd_list,desc="Computing IR Laurent Coefficient"):
			numerator = pair[0]
			denominator = pair[1]
			# Compute the inverse denominator, divided by z**m, so that
			# it is regular in the neighborhood of the origin.
			z_rational, m = self.compute_z_rational(denominator,z)
			# Compute various partial derivatives of z_rational, evaluated
			# at z=0. These are precomputed in the interest of efficiency,
			# and will be accessed repeatedly in the loops below.
			diff_map = self.compute_ir_diff_map(numerator,z,z_rational,zeta,m)
			for mbasis,polynomial in numerator.mdict.items():
				new_symbols = set(list(polynomial.gens)+[z,])
				new_poly = poly(polynomial.as_expr(),new_symbols,domain='QQ_I')
				for p in range(0,new_poly.degree(gen=z)+1):
					index = zeta+m-p
					if index<0:
						continue
					coeff = Rational(1,factorial(p))*(new_poly.diff((z,p))).eval(z,0)
					term = MRational(diff_map[index])
					term *= Rational(1,factorial(index))
					term *= MRing(hmap({mbasis:numerator.poly_one()}))
					term *= coeff
					laurent_coeff += term
		laurent_coeff = laurent_coeff.collect()
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


	def full_string(self):
			string = ''
			for pair in self.nd_list:
				string+="NUM\n"
				string+=pair[0].full_string()
				string+="DEN\n"
				for key in pair[1].keys():
					string+=key.full_string()
					string+=pair[1][key].__str__()+'\n'
				string+='\n'
			return string


	def unify_generators(self):
		if len(self.nd_list)==0:
			return MRational(self),None
		r,upoly = self.nd_list[0][0].unify_generators()
		for pair in self.nd_list:
			r,upoly = pair[0].unify_generators(upoly)
			for key in pair[1].keys():
				r,upoly = key.unify_generators(upoly)
		new_nd_list = hlist()
		for pair in self.nd_list:
			numerator,mypoly = pair[0].unify_generators(upoly)
			denominator = hmap()
			for key,val in pair[1].items():
				denkey,mypoly = key.unify_generators(upoly)
				denominator[denkey] = val
			new_nd_list.append(hlist([numerator,denominator]))
		return MRational(new_nd_list),upoly


	def cull_zeros(self):
		clean_nd_list = hlist()
		for pair in self.nd_list:
			if not pair[0].is_empty():
				clean_nd_list.append(pair)
		return MRational(clean_nd_list)


	def add(self,other,collect=True):
		nd_list = self.nd_list + other.nd_list
		r = MRational(nd_list)
		if collect:
			r = r.collect()
			r = r.cull_zeros()
		return r


	def add_in_place(self,other):
		self.nd_list += other.nd_list


	def __add__(self,other):
		return self.add(other,collect=True)


	def __sub__(self,other):
		return self+(other*-1)


	def __eq__(self,other):
		assert type(self)==type(other)
		diff = self-other
		#Set domain to QQ_I
		diff = diff.set_domain()
		diff = diff.sort_generators()
		diff = diff.collect()
		if len(diff.nd_list)==0:
			return True
		else:
			return False


	def __ne__(self,other):
		return not self.__eq__(other)

	
	def __lt__(self,other):
		self_sort = sorted([tuple(pair) for pair in self.nd_list])
		other_sort = sorted([tuple(pair) for pair in other.nd_list])
		return self_sort < other_sort


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


	def collect(self):
		r = MRational(self)
		#Simplify ones in the denominator and clear zeros in the numerator.
		new_nd_list = hlist()
		#for n,d in tqdm(r.nd_list,desc="Simplify ones and zeros"):
		for n,d in r.nd_list:
			new_denom = hmap()
			for key in d.keys():
				assert not key.is_empty()
				if len(key.mdict)==1 and list(key.mdict.keys())[0]==((0,0),) and list(key.mdict.values())[0] == list(key.mdict.values())[0].one:
					if len(d.keys())==1:
						new_denom[key]=1
					else:
						pass
				else:
					new_denom[key] = d[key]
			if not n.is_empty():
				new_nd_list.append(hlist([n,new_denom]))
		r.nd_list = new_nd_list

		#collect numerators with common denominators.
		nd_keys = []
		nd_vals = []
		for pair in r.nd_list:
			nd_keys.append(pair[1])
			nd_vals.append(pair[0])

		key_index=0
		#for i in tqdm([1,],desc="collect common denom"):
		while True:
			while True:
				if key_index>=len(nd_keys)-1:
					break
				#for i in tqdm([1,],desc="is_dict_in_list"):
				inret = is_dict_in_list(nd_keys[key_index],nd_keys[key_index+1:])
				if inret<0:
					break
				second_index = inret+key_index+1

				#for i in tqdm([1,],desc="adding"):
				#nd_vals[key_index]+=nd_vals[second_index]
				nd_vals[key_index].add_in_place(nd_vals[second_index])

				#for i in tqdm([1,],desc="DoubleDel"):
				del nd_keys[second_index]
				del nd_vals[second_index]
			key_index+=1
			if key_index>=len(nd_keys)-1:
				break

		#for i in tqdm([1,],desc="Instantiate MRational"):
		r = MRational(hlist(list(zip(nd_vals,nd_keys))))

		#Clear zeros in the numerator.
		#for i in tqdm([1,],desc="Clear num zeros"):
		new_nd_list = hlist()
		for n,d in r.nd_list:
			nr = MRing(n)
			nr.cull_zeros()
			if not nr.is_empty():
				new_nd_list.append(hlist([nr,d]))
		r = MRational(new_nd_list)
		#for i in tqdm([1,],desc="clear_signs"):
		r = self.clear_signs(r)

		return r
		#return self.clear_signs(r)


	def clear_signs(self,r):
		rat = MRational(r)
		for n,pair in enumerate(r.nd_list):
				for ring,power in pair[1].items():
					firstkey = sorted(ring.mdict.keys())[0]
					poly_RO = ring.mdict[firstkey].reorder(*sorted(ring.mdict[firstkey].gens,key=str))
					if poly_RO.coeffs()[0]<0:
						rat.nd_list[n][0]=rat.nd_list[n][0]*(-1)**power
						del rat.nd_list[n][1][ring]
						rat.nd_list[n][1][ring*(-1)]=power
		return rat


	def expand(self):
		target_rat = MRational(hlist())
		for pair in self.nd_list:
			for key,poly in pair[0].mdict.items():
				for pterm in pair[0].p_terms(poly):
					term_ring = MRing(hmap({key:pterm}))
					term_rat = MRational(hlist([hlist([term_ring,pair[1]]),]))
					target_rat = target_rat.add(term_rat,collect=False)
		return target_rat


	def merge_denominators(self):
		"""
		This is a refined function which deals directly with pole factors instead
		of expanded MRings. This way, we can multiply by the polynomial LCD to
		merge denominators, cutting down on the ultimate size of the numerator
		MRing, which balloons exponentially in the number of poles merged.

		First, we must compute the LCD. We will iterate through all poles in each
		denominator in the nd_list, and record the maximum power of each given
		pole. The dict containing all distinct poles and their maximum powers is
		the polynomial LCD.

		"""

		LCD = hmap()
		#for pair in tqdm(self.nd_list,desc="Finding LCD"):
		for pair in self.nd_list:
			for pole,multiplicity in zip(pair[1].keys(),pair[1].values()):
				LCD.setdefault(pole,multiplicity)
				if LCD[pole]<multiplicity:
					LCD[pole] = multiplicity

		# Now we iterate through the numerators and multiply them by
		# the LCD mod the factors in the corresponding denominator.
		numerator = MRing(hmap())
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
		return MRational(hlist([hlist([numerator,LCD]),]))


	def free_symbols(self):
		"""
		Returns a `set` of all free symbols appearing in all polynomials
		in both the numerator and the denominator of each pair. Invokes
		`MRing.free_symbols()`.
		"""
		free_symbols = set()
		for pair in self.nd_list:
			# Start with the numerator
			free_symbols |= pair[0].free_symbols()
			# Now the denominator
			free_symbols |= reduce(lambda s,t: s|t, [r.free_symbols()
								   for r in pair[1].keys()])
		return free_symbols


	def tensor_symbols(self):
		"""
		Returns a `set` of all tensor symbols appearing in all polynomials
		in both the numerator and the denominator of each pair.
		First computes set of all free symbols using
		`MRations.tensor_symbols()`, then filters out all symbols without
		"_{" characters, indicative of a tensor index block.
		"""
		return set(filter(lambda s: "_{" in s.__str__(),self.free_symbols()))


	def find_prefix(self,prefix_key):
		"""
		Searches all numerator and denominator polynomials for tensors
		containing `prefix`. Returns a list of all indices carrying this
		prefix.
		"""
		tensor_symbols = self.tensor_symbols()
		keyed_indices = set()
		for tensym in tensor_symbols:
			head,prefix,index = split_tensor_symbol(tensym)
			keyed_indices |= {index[i] for i,v in enumerate(prefix)
							 if v==prefix_key}
		keyed_indices = sorted(list(keyed_indices))
		return keyed_indices


	def bound_indices_to_components(self,
								 first,
								 last,
								 bound_prefix='b',
								 target_prefix='x'):
		"""
		Expand bound indices into sums over component values, between `first`
		and `last` (inclusive). Only applies when no bound indicies appear in
		denominators.
		"""
		target_nd_list = hlist()
		for numerator,denominator in self.nd_list:
			target_numerator = numerator.bound_indices_to_components(first,last,
								bound_prefix,target_prefix)
			target_denominator = hmap()
			for r in denominator.keys():
				assert len(r.find_prefix(bound_prefix))==0, "Bound indices found in a denominator."
			target_nd_list.append(hlist([target_numerator,denominator]))
		return MRational(target_nd_list)


	def free_indices_to_components(self,
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
			source_rat = source_rat.tensor_index_replacement(
														source_index,
														target_index,
														source_prefix='f',
														target_prefix='x')
		return source_rat


	def free_tensor_elements(self,
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
		# tensor_symbols = self.tensor_symbols()
		# free_indices = set()
		# for tensym in tensor_symbols:
		#	 head,prefix,index = split_tensor_symbol(tensym)
		#	 free_indices |= {index[i] for i,v in enumerate(prefix)
		#					  if v==free_prefix}
		# free_indices = sorted(list(free_indices))
		free_indices = self.find_prefix(free_prefix)


		# Next, compute a dictionary mapping all possible combinations of
		# external index components to `MRational` objects evaluated at those
		# components using `MRational.free_indices_to_components`.
		target_dict = {}
		#for tup in tqdm(list(product(range(first,last+1),repeat=len(free_indices))),desc="expanding over free index tuples (tuples)"):
		for tup in list(product(range(first,last+1),repeat=len(free_indices))):
			target_dict[tup] = self.free_indices_to_components(dict(zip(
																free_indices,
																tup)))

		return target_dict


	def numerator_polys(self):
		"""
		Returns a list of numerator polynomials.
		"""
		polys = []
		for pair in self.nd_list:
			polys += list(pair[0].mdict.values())
		return polys


#Some functions for comparison of denominator dictionaries (keyed by MRing objects).
def is_dict_contained(_x,_y):
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


def is_dict_equal(x,y):
	return is_dict_contained(x,y) and is_dict_contained(y,x)


def is_dict_in_list(x,l):
	if len(l)==0:
		return -1
	for i,y in enumerate(l):
		if is_dict_equal(x,y):
			return i
	return -1
