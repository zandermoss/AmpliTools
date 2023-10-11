import shelve
import functools
import itertools
from copy import deepcopy
import sys
from sympy import symbols, poly, Rational, simplify, factor, Poly
from sympy.polys.polytools import div,compose
from tqdm import tqdm
import sympy
from signed_permutations import SignedPermutation, SignedPermutationGroup
from tensor_tools import symbol_group_sort, split_tensor_symbol, index_match
from tensor_tools import first_matching_index, target_index, matching_indices
from functools import reduce
from hashable_containers import hmap


class MRing():

	def __init__(self,arg):
		#Define mathtype
		self.mathtype = "MRing"
		if type(arg) == type(hmap()):
			#Define the zero polynomial.
			if len(arg.keys())>0:
				polynomial = list(arg.values())[0]
				pzero = Poly(0,polynomial.gens,domain='QQ_I')
				#pzero = arg[arg.keys()[0]].zero
			self.mdict = hmap()
			for key in arg.keys():
				sortedkey = self.key_sort(key)
				self.mdict.setdefault(sortedkey,pzero)
				self.mdict[sortedkey]+=arg[key]
			self.cull_zeros()
		elif type(arg) == type(self) and arg.__class__.__name__ == self.__class__.__name__:
			#Define the zero polynomial.
			if len(arg.mdict.keys())>0:
				polynomial = list(arg.mdict.values())[0]
				pzero = Poly(0,polynomial.gens,domain='QQ_I')
				#pzero = arg.mdict[arg.mdict.keys()[0]].zero
			self.mdict = hmap()
			for key in arg.mdict.keys():
				sortedkey = self.key_sort(key)
				self.mdict.setdefault(sortedkey,pzero)
				self.mdict[sortedkey]+=arg.mdict[key]
			self.cull_zeros()
		else:
			sys.exit("Bad argument to MRing")


	def one(self):
		assert len(self.mdict)>0, "MRing is empty!"
		return MRing(hmap({((0,0),):self.poly_one()}))


	def poly_one(self):
		polynomial = list(self.mdict.values())[0]
		return Poly(1,polynomial.gens,domain='QQ_I')


	def poly_zero(self):
		polynomial = list(self.mdict.values())[0]
		return Poly(0,polynomial.gens,domain='QQ_I')


	def cull_zeros(self):
		for key in list(self.mdict.keys()):
			#Need second condition to get around strange bug.
			#if self.mdict[key].is_zero or self.mdict[key].as_expr().__str__()=='0':
			if self.mdict[key].is_zero:
				del self.mdict[key]


	def compare(self,pair1,pair2):
		if pair1[0]!=pair2[0]:
			return pair1[0]-pair2[0]
		else:
			return pair1[1]-pair2[1]


	def key_sort(self,edge):
		sortededge=[]
		for pair in range(len(edge)):
			sortededge.append(tuple(sorted(edge[pair])))
		sortededge = sorted(sortededge,key=functools.cmp_to_key(self.compare))
		return tuple(sortededge)


	def linear_coefficient_replacement_target_keys(self,key,mapkey,targets):
		active=[]
		inert = list(key)
		for pair in key:
			if mapkey in pair:
				active.append(pair)
				inert.remove(pair)
		images=[]
		for pair in active:
			image = []
			for i in range(2):
				if pair[i]==mapkey:
					for target in targets:
						temppair = list(pair)
						temppair[i] = target[1]
						image.append([target[0],tuple(temppair)])
			images.append(image)
		coefftargetkeys=[]
		for pair, ltargets in zip(active,images):
			workactive = list(active)
			workactive.remove(pair)
			for ltarget in ltargets:
				plist = list(inert)+list(workactive)
				plist.append(ltarget[1])
				newkey = self.key_sort(tuple(plist))
				coefftargetkeys.append([ltarget[0],newkey])
		return coefftargetkeys


	def coefficient_replacement_target_keys(self,key,mapkey,targets):
		active=[]
		inert = list(key)
		for pair in key:
			if mapkey in pair:
				active.append(pair)
				inert.remove(pair)
		images=[]
		for pair in active:
			image=[[Rational(1),list(pair)],]
			for i in range(2):
				nextimage=[]
				for p in image:
					if p[1][i] == mapkey:
						for target in targets:
							temppair = list(p[1])
							temppair[i] = target[1]
							coeff = p[0]*target[0]
							nextimage.append([coeff,tuple(temppair)])
					else:
						nextimage.append(p)
				image=nextimage
			images.append(image)
		prodlist = itertools.product(*images)
		coefftargetkeys = []
		for plist in prodlist:
			newplist=[]
			#Handling coeff
			coeff=Rational(1)
			for el in plist:
				coeff*=el[0]
				newplist.append(tuple(el[1]))
			newplist+=inert
			ptuple=self.key_sort(newplist)
			coefftargetkeys.append([coeff,ptuple])
		return coefftargetkeys


	def coefficient_pair_replacement_target_keys(self,key,mappair,targets):
		images=[]
		for pair in key:
			if pair == mappair:
				image = targets
			else:
				image = [[Rational(1),pair],]
			images.append(image)
		prodlist = itertools.product(*images)
		coefftargetkeys = []
		for plist in prodlist:
			newplist=[]
			#Handling coeff
			coeff=1
			for el in plist:
				coeff*=el[0]
				newplist.append(tuple(el[1]))
			ptuple=self.key_sort(newplist)
			coefftargetkeys.append([coeff,ptuple])
		return coefftargetkeys


	def pair_replacement_rule(self,rmap):
		newdict = hmap()
		mappair = tuple(sorted(list(list(rmap.keys())[0])))
		targets = rmap[list(rmap.keys())[0]]
		for key in self.mdict.keys():
			coefftargetkeys = self.coefficient_pair_replacement_target_keys(key, mappair,targets)
			for coeff,targetkey in coefftargetkeys:
				newdict.setdefault(targetkey,self.poly_zero())
				newdict[targetkey]+=coeff*self.mdict[key]
		return newdict


	def replacement_rule(self,rmap):
		newdict = hmap()
		mapkey = list(rmap.keys())[0]
		targets = rmap[mapkey]
		for key in self.mdict.keys():
			coefftargetkeys = self.coefficient_replacement_target_keys(key, mapkey,targets)
			for coeff,targetkey in coefftargetkeys:
				newdict.setdefault(targetkey,self.poly_zero())
				newdict[targetkey]+=coeff*self.mdict[key]
		return newdict


	def linear_replacement_rule(self,rmap):
		newdict = hmap()
		mapkey = list(rmap.keys())[0]
		targets = rmap[mapkey]
		for key in self.mdict.keys():
			coefftargetkeys = self.linear_coefficient_replacement_target_keys(key, mapkey,
																	 targets)
			for coeff,targetkey in coefftargetkeys:
				newdict.setdefault(targetkey,self.poly_zero())
				newdict[targetkey]+=coeff*self.mdict[key]
		return newdict


	def replacement(self,rmap):
		r = MRing(self)
		r.mdict = r.replacement_rule(rmap)
		r.cull_zeros()
		return r


	def linear_replacement(self,rmap):
		r = MRing(self)
		r.mdict = r.linear_replacement_rule(rmap)
		r.cull_zeros()
		return r


	def pair_replacement(self,rmap):
		r = MRing(self)
		r.mdict = r.pair_replacement_rule(rmap)
		r.cull_zeros()
		return r


	def zero_pair(self,pair):
		newdict = hmap([entry for entry in self.mdict.items() if pair not in entry[0]])
		return MRing(newdict)


	def zero_index(self,index):
		newdict = hmap([entry for entry in self.mdict.items() if index not in reduce(lambda x,y:x+y,entry[0])])
		return MRing(newdict)


	def block_replacement(self,blockmap,symbolblocks,source_prefix='f',target_prefix='f'):
		r = MRing(self)
		#Check that inputs are OK.
		assert len(symbolblocks) == len(blockmap)
		#Generate source->temp and temp->target maps.
		flatsymbols = sum(symbolblocks,[])
		map1 = {}
		map2 = {}
		index_map1 = {}
		index_map2 = {}
		for sym in flatsymbols:
			map1[sym]=sym+1000
			if sym<0:
				map2[sym+1000] = -blockmap[abs(sym)]
				index_map1[-sym] = -sym+1000
				index_map2[-sym+1000] = blockmap[-sym]

			elif (0<sym) and (sym<100):
				map2[sym+1000] = blockmap[sym]
			elif 100<sym:
				map2[sym+1000] = 100+blockmap[sym-100]
		#Send source to temp.
		for source,temp in zip(map1.keys(),map1.values()):
			r = r.replacement({source:[[1,temp]]})
		#Send temp to target.
		for temp,target in zip(map2.keys(),map2.values()):
			r = r.replacement({temp:[[1,target]]})

		# FIXME: handle cases in which tensor_index_replacement returns empty MRing
		for source,temp in index_map1.items():
			r = r.tensor_index_replacement(source,temp,source_prefix=source_prefix,
								  target_prefix=target_prefix)
		for temp,target in index_map2.items():
			r = r.tensor_index_replacement(temp,target,source_prefix=source_prefix,
								  target_prefix=target_prefix)

		return r


	def poly_tensor_prefix_index(self, _p, prefix):
		p = Poly(_p)
		for	symbol in list(p.free_symbols):
			symstring = symbol.__str__()
			if "_{" not in symstring:
				continue
			indexblock = list(symstring.split('{')[1].split('}')[0])
			while True:
				char = indexblock.pop(0)
				if char==prefix:
					index = ""
					while True:
						indexchar = indexblock.pop(0)
						if indexchar.isalpha():
							return int(index)
						if indexchar.isdigit() and len(indexblock)==0:
							index+=indexchar
							return int(index)
						else:
							index+=indexchar
				if len(indexblock)==0:
					break
		return False


	def index_dict(self):
		index_dict = {}
		for poly in self.mdict.values():
			poly_index_dict = self.poly_index_dict(poly)
			if len(poly_index_dict)==0:
				continue
			for prefix,index_set in poly_index_dict.items():
				index_dict.setdefault(prefix,set())
				index_dict[prefix] = index_dict[prefix] | index_set
		return index_dict


	def poly_index_dict(self,poly):
		index_dict = {}
		for symbol in poly.free_symbols:
			symstring = symbol.__str__()
			if "_{" not in symstring:
				continue
			indexblock = list(symstring.split('{')[1].split('}')[0])
			assert len(indexblock)>0, "Empty index block."
			char = indexblock.pop(0)
			while len(indexblock)>0:
				prefix = char
				assert prefix.isalpha(), "Prefix not alphabetic."
				istring = ""
				while len(indexblock)>0:
					char = indexblock.pop(0)
					if char.isalpha():
						break
					istring+=char
				index = int(istring)
				index_dict.setdefault(prefix,set())
				index_dict[prefix] = index_dict[prefix] | {index}
		return index_dict


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


	def poly_bound_prefixes(self,poly):
		bound_prefixes = []
		prefixes = []
		for symbol in poly.free_symbols:
			symstring = symbol.__str__()
			if "_{" not in symstring:
				continue
			indexblock = list(symstring.split('{')[1].split('}')[0])
			assert len(indexblock)>0, "Empty index block."
			char = indexblock.pop(0)
			while len(indexblock)>0:
				prefix = char
				assert prefix.isalpha(), "Prefix not alphabetic."
				istring = ""
				while len(indexblock)>0:
					char = indexblock.pop(0)
					if char.isalpha():
						break
					istring+=char
				index = int(istring)
				if (prefix,index) in prefixes:
					bound_prefixes.append(prefix)
				else:
					prefixes.append((prefix,index))
		bound_prefixes = list(set(bound_prefixes))
		if 'x' in bound_prefixes:
			bound_prefixes.remove('x')
		if 'y' in bound_prefixes:
			bound_prefixes.remove('y')
		return bound_prefixes


	def poly_canonize_bound_indices(self,poly):
		new_poly = Poly(0,poly.gens,domain="QQ_I")
		for pterm in self.p_terms(poly):
			new_pterm = Poly(pterm)
			bound_prefixes = self.poly_bound_prefixes(pterm)
			index_dict = self.poly_index_dict(pterm)
			for prefix in bound_prefixes:
				for n,index in enumerate(list(index_dict[prefix])):
					if index!=0:
						new_pterm = self.poly_tensor_index_replacement(new_pterm,
							index,n+1,source_prefix=prefix,target_prefix=prefix)
			new_poly+=new_pterm
		#FIXME: handle case in which poly_tensor_index_replacement has returned only zeros.
		return new_poly


	def canonize_bound_indices(self):
		r = MRing(self)
		for key in r.mdict.keys():
			r.mdict[key] = self.poly_canonize_bound_indices(r.mdict[key])
		return r


	def tensor_prefix_index(self, prefix):
		for key in self.mdict.keys():
			index = self.poly_tensor_prefix_index(self.mdict[key],prefix)
			if index!=False:
				return index
		return False


	def poly_tensor_prefix_check(self, _p, prefix):
		p = Poly(_p)
		for	symbol in list(p.free_symbols):
			symstring = symbol.__str__()
			if "_{" not in symstring:
				continue
			indexblock = symstring.split('{')[1].split('}')[0]
			if prefix in indexblock:
				return True
		return False


	def tensor_prefix_check(self, prefix):
		for key in r.mdict.keys():
			if self.poly_tensor_prefix_check(r.mdict[key]):
				return True
		return False


	def poly_tensor_prefix_replacement(self, _p, source_prefix, target_prefix):
		p = Poly(_p)
		p = p.exclude()
		#p = p.set_domain('QQ_I')
		for	symbol in list(p.free_symbols):
			symstring = symbol.__str__()
			if "_{" not in symstring:
				continue
			indexblock = symstring.split('{')[1].split('}')[0]
			prefix = symstring.split('{')[0]
			symstring = prefix+'{'+indexblock.replace(source_prefix,target_prefix)+'}'
			newsym = symbols(symstring)

			p=p.replace(symbol,newsym)

		return p


	def tensor_prefix_replacement(self, source_prefix, target_prefix):
		r = MRing(self)
		for key in r.mdict.keys():
			r.mdict[key] = self.poly_tensor_prefix_replacement(r.mdict[key],
													  source_prefix,target_prefix)
		return r


	def poly_tensor_index_replacement(self, _p, source, target, source_prefix='f',
										target_prefix='f'):
		#FIXME: Had to comment out the checks here for compatibility with sagemath...
		#assert type(source) == type(int(1))
		#assert type(target) == type(int(1))
		p = Poly(_p)
		p=p.exclude()

		if p == 0:
			return self.poly_zero()
		if p == 1:
			return self.poly_one()

		#p = p.set_domain('QQ_I')
		#freesyms = list(p.free_symbols)
		#freesyms = p.atoms(sympy.Symbol)
		freesyms = list(p.gens)
		for	symbol in freesyms:
			# p = self.loopi_boi(symbol, p,source,target,source_prefix,target_prefix)
			# if p == None:
			#	 continue
			symstring = symbol.__str__()
			if "_{" not in symstring:
				continue

			indexblock = list(symstring.split('{')[1].split('}')[0])
			prefix = symstring.split('{')[0]
			newstring = prefix+'{'

			indexstring=""
			lastprefix=indexblock.pop(0)
			while True:
				char = indexblock.pop(0)
				#if (char in ['f','b']):
				if char.isalpha():
					index = int(indexstring)
					if (index==source) and lastprefix==source_prefix:
						newstring+=target_prefix+str(target)
					else:
						newstring+=lastprefix+str(index)
					lastprefix=char
					indexstring=""
				elif len(indexblock)==0:
					indexstring+=char
					index = int(indexstring)
					if index==source and lastprefix==source_prefix:
						newstring+=target_prefix+str(target)
					else:
						newstring+=lastprefix+str(index)
					break
				else:
					indexstring+=char
			symstring = newstring+'}'

			newsym = symbols(symstring)
			# FIXME: poly.replace fails if the target symbol is already
			# present in the polynomial. Switching to poly.subs here, which
			# does not have this problem.
			#p = p.replace(symbol,newsym)
			p = p.subs(symbol,newsym)


		if p == 0:
			return self.poly_zero()
		if p == 1:
			return self.poly_one()

		p = Poly(p).set_domain('QQ_I')
		#p = Poly(p)
		return p


	def loopi_boi(self,symbol,p,source,target,source_prefix,target_prefix):
		symstring = symbol.__str__()
		if "_{" not in symstring:
			return None

		indexblock = list(symstring.split('{')[1].split('}')[0])
		prefix = symstring.split('{')[0]
		newstring = prefix+'{'

		indexstring=""
		lastprefix=indexblock.pop(0)
		while True:
			char = indexblock.pop(0)
			#if (char in ['f','b']):
			if char.isalpha():
				index = int(indexstring)
				if (index==source) and lastprefix==source_prefix:
					newstring+=target_prefix+str(target)
				else:
					newstring+=lastprefix+str(index)
				lastprefix=char
				indexstring=""
			elif len(indexblock)==0:
				indexstring+=char
				index = int(indexstring)
				if index==source and lastprefix==source_prefix:
					newstring+=target_prefix+str(target)
				else:
					newstring+=lastprefix+str(index)
				break
			else:
				indexstring+=char
		symstring = newstring+'}'

		newsym = symbols(symstring)
		# FIXME: poly.replace fails if the target symbol is already
		# present in the polynomial. Switching to poly.subs here, which
		# does not have this problem.
		# p = p.replace(symbol,newsym)

		p = p.subs(symbol,newsym)

		if p == 0:
			return self.poly_zero()
		else:
			#p = poly(p).set_domain('QQ_I')
			pass
		return p


	def tensor_index_replacement(self, source, target, source_prefix='f', target_prefix='f'):
		r = MRing(self)
		for key in list(r.mdict.keys()):
			replaced_poly = self.poly_tensor_index_replacement(r.mdict[key],source,
													  target,source_prefix,target_prefix)
			if replaced_poly.is_zero:
				del r.mdict[key]
			else:
				r.mdict[key] = replaced_poly

			# Returns MRing with empty mdict if zero.

			return r


	def tensor_product_replacement(self, _source, _target):
		#Prefix replacement for source and target:
		source = self.poly_tensor_prefix_replacement(_source,'x','a')
		source = self.poly_tensor_prefix_replacement(source,'y','c')
		source = self.poly_tensor_prefix_replacement(source,'z','d')
		source = self.poly_tensor_prefix_replacement(source,'w','g')
		source = self.poly_tensor_prefix_replacement(source,'f','h')
		source = self.poly_tensor_prefix_replacement(source,'b','k')
		target = self.poly_tensor_prefix_replacement(_target,'x','a')
		target = self.poly_tensor_prefix_replacement(target,'y','c')
		target = self.poly_tensor_prefix_replacement(target,'z','d')
		target = self.poly_tensor_prefix_replacement(target,'w','g')
		target = self.poly_tensor_prefix_replacement(target,'f','h')
		target = self.poly_tensor_prefix_replacement(target,'b','k')

		#Unify domains and generators.
		#source,target = source.unify(target)
		r = MRing(hmap())

		#for key in tqdm(self.mdict.keys(),desc="Tensor Product replacements (key loop)"):
		for key in self.mdict.keys():
			p = self.mdict[key]
			p_dict = p.as_dict()
			term_generators = p.gens
			source_dict = source.as_dict()
			source_generators = source.gens
			assert len(source_dict)==1, "Source is not monomial."
			source_term = list(source_dict.keys())[0]

			#for term,coefficient in tqdm(p_dict.items(),desc="Tensor Product replacements (poly loop)"):
			for term,coefficient in p_dict.items():
				result = self.match_term(term,term_generators,source_term,
							 source_generators,0,{})
				if result==False:
					target_term = Poly({tuple(term):coefficient},term_generators,domain="QQ_I")
					ring_term = MRing(hmap({key:target_term}))
					r += ring_term
				else:
					stripped_term,xmap = result
					target_term = Poly({tuple(stripped_term):coefficient},term_generators,
						domain="QQ_I")
					target_term *= target
					ring_term = MRing(hmap({key:target_term}))
					for source_pair,target_pair in xmap.items():
						# FIXME: handle case in which tensor_index_replacement returns zero.
						ring_term = ring_term.tensor_index_replacement(source_pair[1],
							target_pair[1],source_prefix = source_pair[0],
							target_prefix = target_pair[0])
					r += ring_term
		return r


	def match_term(self,_term, term_generators, _source, source_generators,
				source_index, _xmap):
		term = list(_term)
		source = list(_source)
		xmap = dict(_xmap)

		if source[source_index] == 0:
			if source_index == len(source)-1:
				#Base case
				return term,xmap
			else:
				source_index += 1
				return self.match_term(term,term_generators, source, source_generators,
						   source_index, xmap)
		else:
			source_symbol = source_generators[source_index]
			source[source_index] -= 1

		term_index = 0
		while term_index < len(term):
			if term[term_index] == 0:
				term_index += 1
				continue
			for source_pair,target_pair in xmap.items():
				# FIXME: doesn't handle case where poly_tensor_index_replacement returns zero
				source_poly = self.poly_tensor_index_replacement(Poly({tuple(source):1},
					source_generators),source_pair[1],target_pair[1],
					source_prefix = source_pair[0], target_prefix = target_pair[0])
				source = list(source_poly.as_dict().keys())[0]
			term_symbol = term_generators[term_index]
			symbol_xmap = self.match_symbols(term_symbol,source_symbol)
			if symbol_xmap != False:
				term[term_index] -= 1
				xmap.update(symbol_xmap)
				"MATCHTERM: RECURSE"
				return self.match_term(term,term_generators,source,source_generators,
						   source_index,xmap)
			else:
				term_index+=1
		return False


	def head_indices(self,symbol,prefixes):
		symstring = symbol.__str__()
		if "_{" not in symstring:
			return False
		head = symstring.split('_{')[0]
		iblock = symstring.split('{')[1].split('}')[0]
		indices = []
		i=0
		while i<len(iblock):
			prefix = iblock[i]
			numstring = ""
			i+=1
			if i==len(iblock):
				break
			while (i<len(iblock)) and (iblock[i] not in prefixes):
				numstring+=iblock[i]
				i+=1
			indices.append((prefix,int(numstring)))
		return head,indices


	def match_symbols(self,term_symbol,source_symbol):
		free_term_prefixes = ['x','y','f']
		bound_term_prefixes = ['z','w','b']
		free_source_prefixes = ['a','c','h']
		bound_source_prefixes = ['d','g','k']
		prefixes = free_term_prefixes + bound_term_prefixes + free_source_prefixes + bound_source_prefixes
		#FIXME: kludge here to avoid scalar matching.
		#term_head,term_indices = self.head_indices(term_symbol,prefixes)
		term_ghi_result = self.head_indices(term_symbol,prefixes)
		if term_ghi_result==False:
			return False
		term_head,term_indices = term_ghi_result

		#source_head,source_indices = self.head_indices(source_symbol,prefixes)
		source_ghi_result = self.head_indices(source_symbol,prefixes)
		if source_ghi_result==False:
			return False
		source_head,source_indices = source_ghi_result

		if term_head != source_head:
			return False
		if len(term_indices)!=len(source_indices):
			return False

		symbol_xmap = {}

		#print "MATCHBOUND-TERM"
		#First, match bound_term_prefixes
		i_source = 0
		while i_source < len(source_indices):
			match=False
			if source_indices[i_source][0] in bound_term_prefixes:
				i_term = 0
				while i_term < len(term_indices):
					if source_indices[i_source][0] == term_indices[i_term][0] and source_indices[i_source][1] == term_indices[i_term][1]:
						match=True
						del source_indices[i_source]
						del term_indices[i_term]
						break
					i_term += 1
				if not match:
					return False
			else:
				i_source +=1

		#print "MATCHBOUND_SOURCE"
		#Next, match bound_source_prefixes
		i_source = 0
		while i_source < len(source_indices):
			match = False
			if source_indices[i_source][0] in bound_source_prefixes:
				i_term = 0
				while i_term < len(term_indices):
					if term_indices[i_term][0] == bound_term_prefixes[bound_source_prefixes.index(source_indices[i_source][0])]:
						match=True
						#symbol_xmap[term_indices[i_term]]=source_indices[i_source]
						symbol_xmap[source_indices[i_source]]=term_indices[i_term]
						del source_indices[i_source]
						del term_indices[i_term]
						break
					i_term += 1
				if not match:
					return False
			else:
				i_source += 1

		#print "MATCHFREE-SOURCE"
		#Finally, match free_source_prefixes
		i_source = 0
		while i_source < len(source_indices):
			match = False
			if source_indices[i_source][0] in free_source_prefixes:
				#First try to match to free_term_prefixes.
				i_term = 0
				while i_term < len(term_indices):
					if term_indices[i_term][0] == free_term_prefixes[free_source_prefixes.index(source_indices[i_source][0])]:
						match = True
						symbol_xmap[source_indices[i_source]] = term_indices[i_term]
						del source_indices[i_source]
						del term_indices[i_term]
						break
					i_term += 1
				#Next, try to match to bound_term_prefixes
				if not match:
					i_term = 0
					while i_term < len(term_indices):
						if term_indices[i_term][0] == bound_term_prefixes[free_source_prefixes.index(source_indices[i_source][0])]:
							match = True
							symbol_xmap[source_indices[i_source]] = term_indices[i_term]
							del source_indices[i_source]
							del term_indices[i_term]
							break
						i_term += 1
				if not match:
					return False
			else:
				i_source += 1

		#Check that we have exhausted source_indices and term_indices.
		assert len(source_indices) == 0, "Didn't exhaust source_indices."
		assert len(term_indices) == 0, "Didn't exhaust term_indices."

		return symbol_xmap


	def contract_deltas(self,deltahead):
		""" This should be called only on MRing elements belonging to a single
			graph. There might be index redundancy otherwise! """
		r = MRing(self)
		for key in r.mdict.keys():
			p = r.mdict[key]
			psymbols = list(p.free_symbols)
			delta_symbols = [sym for sym in psymbols if deltahead in sym.__str__()]
			for delta_sym in delta_symbols:
				indices = (delta_sym.__str__()).split('f')[1:3]
				indices = [index.strip('}') for index in indices]
				r = r.evaluate_poly(delta_sym,1)
				# FIXME: handle case in which tensor_index_replacement returns empty?
				r = r.tensor_index_replacement(int(indices[1]),int(indices[0]))
		return r


	def evaluate_deltas(self,deltahead = "D_"):
		r = MRing(self)
		for key in r.mdict.keys():
			p = r.mdict[key]
			psymbols = list(p.free_symbols)
			delta_symbols = [sym for sym in psymbols if deltahead in sym.__str__()]
			for delta_sym in delta_symbols:
				head,prefixes,indices = split_tensor_symbol(delta_sym)
				assert len(prefixes)==2, "Delta should have two indices!"
				if (prefixes[0]==prefixes[1] and indices[0]==indices[1]):
					r = r.evaluate_poly(delta_sym,1)
				else:
					r = r.evaluate_poly(delta_sym,0)
		return r


	def evaluate_projector(self,projhead,subspace_prefix):
		r = MRing(self)
		for key in r.mdict.keys():
			p = r.mdict[key]
			psymbols = list(p.free_symbols)
			proj_symbols = [sym for sym in psymbols if projhead in sym.__str__()]
			for proj_sym in proj_symbols:
				head,prefixes,indices = split_tensor_symbol(proj_sym)
				assert len(prefixes)==2, "Projector should have two indices!"
				if (prefixes[0]==prefixes[1] and indices[0]==indices[1]
					and prefixes[0]==subspace_prefix):
					r = r.evaluate_poly(proj_sym,1)
				else:
					r = r.evaluate_poly(proj_sym,0)
		return r


	def set_domain(self,domain='QQ_I'):
		r = MRing(self)
		for key in r.mdict.keys():
			r.mdict[key] = r.mdict[key].set_domain(domain)
		return r


	def sort_generators(self):
		r = MRing(self)
		for key,p in r.mdict.items():
			r.mdict[key] = p.reorder(*sorted(p.gens,key=str))
		return r


	def clean_dummy_indices(self):
		""" This should be called only on MRing elements belonging to a single
			graph. There might be index redundancy otherwise! """
		r = MRing(self)
		for key in r.mdict.keys():
			replacements = {}
			p = r.mdict[key]
			pstring = (p.as_expr()).__str__()
			terms = pstring.split(' ')
			for term in terms:
				dummydict ={}
				indexdict = {}
				indexblocks = term.split('{')
				for ib in indexblocks:
					if '}' in ib:
						block = ib.split('}')[0]
						indices = block.split('f')[1:]
						for i in indices:
							indexdict.setdefault(int(i),0)
							indexdict[int(i)]+=1
				#Check for multiplets
				num=1
				for k,n in indexdict.items():
					#assert not (n>2)
					#if n==2:
					if n>1:
						assert k not in dummydict.keys()
						dummydict[k]=num
						num+=1
			psymbols = list(p.free_symbols)
			source_symbols = []
			for index,num in dummydict.items():
				source_symbols += [sym for sym in psymbols if 'f'+str(index) in sym.__str__()]
			source_symbols = list(set(source_symbols))
			for sym in source_symbols:
				target_string = sym.__str__()
				for index,num in dummydict.items():
					target_string = target_string.replace('f'+str(index),'b'+str(num))
				target_poly = poly(symbols(target_string),symbols(target_string),domain='QQ_I')
				replacements[sym] = target_poly
				q = compose(r.mdict[key],target_poly,sym)
				newsymbols =  set(list(r.mdict[key].free_symbols)+list(target_poly.free_symbols))
				q = poly(q.as_expr(),newsymbols,domain='QQ_I')
				q = q.exclude()
				r.mdict[key] = poly(q,domain='QQ_I')
		return r


	def sort_symbol(self,symbol,op_indices):
		symstring = symbol.__str__()
		indexblock = list(symstring.split('{')[1].split('}')[0])
		prefix = symstring.split('{')[0]
		indexlist = []
		indexstring=""
		lastprefix=indexblock.pop(0)
		while True:
			char = indexblock.pop(0)
			if (char.isalpha()):
				indexlist.append(lastprefix+indexstring)
				lastprefix=char
				indexstring=""
			elif len(indexblock)==0:
				indexstring+=char
				indexlist.append(lastprefix+indexstring)
				lastprefix=char
				indexstring=""
				break
			else:
				indexstring+=char

		posmap = {}
		for op,positions in op_indices.items():
			ptypelist = [indexlist[i] for i in positions]
			for i,pos in zip(positions,sorted(ptypelist)):
				posmap[i] = pos
		sorted_iblock = ""
		for i in range(len(indexlist)):
			sorted_iblock += posmap[i]
		newstring = prefix+'{'+sorted_iblock+'}'
		return symbols(newstring)

#	def SymbolPermSort(self,symbol,ranges,signs):
#
#		print "SYMBOL: ",symbol
#		print "RANGES: ",ranges
#		print "SIGNS: ",signs
#
#		symstring = symbol.__str__()
#		indexblock = list(symstring.split('{')[1].split('}')[0])
#		head = symstring.split('{')[0]
#		prefixlist = []
#		indexlist = []
#		indexstring=""
#		lastprefix=indexblock.pop(0)
#		while True:
#			char = indexblock.pop(0)
#			if (char.isalpha()):
#				prefixlist.append(lastprefix)
#				indexlist.append(int(indexstring))
#				lastprefix=char
#				indexstring=""
#			elif len(indexblock)==0:
#				indexstring+=char
#				prefixlist.append(lastprefix)
#				indexlist.append(int(indexstring))
#				break
#			else:
#				indexstring+=char
#
#		print "PREFIXES: ",prefixlist
#		print "INDICES: ",indexlist
#
#
#		symbol_sign = 1
#		index_string = ""
#		for irange,sign in zip(ranges,signs):
#			print "IRANGE: ",irange
#			print "SIGN: ",sign
#			prefix_slice = prefixlist[irange[0]:irange[1]]
#			print "PSLICE: ",prefix_slice
#			#Check for homogeneous indices in block.
#			assert all([prefix==prefix_slice[0] for prefix in prefix_slice]),"Inhomogeneous indices in symmetry block. Split flavors too early?"
#			index_slice = indexlist[irange[0]:irange[1]]
#			print "ISLICE: ",index_slice
#			sorted_index_slice = sorted(index_slice)
#			print "SORTED_ISLICE: ",sorted_index_slice
#			perm = [index_slice.index(p)+1 for p in sorted_index_slice]
#			print "PERM: ",perm
#			if sign==-1:
#				print "SIGNED"
#				perm_sign = GetSign(perm)
#				print "PERMSIGN: ",perm_sign
#				symbol_sign *= perm_sign
#			pi_pairs = [p+str(i) for p,i in zip(prefix_slice,sorted_index_slice)]
#			index_string_slice = reduce(lambda x,y:x+y,pi_pairs)
#			print "ISTRING_SLICE: ",index_string_slice
#			index_string+=index_string_slice
#
#		print "SYMSIGN: ",symbol_sign
#		newstring = head+'{'+index_string+'}'
#		return symbols(newstring),symbol_sign


#	def sort_indices(self,tensor_symmetries):
#		symbolmap = {}
#		for tensorhead, oplist in tensor_symmetries.items():
#			op_indices = {}
#			for i,op in enumerate(oplist):
#				op_indices.setdefault(op,[])
#				op_indices[op].append(i)
#			for mkey,q in self.mdict.items():
#				for freesym in list(q.free_symbols):
#					symstring = freesym.__str__()
#					if "_{" not in symstring:
#						continue
#					if tensorhead in freesym.__str__():
#						sorted_freesym = self.sort_symbol(freesym,op_indices)
#						symbolmap[freesym] = sorted_freesym
#		mr = MRing(self)
#		for source,target in symbolmap.items():
#			#mr = mr.repl_replacement(source,target)
#			mr = mr.replace_symbol(source,target)
#		return mr


	def sort_indices(self,symbol_groups):
		#print("IN Kernel SSI")
		symbolmap = {}
		#for i in tqdm([1,],desc="BuildMap"):
		for head,symgroup in symbol_groups.items():
				#print("Head: {},Symgroup: {}".format(head,symgroup))
			for mkey,q in self.mdict.items():
				for freesym in list(q.free_symbols):
					symstring = freesym.__str__()
					if "_{" not in symstring:
						continue
					if head in freesym.__str__():
						symbolmap[freesym] = symbol_group_sort(freesym,symgroup)
		#print("Done buildmap")
		#for i in tqdm([1,],desc="Copy"):
		mr = MRing(self)
		#print("Done copy")
		#for source,target in tqdm(symbolmap.items(),desc="replace_symbol"):
		for source,target in symbolmap.items():
			mr = mr.replace_symbol(source,target)
		return mr


	def replace_symbol(self,source,target):
		target_symbol,sign = target
		r = MRing(self)
		keys = list(r.mdict.keys())
		for key in keys:
			generators = r.mdict[key].gens
			if type(generators)==None:
				continue
			if target_symbol==0:
				new_generators = generators
			else:
				genlist = list(generators)
				genlist.append(target_symbol)
				new_generators = tuple(set(genlist))
			p = r.mdict[key].subs(source,sign*target_symbol)
			if p.is_zero:
				del r.mdict[key]
			else:
				r.mdict[key] = poly(p,new_generators,domain='QQ_I')
		return r


	def repl_replacement(self,source,target):
		#FIXME: edit in place?
		r = MRing(self)
		for key in r.mdict.keys():
			p = r.mdict[key].exclude()
			#p = p.set_domain('QQ_I')
			if p.is_ground:
				continue
			condition = source in p.gens
			if condition:
				p = p.replace(source,target)
				r.mdict[key] = p.set_domain('QQ_I')
		return r


	def replace_monomial(self,monomial,target_poly):
		r = MRing(self)
		for key in r.mdict.keys():
			if str(r.mdict[key].as_expr())=="1":
				continue
			elif str(r.mdict[key].as_expr())=="0":
				continue
			p = compose(r.mdict[key],target_poly,monomial)
			newsymbols =  set(list(r.mdict[key].free_symbols)+list(target_poly.free_symbols))
			p = poly(p.as_expr(),newsymbols,domain='QQ_I')
			p = p.exclude()
			r.mdict[key] = poly(p,domain='QQ_I')
		return r


	def factor_polynomials(self):
		r = MRing(self)
		for key in r.mdict.keys():
			r.mdict[key] = factor(r.mdict[key])
		return r


	def simplify_polynomials(self):
		r = MRing(self)
		for key in r.mdict.keys():
			r.mdict[key] = simplify(r.mdict[key])
		return r


	def unify_generators(self,ext_poly=None):
		if len(self.mdict.keys())==0:
			return MRing(self),None
		if ext_poly==None:
			upoly = list(self.mdict.values())[0].zero
		else:
			upoly = ext_poly

		for key in self.mdict.keys():
			upoly = (upoly.unify(self.mdict[key]))[0]

		newdict = hmap()
		for key in self.mdict.keys():
			newdict[key] = (upoly.unify(self.mdict[key]))[1]
		r = MRing(hmap())
		r.mdict = newdict
		return r,upoly


	def fuse_dict(self,dictA,dictB):
		"""
		Check that there are no redundant labels. That will
		screw everything up!
		"""
		newdict=hmap()
		for keyA in dictA.keys():
			for keyB in dictB.keys():
				#Handle multiplication by the ring identity (0,0)
				#Remove it if at least one pair in the summed key
				#is not (0,0). If two copies of (0,0) enter, at most
				#one may leave.
				sumkey = list(keyA+keyB)
				non_id = False
				for pair in sumkey:
					if pair!=(0,0):
						non_id=True
						break
				if non_id:
					sumkey = [k for k in sumkey if k!=(0,0)]
				else:
					sumkey = [(0,0)]
				targetkey = self.key_sort(tuple(sumkey))
				newdict.setdefault(targetkey,self.poly_zero())
				newdict[targetkey]+=dictA[keyA]*dictB[keyB]
		return newdict


	def __mul__(self,other):
		dummy = symbols('dummy')
		ptype = type(poly(dummy,dummy,domain='QQ_I'))
		domain_types = [ptype,int]
		#is_sympy_number = ("sympy.core.numbers" in str(type(other)))
		is_sympy_number = ("sympy.core" in str(type(other)))
		assert (type(other) in domain_types) or is_sympy_number or (type(other) == type(self))
		if type(other) == type(self) and other.__class__.__name__ == self.__class__.__name__:
			product = MRing(self.fuse_dict(self.mdict,other.mdict))
		elif ((type(other) in domain_types) or is_sympy_number):
			prod_dict = hmap()
			for key in self.mdict.keys():
				prod_dict[key] = self.mdict[key]*other
			product = MRing(prod_dict)
			product.cull_zeros()
		#product,upoly = product.unify_generators()
		return product


	def __pow__(self,power):
		assert type(power)==int
		assert power>=0
		if self.is_empty():
			return MRing(self)
		if power==0:
			#return MRing({((0,0),):self.mdict[self.mdict.keys()[0]].one})
			return self.one()
		if power==1:
			return MRing(self)
		product = MRing(self)
		for i in range(power-1):
			product*=self
		return product


	def link_reduce(self,key,linkpair):
		#------------Sort------------#
		activeA = [pair for pair in key if (bool(set([linkpair[0]])&set(pair)) and not bool(set([linkpair[1]])&set(pair)))]
		activeB = [pair for pair in key if (bool(set([linkpair[1]])&set(pair)) and not bool(set([linkpair[0]])&set(pair)))]
		activeAB = [pair for pair in key if (bool(set([linkpair[0]])&set(pair)) and bool(set([linkpair[1]])&set(pair)))]
		accumulator = [pair for pair in key if not bool(set(linkpair)&set(pair))]

		#Dump loops to accumulator (handled in upper scope)
		accumulator += activeAB

		#Check for lonely indices
		#This implementation is wrong, but there should still
		#be a check for matching # of indices. FIXME
		#if len(activeA)!=len(activeB):
		#	sys.exit("Uneven number of matching pairs!")

		#Base Case
		if (len(activeA)==0 and len(activeB)==0):
			return accumulator

		#----------Contract----------#
		contraction = list(activeA.pop(0) + activeB.pop(0))
		contraction.remove(linkpair[0])
		contraction.remove(linkpair[1])
		contraction = tuple(contraction)

		accumulator += activeA+activeB+[contraction]
		accumulator = self.link_reduce(accumulator,linkpair)
		return accumulator


	def link(self,linkpairs):
		r = MRing(self)
		D = symbols('D')
		Dp = poly(D,D,domain = 'QQ_I')
		for linkpair in linkpairs:
			rloop = MRing(hmap())
			for key in r.mdict.keys():
				#Do an initial round of sorting to speed up the recursion
				inert = [pair for pair in key if not bool(set(linkpair)&set(pair))]
				active = [pair for pair in key if bool(set(linkpair)&set(pair))]
				accumulator = self.link_reduce(active,linkpair)
				workingkey = accumulator+inert
				activeAB = [pair for pair in workingkey if (bool(set([linkpair[0]])&set(pair)) and bool(set([linkpair[1]])&set(pair)))]
				Nloops = len(activeAB)
				cleanworkingkey = [pair for pair in workingkey if not bool(set(linkpair)&set(pair))]
				finalkey = self.key_sort(cleanworkingkey)
				rloop.quick_add(MRing(hmap({finalkey:r.mdict[key]*(Dp**Nloops)})))
			rloop.cull_zeros()
			r = MRing(rloop)
		return r


	def quick_add(self,other):
		#Doesn't cull zeros. Unsafe, but can save time if this
		#is run in a loop, so long as we remember to cull zeros afterwards.
		for key in other.mdict.keys():
			self.mdict.setdefault(key,other.poly_zero())
			self.mdict[key]+=other.mdict[key]


	def __add__(self,other):
		assert other.mathtype == "MRing"
		sum_dict = hmap()
		if len(self.mdict.keys())>0:
			for key in self.mdict.keys():
				sum_dict.setdefault(key,self.poly_zero())
				sum_dict[key] += self.mdict[key]
		if len(other.mdict.keys())>0:
			for key in other.mdict.keys():
				sum_dict.setdefault(key,other.poly_zero())
				sum_dict[key] += other.mdict[key]
		mysum = MRing(sum_dict)
		#mysum,upoly = mysum.unify_generators()
		mysum.cull_zeros()
		return mysum


	def add_in_place(self,other,cull=False):
		assert other.mathtype == "MRing"
		if len(other.mdict.keys())>0:
			for key in other.mdict.keys():
				self.mdict.setdefault(key,other.poly_zero())
				self.mdict[key] += other.mdict[key]
		#mysum,upoly = mysum.unify_generators()
		if cull:
			self.cull_zeros()


	def __sub__(self,other):
		return self+other*(-1)


	def partial_derivative(self, monomial):
		r = MRing(self)
		for key in r.mdict.keys():
			r.mdict[key]*=poly(1,monomial,domain='QQ_I')
			r.mdict[key] = r.mdict[key].diff(monomial)
		r.cull_zeros()
		return r


	def __str__(self):
		sorted_keys = sorted(self.mdict.keys())
		string = ''
		for key in sorted_keys:
			polydict = self.mdict[key].terms
			string += str(key)+" :"+(self.mdict[key].as_expr()).__str__()+'\n'
		return string


	def full_string(self):
		sorted_keys = sorted(self.mdict.keys())
		string = ''
		for key in sorted_keys:
			polydict = self.mdict[key].terms
			string += str(key)+" :"+(self.mdict[key]).__str__()+'\n'
		return string


	def Print(self):
		sorted_keys = sorted(self.mdict.keys())
		for key in sorted_keys:
			polydict = self.mdict[key].terms
			print(str(key)+" :")
			print(self.mdict[key])
			print()


	def evaluate_poly(self,var,value):
		r = MRing(self)
		for key in r.mdict.keys():
			my_symbols = r.mdict[key].gens
			r.mdict[key]*=poly(1,var,domain='QQ_I')
			r.mdict[key] = poly(r.mdict[key].eval(var,value),my_symbols,domain='QQ_I')
		r.cull_zeros()
		return r


	def proportional(self,other):
		A = self.mdict
		B = other.mdict
		if set(A.keys())!=set(B.keys()):
			return False
		for k,key in enumerate(A.keys()):
			if k==0:
				q,r = div(A[key],B[key])
				if not r.is_zero:
					return False
			elif div(A[key],B[key]) != q:
				return False
		return True


	def __eq__(self,other):
		A = self.mdict
		B = other.mdict
		if set(A.keys())!=set(B.keys()):
			return False
		for key in A.keys():
			p = A[key].set_domain('QQ_I')
			q = B[key].set_domain('QQ_I')
			if not (p-q).is_zero:
				return False
		return True


	#FIXME: Removed all the unify_generators!! Does this cause trouble?
#	def __eq__(self,other):
#		diff = self-other
#		return diff.is_empty()

	def __ne__(self,other):
		return not self.__eq__(other)

	
	def __lt__(self,other):
		self_sort = sorted([(key,hmap(val.as_dict())) for key,val in self.mdict.items()])
		other_sort = sorted([(key,hmap(val.as_dict())) for key,val in other.mdict.items()])
		return self_sort < other_sort


	# FIXME: I'm particularly worried about collisions here, especially for very
	# large keys. Should try to get an estimate of expected # collisions...
	def __hash__(self):
		#Hash "recursively" by stacking (key,hash(Poly)) tuples sorted by key.
		hashlist = []
		for key in sorted(self.mdict.keys()):
			#Remove extraneous generators.
			#self.mdict[key] = self.mdict[key].exclude()
			#self.mdict[key] = self.mdict[key].set_domain('QQ_I')
			#Need to hash the Poly *as an expr*. Otherwise, the ordering of
			#Terms and products in the polynomial will depend on orderings
			#of generators, which is a disaster.
			hashlist.append((key.__hash__(),(self.mdict[key].as_expr()).__hash__()))
		return hash(tuple(hashlist))


	def is_empty(self):
		if len(self.mdict.keys())==0:
			return True
		else:
			return False


	def power_counting(self):
		last_count=[0,0,0]
		first = True
		for key in self.mdict.keys():
			count = [0,0,0]
			for pair in key:
				for el in pair:
					if el<0:
						count[2]+=1
					elif (el>0 and el<10):
						count[0]+=1
					elif el>10:
						count[1]+=1
			#Check that power counting is consistent.
			if first:
				first = False
			else:
				assert count==last_count
			last_count=list(count)
		return count


	def dress_momentum_pairs(self,symbol,longitudinal_modes):
		#FIXME: handle non-scalar cases!
		r = MRing(self)
		for key in r.mdict.keys():
			paircount = 0
			for pair in key:
				if pair[0]!=pair[1]:
					for i in pair:
						if i<0:
							paircount+=1
						if i in longitudinal_modes:
							paircount+=1
			r.mdict[key]*=poly(symbol**paircount,symbol,domain='QQ_I')
		return r


	def dress_momentum(self,symbol,index):
		#FIXME: handle non-scalar cases!
		r = MRing(self)
		for key in r.mdict.keys():
			paircount = 0
			for pair in key:
				if pair[0]!=pair[1]:
					for i in pair:
						if i<0 and -i==index:
							paircount+=1
			r.mdict[key]*=poly(symbol**paircount,symbol,domain='QQ_I')
		return r


	def free_symbols(self):
		"""
		Returns a set of all free symbols appearing in all polynomial values
		of `MRing.mdict`.
		"""
		return reduce(lambda s,t: s|t,
					  [set(p.free_symbols) for p in self.mdict.values()])


	def tensor_symbols(self):
		"""
		Returns a `set` of all tensor symbols appearing in all polynomials
		First computes set of all free symbols using
		`MRing.free_symbols()`, then filters out all symbols without
		"_{" characters, indicative of a tensor index block.
		"""
		return set(filter(lambda s: "_{" in s.__str__(),self.free_symbols()))


	def find_prefix(self,prefix_key):
		"""
		Searches all polynomials for tensors containing `prefix`.
		Returns a list of all indices carrying this prefix.
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
		r = MRing(self)
		for key in r.mdict.keys():
			r.mdict[key] = self.poly_bound_indices_to_components(r.mdict[key],first,last,bound_prefix,target_prefix)
		return r


	def poly_bound_indices_to_components(self,poly,
								 first,
								 last,
								 bound_prefix='b',
								 target_prefix='x'):

		target_poly = poly*0

		for pterm in self.p_terms(poly):
			term = Poly(pterm)

			tensor_symbols = set(filter(lambda s: "_{" in s.__str__(),term.free_symbols))
			bound_indices = set()
			for tensym in tensor_symbols:
				head,prefix,index = split_tensor_symbol(tensym)
				bound_indices |= {index[i] for i,v in enumerate(prefix)
								  if v==bound_prefix}


			if len(bound_indices)==0:
				target_poly += term
			else:
				bound_index = list(bound_indices)[0]
				target_term = term*0
				for target_index in range(first, last+1):
					target_term += self.poly_tensor_index_replacement(term,
													bound_index,
													target_index,
													source_prefix=bound_prefix,
													target_prefix=target_prefix)
				target_poly += self.poly_bound_indices_to_components(target_term,first,
														 last,bound_prefix,target_prefix)
		return target_poly


	def expand_bound_index(self,pattern,targets,match):
		"""
		Prefixes in target indices must all differ from source prefix!
		Otherwise, the recursion will never terminate.
		"""
		if match=="prefix":
			assert type(targets[0])==type(str()),"Prefix match needs string targets."
			assert type(pattern)==type(str()),"Prefix match needs string source."
		elif match=="suffix":
			assert type(targets[0])==type(int()),"Suffix match needs int targets."
			assert type(pattern)==type(int()),"Suffix match needs int source."
		elif match =="index":
			assert type(targets[0])==type(tuple()),"Index match needs tuple targets."
			assert type(pattern[0])==type(tuple()),"Index match needs tuple source."
		else:
			assert False, "[match] must be one of prefix,suffix,index"

		assert pattern not in targets, "Pattern in targets, recursion will never terminate!"

		return self.branch_bound_index(pattern,targets,match)


	def branch_bound_index(self,pattern,targets,match):
		target_mring = MRing(hmap())
		for key,poly in self.mdict.items():
			for pterm in self.p_terms(poly):
				pattern_index = first_matching_index(poly,pattern,match)
				if pattern_index==False:
					target_mring += MRing(hmap({key:poly}))
				else:
					working_mring = MRing(hmap())
					for target in targets:
						target_index = target_index(pattern_index,target,match)
						working_mring+=MRing(hmap({key:poly})).tensor_index_replacement(
												pattern_index[1],target_index[1],
												source_prefix=pattern_index[0],
												target_prefix=target_index[0])
					target_mring += working_mring.branch_bound_index(pattern,
																  targets,match)
		return target_mring


#	def matching_indices(self,pattern,match):
#		matching_indices = set()
#		for key,poly in self.mdict.values():
#			for pterm in self.p_terms(poly):
#				matching_indices |= matching_indices(poly,pattern,match)
#		return matching_indices

#-------------------Functions--------------------#

def join(_A,Alegs,_B,Blegs,pair,_bridge,rank=1):
	"""
	This function joins a pair [_A,_B] of MRings along a pair of legs [pair].
	They are joined "across" an MRing object [_bridge], which plays the role
	of a propagator numerator in the case that [_A] and [_B] are feynman vertices.
	Works for ranks 1 and 2, but does not currently support _bridge objects which
	contain momenta. Essentially, for gravity and Yang-Mills, this restricts us
	to work in Feynman gauge. FIXME: This may not be true, need to think harder.
	The lists of leg labels for _A and _B are given by [Alegs] and [Blegs],
	respectively. The first and second elements of [pair] must then be elements
	of [Alegs] and [Blegs], respectively.
	We use the convention that positive integer leg labels also label the
	corresponding polarization vectors, and that this integer, plus 100, labels
	the dual polarization in the case of rank 2 MRings. The bridge must always
	have legs labelled 1 and 2.
	The function will fail if [_A] and [_B] combined have more than 98 legs,
	but this will probably never occur in the forseen use cases for AmpliTools.
	Note that A and B must both be linear in the various polarization labels!
	Exactly linear, no constant terms!
	"""
	assert (rank==1 or rank==2)
	nA = len(Alegs)
	nB = len(Blegs)
	assert nA+nB<98
	A = MRing(_A)
	B = MRing(_B)
	Astar = max(Alegs)+max(Blegs)-1
	Bstar = max(Alegs)+max(Blegs)+1

	Amap = {Alegs[(Alegs.index(pair[0])+i)%nA]:i for i in range(1,nA)}
	Amap[pair[0]]=Astar
	Bmap = {Blegs[(Blegs.index(pair[1])+i)%nB]:i+nA-1 for i in range(1,nB)}
	Bmap[pair[1]]=Bstar
	#if True:
	#	Amap[pair[0]+100]=Astar+100
	#	for i in range(1,nA):
	#		Amap[Alegs[(Alegs.index(pair[0])+i)%nA]+100]=i+100
	#	Bmap[pair[1]+100]=Bstar+100
	#	for i in range(1,nB):
	#		Bmap[Blegs[(Blegs.index(pair[1])+i)%nB]+100]=i+nA+99

	Ablocks = [[-i,i,i+100] for i in Alegs]
	A = A.block_replacement(Amap,Ablocks)
	Bblocks = [[-i,i,i+100] for i in Blegs]
	B = B.block_replacement(Bmap,Bblocks)

	bridge = MRing(_bridge)
	#bridge.ZeroPadEnd(A.GetPolyLength()-1)
	bridgemap = {1:Astar+1,2:Bstar+1}
	#if rank==2:
	#if True:
	#	bridgemap[101]=Astar+101
	#	bridgemap[102]=Bstar+101
	bridgeblocks = [[-1,1,101],[-2,2,102]]
	bridge = bridge.block_replacement(bridgemap,bridgeblocks)

	C = A.fuse(B.fuse(bridge))
	C = C.link([[Astar,Astar+1],])
	C = C.link([[Bstar,Bstar+1],])
	if rank==2:
		C = C.link([[Astar+100,Astar+101],[Bstar+100,Bstar+101]])
	C = C.replacement({-Astar:[[-1,-i] for i in range(1,nA)]})
	C = C.replacement({-Bstar:[[1,-i] for i in range(1,nA)]})

	s = MRing({((-200,-200),):Poly({(0,):Rational(1)})})
	s = s.replacement({-200:[[1,-i] for i in range(1,nA)]})

	return C,s


def shelve(r,symbolblocks,filename):
	shelf = shelve.open(filename)
	shelf['r'] = r
	shelf['symbolblocks'] = symbolblocks
	shelf.close()


def unshelve(filename):
	shelf = shelve.open(filename)
	r = shelf['r']
	symbolblocks = shelf['symbolblocks']
	shelf.close()
	return r,symbolblocks
