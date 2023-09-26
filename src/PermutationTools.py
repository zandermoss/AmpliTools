from itertools import permutations, combinations, product
from functools import reduce
from MRing import MRing
from sympy import poly, symbols, Rational
import sys

"""
CRUCIAL NOTE: the new PermuteBlocks function bakes in the i,i+100 representation of pairs of rank-2 polarizations! Need to be careful when running old code that uses the i,i+10 representation.
"""

def GetCycles(perm):
	"""
	Takes a tuple of permutation targets, and returns the cycle
	decomposition.

	(perm) should be a tuple (i_1,i_2,...),
	where (..) are permutations of the list (1,...,n). These jth
	entry is interpreted as the target position for the jth symbol
	block.
	[cycles] is a tuple of tuples, each of which is a cycle of the
	permutation (perm).
	"""
	source = range(1,len(perm)+1)
	cycles = []
	while len(source)>0:
		t0 = source[0]
		t = t0
		cycle = []
		looped = False
		while not looped:
			source.remove(t)
			cycle.append(t)
			t = perm[t-1]
			looped = (t==t0)
		cycles.append(tuple(cycle))
	return cycles

def GetSign(perm):
	"""
	Constructs the sign of a permutation by counting cycles.

	(perm) should be a tuple (i_1,i_2,...),
	where (..) are permutations of the list (1,...,n). These jth
	entry is interpreted as the target position for the jth symbol
	block.
	"""
	sign = 1
	for c in GetCycles(perm):
		sign*= (-1)**(len(c)-1)
	return sign

def PermuteBlocks(r,perm,symbolblocks,signed=False,source_prefix='f',target_prefix='f'):
	"""
	(perm) should be a tuple (i_1,i_2,...),
	where (..) are permutations of the list (1,...,n). These jth
	entry is interpreted as the target position for the jth symbol
	block.
	(symbolblocks) should be a list of blocks of symbols, not necessarily
	all of the same size! Allowing this heterogeneity allows us to permute 
	legs of different ranks and spins. That might sound like bad news, but
	remember that we're really only permuting the labels, that is, which leg
	we call 1,2,3, and so on. So, it's completely valid to say, for example,
	"the spin-1 leg, called 1, and the spin-2 leg, called 2, will now have their
	labels swapped. That is, the permuted object will have the spin-2 leg labelled
	1, and the spin-1 leg labelled 2." (signed) indicates whether or not 
	the r should be multiplied by the sign of the permutation. 
	This is useful when permuting legs which live in the adjoint rep. of a lie algebra.
	"""
	#Check that inputs are OK.
	assert len(symbolblocks) == len(perm)
	blockmap = {i:perm[i-1] for i in range(1,len(perm)+1)}
	rnew = r.BlockReplacement(blockmap,symbolblocks,source_prefix=source_prefix,target_prefix=target_prefix)
	#Sign the r if requested.
	if signed:
        #print("SIGNED!")
        #print("PERM: ",perm)
        #print("SIGN: ",GetSign(perm))
		rnew=rnew*GetSign(perm)
	return rnew

#FIXME: Remove after commit
#def OldPermuteBlocks(mvector,perm,symbolblocks,signed=False):
#	"""
#	(perm) should be a tuple (i_1,i_2,...),
#	where (..) are permutations of the list (1,...,n). These jth
#	entry is interpreted as the target position for the jth symbol
#	block.
#	(symbolblocks) should be a list of symbols, partitioned into equal-
#	size blocks, which will be permuted. E.g. [1,2,3,4] -> [[1,2],[3,4]]
#	(signed) indicates whether or not the mvector should be multiplied
#	by the sign of the permutation. This is useful when permuting legs
#	which live in the adjoint rep. of a lie algebra.
#	"""
#	#Check that inputs are OK.
#	assert len(symbolblocks) == len(perm)
#	for n,p in enumerate(perm):
#		if p!=n+1:
#			assert len(symbolblocks[n])==len(symbolblocks[p-1])
#
#	#for block in symbolblocks:
#	#	assert len(block) == len(symbolblocks[0])
#
#	"""
#	The replacement algorithm in MRing doesn't know about
#	blocks. It needs symbol-by-symbol instructions. We first
#	permute the blocks according to perm, and then flatten
#	the list of permuted blocks of symbols into a single list of
#	permuted symbols: e.g. given cycle (12), we turn symbolblock
#	[[1,2],[3,4]] into [[3,4],[1,2]], and then flatten to [3,4,1,2].
#	"""
#	blockperm = [symbolblocks[i-1] for i in perm]
#	flatperm = tuple(reduce(lambda x,y: x+y, blockperm))
#
#	"""
#	Finally, we invoke MRing.Replacement() to do the signed replacements.
#	"""
#	flatsymbols = sum(symbolblocks,[])
#	#Generate a list of temporary symbols to avoid collisions.
#	temps = []
#	for i in range(len(flatsymbols)):
#		temps.append(max(flatsymbols)+1+i)
#	#Send source to temp.   
#	for source,temp in zip(flatsymbols,temps):
#		mvector.Replacement({source:[[1,temp]]})
#	#Send temp to target.   
#	for target,temp in zip(flatperm,temps):
#		mvector.Replacement({temp:[[1,target]]})
#	#Sign the mvector if requested.
#	if signed:
#		sign = GetSign(perm)
#		mvector.ScalarMultiply(sign)

def Orbit(r,perms,symbolblocks,signed=False,source_prefix='f',target_prefix='f'):
	"""
	Given (r) defined over (symbolblocks), compute
	its orbit under (perms), either signed or unsigned, 
	depending on (signed).
	"""
	orbit = MRing({})
	for perm in perms:
		orbit+=PermuteBlocks(r,perm,symbolblocks,signed,source_prefix,target_prefix)
	return orbit

def SymmetricOrbit(r,symbolblocks):
	"""
	Specializes Orbit() to the symmetric group
	over the blocks in symbolblocks. This 
	"symmetrizes" r (without applying a prefactor).
	"""
	permrange = range(1,len(symbolblocks)+1)
	perms = permutations(permrange)
	return Orbit(r,perms,symbolblocks)

def AntiSymmetricOrbit(r,symbolblocks):
	"""
	Specializes Orbit() to the symmetric group,
	including signs, over the blocks in symbolblocks.
	This "antisymmetrizes" r (without applying a prefactor).
	"""
	permrange = range(1,len(symbolblocks)+1)
	perms = permutations(permrange)
	return Orbit(r,perms,symbolblocks,True)

def CyclicOrbit(r,symbolblocks):
	"""
	Specializes Orbit() to the cyclic group
	over the blocks in symbolblocks.
	"""
	n = len(symbolblocks)
	perms = []
	perm = [-1 for k in range(n)]
	j=0
	for i in range(n*n+1):
		if (i%n==0 and i!=0):
			perms.append(tuple(perm))
			j+=1
		perm[i%n] = j%n+1
		j+=1
	return Orbit(r,perms,symbolblocks)

def GetMonomialTag(monomial,symbolblocks):
	"""
	Defines an injective map from monomials to tuples of non-negative 
	integers. The image of the map is called the "tag" of the monomial. 
	The monomial belongs to an MRing defined over a set of symbols, 
	given by (symbolblocks). All symbols in the monomial are transformed 
	to tuples of non-negative integers given by their postitions in the 
	sorted list of symbols. The space of such lists is totally ordered, 
	using element-wise integer ordering, treating tuple[0] as the 
	least-significant element (see TupleCompare()).
	"""
	flatmonomial = sum(monomial,())
	symbols = sorted(tuple(sum(symbolblocks,[])))
	tag = tuple([symbols.index(symbol) for symbol in flatmonomial])
	#orderedmonomial = [symbols.index(symbol) for symbol in flatmonomial]
	#base = len(symbols)
	#tag = 0
	#for i,n in enumerate(orderedmonomial):
	#	tag += n*base**i
	#if tag > sys.maxint/2:
	#	sys.exit("Error: EdgeValue too large!")
	return tag

def TupleCompare(A,B):
	"""
	Compare integer tuple A and B.
	Return -1 if A<B, 0 if A=B, 1 if A>B.
	Compares in descending order, left to right
	(that is, leftmost entry (tuple[0]) is most-significant).
	"""
	# Reverse lists for compatibility with old canonicalizer
	#	Alist = list(A)
	#	Blist = list(B)
	#	Alist.reverse()
	#	Blist.reverse()
	#	A = tuple(Alist)
	#	B = tuple(Blist)

	assert len(A)==len(B)
	for a,b in zip(A,B):
		if a<b:
			return -1
		elif a>b:
			return 1
		else:
			pass
	return 0

def GetMinimalTag(tagtuple):
	"""
	Using TupleCompare(), find the minimal element of the tuple
	of tags, (tagtuple).
	"""
	mintag = tagtuple[0]
	for tag in tagtuple:
		if TupleCompare(tag,mintag)<0:
			mintag=tag
	return mintag

def GetMonomialSignature(monomial,symbolblocks,cyclic=False):
	"""
	Computes the minimum of monomial tags (the signature) over the orbit 
	of the monomial under the action of the symmetric group on the symbol 
	blocks. The tag is useful because it's an injective map to the set of 
	tuples of non-negative integers, which is totally ordered. The signature 
	therefore has a unique monomial preimage. We can use the signatures to 
	remove ``leg labeling'' redundancies in amplitudes.
	"""
	x = symbols('x')
	r = MRing({monomial:poly(1,x,domain='QQ_I')})
	if cyclic==True:
		orbit = CyclicOrbit(r,symbolblocks)
	else:
		orbit = SymmetricOrbit(r,symbolblocks)
	tags=[]
	monomials = orbit.Mdict.keys()
	monotags = tuple([GetMonomialTag(mono,symbolblocks) for mono in monomials])
	signature = GetMinimalTag(monotags)
	return signature

def GetCanonicalMonomial(monomial,symbolblocks):
	"""
	Uses the same methods as GetMonomialSignature() to send (monomial) to
	the preimage of the minimal tag over its orbit.
	"""
	x = symbols('x')
	r = MRing({monomial:poly(1,x,domain='QQ_I')})
	orbit = SymmetricOrbit(r,symbolblocks)
	monomials = orbit.Mdict.keys()
	tagdict = {}
	for mono in monomials:
		tagdict[GetMonomialTag(mono,symbolblocks)] = mono
	signature = GetMinimalTag(tagdict.keys())
	return tagdict[signature]


def SymmetricGroup(n):
	""" Generates a list of lists (called maps), each of which contains
	n [source,target] pair lists. Thould be a tuple (i_1,i_2,...),
	where (..) are permutations of the list (1,...,n). These jth
	entry is interpreted as the target position for the jth symbol
	block. There are n! such lists in maps, each representing one of 
	the elements of S(n).
	"""
	source = range(1,n+1)
	perms = permutations(source)
	maps = []
	for perm in perms:
		maps.append(zip(source,perm))
	return maps

def Compose(A,B):
	assert len(A)==len(B)
	C = tuple([B[A[i]-1] for i in range(len(A))])
	return C

def PartialSymmetricPerms(nlegs,permlegs):
    partial_perms = list(permutations(permlegs))
    perms = []
    for pperm in partial_perms:
        perm = list(range(1,nlegs+1))
        for i,p in zip(permlegs,pperm):
            perm[i-1] = p
        perms.append(tuple(perm))
    return perms

def SymmetricPartitionPerms(partitions):
	nlegs = max(reduce(lambda x,y: x+y, partitions))
	nested_perms=tuple([tuple(PartialSymmetricPerms(nlegs,partition)) for partition in partitions])
	part_perms = [reduce(Compose,perm_sequence) for perm_sequence in list(product(*nested_perms))]
	return part_perms

#------------------------------------------------------------------#
#oldbelow here

#def ApplyPerm(source,perm):
#	target = list(source)
#	for i,s in enumerate(source):
#		target[perm[i]-1] = s
#	return target
#
#def Orbit(glyph,perms):
#	mvector = MRing({glyph:Poly({(0,):Fraction(1)})})
#	sumvec = MRing({})
#	for perm in perms:
#		workvec = mvector.BlockReplacement(mvector,perm)
#		sumvec.Add(workvec)
#	orbit = sumvec.Mdict.keys()
#	return orbit
#
#def SignedOrbit(mvector,rawperms):
#	source = range(1,len(rawperms[0])+1)
#	perms = []
#	for rawperm in rawperms:
#		sign = 1
#		for c in GetCycles(rawperm):
#			sign*= (-1)**(len(c)-1)
#		mymap = zip(source,rawperm)
#		perms.append([sign,mymap])
#	sumvec = MRing({})
#	for perm in perms:
#		sign = perm[0]
#		blockmap = perm[1]
#		workvec = mvector.BlockReplacement(mvector,blockmap)
#		workvec.ScalarMultiply(sign)
#		sumvec.Add(workvec)
#	return sumvec
#"""
#
#def GetUnsignedPMaps(perms):
#	source = range(1,len(perms[0])+1)
#	pmaps = []
#	sign = 1
#	for perm in perms:
#		mymap = zip(source,perm)
#		pmaps.append([sign,mymap])
#	return pmaps
#
#def GetSignedPMaps(perms):
#	source = range(1,len(perms[0])+1)
#	pmaps = []
#	for perm in perms:
#		sign = 1
#		for c in GetCycles(perm):
#			sign*= (-1)**(len(c)-1)
#		mymap = zip(source,perm)
#		pmaps.append([sign,mymap])
#	return pmaps
#
#def GetLPMaps(pmaps,symlist):
#	lpmaps=[]
#	for pmap in pmaps:
#		pairmap = []
#		sign = pmap[0]
#		pairs = pmap[1]
#		for pair in pairs:
#			a = symlist[pair[0]-1]
#			b = symlist[pair[1]-1]
#			pairmap.append((a,b))	
#		lpmaps.append([sign,pairmap])
#	return lpmaps		
#
#def SignedLPOrbit(mvector,perms,symlist):
#	pmaps = GetSignedPMaps(perms)
#	lpmaps = GetLPMaps(pmaps,symlist)
#	sumvec = MRing({})
#	for lpmap in lpmaps:
#		sign = lpmap[0]
#		blockmap = lpmap[1]
#		workvec = mvector.BlockReplacement(mvector,blockmap)
#		workvec.ScalarMultiply(sign)
#		sumvec.Add(workvec)
#	return sumvec
#
#def UnsignedLPOrbit(mvector,perms,symlist):
#	pmaps = GetUnsignedPMaps(perms)
#	lpmaps = GetLPMaps(pmaps,symlist)
#	sumvec = MRing({})
#	for lpmap in lpmaps:
#		blockmap = lpmap[1]
#		print blockmap
#		workvec = mvector.BlockReplacement(mvector,blockmap)
#		sumvec.Add(workvec)
#	return sumvec
#
#def SignedLPOrbit(mvector,perms,symlist):
#	pmaps = GetSignedPMaps(perms)
#	lpmaps = GetLPMaps(pmaps,symlist)
#	sumvec = MRing({})
#	for lpmap in lpmaps:
#		sign = lpmap[0]
#		blockmap = lpmap[1]
#		workvec = mvector.LegReplacement(mvector,blockmap,symlist)
#		workvec.ScalarMultiply(sign)
#		sumvec.Add(workvec)
#	return sumvec
#
#def UnsignedLPOrbit(mvector,perms,symlist):
#	pmaps = GetUnsignedPMaps(perms)
#	lpmaps = GetLPMaps(pmaps,symlist)
#	sumvec = MRing({})
#	for lpmap in lpmaps:
#		blockmap = lpmap[1]
#		workvec = mvector.LegReplacement(mvector,blockmap,symlist)
#		sumvec.Add(workvec)
#	return sumvec
#
#
#def ListSignedSymmetric(mylist):
#	permmaps = SignedSymmetric(len(mylist))
#	listpermmaps = []
#	for mymap in permmaps:
#		listpairmap = []
#		sign = mymap[0]
#		pairs = mymap[1]
#		for pair in pairs:
#			a = mylist[pair[0]-1]
#			b = mylist[pair[1]-1]
#			listpairmap.append((a,b))	
#		listpermmaps.append([sign,listpairmap])
#	return listpermmaps
#
#def IsInMinusPermOrbit(glyph):
#	symlist = list(sum(glyph,()))
#	symlist = [abs(x) for x in symlist]
#	symlist = list(set(symlist))
#	rawperms = ListSignedSymmetric(symlist)
#	negperms = [perm[1] for perm in rawperms if perm[0]<0]
#	return glyph in Orbit(glyph,negperms)
#
#"""
#def GetCycles(perm):
#	n = len(perm)
#	source = range(1,n+1)
#	target = list(source)
#	cycles = []
#	for i in range(n):
#		cycles.append([])
#	for i in range(n):
#		for j in range(n):
#			if target[j] not in cycles[j]:
#				cycles[j].append(target[j])
#		target = ApplyPerm(target,perm)
#	canon_cycles = []
#	for c in cycles:
#		ncycle = []
#		for i in range(1,len(c)):
#			ncycle.append(i+1)
#		ncycle.append(1)
#		orbit = []
#		target = c
#		for i in range(n):
#			orbit.append(target)
#			target = ApplyPerm(target,ncycle)
#		orbit = sorted(orbit, key=lambda cyc: cyc[0])		
#		canon_cycles.append(tuple(orbit[0]))
#	cycles = sorted(list(set(canon_cycles)),key = lambda cyc: cyc[0])
#	return cycles
#"""
#
#def SignedSymmetric(n):
#	source = range(1,n+1)
#	perms = list(permutations(source))
#	signedmaps = []
#	for perm in perms:
#		sign = 1
#		for c in GetCycles(perm):
#
#
#
#			sign*= (-1)**(len(c)-1)
#		mymap = zip(source,perm)
#		signedmaps.append([sign,mymap])
#	return signedmaps
#
#
#def ListSignedSymmetric(mylist):
#	permmaps = SignedSymmetric(len(mylist))
#	listpermmaps = []
#	for mymap in permmaps:
#		listpairmap = []
#		sign = mymap[0]
#		pairs = mymap[1]
#		for pair in pairs:
#			a = mylist[pair[0]-1]
#			b = mylist[pair[1]-1]
#			listpairmap.append((a,b))	
#		listpermmaps.append([sign,listpairmap])
#	return listpermmaps
#
##Working on a function to return partitions of a list into two sublists,ignoring order.
#
#def GetPartitions(mylist,size):
#	#Check that the size is reasonable:
#	assert size<len(mylist)
#	assert size>=0
#	#Arguments OK. Generate the partitions.
#	partlist =[]
#	flatlist =[]
#	for com in combinations(mylist,size):
#		complement = list(mylist)
#		for c in com:
#			complement.remove(c)
#		complement = tuple(complement)
#		if ((com not in flatlist) and (complement not in flatlist)):
#			partlist.append((com,complement))
#		flatlist.append(com)
#	return tuple(partlist)
