from itertools import permutations, combinations, product
from functools import reduce
from mring import MRing
from sympy import poly, symbols, Rational
import sys
from hashable_containers import hmap

"""
CRUCIAL NOTE: the new permute_blocks function bakes in the i,i+100 representation of pairs of rank-2 polarizations! Need to be careful when running old code that uses the i,i+10 representation.
"""

def get_cycles(perm):
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


def get_sign(perm):
	"""
	Constructs the sign of a permutation by counting cycles.

	(perm) should be a tuple (i_1,i_2,...),
	where (..) are permutations of the list (1,...,n). These jth
	entry is interpreted as the target position for the jth symbol
	block.
	"""
	sign = 1
	for c in get_cycles(perm):
		sign*= (-1)**(len(c)-1)
	return sign


def permute_blocks(r,perm,symbolblocks,signed=False,source_prefix='f',target_prefix='f'):
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
	rnew = r.block_replacement(blockmap,symbolblocks,source_prefix=source_prefix,target_prefix=target_prefix)
	#Sign r if requested.
	if signed:
		rnew=rnew*get_sign(perm)
	return rnew


def orbit(r,perms,symbolblocks,signed=False,source_prefix='f',target_prefix='f'):
	"""
	Given (r) defined over (symbolblocks), compute
	its orbit under (perms), either signed or unsigned, 
	depending on (signed).
	"""
	orbit = MRing(hmap())
	for perm in perms:
		orbit+=permute_blocks(r,perm,symbolblocks,signed,source_prefix,target_prefix)
	return orbit


def symmetric_orbit(r,symbolblocks):
	"""
	Specializes orbit() to the symmetric group
	over the blocks in symbolblocks. This 
	"symmetrizes" r (without applying a prefactor).
	"""
	permrange = range(1,len(symbolblocks)+1)
	perms = permutations(permrange)
	return orbit(r,perms,symbolblocks)


def antisymmetric_orbit(r,symbolblocks):
	"""
	Specializes orbit() to the symmetric group,
	including signs, over the blocks in symbolblocks.
	This "antisymmetrizes" r (without applying a prefactor).
	"""
	permrange = range(1,len(symbolblocks)+1)
	perms = permutations(permrange)
	return orbit(r,perms,symbolblocks,True)


def cyclic_orbit(r,symbolblocks):
	"""
	Specializes orbit() to the cyclic group
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
	return orbit(r,perms,symbolblocks)


def monomial_tag(monomial,symbolblocks):
	"""
	Defines an injective map from monomials to tuples of non-negative 
	integers. The image of the map is called the "tag" of the monomial. 
	The monomial belongs to an MRing defined over a set of symbols, 
	given by (symbolblocks). All symbols in the monomial are transformed 
	to tuples of non-negative integers given by their postitions in the 
	sorted list of symbols. The space of such lists is totally ordered, 
	using element-wise integer ordering, treating tuple[0] as the 
	least-significant element (see tuple_compare()).
	"""
	flatmonomial = sum(monomial,())
	symbols = sorted(tuple(sum(symbolblocks,[])))
	tag = tuple([symbols.index(symbol) for symbol in flatmonomial])
	return tag


def tuple_compare(A,B):
	"""
	Compare integer tuple A and B.
	Return -1 if A<B, 0 if A=B, 1 if A>B.
	Compares in descending order, left to right
	(that is, leftmost entry (tuple[0]) is most-significant).
	"""

	assert len(A)==len(B)
	for a,b in zip(A,B):
		if a<b:
			return -1
		elif a>b:
			return 1
		else:
			pass
	return 0


def minimal_tag(tagtuple):
	"""
	Using tuple_compare(), find the minimal element of the tuple
	of tags, (tagtuple).
	"""
	mintag = tagtuple[0]
	for tag in tagtuple:
		if tuple_compare(tag,mintag)<0:
			mintag=tag
	return mintag


def monomial_signature(monomial,symbolblocks,cyclic=False):
	"""
	Computes the minimum of monomial tags (the signature) over the orbit 
	of the monomial under the action of the symmetric group on the symbol 
	blocks. The tag is useful because it's an injective map to the set of 
	tuples of non-negative integers, which is totally ordered. The signature 
	therefore has a unique monomial preimage. We can use the signatures to 
	remove ``leg labeling'' redundancies in amplitudes.
	"""
	x = symbols('x')
	r = MRing(hmap({monomial:poly(1,x,domain='QQ_I')}))
	if cyclic==True:
		orbit = cyclic_orbit(r,symbolblocks)
	else:
		orbit = symmetric_orbit(r,symbolblocks)
	tags=[]
	monomials = orbit.mdict.keys()
	monotags = tuple([monomial_tag(mono,symbolblocks) for mono in monomials])
	signature = minimal_tag(monotags)
	return signature


def canonical_monomial(monomial,symbolblocks):
	"""
	Uses the same methods as monomial_signature() to send (monomial) to
	the preimage of the minimal tag over its orbit.
	"""
	x = symbols('x')
	r = MRing(hmap({monomial:poly(1,x,domain='QQ_I')}))
	orbit = symmetric_orbit(r,symbolblocks)
	monomials = orbit.mdict.keys()
	tagdict = {}
	for mono in monomials:
		tagdict[monomial_tag(mono,symbolblocks)] = mono
	signature = minimal_tag(tagdict.keys())
	return tagdict[signature]


def symmetric_group(n):
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


def compose(A,B):
	assert len(A)==len(B)
	C = tuple([B[A[i]-1] for i in range(len(A))])
	return C


def partial_symmetric_permutations(nlegs,permlegs):
    partial_perms = list(permutations(permlegs))
    perms = []
    for pperm in partial_perms:
        perm = list(range(1,nlegs+1))
        for i,p in zip(permlegs,pperm):
            perm[i-1] = p
        perms.append(tuple(perm))
    return perms


def symmetric_partition_permutations(partitions):
	nlegs = max(reduce(lambda x,y: x+y, partitions))
	nested_perms=tuple([tuple(partial_symmetric_permutations(nlegs,partition)) for partition in partitions])
	part_perms = [reduce(compose,perm_sequence) for perm_sequence in list(product(*nested_perms))]
	return part_perms
