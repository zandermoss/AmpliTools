#! /usr/bin/python

# from sympy import symbols, poly, Rational, simplify, factor, Poly
import sys
import sympy
from sympy import *
from tqdm import tqdm
from TensorTools import SplitTensorSymbol
from SymPyTools import IsSympySymbolic, FuncType

"""
Here, we collect functions related to the treatment of amplitudes using a covariance-breaking
formalism. In particular, we have need to scale the maginitude of three-momenta by various and
potentially complex factors. To conserve 4-momentum and keep all external legs on-shell, we'll
have to scale energies accordingly. Generally, if p3_i are three momenta, and p4_i four momenta,
p4_0 = sign*(m^2_i + z^2*p3_i^2)^(1/2). Note that we do not need to break three-space rotational
invariance, but we do pick out a direction in time. Furthermore, our amplitudes, even at tree
level, will no longer be rational functions in external kinematic data. We need to account for the
covariance breaking in two ways. First, we must convert from the representation of amplitudes in terms
of fraction fields over mandelstam-invariant generators to irrational expressions given in terms of
three-momentum invariants (p3_i dot p3_j), etc. Second, we must update our laurent coefficient
calculations to handle the square roots.
To accomplish both goals, we move away from the original representation used by AmpliTools:
an MRing dictionary with kinematic ID pairs as keys and sympy polynomials as values, and the
corresponding fraction field MRational. Instead, we'll just use symbolic expressions directly
in sympy, representing the 3D mandelstam invariants as indexed polynomial generators (e.g. pp_ij),
and manipulate them with the same string parsing machinery that we use to handle color tensors.
Below, we collect functions used to perform the conversion from the original AmpliTools representation
to the sympy representation, and some tools for manipulating the new representation. Unfortunately,
the square roots cause sympy's internal series methods to throw a fit, so we have written a custom
PBLaurentCoefficient() function to compute these coefficients.
"""

def IdemList(x):
    """
    Idempotent function for converting data types other than lists
    into lists of that data type. If [x] is a list, return x. Else,
    embed x into a list with one element, [x,], and return this list.
    """
    if type(x) == list:
        return x
    else:
        return [
            x,
        ]


def BuildSymbol(prefix, indices, index_type):
    # FIXME: note that we can't do spectrum splitting after the conversion!
    # Can we? Would need to double check the splitting code.
    """
    Build a sympy monomial (tensor in prefix_{indices} format). This is the
    same format used to describe color tensors throughout AmpliTools. This
    function will be used to construct masses, m_i, as well as kinematic
    invariants pp_{ij},pe_{ij},ee_{ij} in 3D and 4D, and ll_{ij},pl_{ij},le_{ij} in 4D.
    Takes a prefix string [prefix], index list [index] (list of non-negative
    integers), and an argument [index_type] which can be either a string or
    a list. If string, all indices in the {...} block will be prefaced with the
    string. e.g.

    | prefix = "pp"	 |
    | indices = "[1,3]" |  ===>  pp_{i1i3}
    | index_type = "i"  |

    On the other hand, a dictionary keyed with non-negative integers and valued
    with strings will prepend a given index value with the matching string. e.g.

    | prefix = "pp"					|
    | indices = "[1,3]"				|  ===>  pp_{x1z3}
    | index_type = {1:"x",2:"y",3:"z"} |

    This is very useful for labeling masses with their external particle type.
    For example, if leg 1 is a pion, and we're labeling pion indices by prepending
    them with "y" instead of "i", we can supply the dictionary matching external
    leg indices to these particle type strings, and automatically get m_y1 instead
    of m_i1.
    *** We need to be careful about applying the spectrum splitting operation after ***
    *** these conversions. It might work out, but has not yet been tested.		  ***
    """
    indexblock = ""
    for index in IdemList(indices):
        if type(index_type) == type(""):
            indexblock += "i" + str(index)
        elif type(index_type) == type({}):
            indexblock += index_type[index] + str(index)
        else:
            assert False, "Argument [index_type] is neither string nor dict."
    symstring = prefix + "_{" + indexblock + "}"
    return symbols(symstring)


def PairSymbols(pair, polarization_map={}):
    """
    Converts a kinematic index pair (i,j) of the type used to index mandelstam
    generators in the covariant, AmpliTools representation into a sympy monomial
    with prefix (pp,pe,ee) and indices (i,j). These monomials are then manipulated
    just as the color tensor monomials are in the original AmpliTools library.
    [polarization_map] contains elements {particle_id:'e'/'l'} mapping external vector
    legs to either transverse 'e' or longitudinal 'l' modes.
    """
    if pair[0] == 0 and pair[1] == 0:
        return 1

    elif pair[0] < 0 and pair[1] < 0:
        return BuildSymbol("pp", sorted([abs(x) for x in pair]), "i")

    elif pair[0] < 0 and pair[1] > 0:
        assert (
            pair[1] in polarization_map.keys()
        ), "Missing polarization map entry for leg " + str(pair[1])
        if polarization_map[pair[1]] == "e":
            return BuildSymbol("pe", [abs(x) for x in pair], "i")
        elif polarization_map[pair[1]] == "l":
            return BuildSymbol("pl", [abs(x) for x in pair], "i")
        else:
            assert False, "Unrecognized polarization type for leg " + \
                str(pair[1])

    elif pair[0] > 0 and pair[1] > 0:
        for i in range(2):
            assert (
                pair[i] in polarization_map.keys()
            ), "Missing polarization map entry for leg " + str(pair[i])
        pol = [polarization_map[pair[i]] for i in range(2)]
        if pol == ["e", "e"]:
            return BuildSymbol("ee", sorted([abs(x) for x in pair]), "i")
        elif pol == ["l", "e"]:
            return BuildSymbol("le", [abs(pair[0]), abs(pair[1])], "i")
        elif pol == ["e", "l"]:
            return BuildSymbol("le", [abs(pair[1]), abs(pair[0])], "i")
        elif pol == ["l", "l"]:
            return BuildSymbol("ll", sorted([abs(x) for x in pair]), "i")
        else:
            assert False, (
                "Unrecognized polarization types for legs ("
                + str(pair[0])
                + ","
                + str(pair[1])
                + ")"
            )


def MRingToPoly(r, polarization_map={}):
    """
    Convert an MRing object [r] to a sympy polynomial.
    Converts mandelstam pair generators (i,j) to sympy monomials pp_ij
    using the PairSymbols() function.
    """
    mypoly = 0
    for key, val in r.Mdict.items():
        key_product = 1
        for pair in key:
            key_product *= PairSymbols(pair, polarization_map)
        mypoly += key_product * val
    return mypoly


def MRatToExpr(q, polarization_map={}):
    """
    Convert an MRational object [q] to a sympy expression (rational expression
    generated from sympy polynomials). Each MRing object within the MRational
    object is converted to a sympy polynomial using MRingToPoly().
    """
    mypoly = 0
    for pair in q.nd_list:
        numerator = MRingToPoly(pair[0], polarization_map)
        denominator = 1
        for r, p in pair[1].items():
            dfactor = MRingToPoly(r, polarization_map) ** p
            denominator *= dfactor
        # kludge
        mypoly += numerator / (factor(denominator))
    return mypoly


def SplitSymbol(sym):
    """
    Variation of the color tensor parsing algorithm. This splits the header, or prefix
    (the tensor ID) from the index substring and returns the header as a string, and a
    list of indices as integers.
    """
    symstring = sym.__str__()
    if "_{" not in symstring:
        return False

    indexblock = list(symstring.split("{")[1].split("}")[0])
    prefix = symstring.split("_")[0]

    indices = []
    indexstring = ""
    lastprefix = indexblock.pop(0)
    while True:
        char = indexblock.pop(0)
        if char.isalpha():
            index = int(indexstring)
            indices.append((lastprefix, index))
            lastprefix = char
            indexstring = ""
        elif len(indexblock) == 0:
            indexstring += char
            index = int(indexstring)
            indices.append((lastprefix, index))
            break
        else:
            indexstring += char
    return prefix, indices


def KillZeroMasses(q, massless_prefixes):
    """
    Kill all masses with indices bearing the given prefixes.
    This is very useful for killing all masses corresponding
    to pion indices, for example.
    """
    subdict = {}
    for sym in q.free_symbols:
        head, prefixlist, indexlist = SplitTensorSymbol(sym)
        if head == "m_":
            assert len(prefixlist) == 1
            print("PREFIX: ", prefixlist[0])
            print("MASSLESSPREFIXES: ", massless_prefixes)
            if prefixlist[0] in massless_prefixes:
                subdict[sym] = 0
    q = q.subs(subdict)
    return q


# def KillZeroMasses(q,zeromass_list):
# 	"""
# 	Given a list of mass indices which should be set to zero,
# 	searches a sympy rational for mass tensors m_i and kills
# 	those with indices in the list.
# 	"""
# 	subdict={}
# 	for sym in q.free_symbols:
# 		prefix,indexpairs = SplitSymbol(sym)
# 		if ((prefix=='m') and (indexpairs[0][0] in zeromass_list)):
# 			subdict[sym]=0
# 	q = q.subs(subdict)
# 	return q
#


def SymbolFourToThree(sym, particle_type_map):
    """
    Converts xy_{ij} mandelstam 4-invariants into non-covariant
    expressions in terms of the 3-invariants xy3_{ij},masses, and signs.
    Unit Tested 1/30/20
    """
    prefix, indexpairs = SplitSymbol(sym)
    indexpair = (indexpairs[0][1], indexpairs[1][1])

    if prefix == "pp":
        s1 = BuildSymbol("s", indexpair[0], particle_type_map)
        s2 = BuildSymbol("s", indexpair[1], particle_type_map)
        m1 = BuildSymbol("m", indexpair[0], particle_type_map)
        m2 = BuildSymbol("m", indexpair[1], particle_type_map)
        pp11 = BuildSymbol(
            "pp3", [indexpair[0], indexpair[0]], particle_type_map)
        pp22 = BuildSymbol(
            "pp3", [indexpair[1], indexpair[1]], particle_type_map)
        pp12 = BuildSymbol(
            "pp3", [indexpair[0], indexpair[1]], particle_type_map)
        target = (
            s1
            * s2
            * (
                (m1 ** 2 + pp11) ** Rational(1, 2)
                * (m2 ** 2 + pp22) ** Rational(1, 2)
                - pp12
            )
        )
    elif prefix == "pe":
        s1 = BuildSymbol("s", indexpair[0], particle_type_map)
        pe12 = BuildSymbol(
            "pe3", [indexpair[0], indexpair[1]], particle_type_map)
        target = (-1) * s1 * pe12
    elif prefix == "pl":
        s1 = BuildSymbol("s", indexpair[0], particle_type_map)
        s2 = BuildSymbol("s", indexpair[1], particle_type_map)
        m1 = BuildSymbol("m", indexpair[0], particle_type_map)
        m2 = BuildSymbol("m", indexpair[1], particle_type_map)
        pp11 = BuildSymbol(
            "pp3", [indexpair[0], indexpair[0]], particle_type_map)
        pp22 = BuildSymbol(
            "pp3", [indexpair[1], indexpair[1]], particle_type_map)
        pp12 = BuildSymbol(
            "pp3", [indexpair[0], indexpair[1]], particle_type_map)
        target = (s1 * s2 / m2) * (
            sqrt(m1 ** 2 + pp11) * sqrt(pp22)
            - sqrt(m2 ** 2 + pp22) * pp12 / sqrt(pp22)
        )
    elif prefix == "ee":
        ee12 = BuildSymbol(
            "ee3", [indexpair[0], indexpair[1]], particle_type_map)
        target = (-1) * ee12
    elif prefix == "le":
        s1 = BuildSymbol("s", indexpair[0], particle_type_map)
        m1 = BuildSymbol("m", indexpair[0], particle_type_map)
        pp11 = BuildSymbol(
            "pp3", [indexpair[0], indexpair[0]], particle_type_map)
        pe12 = BuildSymbol(
            "pe3", [indexpair[0], indexpair[1]], particle_type_map)
        target = (-1) * s1 * sqrt(m1 ** 2 + pp11) * pe12 / (m1 * sqrt(pp11))
    elif prefix == "ll":
        s1 = BuildSymbol("s", indexpair[0], particle_type_map)
        s2 = BuildSymbol("s", indexpair[1], particle_type_map)
        m1 = BuildSymbol("m", indexpair[0], particle_type_map)
        m2 = BuildSymbol("m", indexpair[1], particle_type_map)
        pp11 = BuildSymbol(
            "pp3", [indexpair[0], indexpair[0]], particle_type_map)
        pp22 = BuildSymbol(
            "pp3", [indexpair[1], indexpair[1]], particle_type_map)
        pp12 = BuildSymbol(
            "pp3", [indexpair[0], indexpair[1]], particle_type_map)
        target = (s1 * s2 / (m1 * m2)) * (
            sqrt(pp11) * sqrt(pp22)
            - sqrt(m1 ** 2 + pp11)
            * sqrt(m2 ** 2 + pp22)
            * pp12
            / (sqrt(pp11) * sqrt(pp22))
        )
    else:
        assert False, "Unregognized kinematic tensor prefix: " + prefix

    return target


def FourToThree(q, particle_type_map):
    """
    Converts full sympy expression involving xy_{ij} 4-invariants into corresponding
    expressions in terms of xy3_{ij} 3-invariants, m_{i} masses, and s_{i} momentum
    sign invariants. Calls SymbolFourToThree() on all 4-invariants in the expression.
    Takes sympy expression [q] and dict [particle_type_map].
    """
    # Filter kinematic tensor symbols.
    kinematic_prefixes = ["pp", "pe", "pl", "ee", "le", "ll"]
    subdict = {
        sym: SymbolFourToThree(sym, particle_type_map)
        for sym in q.free_symbols
        if any([prefix + "_{" in sym.__str__() for prefix in kinematic_prefixes])
    }
    q = q.subs(subdict)
    return q


# def ZScale3D(q,z,prefix,ip='all'):
# 	"""
# 	Perform z-scaling in the 3-invariant representation. We have set up this covariance-breaking representation in order to accommodate such scalings!
# 	[q] is the expression to scale, [prefix] the prefixes of the monomials in [q] which need scaling, and [ip] determines which external leg 3-momentum
# 	is to be scaled. If [ip] is an integer, then the corresponding 3-momentum is scaled. If it is the string 'all', then all 3-momenta are scaled.
# 	The scalings are applied by multiplying the various pp3 by the appropriate power of z.
# 	"""
# 	#FIXME: currently only handling pp symbols. Need to generalize to include spin.
# 	#NOTE: z must be real and positive, so we can trust PBLaurentCoefficient().
# 	if ip=='all':
# 		subdict = {sym:z**2*sym for sym in q.free_symbols if (prefix in sym.__str__())}
# 	else:
# 		subdict = {}
# 		for sym in q.free_symbols:
# 			power=0
# 			if prefix in sym.__str__():
# 				prefix,indices = SplitSymbol(sym)
# 				for tag,index in indices:
# 					if index==ip:
# 						power+=1
# 			if power>0:
# 				subdict[sym] = z**power*sym
#
# 		#subdict = {sym:symbols('z',real=True,positive=True)*sym for sym in q.free_symbols if (('pp' in sym.__str__()) and ('i'+str(ip) in sym.__str__()))}
# 	q = q.subs(subdict)
# 	return q


def ZScale(q, z, ip="all", dim=4):
    """
    Perform z-scaling in the 4-invariant representation. [q] is the expression to scale, [prefix] the prefixes of the monomials in [q] which need
    scaling, and [ip] determines which external leg 4-momentum is to be scaled. If [ip] is an integer, then the corresponding 4-momentum is scaled.
    If it is the string 'all', then all 4-momenta are scaled. The scalings are applied by multiplying the various pp by the appropriate power of z.
    """
    # NOTE: z must be real and positive, so we can trust PBLaurentCoefficient().

    if dim == 4:
        pp_prefix = "pp_"
        pe_prefix = "pe_"
    elif dim == 3:
        pp_prefix = "pp3_"
        pe_prefix = "pe3_"
    else:
        assert (
            False
        ), "Spacetimes of dimension other than three and four not currently supported"

    subdict = {}
    for sym in q.free_symbols:
        if pp_prefix in sym.__str__():
            if ip == "all":
                zpow = 2
            else:
                prefix, indices = SplitSymbol(sym)
                labels = [indices[i][1] for i in range(len(indices))]
                zpow = 0
                for l in labels:
                    if l == ip:
                        zpow += 1
            subdict[sym] = z ** zpow * sym

        elif pe_prefix in sym.__str__():
            if ip == "all":
                zpow = 1
            else:
                prefix, indices = SplitSymbol(sym)
                labels = [indices[i][1] for i in range(len(indices))]
                if labels[0] == ip:
                    zpow = 1
                else:
                    zpow = 0
            subdict[sym] = z ** zpow * sym

    q = q.subs(subdict)
    return q


"""
This was a hybrid attempt to project out components of polarizations parallel to 4-momenta without breaking covariance.
In the high energy limit, these expressions become complex. This might not be a problem, but since we have a better
grasp of the covariance-broken expressions, we'll stick with that approach for now.
"""
# def Spin1Projection(q):
# 	#Project epsilon.p cross terms.
# 	subdict = {}
# 	for sym in q.free_symbols:
# 		if 'pe_' in sym.__str__():
# 			prefix,indices = SplitSymbol(sym)
# 			i,j = [indices[i][1] for i in range(len(indices))]
# 			if (i==j):
# 				subdict[sym]=0
# 			else:
# 				subdict[sym]=(BuildSymbol('pe',[i,j],"i") - BuildSymbol('pe',[j,j],"i")*BuildSymbol('pp',[i,j],"i")/BuildSymbol('pp',[j,j],"i"))*(1-BuildSymbol('pe',[j,j],"i")**2/BuildSymbol('pp',[j,j],"i"))**Rational(-1,2)
# 		elif 'ee_' in sym.__str__():
# 			prefix,indices = SplitSymbol(sym)
# 			i,j = [indices[i][1] for i in range(len(indices))]
# 			if (i==j):
# 				subdict[sym]=1
# 			else:
# 				subdict[sym]=(BuildSymbol('ee',[i,j],"i") - BuildSymbol('pe',[i,i],"i")*BuildSymbol('pe',[i,j],"i")/BuildSymbol('pp',[i,i],"i") - BuildSymbol('pe',[j,i],"i")*BuildSymbol('pe',[i,j],"i")/BuildSymbol('pp',[i,i],"i"))*(1-BuildSymbol('pe',[j,j],"i")**2/BuildSymbol('pp',[j,j],"i"))**Rational(-1,2)


# def Spin1OnShell(q,n):
# 	#First, project epsilons.
# 	#Next, enforce p_n on-shell.
# 	#Handle e.e invariants.
# 	subdict = {}
# 	for sym in q.free_symbols:
# 		if 'ee_' in sym.__str__():
# 			prefix,indices = SplitSymbol(sym)
# 			i,j = sorted([indices[i][1] for i in range(len(indices))])
# 			if (i==j):
# 				subdict[sym] = 1
# 			else:
# 				subdict[sym] = (BuildSymbol('ee',[i,j],"i") - BuildSymbol('pe',[i,i],"i")*BuildSymbol('pe',[i,j],"i")/BuildSymbol('pp',[i,i],"i") - BuildSymbol('pe',[j,j],"i")*BuildSymbol('pe',[j,i],"i")/BuildSymbol('pp',[j,j],"i") + BuildSymbol('pe',[i,i],"i")*BuildSymbol('pe',[j,j],"i")*BuildSymbol('pp',[i,j],"i")/(BuildSymbol('pp',[i,i],"i")*BuildSymbol('pp',[j,j],"i")))/((1 - BuildSymbol('pe',[i,i],"i")**2/BuildSymbol('pp',[i,i],"i"))**Rational(1,2)*(1 - BuildSymbol('pe',[j,j],"i")**2/BuildSymbol('pp',[j,j],"i"))**Rational(1,2))
#
# 	#q = q.subs(subdict)
#
#
# 	#Handle p.e invariants.
# 	#subdict = {}
# 	for sym in q.free_symbols:
# 		if 'pe_' in sym.__str__():
# 			prefix,indices = SplitSymbol(sym)
# 			i,j = [indices[i][1] for i in range(len(indices))]
# 			if (i==j):
# 				subdict[sym] = 0
# 			else:
# 				subdict[sym] = (BuildSymbol('pe',[i,j],"i") - BuildSymbol('pe',[j,j],"i")*BuildSymbol('pp',[i,j],"i")/BuildSymbol('pp',[j,j],"i"))/((1 - BuildSymbol('pe',[j,j],"i")**2/BuildSymbol('pp',[j,j],"i"))**Rational(1,2))
#
# 	q = q.subs(subdict)
#
# 	#Next, enforce momentum conservation.
# 	subdict = {}
# 	for sym in q.free_symbols:
# 		if 'pp_' in sym.__str__():
# 			prefix,indices = SplitSymbol(sym)
# 			i,j = sorted([indices[i][1] for i in range(len(indices))])
# 			if (i!=n and j!=n):
# 				pass
# 			elif (i!=n and j==n):
# 				subdict[sym]=(-1)*sum([BuildSymbol('pp',sorted([i,k]),"i") for k in range(1,n)])
# 			elif (i==n and j==n):
# 				pass
#
# 		elif 'pe_' in sym.__str__():
# 			prefix,indices = SplitSymbol(sym)
# 			i,j = [indices[i][1] for i in range(len(indices))]
# 			if i==n:
# 				subdict[sym]=(-1)*sum([BuildSymbol('pe',[k,j],"i") for k in range(1,n)])
# 			else:
# 				pass
#
# 		else:
# 			pass
#
# 	q = q.subs(subdict)
#
# 	#Next, enforce p_n on-shell.
# 	subdict = {}
# 	for sym in q.free_symbols:
# 		if 'pp_' in sym.__str__():
# 			prefix,indices = SplitSymbol(sym)
# 			i,j = sorted([indices[i][1] for i in range(len(indices))])
# 			if (i==n-2 and j==n-1):
# 				subdict[sym] = Rational(1,2)*(BuildSymbol('pp',[n,n],"i") - sum([BuildSymbol('pp',[r,r],"i") for r in range(1,n)]))-(sum([sum([BuildSymbol('pp',[r,s],"i") for r in range(1,s)]) for s in range(2,n)])- BuildSymbol('pp',[n-2,n-1],"i"))
# 			else:
# 				pass
# 	q = q.subs(subdict)
#
# 	#Finally, eject masses from kinematic invariant ring.
# 	subdict = {}
# 	for sym in q.free_symbols:
# 		if 'pp_' in sym.__str__():
# 			prefix,indices = SplitSymbol(sym)
# 			i,j = sorted([indices[i][1] for i in range(len(indices))])
# 			if (i==j):
# 				subdict[sym] = symbols('m_{i'+str(i)+'}')**2
# 			else:
# 				pass
# 	q = q.subs(subdict)
#
# 	return q


def Spin1OnShell(q, n):
    """
    Enforce on-shell relations for mandelstam 4-invariants.
    Passed unit tests 1/30/20
    """
    # First, enforce momentum conservation.
    subdict = {}
    for sym in q.free_symbols:
        if "pp_" in sym.__str__():
            prefix, indices = SplitSymbol(sym)
            i, j = sorted([indices[i][1] for i in range(len(indices))])
            if i != n and j == n:
                subdict[sym] = (-1) * sum(
                    [BuildSymbol("pp", sorted([i, k]), "i")
                     for k in range(1, n)]
                )
            elif i == n and j == n:
                # This case will be mapped to a mass.
                # We will get the same result whether or not we make this replacment here, but
                # we reduce expression bloat by avoiding it.
                pass
            else:
                pass
        elif "pe_" in sym.__str__():
            prefix, indices = SplitSymbol(sym)
            i, j = [indices[i][1] for i in range(len(indices))]
            if i == n and j != n:
                subdict[sym] = (-1) * sum(
                    [BuildSymbol("pe", [k, j], "i") for k in range(1, n)]
                )
            if i == n and j == n:
                # This case will be mapped to zero.
                # We will get the same result whether or not we make this replacment here, but
                # we reduce expression bloat by avoiding it.
                pass
            else:
                pass
        elif "pl_" in sym.__str__():
            prefix, indices = SplitSymbol(sym)
            i, j = [indices[i][1] for i in range(len(indices))]
            if i == n and j != n:
                subdict[sym] = (-1) * sum(
                    [BuildSymbol("pl", [k, j], "i") for k in range(1, n)]
                )
            if i == n and j == n:
                # This case will be mapped to zero.
                # We will get the same result whether or not we make this replacment here, but
                # we reduce expression bloat by avoiding it.
                pass
            else:
                pass
    q = q.subs(subdict)

    subdict = {}
    # Enforce p_n*p_n = m_n**2.
    for sym in q.free_symbols:
        if "pp_" in sym.__str__():
            prefix, indices = SplitSymbol(sym)
            i, j = sorted([indices[i][1] for i in range(len(indices))])
            if i == n - 2 and j == n - 1:
                subdict[sym] = Rational(1, 2) * (
                    BuildSymbol("pp", [n, n], "i")
                    - sum([BuildSymbol("pp", [r, r], "i")
                           for r in range(1, n)])
                ) - (
                    sum(
                        [
                            sum([BuildSymbol("pp", [r, s], "i")
                                 for r in range(1, s)])
                            for s in range(2, n)
                        ]
                    )
                    - BuildSymbol("pp", [n - 2, n - 1], "i")
                )
            else:
                pass
    # Enforce p_n*e_n=0.
    for sym in q.free_symbols:
        if "pe_" in sym.__str__():
            prefix, indices = SplitSymbol(sym)
            i, j = [indices[i][1] for i in range(len(indices))]
            if i == n - 1 and j == n:
                subdict[sym] = (-1) * sum(
                    [BuildSymbol("pe", [r, n], "i") for r in range(1, n - 1)]
                )
            else:
                pass
    # Enforce p_n*l_n=0.
    for sym in q.free_symbols:
        if "pl_" in sym.__str__():
            prefix, indices = SplitSymbol(sym)
            i, j = [indices[i][1] for i in range(len(indices))]
            if i == n - 1 and j == n:
                subdict[sym] = (-1) * sum(
                    [BuildSymbol("pl", [r, n], "i") for r in range(1, n - 1)]
                )
            else:
                pass
    q = q.subs(subdict)

    subdict = {}
    for sym in q.free_symbols:
        # Handle e.e invariants.
        if "ee_" in sym.__str__():
            prefix, indices = SplitSymbol(sym)
            i, j = [indices[i][1] for i in range(len(indices))]
            if i == j:
                subdict[sym] = -1
        elif "le_" in sym.__str__():
            prefix, indices = SplitSymbol(sym)
            i, j = [indices[i][1] for i in range(len(indices))]
            if i == j:
                subdict[sym] = 0
        elif "ll_" in sym.__str__():
            prefix, indices = SplitSymbol(sym)
            i, j = [indices[i][1] for i in range(len(indices))]
            if i == j:
                subdict[sym] = -1
        # Handle p.e invariants.
        elif "pe_" in sym.__str__():
            prefix, indices = SplitSymbol(sym)
            i, j = [indices[i][1] for i in range(len(indices))]
            if i == j:
                subdict[sym] = 0
        elif "pl_" in sym.__str__():
            prefix, indices = SplitSymbol(sym)
            i, j = [indices[i][1] for i in range(len(indices))]
            if i == j:
                subdict[sym] = 0
        # Handle masses.
        elif "pp_" in sym.__str__():
            prefix, indices = SplitSymbol(sym)
            i, j = [indices[i][1] for i in range(len(indices))]
            if i == j:
                subdict[sym] = symbols("m_{i" + str(i) + "}") ** 2
        else:
            pass
    q = q.subs(subdict)

    return q


def ComputeLeadingSingularPower(expr, sym, algcheck=False):
    """
            WARNING:: Assumes [sym] in principal branch of [expr].
            See docs for PBLaurentCoefficient() for details.
            Compute the leading power of sym in the limit sym->zero.
    """

    # Check for algebraic expression.
    # Conditional is to prevent redundant checking at every recursive step.
    if not algcheck:
        assert (
            expr.is_algebraic_expr()
        ), "Leading power simplification only applies to algebraic expressions."
        algcheck = True

    # Recursive step, compute extracted expressions for all args (daughters) in expr,
    # as well as leading powers list for all args.
    # If expr is a leaf, loop will not execute, as args=().
    leading_powers = []
    for arg in expr.args:
        leading_power = ComputeLeadingSingularPower(arg, sym, algcheck)
        leading_powers.append(leading_power)

    # Work step. Extract leading power of sym for each of Add,Mul,Pow,Sym, and Const/other symbol cases.
    # Each case yields new_expr (extracted expression) and leading_power of sym in expr.
    # if expr.func==sympify("f+g").func: #Add case
    if FuncType(expr) == "add":  # Add case
        leading_power = min(leading_powers)
    # elif expr.func==sympify("f*g").func: #Mul case
    elif FuncType(expr) == "mul":  # Mul case
        leading_power = sum(leading_powers)
    # elif expr.func==sympify("f**2").func: #Pow case
    elif FuncType(expr) == "pow":  # Pow case
        leading_power = leading_powers[0] * expr.args[1]
    # Leaf cases below.
    elif expr == sym:  # Symbol case
        leading_power = 1
    else:  # Should just be other symbols and constants remaining, if expression is algebraic.
        leading_power = 0

    return leading_power


def UnitizeTrailingPowers(expr, sym, algcheck=False):
    """
            WARNING:: Assumes [sym] in principal branch of [expr].
            See docs for PBLaurentCoefficient() for details.
            Extracts inverse powers of sym from all nested expressions.
            This isolates the singularity in the sym->0 limit as a prefactor sym**pow,
            and greatly simplifies the computation of a laurent series.
    """

    # Check for algebraic expression.
    # Conditional is to prevent redundant checking at every recursive step.
    if not algcheck:
        assert (
            expr.is_algebraic_expr()
        ), "Leading power simplification only applies to algebraic expressions."
        algcheck = True

    # Recursive step, compute extracted expressions for all args (daughters) in expr,
    # as well as trailing powers list for all args.
    # If expr is a leaf, loop will not execute, as args=().
    new_args = []
    trailing_powers = []
    for arg in expr.args:
        new_arg, trailing_power = UnitizeTrailingPowers(arg, sym, algcheck)
        new_args.append(new_arg)
        trailing_powers.append(trailing_power)

    # Work step. Unitize trailing power of sym for each of Add,Mul,Pow,Sym, and Const/other symbol cases.
    # Each case yields new_expr (extracted expression) and trailing_power of sym in expr.
    # if expr.func==sympify("f+g").func: #Add case
    if FuncType(expr) == "add":  # Add case
        trailing_power = min(trailing_powers)
        if trailing_power != 0:
            new_expr = Mul(
                Pow(sym, trailing_power),
                Add(*[Mul(arg, Pow(sym, -trailing_power))
                      for arg in new_args]),
            )
        else:
            new_expr = Add(*new_args)
    # elif expr.func==sympify("f*g").func: #Mul case
    elif FuncType(expr) == "mul":  # Mul case
        trailing_power = sum(trailing_powers)
        new_expr = Mul(*new_args)
    # elif expr.func==sympify("f**2").func: #Pow case
    elif FuncType(expr) == "pow":  # Pow case
        trailing_power = trailing_powers[0] * new_args[1]
        new_expr = Mul(
            Pow(sym, trailing_power),
            Pow(Mul(Pow(sym, -trailing_powers[0]), new_args[0]), new_args[1]),
        )
    # Leaf cases below.
    elif expr == sym:  # Symbol case
        trailing_power = 1
        new_expr = expr
    else:  # Should just be other symbols and constants remaining, if expression is algebraic.
        trailing_power = 0
        new_expr = expr

    return new_expr, trailing_power


def PBLaurentCoefficient(expr, var, origin, order):
    """
    Short for "Principal Branch Laurent Coefficient"
    Computes the laurent coefficient of function [expr] of variable [var]
    about [origin] at order [order] in [var]. We assume expr does not have
    an essential singularity at [origin], and that expr is algebraic. Furthermore,
    we must assume [origin] is not a branch point and that there exists a simple contour
    about [origin] in the complex plane which is contained within the principal branch of
    all irrational functions within [expr]. Concretely, if [expr] contains sqrt(var), then
    [origin] must lie within a neighborhood of the complex plane contained within the half-plane
    with positive real coefficient. This way, (var^2)^(1/2)=var, and there is no pi-phase
    ambiguitiy. [expr] may contain any number of irrational functions, and the intersection
    of their principal branchs will give the principal branch of [expr] (FIXME: check this)
    This ambiguity-free condition is *critical* to the validity of the workhorse
    algorithm, UnitizeTrailingPowers(). (There may be an overall phase ambiguity in the LC
    when taking coefficients around the infinite pole, but these are irrelevant for our unitarity
    arguments).

    None of the limit points we're interested in for kinematic purposes are termini of branch cuts
    (the singularity is always isolated, wherever a square root appears, by the m^2 term
    in E^2; when m^2=0, there's no square root in the first place).

    Tested with expr = (a*x**(-2) + b*x**2 + c*x+d)/sqrt(c*x**2+1)
    at origin=0 and oo and const. point g.

    A final note: why not just use the sympy series() function after simplification?
    It works fine for [origin] = 0, but it struggles intermittently with oo, and eats up
    enormous amounts of time working on symbolic origins. For example, the only way to
    compute a laurent series about oo of an expression containing square roots of the
    form sqrt(a+b*z^2) was to assume that every single symbol in the expression was
    real and positive. Making that assumption in cases where it does not apply could lead
    to incorrect manipulations of roots in the expressions. This algorithm has been written
    *only for algebraic functions*, and *only when [var] is in the principal branch of the
    expression*. Narrowing down the domain of applicability allows us to write a simple
    algorithm that finds the answer quickly.
    """

    # Change variables to take contour about (1/z)=0.
    if origin == oo:
        expr = expr.subs({var: 1 / var})
        return PBLaurentCoefficient(expr, var, 0, -order)
    else:
        pass

    # Regularize expr and keep track of the power required to do so.
    # This yields the highest-order singularity.
    expr = expr.subs({var: var + origin})
    expr, n_singular = UnitizeTrailingPowers(expr, var)
    if n_singular > 0:
        n_singular = 0
    if n_singular < 0:
        expr *= var ** (-n_singular)
        # expr = expr.cancel()
        # Don't appear to need cancel()! Mul() autocancels.
        # Plus, cancel() consumes an *absurd* number of cycles.

    # If the highest-order singularity is still smaller than the requested order,
    # the z^order laurent coefficient is zero, and we return accordingly.
    # Otherwise, apply the cauchy integral formula to represent the laurent coefficient
    # in terms of a derivative. Evaluate, then return.
    if order < n_singular:
        return 0
    else:
        diff_order = order - n_singular
        result = expr.diff(var, diff_order)
        # Check that there's no singularity out front which will blow up when we evaluate
        # [var] at [origin].
        # One might wonder, why not just use sympy's limit() function?
        # We tried this, but found that it eats up unnecessary cycles, probably trying to handle
        # cases that we've excluded. Since we've done all the work to isolate the 1/[var]^p singularity
        # which indexes the laurent coefficient, we should reap the speed reward by performing subs()
        # instead of .limit().
        leading_power = ComputeLeadingSingularPower(result, var)
        # This will catch cases like 1/sqrt([var]) @ [var]->0, where the laurent coefficient is undefined.
        assert leading_power >= 0, (
            "Leading power in "
            + var.__str__()
            + "-> 0 limit is "
            + str(leading_power)
            + ". The laurent coefficient is, therefore, undefined at this order."
        )
        result = result.subs({var: 0})
        result *= Rational(1, factorial(diff_order))
        return result


# Below, we define some functions for extracting coefficient constraints from the condition of a vanishing UV or IR
# laurent coefficient of some amplitude.


def ExtractNumerator(expr):
    """
    Extract numerator from cancelled expression in p/q form. Remember to cancel the arg to this function
    with sympy.cancel() *and* check if we've violated a hilbert space contraction rule in the denominators
    by doing so!
    """
    stripped_args = []
    for arg in expr.args:
        # if arg.func==sympify('x**2').func:
        if FuncTools(arg) == "pow":
            if arg.args[-1] < 0:
                pass
            else:
                stripped_args.append(arg)
        else:
            stripped_args.append(arg)
    numerator_expr = Mul(*stripped_args)
    return numerator_expr


def CollectKinematicDenominators(expr, dim=4):
    """
    Multiply all terms by denominator LCM. Return resulting numerator and LCM.
    """
    # Fixme, should really be n or n-1...
    assert dim in [3, 4], "Dimensions other than 3,4 not supported"
    if dim == 4:
        kinematic_prefixes = ["pp", "pe", "pl", "ee", "le", "ll"]
    elif dim == 3:
        kinematic_prefixes = ["pp3", "pe3", "ee3"]

    # Define some functions for use within this function
    def IsKinematic(expr):
        for prefix in kinematic_prefixes:
            if prefix + "_{" in expr.__str__():
                return True
        return False

    def KinematicContent(expr):
        symbol_bools = [IsKinematic(sym) for sym in expr.free_symbols]
        if all(symbol_bools):
            return "kin"
        else:
            if any(symbol_bools):
                return "mixed"
            else:
                return "coeff"

    def DenominatorKinematicFactor(expr):
        expr = expr.factor()
        # if expr.func==sympify("x*y").func:
        if FuncTools(expr) == "mul":
            kinematic_factor = 1
            algebraic_factor = 1
            for factor in expr.args:
                kinematic_content = KinematicContent(factor)
                assert kinematic_content != "mixed"
                if kinematic_content == "kin":
                    kinematic_factor *= factor
                else:
                    algebraic_factor *= factor
            return kinematic_factor
        else:
            kinematic_content = KinematicContent(expr)
            assert kinematic_content != "mixed"
            if kinematic_content == "kin":
                return expr
            else:
                return 1

    def TermKinematicFactor(term):
        # assert term.func!=sympify("x+y").func
        assert FuncType(term) != "add", "Unexpected sum."
        kinematic_factor = 1
        # if term.func==sympify("x*y").func:
        if FuncType(term) == "mul":
            for factor in term.args:
                # if factor.func==sympify("x**2").func:
                if FuncType(factor) == "pow":
                    if factor.args[1] < 0:
                        kinematic_factor *= DenominatorKinematicFactor(
                            factor.args[0])
        # elif term.func==sympify("x**2").func:
        elif FuncType(term) == "pow":
            if factor.args[1] < 0:
                kinematic_factor *= DenominatorKinematicFactor(factor.args[0])
        else:
            pass
        return kinematic_factor

    def LCMReduceTerm(term, expr_lcm):
        # assert term.func!=sympify("x+y").func
        assert FuncType(term) != "add", "Unexpected sum."
        reduced_term = 1
        kinematic_factor = 1
        # if term.func==sympify("x*y").func:
        if FuncType(term) == "mul":
            for factor in term.args:
                # if factor.func==sympify("x**2").func:
                if FuncType(factor) == "pow":
                    print("POWFAC: ", factor)
                    if factor.args[1] < 0:
                        print("NEGATIVE")
                        print("DKF: ", DenominatorKinematicFactor(
                            factor.args[0]))
                        kinematic_factor *= DenominatorKinematicFactor(
                            factor.args[0])
                    else:
                        print("POSITIVE")
                        reduced_term *= factor
                else:
                    reduced_term *= factor
        # elif term.func==sympify("x**2").func:
        elif FuncType(term) == "pow":
            if factor.args[1] < 0:
                kinematic_factor *= DenominatorKinematicFactor(factor.args[0])
            else:
                reduced_term = term
        else:
            reduced_term = term

        print("REDTERM: ", reduced_term)

        print("KINFAC: ", kinematic_factor)

        multiplier = expr_lcm / kinematic_factor
        multiplier = multiplier.simplify()
        print("MULT: ", multiplier)
        reduced_term *= multiplier

        return reduced_term

    expr = expr.expand()
    kinematic_denominator_factors = [
        1,
    ]
    # if expr.func==sympify("x+y").func:
    if FuncType(expr) == "add":
        for term in expr.args:
            kinematic_denominator_factors.append(TermKinematicFactor(term))
    else:
        # kinematic_denominator_factors.append(TermKinematicFactor(term))
        kinematic_denominator_factors.append(TermKinematicFactor(expr))

    kinematic_denominator_factors = list(set(kinematic_denominator_factors))

    denom_lcm = 1
    for factor in kinematic_denominator_factors:
        denom_lcm = lcm(denom_lcm, factor)
    print("COMPUTED DLCM")

    print("DLCM: ")
    print(denom_lcm)
    denom_lcm = denom_lcm.factor()
    print("FACTORED DLCM")
    # assert denom_lcm.func!=sympify("x+y").func
    assert FuncType(denom_lcm) != "add", "Unexpected sum."
    # assert denom_lcm.func!=sympify("x**2").func
    assert FuncType(denom_lcm) != "pow", "Unexpected power."

    reduced_expr = 0
    print("BEGINNING LCMREDUCTIONS.")
    print("EXPR: ", expr)
    # if expr.func==sympify("x+y").func:
    if FuncType(expr) == "add":
        for term in expr.args:
            reduced_expr += LCMReduceTerm(term, denom_lcm)
    else:
        reduced_expr = LCMReduceTerm(expr, denom_lcm)

    return reduced_expr, kinematic_denominator_factors


def KinematicRingCollect(expr, dim=4):
    """
    Split numerator_expr into dictionary of kinematic monomials and corresponding coefficients.
    """
    # Fixme, should really be n or n-1...
    assert dim in [3, 4], "Dimensions other than 3,4 not supported"
    if dim == 4:
        kinematic_prefixes = ["pp", "pe", "pl", "ee", "le", "ll"]
    elif dim == 3:
        kinematic_prefixes = ["pp3", "pe3", "ee3"]

    # Define some functions for use within this function
    def IsKinematic(expr):
        for prefix in kinematic_prefixes:
            if prefix + "_{" in expr.__str__():
                return True
        return False

    def KinematicContent(expr):
        symbol_bools = [IsKinematic(sym) for sym in expr.free_symbols]
        # If the only factor is "i", that will show up as a ground domain element,
        # not as a free symbol. Need to include this in the "coeff" case.
        if list(expr.free_symbols) == []:
            return "coeff"
        elif all(symbol_bools):
            return "kin"
        else:
            if any(symbol_bools):
                return "mixed"
            else:
                return "coeff"

    def ExprKinematicFactor(expr):
        expr = expr.factor()
        # if expr.func==sympify("x*y").func:
        if FuncType(expr) == "mul":
            kinematic_factor = 1
            algebraic_factor = 1
            for factor in expr.args:
                kinematic_content = KinematicContent(factor)
                assert kinematic_content != "mixed"
                if kinematic_content == "kin":
                    kinematic_factor *= factor
                else:
                    algebraic_factor *= factor
            return kinematic_factor, algebraic_factor
        else:
            kinematic_content = KinematicContent(expr)
            assert kinematic_content != "mixed"
            if kinematic_content == "kin":
                return expr, 1
            else:
                return 1, expr

    expr = expr.expand()
    kinebraic_dict = {}
    # if expr.func==sympify("x+y").func:
    if FuncType(expr) == "add":
        for term in expr.args:
            # assert term.func!=sympify("x+y").func
            assert FuncType(term) != "add", "Unexpected sum."
            kinematic_factor, algebraic_factor = ExprKinematicFactor(term)
            if not type(kinematic_factor) == type(1):
                kinematic_factor = kinematic_factor.simplify()
            kinebraic_dict.setdefault(kinematic_factor, 0)
            kinebraic_dict[kinematic_factor] += algebraic_factor
    else:
        kinematic_factor, algebraic_factor = ExprKinematicFactor(expr)
        if not type(kinematic_factor) == type(1):
            kinematic_factor = kinematic_factor.simplify()
        kinebraic_dict.setdefault(kinematic_factor, 0)
        kinebraic_dict[kinematic_factor] += algebraic_factor

    return kinebraic_dict


def ReduceRedundantConstraints(kinebraic_dict):
    algebraic_constraints = []
    # Try factorization?
    for constraint in kinebraic_dict.values():
        # if constraint.func==sympify("x*y").func:
        if FuncType(constraint) == "mul":
            stripped_constraint = 1
            for factor in constraint.args:
                if IsSympySymbolic(factor):
                    stripped_constraint *= factor
        else:
            stripped_constraint = constraint

        # algebraic_constraints.append(stripped_constraint.simplify())
        algebraic_constraints.append(stripped_constraint)
    return list(set(algebraic_constraints))
