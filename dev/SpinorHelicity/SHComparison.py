#! /usr/bin/python

from tqdm import tqdm
import random
import pickle
import numpy as np
from TensorFeynman import MVector
from Poly import Poly
from fractions import Fraction
from Spin1Basis import Basis
from PermutationTools import PermuteBlocks
from SpinorHelicity import SpinorHelicity
from math import sqrt, cos, acos, sin, asin, pi, atan, floor
import cmath
import matplotlib.pyplot as plt
from itertools import permutations

"""
This script performs exact comparisons of tree amplitudes in mandelstam-vector and spinor-helicity format. At tree level, both are complex functions over all input helicities and momenta. In particular, these amplitudes take python [complex] values, that is, pairs of floats. Roughly, the plan is to generate lists of random integers, and use them to build complex numbers with real and imaginary components taken as random samples from the set of primitive pythagorean triples. These numbers can represent any complex phase. Combining these random phases with a random rational magnitude appropriately, we can sample all possible square and angle brackets corresponding to an n-point tree amplitude. We generate many such samples, and use them to evaluate a spinor-helicity expression for a tree amplitude on one hand. On the other, we evaluate a mandelstam-vector amplitude in terms of the spinor-helicity brackets. We can then compare the random samples of the two amplitudes repeatedly, to determine whether they are eqivalent (or, at least proportional up to some real scalar).
""" 

#NOTE: Code below should work for 5 point. Fails for 4 point, need to think harder for higher point.

def EvaluatePair(pair,bracket_dict,pos_legs,neg_legs):
	r_plus = neg_legs[0]
	r_minus = pos_legs[0]
	i = abs(pair[0])
	j = abs(pair[1])
	if (pair[0]<0 and pair[1]<0):
		monomial = SpinorHelicity({(((1,(i,j)),(-1,(i,j))),()):complex(0.5,0)})
	elif (pair[0]<0 and pair[1]>0):
		if j in neg_legs:
			monomial = SpinorHelicity({(((1,(i,j)),(-1,(i,r_minus))),((-1,(j,r_minus)),)):1.0/sqrt(2)})
		elif j in pos_legs:
			monomial = SpinorHelicity({(((1,(r_plus,i)),(-1,(i,j))),((1,(r_plus,j)),)):1.0/sqrt(2)})
		else:
			assert False
	elif (pair[0]>0 and pair[1]>0):
		if ((i in neg_legs) and (j in pos_legs)):
			monomial = SpinorHelicity({(((1,(i,r_plus)),(-1,(r_minus,j))),((1,(r_plus,j)),(-1,(r_minus,i)))):complex(1,0)})
		elif ((i in pos_legs) and (j in neg_legs)):
			monomial = SpinorHelicity({(((1,(j,r_plus)),(-1,(r_minus,i))),((1,(r_plus,i)),(-1,(r_minus,j)))):complex(1,0)})
		elif ((i in pos_legs) and (j in pos_legs)):
			monomial = SpinorHelicity({(((1,(r_plus,r_plus)),(-1,(i,j))),((1,(r_plus,i)),(1,(r_plus,j)))):complex(1,0)})
		elif ((i in neg_legs) and (j in neg_legs)):
			monomial = SpinorHelicity({(((1,(i,j)),(-1,(r_minus,r_minus))),((-1,(r_minus,i)),(-1,(j,r_minus)))):complex(1,0)})
		else:
			assert False
	else:
		assert False
	
	return monomial.Evaluate(bracket_dict)

def EvaluateMVector(mv,bracket_dict,pos_legs,neg_legs):
	value = complex(0,0)
	for key in mv.Mdict.keys():
		assert len(mv.Mdict[key].terms.values())==1
		term  = complex(mv.Mdict[key].terms.values()[0],0)
		for pair in key:
			term*=EvaluatePair(pair,bracket_dict,pos_legs,neg_legs)
		value+=term
	return value

def GenerateRandom(extlegs):
	""" Generate lists of random complex floats, one for each
	external leg. These will be used to calcluate the square and angle
	bracket values used to compute a random sample of a spinor-helicity
	expression. """

	#Initialize the momentum constraint matrix.
	M = np.zeros([4,extlegs])

	halftheta_list = []
	phi_list = []
	for i in range(extlegs):
		theta = random.uniform(0.0,pi)
		halftheta_list.append(theta/2.0)
		phi = random.uniform(0.0,2.0*pi)
		phi_list.append(phi)
		M[0,i]=1.0
		M[1,i]=sin(theta)*cos(phi)
		M[2,i]=sin(theta)*sin(phi)
		M[3,i]=cos(theta)

	U,S,VH = np.linalg.svd(M)
	energy_list = VH.conj()[-1]

	cos_list = []
	sin_list = []
	phase_list = []
	modulus_list = []

	for i in range(extlegs):
		cos_list.append(complex(cos(halftheta_list[i]),0))
		sin_list.append(complex(sin(halftheta_list[i]),0))
		phase_list.append(complex(cos(phi_list[i]),sin(phi_list[i])))	
		modulus_list.append(cmath.sqrt(2.0*energy_list[i]))
	
	return cos_list,sin_list,phase_list,modulus_list	

def BuildBracketDict(extlegs):
	""" Uses the values generated in GenerateRandom() to compute <ij>
	and [ij] samples over the complex floats. Outputs a dictionary
	keyed with representations of the brackets, with these random samples
	as values. """
	cos_list,sin_list,phase_list,modulus_list = GenerateRandom(extlegs)
	bracket_dict = {}
	for i in range(extlegs):
		#Antisymmetry kills these two:
		bracket_dict[(1,(i+1,i+1))] = complex(0.0,0.0)
		bracket_dict[(-1,(i+1,i+1))] = complex(0.0,0.0)
		#[]=-<>* IFF p0>0!!!!
		for j in range(i+1,extlegs):
			T1 = cos_list[i]*sin_list[j]*phase_list[j]
			T2 = cos_list[j]*sin_list[i]*phase_list[i]
			T1-=T2
			T1*=modulus_list[i]*modulus_list[j]
			bracket_dict[(1,(i+1,j+1))] = T1
		for j in range(i+1,extlegs):
			T1 = cos_list[i]*sin_list[j]*(phase_list[j].conjugate())
			T2 = cos_list[j]*sin_list[i]*(phase_list[i].conjugate())
			T1-=T2
			T1*=modulus_list[i]*modulus_list[j]
			T1*=complex(-1,0)
			bracket_dict[(-1,(i+1,j+1))] = T1
	return bracket_dict


def CheckProportionality(mvector,shvector,poles,extlegs,pos_legs,neg_legs,nchecks):
	verbosity=0
	epsilon = 1e-6
	ratio=complex(0,0)
	#for i in tqdm(range(nchecks)):
	for i in range(nchecks):
		bracket_dict = BuildBracketDict(extlegs)
		mvector_value = EvaluateMVector(mvector,bracket_dict,pos_legs,neg_legs)
		shvector_value = shvector.Evaluate(bracket_dict) 
		shvector_value *= EvaluateMVector(poles,bracket_dict,pos_legs,neg_legs)
		new_ratio = mvector_value/shvector_value
		#FIXME: need to set some epsilon threshold for comparison!
		mod_diff = abs(ratio-new_ratio)
		if (i>0 and mod_diff>epsilon):
			return False
		if verbosity>0:
			print "========================================================"
			print "---------values---------"
			print "MV: ",mvector_value
			print "SH: ",shvector_value
			print
			print "-------Ratios------"
			print "Ratio: ",new_ratio
			print "ModDiff: ",mod_diff
			print "Epsilon: ",epsilon
			if verbosity>1:
				print
				print "Bracket Dict: "
				for key in bracket_dict.keys():
					print key, "   ",bracket_dict[key]
		ratio = new_ratio
	return True	

def CheckMHV(mv,poles,extlegs,nchecks=10):
	numerator = [(1,(1,2)) for k in range(4)]
	denominator = [(1,(k,k+1)) for k in range(1,extlegs)]
	denominator.append((1,(extlegs,1)))
	sh = SpinorHelicity({(tuple(numerator),tuple(denominator)):complex(1.0,0.0)})
	neg_legs = [1,2]
	pos_legs = [i for i in range(3,extlegs+1)]
	return CheckProportionality(mv,sh,poles,extlegs,pos_legs,neg_legs,nchecks)
