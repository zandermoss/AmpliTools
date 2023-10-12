from .mring import MRing
from .permutation_tools import monomial_signature, monomial_tag, symmetric_orbit, minimal_tag, tuple_compare
from itertools import permutations,product
import pickle
from sympy import Rational, poly, symbols
from hashable_containers import hmap


class Basis():
	"""A minimal basis of Mandelstam invariants.

	Provides methods for conversion of ``MRational`` and ``MRing`` objects
	to a minimal, on-shell basis of invariants. Currently handles scalars only.
	"""
	def __init__(self,spin,npoint,symmetric=False):
		self.spin = spin
		self.npoint = npoint
		self.symmetric = symmetric
		self.symbolblocks = []
		for i in range(1,self.npoint+1):
			self.symbolblocks.append([-i,i,100+i])

	def eject_masses(self,r):
		""" Eject masses from MRing generator pairs to polynomial ring. """
		rnew = MRing(hmap())
		m_1 = symbols('m_1')
		for key in r.mdict.keys():
			masspoly = poly(1,m_1,domain='QQ_I')
			newkey = []
			for pair in key:
				if (pair[0]==pair[1]) and (pair[0]<0):
					symbol = symbols('m_'+str(abs(pair[0])))
					massfactor = poly(symbol**2,symbol,domain='QQ_I') 
					masspoly*=massfactor
				else:
					newkey.append(pair)
			if len(newkey)==0:
				newkey.append((0,0))
			newkey = tuple(newkey)
			p = r.mdict[key]
			masspoly = masspoly.exclude()
			masspoly = masspoly.set_domain('QQ_I')
			p*=masspoly
			rnew.mdict.setdefault(newkey,r.poly_zero())
			rnew+=MRing(hmap({newkey:p}))
		rnew.cull_zeros()
		return rnew

	def zero_masses(self,r,masses):
		rnew = MRing(r)
		for mass in masses:
			rnew = rnew.evaluate_poly(mass,0)
		rnew.cull_zeros()
		return rnew

	def group_masses(self,r,massmap):
		rnew = MRing(r)
		for key in massmap:
			for mass in key:
				target_poly = poly(massmap[key],massmap[key],domain='QQ_I')
				#rnew = rnew.MonomialReplacement(mass,target_poly)
				#rnew = rnew.repl_replacement(mass,target_poly)
				rnew = rnew.repl_replacement(mass,massmap[key])
		rnew.cull_zeros()
		return rnew

	def onshell_restriction(self,r):
			rnew = MRing(r)
			momentumtarget = [[-1,-i] for i in range(1,self.npoint)]
			rnew = rnew.replacement({-self.npoint:momentumtarget})
			pairtarget = list()
			pairtarget.append([Rational(1,2),(-self.npoint,-self.npoint)])
			pairtarget+=[[Rational(-1,2),(-i,-i)] for i in range(1,self.npoint)]
			for i in range(1,self.npoint-2):
				for j in range(i+1,self.npoint):
						pairtarget.append([-1,(-j,-i)])
			rnew = rnew.pair_replacement({(-(self.npoint-1),-(self.npoint-2)):pairtarget})

			#Applies only to symmetric tensors!
			if self.spin==1 or self.spin==2:
				for i in range(1,self.npoint):
					rnew = rnew.zero_pair((-i,i))
				pairtarget=[[Rational(-1,1),(-i,self.npoint)] for i in range(1,self.npoint-1)]
				rnew = rnew.pair_replacement({(-(self.npoint-1),self.npoint):pairtarget})

			rnew.cull_zeros()
			return rnew
