#! /usr/bin/python

from TensorFeynman import MVector
from Poly import Poly
from fractions import Fraction
from PermutationTools import GetMonomialSignature, GetMonomialTag, SymmetricOrbit, GetMinimalTag, TupleCompare
from itertools import permutations,product
#from tqdm import tqdm
import pickle

class Basis():
	def __init__(self,spin,npoint,symmetric=False):
		self.spin = spin
		self.npoint = npoint
		self.symmetric = symmetric
		#FIXME: generalize to higher spin.
		assert self.spin==1
		#FIXME: pythonize!
		self.symbolblocks = []
		for i in range(1,self.npoint+1):
			self.symbolblocks.append([-i,i,100+i])

	def ScrubCoefficients(self,mv):
	   for key in mv.Mdict.keys():
		   #NOTE: Changed this from (1,) poly.
		   mv.Mdict[key] = Poly({(0,):Fraction(1)})

	def KeyCompare(self,A,B):
		"""
		Takes keys A and B, computes their tags, and returns the ordering
		of the tags using TupleCompare().
		"""
		TA = GetMonomialTag(A,self.symbolblocks)
		TB = GetMonomialTag(B,self.symbolblocks)
		return TupleCompare(TA,TB)
	
#===========================Power Counting============================#
	#Below, we generate all monomial countings consistent with the 
	#given power counting.
	
	def ZListSum(self,A,B):
		return [a+b for a,b in zip(A,B)]
	def ZListMultiply(self,A,x):
		return [a*x for a in A]
	
	def FlattenRound(self,tlist):
		flat_tlist = []
		for x in tlist:
			if type(x)==int:
				flat_tlist.append((x,))
			else:
				flat_tlist.append(x)
		return sum(flat_tlist,())
	
	def IsFlat(self,tlist):
		flat = True
		for x in tlist:
			if type(x)!=int:
				flat = False
		return flat
	
	def Flatten(self,tlist):
		while self.IsFlat(tlist)==False:
			tlist = self.FlattenRound(tlist)
		return tlist
	
	def GetMaxPower(self,source,target):
		less = True
		n=0
		while less:
			n+=1
			test = self.ZListMultiply(source,n)
			for k,m in zip(test,target):
				if k>m:
					less = False
		return n-1
	
	def GetPowerCountings(self,basis_powers,generator_powers):
		#Generate an overcomplete list of powers of the generators.
		bases = [self.GetMaxPower(power, basis_powers) for power in generator_powers]
		ranges = [range(b+1) for b in bases]
		prod = list(reduce(product, ranges))
		pcombos = [self.Flatten(p) for p in prod]
		#Prune the list to those monomials with the correct power counting.
		powercountings=[]
		for plist in pcombos:
			sumlist = [0 for i in range(len(plist))]
			for n,p in zip(plist,generator_powers):
				sumlist = self.ZListSum(sumlist,self.ZListMultiply(p,n))
			if sumlist==basis_powers:
				powercountings.append(plist)
		return powercountings
	
#=========================Invariant Generation=====================#

	def GetInvariants(self):
		""" Generates a complete set (off-shell) of mandelstam invariants
			for spin-1 at n-point. FIXME: Implement for higher spin. """

		assert self.spin==1
		
		e = set([i for i in range(1,self.npoint+1)])
		p = set([-i for i in range(1,self.npoint+1)])
		
		ee = list(product(e,e))
		ee = list(set([tuple(sorted(pair)) for pair in ee]))
		ep = list(product(e,p))
		ep = list(set([tuple(sorted(pair)) for pair in ep]))
		pp = list(product(p,p))
		pp = list(set([tuple(sorted(pair)) for pair in pp]))
		
		eedict = {(pair,):Poly({(0,):Fraction(1)}) for pair in ee}
		mvee = MVector(eedict)
		epdict = {(pair,):Poly({(0,):Fraction(1)}) for pair in ep}
		mvep = MVector(epdict)
		ppdict = {(pair,):Poly({(0,):Fraction(1)}) for pair in pp}
		mvpp = MVector(ppdict)
		
		generators = [mvee,mvep,mvpp]
		return generators


		
#============================Set Products==========================#
	#Here, we define some functions to prune out basis elements with more than one copy of any given epsilon or epsilon bar.
	def IndexMap(self,n):
		#FIXME: will break for hi-point
		if n<10:
			m = n-1
		if n>10:
			m = n-11 +4
		return m

	#FIXME: Testme!
	def PruneNonlinearEpsilon(self,mv):
		if (self.spin==2 and self.symmetric==True):
			epsilon_count = 2
		else:
			epsilon_count = 1
		workvec = MVector({})
		for key in mv.Mdict.keys():
			if (self.spin==2 and self.symmetric==False):
				intlist = [0 for i in range(2*self.npoint)]
			else:
				intlist = [0 for i in range(self.npoint)]
			for pair in key:
				for p in pair:
					if p>0:
						intlist[self.IndexMap(p)]+=1
			multilinear=True
			for i in intlist:
				if i!=epsilon_count:
					multilinear=False
			if multilinear:
				workvec.Mdict[key] = mv.Mdict[key]
		return workvec
	
	#Now, we're ready to step through each of the possible power countings, and take the appropriate set products. We do a round of pruning for polarization multilinearity after each fusion operation. Otherwise, if we leave the pruning to the end, the size of the intermediate Mvectors balloons, and the script will run all day.
	def GetMultilinearSet(self,powercountings,generators):
		mlset = MVector({})
		#for counting in tqdm(powercountings):
		for counting in powercountings:
			first = True
			mv = MVector({})
			for i,n in enumerate(counting):
				if n>0:
					if first:
						mv.Add(generators[i])
						n-=1
						first=False
					for k in range(n):
						mv = mv.Fuse(generators[i])
						#mv = self.PruneNonlinearEpsilon(mv)
			mv = self.PruneNonlinearEpsilon(mv)
			mlset.Add(mv)
		self.ScrubCoefficients(mlset)
		return mlset
	
#=========================Labelling Redundancies=====================#
	
	def GetBasis(self,mlset):
		sigs = {}
#		for key in tqdm(mlset.Mdict.keys()):
		for key in mlset.Mdict.keys():
			tag = GetMonomialTag(key,self.symbolblocks)
			sig = GetMonomialSignature(key,self.symbolblocks,True)
			sigs[key] = (tag,sig)
		
		simdict={}
		for key in sigs:
			tag,sig = sigs[key]
			simdict.setdefault(sig, {tag:key})
			simdict[sig][tag]=key
		
		reducedmonomials=[]
		for sig in simdict:
			mintag = GetMinimalTag(simdict[sig].keys())
			reducedmonomials.append(simdict[sig][mintag])
		
		basis = MVector({})
		for monomial in reducedmonomials:
			#mv = MVector({monomial:Poly({(1,):Fraction(1)})})
			mv = MVector({monomial:Poly({(0,):Fraction(1)})})
			basis.Add(mv)
		self.ScrubCoefficients(basis)
		return basis
	
#======================Label basis with coefficients=================#
	
	def LabelBasis(self,basis, extra_labels):
		#We need enough coefficients to accommodate all basis elements.
		#We also allow for extra coefficients to carry other data (like
		#explicit dimensional dependence).
		clist_size = len(basis.Mdict)+extra_labels
		clist_base = [0 for i in range(clist_size)]
		count=0
		labelledbasis = MVector({})
		sorted_keys = sorted(basis.Mdict.keys(),cmp=self.KeyCompare)
		for key in sorted_keys:
			clist = list(clist_base)
			clist[count]=1
			mv = MVector({key:Poly({tuple(clist):Fraction(1)})})
			labelledbasis.Add(mv)
			count+=1
		return labelledbasis
	
	
#================Utilities for eliminating forbidden pairs============#
	def SetForbiddenPairs(self,fpairs):
		self.forbiddenpairs = fpairs
	
#	def PruneForbiddenPairs(self,mv):
#		workvec = MVector({})
#		for key in mv.Mdict.keys():
#			keep=True
#			for pair in key:
#				#FIXME: generalize for higher-point
#				if (-4) in pair:
#					keep=False
#				if pair in self.forbiddenpairs:
#					keep=False 
#				assert (pair not in self.errorpairs)
#			if keep:
#				workvec.Mdict[key] = mv.Mdict[key]
#		return workvec

#	def ReplaceForbiddenPairs(self,mv):
#		#FIXME: generalize for higher-point
#		#First, replace p4
#		mv.Replacement({-4:[[-1,-1],[-1,-2],[-1,-3]]})
#		#Next, replace forbidden pairs related into our minimal
#		#basis by (p4.e4)=0, (p4.ebar4)=0, (p4.p4)=0.
#		#FIXME: generalize! Don't hardcode!
#		mv.PairReplacement({(-3,1):[[-1,(-2,-1)],[-1,(-3,-2)]]})
#		mv.PairReplacement({(-3,4):[[-1,(-1,4)],[-1,(-2,4)]]})
#		mv.PairReplacement({(-3,14):[[-1,(-1,14)],[-1,(-2,14)]]})
#		#Check that no forbidden pairs remain!
#		for key in mv.Mdict.keys():
#			for pair in key:
#				assert (pair not in self.forbiddenpairs)
#				assert (pair not in self.errorpairs)


	def ThreePointRestriction(self,mv):
		#FIX CONVENTIONS
		#FIXME: generalize for higher-point
		#First, replace p4
		mv.Replacement({-3:[[-1,-1],[-1,-2]]})
		mv.PairReplacement({(-1,3):[[-1,(-2,3)],]})
		mv.PairReplacement({(-1,103):[[-1,(-2,103)],]})
		mv.ZeroPair((-2,-1))
		mv.ZeroPair((-1,-1))
		mv.ZeroPair((-2,-2))
		mv.ZeroPair((-1,1))
		mv.ZeroPair((-2,2))
		mv.ZeroPair((-1,101))
		mv.ZeroPair((-2,102))
		mv.ZeroPair((1,1))
		mv.ZeroPair((2,2))
		mv.ZeroPair((3,3))
		mv.ZeroPair((101,101))
		mv.ZeroPair((102,102))
		mv.ZeroPair((103,103))

	def FourPointRestriction(self,mv):
		#FIXME: generalize for higher-point
		#First, replace p4
		mv.Replacement({-4:[[-1,-1],[-1,-2],[-1,-3]]})
		#Next, replace forbidden pairs related into our minimal
		#basis by (p4.e4)=0, (p4.ebar4)=0, (p4.p4)=0.
		#FIXME: generalize! Don't hardcode!

		mv.PairReplacement({(-3,-1):[[-1,(-2,-1)],[-1,(-3,-2)]]})
		mv.PairReplacement({(-3,4):[[-1,(-1,4)],[-1,(-2,4)]]})
		mv.PairReplacement({(-3,104):[[-1,(-1,104)],[-1,(-2,104)]]})
		mv.ZeroPair((-1,-1))
		mv.ZeroPair((-2,-2))
		mv.ZeroPair((-3,-3))
		mv.ZeroPair((-1,1))
		mv.ZeroPair((-2,2))
		mv.ZeroPair((-3,3))
		mv.ZeroPair((-1,101))
		mv.ZeroPair((-2,102))
		mv.ZeroPair((-3,103))
		mv.ZeroPair((1,1))
		mv.ZeroPair((2,2))
		mv.ZeroPair((3,3))
		mv.ZeroPair((4,4))
		mv.ZeroPair((101,101))
		mv.ZeroPair((102,102))
		mv.ZeroPair((103,103))
		mv.ZeroPair((104,104))


	def FivePointRestriction(self,mv):
		#FIXME: generalize for higher-point
		#First, replace p4
		mv.Replacement({-5:[[-1,-1],[-1,-2],[-1,-3],[-1,-4]]})
		#Next, replace forbidden pairs related into our minimal
		#basis by (p4.e4)=0, (p4.ebar4)=0, (p4.p4)=0.
		#FIXME: generalize! Don't hardcode!

		mv.PairReplacement({(-4,-3):[[-1,(-2,-1)],[-1,(-3,-1)],[-1,(-4,-1)],[-1,(-3,-2)],[-1,(-4,-2)]]})
		mv.PairReplacement({(-4,5):[[-1,(-1,5)],[-1,(-2,5)],[-1,(-3,5)]]})
		mv.ZeroPair((-1,-1))
		mv.ZeroPair((-2,-2))
		mv.ZeroPair((-3,-3))
		mv.ZeroPair((-4,-4))
		mv.ZeroPair((-1,1))
		mv.ZeroPair((-2,2))
		mv.ZeroPair((-3,3))
		mv.ZeroPair((-4,4))
		mv.ZeroPair((1,1))
		mv.ZeroPair((2,2))
		mv.ZeroPair((3,3))
		mv.ZeroPair((4,4))
		mv.ZeroPair((5,5))

	def OnShellRestriction(self,mv):
		if self.npoint==3:
			self.ThreePointRestriction(mv)
		elif self.npoint==4:
			self.FourPointRestriction(mv)
		elif self.npoint==5:
			self.FivePointRestriction(mv)
		else:
			print ">5 POINT NOT IMPLEMENTED"
		


	
#==========================Generate a basis!==========================#
	
	def GenerateBasis(self,basis_powers,extra_labels):
		assert self.spin==1
		print "--------------------Basis Generation--------------------"
		#Kludge for spin 1:
		generator_powers = [[2,0],[1,1],[0,2]]
		print "Computing Invariants"
		generators = self.GetInvariants()
		print "Generating power countings."
		powercountings = self.GetPowerCountings(basis_powers,generator_powers)
		print "Generating multilinear set. Iterating over power countings."
		multilinearset = self.GetMultilinearSet(powercountings,generators)
		#print "Removing relabelling redundancy."
		#unlabelledbasis = self.GetBasis(multilinearset,symbolblocks)
		print "Appending coefficients to basis elements."
		#basis = self.LabelBasis(unlabelledbasis,extra_labels)
		basis = self.LabelBasis(multilinearset,extra_labels)
		pickle.dump(basis, open( "basis.p", "wb" ) )
		pickle.dump(self.symbolblocks, open( "symbolblocks.p", "wb" ) )
		print "--------------------------------------------------------"

	def LoadBasis(self):
		self.basis = pickle.load( open( "basis.p", "rb" ) )	
		self.symbolblocks = pickle.load( open( "symbolblocks.p", "rb" ) )	
