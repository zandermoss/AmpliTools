from mring import MRing
from permutation_tools import monomial_signature, monomial_tag, symmetric_orbit, minimal_tag, tuple_compare
from itertools import permutations,product
import pickle
from sympy import Rational, poly, symbols


class Basis():
	def __init__(self,spin,npoint,symmetric=False):
		self.spin = spin
		self.npoint = npoint
		self.symmetric = symmetric
		self.symbolblocks = []
		for i in range(1,self.npoint+1):
			self.symbolblocks.append([-i,i,100+i])

	def eject_masses(self,r):
		""" Eject masses from MRing generator pairs to polynomial ring. """
		rnew = MRing({})
		m_1 = symbols('m_1')
		for key in r.Mdict.keys():
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
			p = r.Mdict[key]
			masspoly = masspoly.exclude()
			masspoly = masspoly.set_domain('QQ_I')
			p*=masspoly
			rnew.Mdict.setdefault(newkey,r.PolyZero())
			rnew+=MRing({newkey:p})
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
				#rnew = rnew.ReplReplacement(mass,target_poly)
				rnew = rnew.ReplReplacement(mass,massmap[key])
		rnew.cull_zeros()
		return rnew

	def onshell_restriction(self,r):
			rnew = MRing(r)
			momentumtarget = [[-1,-i] for i in range(1,self.npoint)]
			rnew = rnew.Replacement({-self.npoint:momentumtarget})
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
	
#
#	def ScrubCoefficients(self,mr):
#	   for key in mr.Mdict.keys():
#		   #NOTE: Changed this from (1,) poly.
#		   mr.Mdict[key] = Poly({(0,):Rational(1)})
#
#	def KeyCompare(self,A,B):
#		"""
#		Takes keys A and B, computes their tags, and returns the ordering
#		of the tags using tuple_compare().
#		"""
#		TA = monomial_tag(A,self.symbolblocks)
#		TB = monomial_tag(B,self.symbolblocks)
#		return tuple_compare(TA,TB)
#	
##===========================Power Counting============================#
#	#Below, we generate all monomial countings consistent with the 
#	#given power counting.
#	
#	def ZListSum(self,A,B):
#		return [a+b for a,b in zip(A,B)]
#	def ZListMultiply(self,A,x):
#		return [a*x for a in A]
#	
#	def FlattenRound(self,tlist):
#		flat_tlist = []
#		for x in tlist:
#			if type(x)==int:
#				flat_tlist.append((x,))
#			else:
#				flat_tlist.append(x)
#		return sum(flat_tlist,())
#	
#	def IsFlat(self,tlist):
#		flat = True
#		for x in tlist:
#			if type(x)!=int:
#				flat = False
#		return flat
#	
#	def Flatten(self,tlist):
#		while self.IsFlat(tlist)==False:
#			tlist = self.FlattenRound(tlist)
#		return tlist
#	
#	def GetMaxPower(self,source,target):
#		less = True
#		n=0
#		while less:
#			n+=1
#			test = self.ZListMultiply(source,n)
#			for k,m in zip(test,target):
#				if k>m:
#					less = False
#		return n-1
#	
#	def GetPowerCountings(self,basis_powers,generator_powers):
#		#Generate an overcomplete list of powers of the generators.
#		bases = [self.GetMaxPower(power, basis_powers) for power in generator_powers]
#		ranges = [range(b+1) for b in bases]
#		prod = list(reduce(product, ranges))
#		pcombos = [self.Flatten(p) for p in prod]
#		#Prune the list to those monomials with the correct power counting.
#		powercountings=[]
#		for plist in pcombos:
#			sumlist = [0 for i in range(len(plist))]
#			for n,p in zip(plist,generator_powers):
#				sumlist = self.ZListSum(sumlist,self.ZListMultiply(p,n))
#			if sumlist==basis_powers:
#				powercountings.append(plist)
#		return powercountings
#	
##=========================Invariant Generation=====================#
#
#	def GetInvariants(self):
#		""" Generates a complete set (off-shell) of mandelstam invariants
#			for spin-1 at n-point. FIXME: Implement for higher spin. """
#
#		assert self.spin==1
#		
#		e = set([i for i in range(1,self.npoint+1)])
#		p = set([-i for i in range(1,self.npoint+1)])
#		
#		ee = list(product(e,e))
#		ee = list(set([tuple(sorted(pair)) for pair in ee]))
#		ep = list(product(e,p))
#		ep = list(set([tuple(sorted(pair)) for pair in ep]))
#		pp = list(product(p,p))
#		pp = list(set([tuple(sorted(pair)) for pair in pp]))
#		
#		eedict = {(pair,):Poly({(0,):Rational(1)}) for pair in ee}
#		mree = MRing(eedict)
#		epdict = {(pair,):Poly({(0,):Rational(1)}) for pair in ep}
#		mrep = MRing(epdict)
#		ppdict = {(pair,):Poly({(0,):Rational(1)}) for pair in pp}
#		mrpp = MRing(ppdict)
#		
#		generators = [mree,mrep,mrpp]
#		return generators
#
#
#		
##============================Set Products==========================#
#	#Here, we define some functions to prune out basis elements with more than one copy of any given epsilon or epsilon bar.
#	def IndexMap(self,n):
#		#FIXME: will break for hi-point
#		if n<10:
#			m = n-1
#		if n>10:
#			m = n-11 +4
#		return m
#
#	#FIXME: Testme!
#	def PruneNonlinearEpsilon(self,mr):
#		if (self.spin==2 and self.symmetric==True):
#			epsilon_count = 2
#		else:
#			epsilon_count = 1
#		workvec = MRing({})
#		for key in mr.Mdict.keys():
#			if (self.spin==2 and self.symmetric==False):
#				intlist = [0 for i in range(2*self.npoint)]
#			else:
#				intlist = [0 for i in range(self.npoint)]
#			for pair in key:
#				for p in pair:
#					if p>0:
#						intlist[self.IndexMap(p)]+=1
#			multilinear=True
#			for i in intlist:
#				if i!=epsilon_count:
#					multilinear=False
#			if multilinear:
#				workvec.Mdict[key] = mr.Mdict[key]
#		return workvec
#	
#	#Now, we're ready to step through each of the possible power countings, and take the appropriate set products. We do a round of pruning for polarization multilinearity after each fusion operation. Otherwise, if we leave the pruning to the end, the size of the intermediate Mvectors balloons, and the script will run all day.
#	def GetMultilinearSet(self,powercountings,generators):
#		mlset = MRing({})
#		#for counting in tqdm(powercountings):
#		for counting in powercountings:
#			first = True
#			mr = MRing({})
#			for i,n in enumerate(counting):
#				if n>0:
#					if first:
#						mr.Add(generators[i])
#						n-=1
#						first=False
#					for k in range(n):
#						mr = mr.Fuse(generators[i])
#						#mr = self.PruneNonlinearEpsilon(mr)
#			mr = self.PruneNonlinearEpsilon(mr)
#			mlset.Add(mr)
#		self.ScrubCoefficients(mlset)
#		return mlset
#	
##=========================Labeling Redundancies=====================#
#	
#	def GetBasis(self,mlset):
#		sigs = {}
##		for key in tqdm(mlset.Mdict.keys()):
#		for key in mlset.Mdict.keys():
#			tag = monomial_tag(key,self.symbolblocks)
#			sig = monomial_signature(key,self.symbolblocks,True)
#			sigs[key] = (tag,sig)
#		
#		simdict={}
#		for key in sigs:
#			tag,sig = sigs[key]
#			simdict.setdefault(sig, {tag:key})
#			simdict[sig][tag]=key
#		
#		reducedmonomials=[]
#		for sig in simdict:
#			mintag = minimal_tag(simdict[sig].keys())
#			reducedmonomials.append(simdict[sig][mintag])
#		
#		basis = MRing({})
#		for monomial in reducedmonomials:
#			#mr = MRing({monomial:Poly({(1,):Rational(1)})})
#			mr = MRing({monomial:Poly({(0,):Rational(1)})})
#			basis.Add(mr)
#		self.ScrubCoefficients(basis)
#		return basis
#	
##======================Label basis with coefficients=================#
#	
#	def LabelBasis(self,basis, extra_labels):
#		#We need enough coefficients to accommodate all basis elements.
#		#We also allow for extra coefficients to carry other data (like
#		#explicit dimensional dependence).
#		clist_size = len(basis.Mdict)+extra_labels
#		clist_base = [0 for i in range(clist_size)]
#		count=0
#		labelledbasis = MRing({})
#		sorted_keys = sorted(basis.Mdict.keys(),cmp=self.KeyCompare)
#		for key in sorted_keys:
#			clist = list(clist_base)
#			clist[count]=1
#			mr = MRing({key:Poly({tuple(clist):Rational(1)})})
#			labelledbasis.Add(mr)
#			count+=1
#		return labelledbasis
#	
#	
##================Utilities for eliminating forbidden pairs============#
#	def SetForbiddenPairs(self,fpairs):
#		self.forbiddenpairs = fpairs
#	
#	def PruneForbiddenPairs(self,mr):
#		workvec = MRing({})
#		for key in mr.Mdict.keys():
#			keep=True
#			for pair in key:
#				#FIXME: generalize for higher-point
#				if (-4) in pair:
#					keep=False
#				if pair in self.forbiddenpairs:
#					keep=False 
#				assert (pair not in self.errorpairs)
#			if keep:
#				workvec.Mdict[key] = mr.Mdict[key]
#		return workvec

#	def ReplaceForbiddenPairs(self,mr):
#		#FIXME: generalize for higher-point
#		#First, replace p4
#		mr.Replacement({-4:[[-1,-1],[-1,-2],[-1,-3]]})
#		#Next, replace forbidden pairs related into our minimal
#		#basis by (p4.e4)=0, (p4.ebar4)=0, (p4.p4)=0.
#		#FIXME: generalize! Don't hardcode!
#		mr.pair_replacement({(-3,1):[[-1,(-2,-1)],[-1,(-3,-2)]]})
#		mr.pair_replacement({(-3,4):[[-1,(-1,4)],[-1,(-2,4)]]})
#		mr.pair_replacement({(-3,14):[[-1,(-1,14)],[-1,(-2,14)]]})
#		#Check that no forbidden pairs remain!
#		for key in mr.Mdict.keys():
#			for pair in key:
#				assert (pair not in self.forbiddenpairs)
#				assert (pair not in self.errorpairs)
#==========================Generate a basis!==========================#
#	
#	def GenerateBasis(self,basis_powers,extra_labels):
#		assert self.spin==1
#		print "--------------------Basis Generation--------------------"
#		#Kludge for spin 1:
#		generator_powers = [[2,0],[1,1],[0,2]]
#		print "Computing Invariants"
#		generators = self.GetInvariants()
#		print "Generating power countings."
#		powercountings = self.GetPowerCountings(basis_powers,generator_powers)
#		print "Generating multilinear set. Iterating over power countings."
#		multilinearset = self.GetMultilinearSet(powercountings,generators)
#		#print "Removing relabelling redundancy."
#		#unlabelledbasis = self.GetBasis(multilinearset,symbolblocks)
#		print "Appending coefficients to basis elements."
#		#basis = self.LabelBasis(unlabelledbasis,extra_labels)
#		basis = self.LabelBasis(multilinearset,extra_labels)
#		pickle.dump(basis, open( "basis.p", "wb" ) )
#		pickle.dump(self.symbolblocks, open( "symbolblocks.p", "wb" ) )
#		print "--------------------------------------------------------"
#
#	def LoadBasis(self):
#		self.basis = pickle.load( open( "basis.p", "rb" ) )	
#		self.symbolblocks = pickle.load( open( "symbolblocks.p", "rb" ) )	
