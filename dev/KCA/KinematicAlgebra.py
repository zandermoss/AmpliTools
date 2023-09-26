#! /usr/bin/python

import shelve
from TensorFeynman import MVector, Join
from Poly import Poly
from fractions import Fraction
from PermutationTools import PermuteBlocks
import numpy as np
from itertools import permutations
from tqdm import tqdm

class TensorIterator(object):
	def __init__(self,rank,mod):
		self.rank = rank
		self.mod = mod
		self.counter = 0
		self.indices = [0 for i in range(self.rank)]
	def I(self,n):
		return self.indices[n]
	def Next(self):
		self.counter+=1
		r = self.counter
		for i in range(self.rank):
			d = (self.mod)**(self.rank-i-1)
			q = r/d
			r = r%d
			self.indices[i] = q
	def Done(self):
		if self.counter>=(self.mod)**(self.rank):
			return True
		else:
			return False
	def Reset(self):
		self.counter = 0

def FillTensor(T,_mv,symbolblocks,indices):
	perms = list(set(permutations(range(1,4))))
	for perm in perms:
		indexperm = [0,0,0]
		for i in range(3):
			indexperm[perm[i]-1] = indices[i]
		mv = MVector(_mv)
		PermuteBlocks(mv,perm,symbolblocks,True)
		if T[tuple(indexperm)]==None:
			T[tuple(indexperm)] = MVector(mv)
		else:
			T[tuple(indexperm)].Add(mv)
		#CULL ZEROS
		TI = TensorIterator(3,3)
		while not TI.Done():
			if T[tuple(TI.indices)]!=None:
				if T[tuple(TI.indices)].IsEmpty():
					T[tuple(TI.indices)] = None
			TI.Next()

def TensorPrint(T):
	""" Assuming rank 3 """
	TI = TensorIterator(3,3)
	while not TI.Done():
		print "({},{},{})".format(TI.I(0),TI.I(1),TI.I(2))
		if T[tuple(TI.indices)]==None:
			print "NONE"
		else:
			T[tuple(TI.indices)].Print()
		print
		TI.Next()

def TensorPrint4(T):
	""" Assuming rank 4 """
	TI = TensorIterator(4,3)
	while not TI.Done():
		print "({},{},{},{})".format(TI.I(0),TI.I(1),TI.I(2),TI.I(3))
		if T[tuple(TI.indices)]==None:
			print "NONE"
		else:
			T[tuple(TI.indices)].Print()
		print
		TI.Next()

def TensorSum(T):
	mvsum = MVector({})
	""" Assuming rank 4 """
	TI = TensorIterator(4,3)
	while not TI.Done():
		mvsum.Add(T[tuple(TI.indices)])
		TI.Next()
	return mvsum

def TensorJoin(A,B,G,pair):
	Ashape = list(A.shape)
	Arank = len(Ashape)
	Alegs = [i for i in range(1,Arank+1)]
	Ashape.pop()
	Bshape = list(B.shape)
	Brank = len(Bshape)
	Blegs = [i for i in range(1,Brank+1)]
	Bshape.pop(0)
	Cshape = Ashape+Bshape
	C = np.empty(Cshape,dtype=object)
	def index2rank(index):
		if i==0:
			return 1
		return 2	
	TI = TensorIterator(len(Cshape),3)
	total = 3**len(Cshape)
	while not TI.Done():
		d = total/10
		q = TI.counter/d
		r = TI.counter%d
		if r==0: 
			print "TensorJoin: {}% Complete".format(q*10)
	
		C[tuple(TI.indices)] = MVector({})
		#print "({},{},{},{})".format(TI.I(0),TI.I(1),TI.I(2),TI.I(3))
		for i in range(3):
			for j in range(3):
				if A[tuple(TI.indices[:Arank-1]+[i,])]==None:
					continue
				if B[tuple(TI.indices[Arank-1:]+[j,])]==None:
					continue
				if G[i,j]==None:
					continue
				#print "	[{},{}]".format(i,j)
				#print TI.indices[:Arank-1]+[i,]
				#A[tuple(TI.indices[:Arank-1]+[i,])].Print()
				#print "-----------"
				#print TI.indices[Arank-1:]+[j,]
				#B[tuple(TI.indices[Arank-1:]+[j,])].Print()
				product,s = Join(A[tuple(TI.indices[:Arank-1]+[i,])],Alegs,B[tuple(TI.indices[Arank-1:]+[j,])],Blegs,pair,G[i,j],index2rank(i))
				C[tuple(TI.indices)].Add(product)
				#print "*********************"
				#product.Print()
				#print "==============================================="
		TI.Next()
	return C
	
def GetPolyFormat(mvectors,mvector_names):
	lengths = [mv.GetPolyLength() for mv in mvectors]
	cumlengths = [sum(lengths[:n]) for n in range(0,len(lengths))]
	bounds = [[cumlengths[n]+1,cumlengths[n+1]] for n in range(0,len(cumlengths)-1)]
	bounds.append((cumlengths[-1]+1,sum(lengths)))
	format_dict = {tuple(key):val for key,val in zip(bounds,mvector_names)}
	print zip(bounds,mvector_names)
	return format_dict

def PrintBasisMVector(_reference,index,format_dict):
	reference = MVector(_reference)
	values = [0 for i in range(reference.GetPolyLength())]
	assert index>0
	values[index-1]=1
	reference.FullEvaluatePolynomials(values)
	def GetName(format_dict,index):
		for key in format_dict.keys():
			if (index>=key[0]) and (index<=key[1]):
				return format_dict[key]
		assert False

	print
	print "===================== "+str(index)+" from "+GetName(format_dict,index)+" ===================== "
	reference.Print()
	print "============================================== "

