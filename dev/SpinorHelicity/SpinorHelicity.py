#! /usr/bin/python

import functools
from fractions import Fraction


"""
A class to store and evaluate spinor helicity expressions. 
Angle brackets are denoted (+1,(a,b)), square brackets denoted
(-1,(a,b)).
We are constructing elements of a vector space with a basis
defined by rational functions of the spinor helicity variables
over the complex field (complex floats) 
FIXME: big caveat here, we're not checking that the given fractions
of spinor helicity variables are actually independent, nor are we, 
for the time being, implementing the necessary algebraic techniques 
to reduce some spanning list of these functions to a basis for the span.
This means that all comparisons of amplitudes expressed in this format
will need to be done by evaluation over the complex numbers.
Functions to to exactly this are given in Comparison.py.
"""
class SpinorHelicity(object):
	def __init__(self,arg):
		if type(arg) == type({}):
			self.shdict = {}
			for key in arg.keys():
				sign,sortedkey = self.KeySort(key)
				self.CheckInitKey(self.shdict,sortedkey)
				if sign==1:
					self.shdict[sortedkey]+=arg[key]
				else:
					self.shdict[sortedkey]-=arg[key]
					
			#self.CullZeros()
		elif type(arg) == type(self) and arg.__class__.__name__ == self.__class__.__name__:
			self.shdict = {}
			for key in arg.shdict.keys():
				sortedkey = self.KeySort(key)
				self.CheckInitKey(self.shdict,sortedkey)
				if sign==1:
					self.shdict[sortedkey]+=arg.shdict[key]
				else:
					self.shdict[sortedkey]-=arg.shdict[key]
			#self.CullZeros()
		else:
			assert False
	
	def Compare(self,var1,var2):
		if var1[0]!=var2[0]:
			return var1[0] - var2[0]
		elif var1[1][0]!=var2[1][0]:
			return var1[1][0]-var2[1][0]
		else:
			return var1[1][1]-var2[1][1]
		
	def KeySort(self,key):
		key_sorted=[]	
		sign=1
		for m in key:
			if len(m)==0:
				m_sorted = m
			else:
				m_sorted=[]
				for var in range(len(m)):
					if tuple(sorted(m[var][1]))!=m[var][1]:
						#print "KEYSORT: ",m[var][1], "    ",tuple(sorted(m[var][1]))
						sign*=-1
					m_sorted.append((m[var][0],tuple(sorted(m[var][1]))))
				m_sorted = sorted(m_sorted,key=functools.cmp_to_key(self.Compare))
			key_sorted.append(tuple(m_sorted))
		return sign,tuple(key_sorted)
	
	def CheckInitKey(self,mydict,key):
		mydict.setdefault(key,complex(0,0))

	def Evaluate(self,bracket_dict):
		"""
		Evaluate the spinor helicity expression given dictionaries of values 		 of all spinor helicity variables over the complex floats.
		"""
		target_scalar = complex(0,0)
		for key in self.shdict.keys():
			target_numerator = complex(1,0)
			for term in key[0]:
				target_numerator*=bracket_dict[term]
			target_denominator = complex(1,0)
			for term in key[1]:
				target_denominator*=bracket_dict[term]
			target_numerator/=target_denominator
			target_numerator*=self.shdict[key]
			target_scalar+=target_numerator
		return target_scalar

	def Print(self):
		for key in self.shdict.keys():
			numerator_string=""
			for var in key[0]:
				if var[0]==1:
					numerator_string+="<"+str(var[1][0])+str(var[1][1])+">"	
				else:
					numerator_string+="["+str(var[1][0])+str(var[1][1])+"]"	
			denominator_string=""
			for var in key[1]:
				if var[0]==1:
					denominator_string+="<"+str(var[1][0])+str(var[1][1])+">"	
				else:
					denominator_string+="["+str(var[1][0])+str(var[1][1])+"]"	
			output = "("+str(self.shdict[key])+")"+"("+numerator_string+"/"+denominator_string+")"
			print output

	def CullZeros(self):
		for key in self.shdict.keys():
			#FIXME: Comparison to zero is now a problem!
			if self.shdict[key]==complex(0,0):
				del self.shdict[key]

#	def Add(self,shvec):
#		for key in shvec.shdict.keys():
#			self.CheckInitKey(self.shdict,key)
#			self.shdict[key].Add(shvec.shdict[key])
#		self.CullZeros()
#
#	def FuseDict(self,dictA,dictB):
#		"""
#		Check that there are no redundant labels. That will
#		screw everything up!
#		"""
#		newdict={}
#		for keyA in dictA.keys():
#			for keyB in dictB.keys():
#				workgr = GaussRational(dictA[keyA])
#				workgr.Multiply(dictB[keyB])
#				numerator = keyA[0]+keyB[0]
#				denominator = keyA[1]+keyB[1]
#				newkey = self.KeySort((numerator,denominator))
#				self.CheckInitKey(newdict,newkey)
#				newdict[newkey].Add(workgr)
#		self.CullZeros()
#		return newdict
#
#	def Fuse(self,shvec):
#		return SpinorHelicity(self.FuseDict(self.shdict,shvec.shdict))
