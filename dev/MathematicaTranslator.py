#! /usr/bin/python
import re
import sys
from PermutationTools import GetCanonicalMonomial, GetMonomialTag, TupleCompare
from MRing import MRing
from itertools import permutations
from fractions import Fraction
#from Poly import Poly
from sympy import poly,symbols,Rational
import re
import pickle

#-----------------------Function Definitions----------------------#
def StripCoefficient(term):
	""" Separate tensors from coefficient of the term.
	We just look for the three possible symbols
	that could start the tensor block, and separate
	whatever comes before the earliest of these symbols.
	That's the coefficient.
	"""
	
	init_symlist = ["h","C","["]
	init_sympos = []
	for s in init_symlist:
		pos = term.find(s)
		if pos>-1:
			init_sympos.append(pos)
	first_sym_pos = min(init_sympos)

	revstring = term[::-1]
	rev_final_sympos = revstring.find("]")
	final_sympos = len(term)-1-rev_final_sympos
	coeff1 = term[0:first_sym_pos]
	if coeff1=='':
		coeff1 = str(1)
	if coeff1=='-':
		coeff1 = str(-1)

	coeff2 = term[final_sympos+1:]
	coeff = coeff1+coeff2
	if '/' in coeff:
		slashpos = coeff.find('/')
		num = int(coeff[0:slashpos])
		denom = int(coeff[slashpos+1:])
		coeff = Rational(num,denom)
	else:
		coeff = Rational(int(coeff),1)
	tensors = term[first_sym_pos:]
	return [coeff,tensors]

def GetL1BracketPositions(mystring):
	""" Here, we search through the string, looking for
	matching pairs of square brackets ("Catalan style").
	Each additional nesting takes us up a "nestlevel".
	This function finds the positions of the pairs of 
	[,] brackets that cross into the first nesting level.
	This function allows us to break up the tensor expression. 
	"""
	L1=[]
	myL1=[]
	nestlevel=0
	for i,c in enumerate(mystring):
		if c=='[':
			if nestlevel==0:
				myL1.append(i)	
			nestlevel+=1
		elif c==']':
			nestlevel-=1
			if nestlevel==0:
				myL1.append(i)
				L1.append(myL1)
				myL1=[]
	return L1

def GetL1Chunks(mystring):
	""" This function recursively un-nests square brackets that 
	mathematica uses to express repeated application of CD
	(covariant derivative). This function does not, however,
	remove the square brackets that enclose indices.
	"""
	L1 = GetL1BracketPositions(mystring)
	start=0
	chunks=[]
	for myL1 in L1:
		if mystring[start]!="[": 
			chunks.append(mystring[start:myL1[1]+1])
		else:
			chunks+=GetL1Chunks(mystring[start:myL1[1]+1].strip("["))
		start = myL1[1]+1
	return chunks



def GetHDict(tensorchunks):
	""" This function reads the un-nested tensor string,
		and creates an "HDict": a dictionary with integer
		keys labeling the various h tensors that appear in 
		the string (in arbitary order). Each value contains
		two lists. The first contains the lorentz index strings
		of the h tensor. The second contains the lorentz index
		strings of the derivatives acting on this h tensor, if any.
	"""
	hdict={}
	pindexlist=[]
	hnum=1
	for chunk in tensorchunks:
		if chunk[0]=="h":
			stripstring = chunk[2:].strip("[]")
			hdict[hnum]=[stripstring.split(","),pindexlist]
			hnum+=1
			pindexlist=[]
		elif chunk[0:2]=="CD":
			stripstring = chunk[2:].strip("[]")
			pindexlist.append(stripstring)
	return hdict

def GetEdges(hdict):
	""" GetEdges reads the indices of an hdict, and writes the 
	corresponding labelled graph as a tuple of edge tuples,
	(v1,v2), which is the (un-ordered) format used to key the 
	MDict object (see TensorFeynman module).
	"""
	pairs=[]
	signpair=["",""]
	#loop through the vertex keys.
	for key1 in hdict.keys():
		for i, sourcelist in enumerate(hdict[key1]):
			#Sign the index if it belongs to a momentum.
			if i==0:
				signpair[0]=""
			else:
				signpair[0]="-"
			#Loop through the index list, find the complementary
			#index elsewhere in the hdict (possibly in the same
			#index list), append the pair to pairs[], and remove each
			#element from the respective list. We use while loops
			#because removal mutates the list while we operate on it.
			while len(sourcelist)>0:
				index = sourcelist[0]
				if index[0]=="-":
					complement = index[1]
				else:
					complement = "-"+index		
		
				#Kludge to break out of nested loops upon finding 
				#the complementary index.	
				breakflag=False	
				for key2 in hdict.keys():
					for j,targetlist in enumerate(hdict[key2]):
						if complement in targetlist:
							if j==0:
								signpair[1]=""
							else:
								signpair[1]="-"
								
#							#-------------------------------
#							#Promoting to a "two-fold" theory
#							if signpair[1]=="":
#								modkey = key2+10
#							else:
#								modkey = key2
#							pair = (int(signpair[0]+str(key1)),int(signpair[1]+str(modkey)))
#							#--------------------------------

							pair = (int(signpair[0]+str(key1)),int(signpair[1]+str(key2)))
							pairs.append(pair)
							targetlist.remove(complement)
							breakflag=True
							break
					if breakflag:
						break	
				sourcelist.remove(index)
	return tuple(pairs)

def ParseInteractions(mystring):
	#Strip the trailing $.
	mystring = mystring.strip("$")
	#Remove spaces.
	mystring = mystring.replace(" ","")
	#Remove parentheses.
	mystring = mystring.replace("(","")
	mystring = mystring.replace(")","")
	#Remove product star.
	mystring = mystring.replace("*","")
	#Remove perturbation order information (we know it's 1, by construction).
	mystring = mystring.replace("LI[1],","")
	#Split string into list of terms
	termlist = mystring.split("$")
	#FIXME: number of h tensors!
	#(Check that it's uniform)!
	
	accum = MRing({})	
	for term in termlist:
		coeff,tensorstring = StripCoefficient(term)
		tensorchunks = GetL1Chunks(tensorstring)
		hdict = GetHDict(tensorchunks)
		edges = GetEdges(hdict)
		#Patch for new canonicalizer
		maxn = max([abs(x) for x in sum(edges,())])
		symbolblocks = [[-i,i] for i in range(1,maxn+1)]
		#canon_edges = GetCanonicalEdges(edges)
		canon_edges = GetCanonicalMonomial(edges,symbolblocks)

		r = MRing({canon_edges:poly(coeff,symbols('x'),domain='QQ(I)')})
		accum+=r
	return accum

#def EdgeTuple2SKEString(edges):
#	string = ""
#	factor = Fraction(1)
#	for edge in edges:
#		if edge[0]<0 and edge[1]<0:
#			var = "s"
#			factor*=Fraction(1,2)
#		elif edge[0]<0 and edge[1]>0:
#			var = "ke"
#		elif edge[0]>0 and edge[1]>0:
#			var = "ee"
#		num = str(abs(edge[0]))+str(abs(edge[1]))
#		string+=var+"["+str(abs(edge[0]))+","+str(abs(edge[1]))+"]*"
#	return string.strip('*'),factor
#
#def EdgeTuple2PEString(edges):
#	string = ""
#	for edge in edges:
#		if edge[0]<0 and edge[1]<0:
#			var = "pp"
#		elif edge[0]<0 and edge[1]>0:
#			var = "ke"
#		elif edge[0]>0 and edge[1]>0:
#			var = "ee"
#		num = str(abs(edge[0]))+str(abs(edge[1]))
#		string+=var+"["+str(abs(edge[0]))+","+str(abs(edge[1]))+"]*"
#	return string.strip('*')
#
#def EdgeTuple2FGHString(edges):
#	string = ""
#	for edge in edges:
#		if edge[0]<0 and edge[1]<0:
#			var = "F"
#		elif edge[0]<0 and edge[1]>0:
#			var = "G"
#		elif edge[0]>0 and edge[1]>0:
#			var = "H"
#		num = str(abs(edge[0]))+str(abs(edge[1]))
#		string += "Subscript["+var+","+num+"]"
#	return string
#
#def MVector2Mathematica(mvec):
#	string = ""
#	for key in mvec.Mdict.keys():
#		string += "("+mvec.Mdict[key].GetMathematicaString()+")"
#		string += "*"+EdgeTuple2FGHString(key) + "+"
#	string = string.strip("+")
#	return string
#
#def ReplaceDelimiters(source_string,source_delimiters,target_delimiters):
#	target_string = ""
#	i=0
#	while i<len(source_string):
#		if source_string[i]==source_delimiters[0][0]:
#			i+=len(source_delimiters[0])
#			index_target = str(target_delimiters[0])
#			while source_string[i]!=source_delimiters[1][0]:
#				index_target+=source_string[i]
#				i+=1
#			i+=len(source_delimiters[1])
#			index_target+=target_delimiters[1]
#			target_string+=index_target
#		else:
#			target_string+=source_string[i]
#			i+=1
#	return target_string
#
#def SPPoly2Mathematica(poly):
#	polystring = poly.as_expr()
#	polystring = polystring.replace('_{','[')
#	polystring = polystring.replace('}',']')
#	polystring = polystring.replace('**','^')
#	return polystring
#
#def MRing2MathematicaPE(ring):
#	string = ""
#	for key in ring.Mdict.keys():
#		string += "("+SPPoly2Mathematica(ring.Mdict[key])+")"
#		string += "*"+EdgeTuple2PEString(key) + "+"
#	string = string.strip("+")
#	return string
#
#def MRational2MathematicaPE(rat):
#	string = ""
#	for pair in rat.nd_list:
#		string+='('+MRing2MathematicaPE(pair[0])+')/'
#		dstring = ""
#		for ring,power in rat.nd_list[1]:
#			dstring+='('+MRing2MathematicaPE(ring)+')^'+str(power)
#		string+='('+dstring+')+'
#	string = string.strip('+')
#	return string
#
#def MVector2MathematicaSKE(mvec):
#	string = ""
#	for key in mvec.Mdict.keys():
#		edgestring,factor = EdgeTuple2SKEString(key)
#		coeff = Poly(mvec.Mdict[key].terms)
#		coeff.ScalarMultiply(factor)
#		string += "("+coeff.GetMathematicaString()+")"
#		string += "*"+edgestring + "+"
#	string = string.strip("+")
#	return string
#
##--------------------Mathematica SKE Format to MVector-----------------#
#
#def PMSplit(string):
#	splitlist = []
#	chunk=""
#	for char in string:
#		if char=="+":
#			splitlist.append(chunk)
#			chunk = "+"
#		elif char=="-":
#			splitlist.append(chunk)
#			chunk = "-"
#		elif char == "\n":
#			pass
#		else:
#			chunk+=char
#	splitlist.append(chunk)
#	if "" in splitlist:
#		splitlist.remove("")
#	return splitlist		
#
#def GetMTuple(factor,keymap):
#	for key in keymap.keys():
#		if key in factor:
#			signpair = keymap[key]
#			factor = factor.replace(key,"")
#			pair = factor.split(",")	
#			pairtuple = (int(pair[0][-1])*signpair[0],int(pair[1][0])*signpair[1])
#	return pairtuple
#
#def MathematicaSKE2MVector(ske_string):
#	""" Note, this will only read in mvectors with constant coefficients,
#	not polynomial coefficients. Make sure to call Togther[] before writing
#	mathematica string!"""
#
#	ske_string = ske_string.replace(" ","")
#	if "(" in ske_string:
#		superterms = re.split('\(|\)', ske_string)
#		if '' in superterms:
#			empty_index = superterms.index('')
#			if empty_index==2:
#				prefactor = Fraction(superterms[0].strip('*'))
#			elif empty_index==0:
#				prefactor = Fraction(1,int(superterms[2].strip('/')))
#			else:
#				assert False
#		else:
#			A = Fraction(superterms[0].strip('*'))
#			B = Fraction(superterms[2].strip('/'))
#			prefactor = A/B
#		expression = superterms[1]
#	else:
#		prefactor = Fraction(1)
#		expression = ske_string
#		
#	terms= PMSplit(expression) 
#	
#	keymap = {"ee":(1,1),"ke":(-1,1),"s":(-1,-1)}
#	keys = keymap.keys()
#	
#	coefficients = []
#	mandelstams = []
#	
#	for term in terms:
#		factors = term.split("*")
#		#Expand powers
#		for n,factor in enumerate(factors):
#			if "^" in factor:
#				factors.remove(factor)
#				pair = factor.split("^")
#				for n in range(int(pair[1])):
#					factors.append(pair[0]) 
#		#Extract Coefficients
#		for n,factor in enumerate(factors):
#			purecoeff = True
#			for key in keys:
#				if key in factor:
#					purecoeff = False
#			if purecoeff:
#				coefficients.append(int(factor))
#				factors.remove(factor)
#				mandelstams.append(factors)
#				break
#	
#			if "+" in factor:
#				coefficients.append(1)
#				factors[n] = factor.replace("+","")
#				mandelstams.append(factors)
#				break
#	
#			if "-" in factor:
#				coefficients.append(-1)
#				factors[n] = factor.replace("-","")
#				mandelstams.append(factors)
#				break
#	
#			coefficients.append(1)
#			mandelstams.append(factors)
#	
#	mtuples=[]
#	for mandel in mandelstams:	
#		mtuples.append(tuple([GetMTuple(pair,keymap) for pair in mandel]))
#	
#	mv = MVector({})
#	for mtuple,coeff in zip(mtuples,coefficients):
#		nfactors = 0
#		for pair in mtuple:
#			if pair[0]<0 and pair[1]<0:
#				nfactors+=1
#		mvterm = MVector({mtuple:Poly({(0,):Fraction(prefactor*(2**nfactors)*coeff)})})
#		mv.Add(mvterm)
#
#	return mv
#
#def KeyCompare(A,B,symbolblocks):
#	""" 
#	Takes keys A and B, computes their tags, and returns the ordering 
#	of the tags using TupleCompare(). 
#	"""
#	TA = GetMonomialTag(A,symbolblocks)
#	TB = GetMonomialTag(B,symbolblocks)
#	return TupleCompare(TA,TB)
#
#def GetSortedKeys(mv,symbolblocks):
#	#Curry the comparison function, fixing symbolblocks.
#	sorted_keys = sorted(mv.Mdict.keys(),cmp=lambda A,B: KeyCompare(A,B,symbolblocks))
#	return sorted_keys
#
#
#def Coeffs2Mathematica(mv,symbolblocks,filename):
#	polylist=[]
#	sorted_keys = sorted(mv.Mdict.keys(),cmp=lambda A,B:KeyCompare(A,B,symbolblocks))
#	pickle.dump(sorted_keys, open( "sorted_keys.p", "wb" ) )
#
#	for key in sorted_keys:
#		polylist.append(mv.Mdict[key])
#
#	pstringlist = []
#	for p in polylist:
#		pstringlist.append(p.GetMathematicaString())
#	eqnlist = [pstring for pstring in pstringlist]
#	eqnstring="{"
#	for neqn,eqn in enumerate(eqnlist):
#		if neqn>0:
#			eqnstring+=","
#		eqnstring+=eqn
#	eqnstring+="}"
#	text_file = open(filename, "w")
#	text_file.write(eqnstring)
#	text_file.close()
#
#def PolyList2Mathematica(polylist,filename):
#	pstringlist = []
#	for p in polylist:
#		pstringlist.append(p.GetMathematicaString())
#	eqnlist = [pstring+" == 0" for pstring in pstringlist]
#	eqnstring="{"
#	for neqn,eqn in enumerate(eqnlist):
#		if neqn>0:
#			eqnstring+=","
#		eqnstring+=eqn
#	eqnstring+="}"
#	text_file = open(filename, "w")
#	text_file.write(eqnstring)
#	text_file.close()
#
#
#
