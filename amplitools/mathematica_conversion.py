from . import tensor_tools
from functools import reduce


"""Utilities for conversion of ``MRational``, ``MRing``, and ``sympy.poly``
expressions to and from a format compatible with Mathematica.

"""


def format_indices(source_string,delimiters):
	target_string = ""
	i=0
	while i<len(source_string):
		if source_string[i]==delimiters[0][0]:
			i+=len(delimiters[0])
			index_target=""
			#index_target = str(delimiters[0])
			while source_string[i]!=delimiters[1][0]:
				if source_string[i]=='i' or source_string[i]=='j':
					index_target+=','
				if source_string[i]=='i':
					i+=1
				index_target+=source_string[i]
				i+=1
			i+=len(delimiters[1])
			#index_target+=delimiters[1]
			index_target = index_target.strip(",")
			index_target = delimiters[0]+index_target+delimiters[1]
			target_string+=index_target
		else:
			target_string+=source_string[i]
			i+=1
	return target_string


def sympy_poly_to_mathematica(poly):
	polystring = str(poly.as_expr())
	polystring = polystring.replace('_{','[')
	polystring = polystring.replace('}',']')
	polystring = polystring.replace('**','^')
	polystring = format_indices(polystring,['[',']'])
	polystring = polystring.replace('alpha','[Alpha]')
	polystring = polystring.replace('beta','[Beta]')
	return polystring

def tensor_poly_to_mathematica(r):
	"""
	Convert a polynomial of A_{x1x2x3...} tensors to mathematica polynomial
	with variables x_{123...}. Takes a `MRational` object of the form poly/1.
	Returns string.
	"""
	# Check that r is `MRational` of the form poly/1
	if len(r.nd_list)==0:
		return "0",[]
	assert len(r.nd_list)==1, "Not a simple fraction"
	pair = r.nd_list[0]
	assert len(pair[1])==1 and list(pair[1].values())[0]==1, "Denominator not one."
	r_denom = list(pair[1].keys())[0]
	assert r_denom == r_denom.one(), "Denominator not one."
	r_num = pair[0]
	assert r_num.is_proportional(r_num.one()), "Numerator not poly"

	# Compute the mapping from AT tensor notation to Mathematica subscripts.
	tensor_symbols = r.tensor_symbols()
	string_map = {}
	for sym in tensor_symbols:
		head,prefix,index = tensor_tools.split_tensor_symbol(sym)
		istring = reduce(lambda x,y:str(x)+str(y),index)
		#Fix in case of single character
		istring = str(istring)
		string_map[str(sym)] = "Subscript["+head.replace("_","")+","+istring+"]"


	#Extract the numerator poly string and replace.
	pstring = str(list(r_num.mdict.values())[0].as_expr())

	target_string = pstring
	for src,tar in string_map.items():
		target_string = target_string.replace(src,tar)
	target_string = target_string.replace("**","^")
	return target_string,list(string_map.values())


def subspace_tensor_poly_to_mathematica(r):
	# Check that r is `MRational` of the form poly/1
	if len(r.nd_list)==0:
		return "0",[]
	assert len(r.nd_list)==1, "Not a simple fraction"
	pair = r.nd_list[0]
	assert len(pair[1])==1 and list(pair[1].values())[0]==1, "Denominator not one."
	r_denom = list(pair[1].keys())[0]
	assert r_denom == r_denom.one(), "Denominator not one."
	r_num = pair[0]
	assert r_num.is_proportional(r_num.one()), "Numerator not poly"

	# Compute the mapping from AT tensor notation to Mathematica subscripts.
	tensor_symbols = r.tensor_symbols()
	string_map = {}
	for sym in tensor_symbols:
		head,prefix,index = tensor_tools.split_tensor_symbol(sym)
		istring = ""
		for p,s in zip(prefix,index):
			istring+=str(p)+str(s)
		#istring = reduce(lambda x,y:str(x)+str(y),index)
		#Fix in case of single character
		istring = str(istring)
		string_map[str(sym)] = "Subscript["+head.replace("_","")+","+istring+"]"


	#Extract the numerator poly string and replace.
	pstring = str(list(r_num.mdict.values())[0].as_expr())

	target_string = pstring
	for src,tar in string_map.items():
		target_string = target_string.replace(src,tar)
	target_string = target_string.replace("**","^")
	return target_string,list(string_map.values())


def subspace_tensor_poly_to_sage(r):
	# Check that r is `MRational` of the form poly/1
	if len(r.nd_list)==0:
		return "0",[]
	assert len(r.nd_list)==1, "Not a simple fraction"
	pair = r.nd_list[0]
	assert len(pair[1])==1 and list(pair[1].values())[0]==1, "Denominator not one."
	r_denom = list(pair[1].keys())[0]
	assert r_denom == r_denom.one(), "Denominator not one."
	r_num = pair[0]
	assert r_num.is_proportional(r_num.one()), "Numerator not poly"

	# Compute the mapping from AT tensor notation to Mathematica subscripts.
	tensor_symbols = r.tensor_symbols()
	string_map = {}
	for sym in tensor_symbols:
		head,prefix,index = tensor_tools.split_tensor_symbol(sym)
		istring = ""
		for p,s in zip(prefix,index):
			istring+=str(p)+str(s)
		#istring = reduce(lambda x,y:str(x)+str(y),index)
		#Fix in case of single character
		istring = str(istring)
		string_map[str(sym)] = head+istring.replace("{","").replace("}","")


	#Extract the numerator poly string and replace.
	pstring = str(list(r_num.mdict.values())[0].as_expr())

	target_string = pstring
	for src,tar in string_map.items():
		target_string = target_string.replace(src,tar)
	target_string = target_string.replace("**","^")
	return target_string,list(string_map.values())



def edge_tuple_to_pestring(edges):
	string = ""
	for edge in edges:
		if edge[0]==0 and edge[1]==0:
			pass
		else:
			if edge[0]<0 and edge[1]<0:
				var = "pp"
			elif edge[0]<0 and edge[1]>0:
				var = "ke"
			elif edge[0]>0 and edge[1]>0:
				var = "ee"
			num = str(abs(edge[0]))+str(abs(edge[1]))
			string+=var+"["+str(abs(edge[0]))+","+str(abs(edge[1]))+"]*"
	return string.strip('*')

def mring_to_mathematica_pe(ring):
	string = ""
	for key in ring.mdict.keys():
		string += "("+sympy_poly_to_mathematica(ring.mdict[key])+")"
		edgestring = edge_tuple_to_pestring(key)
		if edgestring=="":
			string+=edgestring+"+"
		else:
			string += "*"+edge_tuple_to_pestring(key) + "+"
	string = string.strip("+")
	string = string.strip("*")
	return string

def mrational_to_mathematica_pe(rat):
	string = ""
	for pair in rat.nd_list:
		string+='('+mring_to_mathematica_pe(pair[0])+')/'
		dstring = ""
		for ring,power in pair[1].items():
			if power==1:
				dstring+='('+mring_to_mathematica_pe(ring)+')'
			else:
				dstring+='('+mring_to_mathematica_pe(ring)+')^'+str(power)
		string+='('+dstring+')+'
	string = string.strip('+')
	return string

def mrational_to_file(rat,filename):
	text_file = open(filename, "w")
	text_file.write(mrational_to_mathematica_pe(rat))
	text_file.close()
