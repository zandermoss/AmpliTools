#! /usr/bin/python

from Graph import *
from MRational import MRational
from sympy import *
from Interface import Interface
from Spin0Basis import Basis

import os
import psutil
from tqdm import tqdm
from FormatTools import hline,title

def ProcMem(desc=None):
	process = psutil.Process(os.getpid())
	string = "Process Memory: {} Mb"
	if desc!=None:
		string+=" ("+desc+")"
	print string.format((process.memory_info().rss)/1000000)  # in mb

ProcMem("Initial")

basis = Basis(0,4)

p1,p2,p3,p4 = symbols('p1 p2 p3 p4')
x = symbols('x')

momentum_symbols = [p1,p2,p3,p4]
coefficient_symbols = [x,]
coefficient_display_map = {'x':'x'}
io = Interface(momentum_symbols,coefficient_symbols,coefficient_display_map)
#Define convenience functions.
def mring(expr):
    return io.ExprToMRing(expr)
def mrat(num,den):
    return io.ExprToMRational(num,den,basis)
def mprint(malg):
    io.Print(malg)

numerator = mring(p1*p2)
denominator = {mring(1):1}
r = mrat(numerator,denominator)

numerator = mring(1)
denominator = {mring(p1*p1):1}
prop = mrat(numerator,denominator)

propagators = {'s':prop}

v = Vertex(r,3)
v1 = Vertex(r,1,ext_label=1)
v2 = Vertex(r,1,ext_label=2)
v3 = Vertex(r,1,ext_label=3)
opv1 = Operator((v,1),"s",None)
opv2 = Operator((v,2),"s",None)
opv3 = Operator((v,3),"s",None)
op1 = Operator((v1,1),"s",None)
op2 = Operator((v2,1),"s",None)
op3 = Operator((v3,1),"s",None)
g = GRing([Graph([v,],{opv1.OID:opv1,opv2.OID:opv2,opv3.OID:opv3}),])
h1 = GRing([Graph([v1,],{op1.OID:op1}),])
h2 = GRing([Graph([v2,],{op2.OID:op2}),])
h3 = GRing([Graph([v3,],{op3.OID:op3}),])
#k = h1*(h2*(h3*g))
#k = h3*g
hline()
for i in tqdm([1,],desc="Contractions"):
	res = h3*(h2*(h1*g))
ProcMem("Final")

rat = res.ComputeMRational(propagators,v3)
print rat


