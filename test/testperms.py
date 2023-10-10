#! /usr/bin/python

from SignedPermutations import SignedPermutation,SignedPermutationGroup

perm0 = SignedPermutation([1,3,2],1)
perm = perm0
print perm.permlist,perm.sign
perm = perm0*perm0
print perm.permlist,perm.sign
perm = perm0*perm0*perm0
print perm.permlist,perm.sign

print "---------------"
print perm.ID().permlist,perm.ID().sign
print perm.len()
print "-----------------"
print perm0.permlist,perm0.sign
print perm0.Inverse().permlist, perm0.Inverse().sign
prod1 = perm0*(perm0.Inverse())
prod2 = (perm0.Inverse())*perm0
print "====================="
print prod1.permlist, prod1.sign
print prod2.permlist, prod2.sign
print "*********************"

permA = SignedPermutation([1,2,4,3],-1)
permB = SignedPermutation([2,1,3,4],-1)
print permA.permlist,permA.sign
print permA.Inverse().permlist,permA.Inverse().sign
prodA1 = permA*(permA.Inverse())
prodA2 = (permA.Inverse())*permA
print "----------"
print prodA1.permlist,prodA1.sign
print prodA2.permlist,prodA2.sign

print "=======================+"
cyc1 = perm0.Cycles()
print cyc1
print perm0.Parity()
cyc2 = permA.Cycles()
print cyc2
print permA.Parity()
print "++++++++++++++++++++++"
print permA

print "*********************************"
gp1 = SignedPermutationGroup([permA,permB])
print "GP1: "
print gp1
print "LEN: ",gp1.len()
ituple = (4,1,2,3)
gpsorted_ituple, sorting_perms = gp1.GroupSort(ituple)
print "%%%%%%%%"
print "ITUP: ",ituple
print "ITUP_GPSRT: ",gpsorted_ituple
print "SORTPERMS: "
if type(sorting_perms)==type([]):
	for perm in sorting_perms:
		print perm
else:
	print sorting_perms
