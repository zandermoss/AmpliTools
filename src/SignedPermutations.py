


class SignedPermutation():
    def __init__(self,_permlist,_sign):
        self.permlist = _permlist
        self.sign = _sign

    def __len__(self):
        """
        Return the length of the permutation.
        """
        return len(self.permlist)

    def ID(self):
        """
        Return the identity permutation of appropriate length.
        """
        permlist = range(1,len(self)+1)
        sign = 1
        return SignedPermutation(permlist,sign)

    def Permute(self,_iterable):
        """
        Apply permutation (disregarding sign) to iterable [_iterable], return as list.
        """
        assert len(self) == len(_iterable),"List length doesn't match permutation length!"
        return [_iterable[x-1] for x in self.permlist]

    def __mul__(self,other):
        """
        Return the product of self permutation and other permutation.
        This product is signed and *non-commutative*. In terms of
        composition of permutations, this product returns:
          			(self)o(other)
        """
        permlist = self.Permute(other.permlist)
        #permlist = [other.permlist[x-1] for x in self.permlist]
        sign = self.sign*other.sign
        return SignedPermutation(permlist,sign)

    def __str__(self):
        return self.permlist.__str__()+" "+self.sign.__str__()

    def __eq__(self,other):
        return (self.permlist==other.permlist and self.sign==other.sign)

    def Inverse(self):
        """
        Return the inverse of the self permutation. The sign remains the same.
        """
        permlist = [self.permlist.index(i)+1 for i in range(1,len(self)+1)]
        sign = self.sign
        return SignedPermutation(permlist,sign)

    def Cycles(self):
        """
        Returns the cycle decomposition of self.
        [cycles] is a tuple of tuples, each of which is a cycle of the
        permutation self.permlist.
        """
        source = range(1,len(self)+1)
        cycles = []
        while len(source)>0:
            t0 = source[0]
            t = t0
            cycle = []
            looped = False
            while not looped:
                source.remove(t)
                cycle.append(t)
                t = self.permlist[t-1]
                looped = (t==t0)
            cycles.append(tuple(cycle))
        return cycles

    def Parity(self):
        """
        Constructs the parity of a permutation by counting cycles.
        Don't confuse the parity of a signed permutation with the
        sign! The sign is an additional element of Z_2 which we've
        tacked on to handle signed symmetries of tensors. The
        parity is what we traditionally think of as the "sign" of
        a permutation.
        """
        sign = 1
        for c in self.Cycles():
            sign*= (-1)**(len(c)-1)
        return sign

class SignedPermutationGroup():

    def Generate(self,element):
        for generator in self.generators:
            next_element = generator*element
            if tuple(next_element.permlist) in self.perm_sign_dict.keys():
                #Check for sign consistency.
                assert next_element.sign == self.perm_sign_dict[tuple(next_element.permlist)]
            else:
                self.perm_sign_dict[tuple(next_element.permlist)] = next_element.sign
                self.Generate(next_element)
        return

    def __init__(self,generators):
        """
        Takes a set of SignedPermutation objects and generates the
        corresponding group by way of exhaustive multiplication.
        The group is stored as a dictionary {permlist:sign}.
        Note: the order of any generator with negative size must be even!
        Otherwise, we will generate the same permutation with both signs.
        The assertion in Generate() will catch this case.
        """
        self.generators = generators
        self.perm_sign_dict = {}
        self.Generate(generators[0].ID())

    def __len__(self):
        return len(self.perm_sign_dict)

    def PermLen(self):
        return len(list(self.perm_sign_dict.keys())[0])

    def __str__(self):
        retstr = ""
        for perm in self.perm_sign_dict.keys():
            retstr+=perm.__str__()+" "+self.perm_sign_dict[perm].__str__()+"\n"
        return retstr

    def __eq__(self,other):
        return self.perm_sign_dict==other.perm_sign_dict

    def GroupSort(self,ituple):
        """
        This function solves a problem in which we are given a list of signed
        permutations and a tuple of strings (which are an ordered set relative
        to Python's cmp() function). We are asked to find
        the permutation	which "best sorts" the tuple (in ascending order), and
        return the sorted tuple and the signed permutation which "best sorted" it.

        When we feed a pair of tuples to python's cmp() function, it will
        employ a R^n ordering. That is, when a and b are tuples of the same
        length, cmp() returns cmp(a[i],b[i]) where i is the smallest index at which
        a[i]!=b[i]. This is a very nice default behavior, because an ordered
        tuple x will always be the minimum of any set of permutations of that
        tuple.

        Given a set of permutations (in this case, the group of permutations, self).
        we compute the action of each of these permutations on ituple and then find
        the minimum of this set relative to python's cmp() operator. The result is
        the "best sorted" in the sense that completing our permutation subgroup to
        S(n) and repeating this process will always yield ituple sorted in ascending
        order as the minimum. When we order the permuted ituples with cmp(), the
        minimum element possible in this ordered set is the ascending sort of ituple,
        and all other permutations are ordered based on "how well" they approximate
        an ascending sort.

        Why play this abstract, nonsense game? Given a tensor with a (potentially)
        signed index symmetry, we can take any index tuple ituple and find a canonical
        ordering together with the permutation (and its sign!) required to get us
        there. This allows us to systematically exploit tensor symmetries to simplify
        expressions involving contracted tensors by canonicalizing their indices
        *while keeping track of signs*, which is crucial if we have antisymmetric
        indices, as in the case where our tensor is actually a structure constant!

        Note that this function may return a list of permutations if more than one
        best sorts ituple. This is possible if ituple has redundancies.
        """

        assert len(ituple) == self.PermLen(), "Mismatch in tuple and permutation lengths!"

        tuple_perm_dict = {}
        #Generate the list of group-sorted ituples.
        for permlist in self.perm_sign_dict.keys():
            tuple_image = tuple([ituple[x-1] for x in permlist])
            tuple_perm_dict.setdefault(tuple_image,[])
            tuple_perm_dict[tuple_image].append(permlist)
        min_key = min(tuple_perm_dict.keys())
        best_perms = tuple_perm_dict[min_key]
        if len(best_perms)==1:
            return min_key,SignedPermutation(best_perms[0],self.perm_sign_dict[best_perms[0]])
        else:
            best_signed_perms = [SignedPermutation(best_perm,self.perm_sign_dict[best_perm]) for best_perm in best_perms]
            return min_key,best_signed_perms
