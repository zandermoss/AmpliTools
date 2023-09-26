"""
Some simple tools for working with sympy types.
"""

from sympy import *


def IsSympy(expr):
    """
    Check if the object [expr] belongs to the SymPy class heirarchy.
    """
    typestring = str(type(expr))
    return ("sympy" in typestring)

def IsSympySymbolic(expr):
    """
    Check if the object [expr] is a sympy number. If not, return True.
    """
    if not IsSympy(expr):
        return False
    return (not expr.is_number)

def FuncType(expr):
    """
    Partially classify [expr] by sympy type.
    """
    if (not IsSympySymbolic(expr)):
        return "number"
    elif expr.func==sympify("x**2").func:
        return "pow"
    elif expr.func==sympify("x*y").func:
        return "mul"
    elif expr.func==sympify("x+y").func:
        return "add"
    else:
        return "other"

def LinearSystemToMatrix(exprs,var_head):
    """
    Utility function for converting a system of linear equations [exprs]
    (a list or tuple of sympy expressions) over a list of variables [vars]
    (a list of sympy symbols) into a coefficient matrix and constant vector.
    More precisely, given a system A.x + b = 0, specified by a list of sympy
    exprs (one for each row), returns a pair of sympy matrices (A,b).
    """

    #Extract variables
    vars = set()
    for poly in exprs:
        vars |= set(filter(lambda s: var_head in s.__str__(),poly.free_symbols))
    vars = list(vars)
    print("VARS ",vars)

    #Initialize the coefficient matrix and constant vectors here. We will
    #cast them to sympy matrices at the very end.
    coefficient_matrix = [[Poly(0,expr.gens) for var in vars]
                          for expr in exprs]
    constant_vector = [Poly(0,expr.gens) for expr in exprs]

    #Verify that the system is linear.
    for expr in exprs:
        for monom in (expr.as_dict()).keys():
            var_count = 0
            for var in vars:
                if not (var in expr.gens):
                    continue
                var_index = (expr.gens).index(var)
                var_count += monom[var_index]
            assert var_count<2, "System [exprs] is non-linear!"

    #Do the conversion.
    for nexp,expr in enumerate(exprs):
        last_dict = expr.as_dict()
        for nv,var in enumerate(vars):
            next_dict = {}
            if not(var in expr.gens):
                next_dict = last_dict
                continue
            var_index = (expr.gens).index(var)
            for monom,coeff in last_dict.items():
                if monom[var_index]>0:
                    quotient_monom = list(monom)
                    quotient_monom[var_index]=0
                    var_coefficient = Poly({tuple(quotient_monom):coeff},
                                           expr.gens)
                    coefficient_matrix[nexp][nv] += var_coefficient
                else:
                    next_dict.setdefault(monom,0)
                    next_dict[monom]+=coeff
            last_dict = next_dict
        constant_vector[nexp] = Poly(last_dict,expr.gens)

    #Exclude superfluous generators.
    for i in range(0,len(exprs)):
        for j in range(0,len(vars)):
            coefficient_matrix[i][j] = coefficient_matrix[i][j].exclude()
        constant_vector[i] = constant_vector[i].exclude()

    #Construct sympy matrices from 2d arrays and return
    return Matrix(coefficient_matrix),Matrix(constant_vector)

def PolyMatrixToExprs(matrix):
    expr_matrix = []
    for i in range(matrix.shape[0]):
        expr_row = []
        for j in range(matrix.shape[1]):
            expr_row.append((matrix[i,j]).as_expr())
        expr_matrix.append(expr_row)
    return Matrix(expr_matrix)


def test():
    x,y = symbols('x y')
    owl = x*y+I
    pussycat = I*x + y
    piggywig = x**y
    print("FUNCTYPE")
    print(owl,FuncType(owl))
    print(pussycat,FuncType(pussycat))
    print(piggywig,FuncType(piggywig))
