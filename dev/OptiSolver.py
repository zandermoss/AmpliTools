from MRing import MRing
from MRational import MRational
from sympy import symbols, poly, Rational, factor, simplify, Poly,lambdify
from scipy.optimize import minimize,differential_evolution,NonlinearConstraint
from sympy import *
from scipy import linalg
import numpy as np

class OptiSolver():
    def __init__(self,eq_rats,ineq_rats,decoupling_mat = None,
                 linearization_mat = None, linearization_vec = None,
                 epsilon=1e-6, decoupling_epsilon = 1e-5,
                 linearization_epsilon = 1e-5):
        self.eq_rats = eq_rats
        self.ineq_rats = ineq_rats
        self.decoupling_mat = decoupling_mat
        self.linearization_mat = linearization_mat
        self.linearization_vec = linearization_vec
        self.epsilon = epsilon
        self.decoupling_epsilon = decoupling_epsilon
        self.linearization_epsilon = linearization_epsilon

        self.BuildFunctions()

    def BuildFunctions(self):
        #Construct function from list of MRational expressions.
        self.generators = set()
        self.eq_expressionmap = {}
        self.ineq_expressionmap = {}
        for rat in self.eq_rats:
            expr,gens = self.ExtractExprAndGens(rat)
            self.eq_expressionmap[expr] = gens
            self.generators |= set(gens)
        for irat in self.ineq_rats:
            expr,gens = self.ExtractExprAndGens(irat)
            self.ineq_expressionmap[expr] = gens
            self.generators |= set(gens)


        if self.decoupling_mat != None:
            self.generators |= set(self.ExtractMatrixGens(self.decoupling_mat))


        if self.linearization_mat != None:
            assert self.linearization_vec != None, "Missing vec."
            self.generators |= set(self.ExtractMatrixGens(
                                                    self.linearization_mat))

        self.ordered_generators = tuple(self.generators)
        self.eq_lambdas=[]
        for expr,gens in self.eq_expressionmap.items():
            self.eq_lambdas.append(self.BuildLambda(expr,gens))
        self.ineq_lambdas=[]
        for expr,gens in self.ineq_expressionmap.items():
            self.ineq_lambdas.append(self.BuildLambda(expr,gens))

        self.objective = self.Objective()
        self.constraints = self.Constraints()

        if self.decoupling_mat != None:
            self.decoupling_mat_lambda = self.BuildMatrixLambda(
                                                        self.decoupling_mat)
            self.decoupling_lambda = self.BuildDecouplingConstraint()
            self.constraints.append(self.decoupling_lambda)
        if self.linearization_mat != None:
            self.linearization_mat_lambda = self.BuildMatrixLambda(
                                                        self.linearization_mat)
            self.linearization_vec_const = self.BuildConstVector(
                                                        self.linearization_vec)
            self.linearization_lambda = self.BuildLinearizationConstraint()
            self.constraints.append(self.linearization_lambda)

    def SetEpsilon(self,epsilon):
        self.epsilon = epsilon
        self.BuildFunctions()

    def Solve(self,x0,tol=1e-12):
        res = minimize(self.objective, x0, method='SLSQP',
                       constraints = self.constraints,tol=tol,
                       options = {'maxiter':2000,'ftol':1e-12})
        return res,x0

    def GlobalSolve(self,bounds):
        deco_func = self.BuildDecouplingConstraint()['fun']
        decocon = NonlinearConstraint(deco_func,0.0,np.inf)
        lin_func = self.BuildLinearizationConstraint()['fun']
        lincon = NonlinearConstraint(lin_func,0.0,np.inf)
        res = differential_evolution(self.objective,bounds,maxiter=4000,
                                     constraints = (decocon,lincon),disp=True)
        return res

    def ExtractExprAndGens(self,expr):
        assert len(expr.nd_list)==1,"Too many MRational Terms."
        num = list(expr.nd_list[0][0].Mdict.values())[0].exclude()
        return num, num.gens

    def ExtractMatrixGens(self,mat):
        gens = set()
        for i in range(0,mat.shape[0]):
            for j in range(0,mat.shape[1]):
                gens |= set(mat[i,j].gens)
        return tuple(gens)


    def BuildLambda(self,expr,gens):
        lamfunc = lambdify(gens,expr.as_expr(),'numpy')
        indices = [self.ordered_generators.index(gen) for gen in gens]

        def expr_lambda(arg_array):
            return lamfunc(*[arg_array[index] for index in indices])
        return expr_lambda

    def Objective(self):
        def obj_func(arg_array):
            square_sum = 0.0
            for expr_lambda in self.eq_lambdas:
                square_sum += expr_lambda(arg_array)**2

            return square_sum
        return obj_func

    def Constraints(self):
        constraint_funcs = [lambda arg_array,myfunc=expr_lambda,
                            epsilon=self.epsilon:
                            myfunc(arg_array)**2 - epsilon**2
                            for expr_lambda in self.ineq_lambdas]
        constraints = [{'type':'ineq','fun':cfunc} for cfunc
                       in constraint_funcs]
        return constraints

    def BuildMatrixLambda(self,_mat):
        def mat_lambda(arg_array):
            mat = _mat
            target_matrix = [[0 for j in range(0,mat.shape[1])]
                             for i in range(0,mat.shape[0])]
            for i in range(0,mat.shape[0]):
                for j in range(0,mat.shape[1]):
                    expr = mat[i,j]
                    lamfunc = lambdify(expr.gens,expr.as_expr(),'numpy')
                    indices = [self.ordered_generators.index(gen)
                               for gen in expr.gens]
                    target_matrix[i][j] = lamfunc(*[arg_array[index]
                                                 for index in indices])
            return np.array(target_matrix)
        return mat_lambda

    def BuildConstVector(self,poly_vec):
        vec = [0.0 for i in range(0,poly_vec.shape[0])]
        for i in range(0,poly_vec.shape[0]):
            vec[i] = float(poly_vec[i].as_expr())
        return np.array(vec)

    def BuildDecouplingConstraint(self):
        def decoupling_lambda(arg_array):
            mat = self.decoupling_mat_lambda(arg_array)
            decoupling_epsilon = self.decoupling_epsilon
            return (linalg.det(np.dot(np.transpose(mat),mat))**2
                    - decoupling_epsilon**2)
        return {'type':'ineq','fun':decoupling_lambda}


    def BuildLinearizationConstraint(self):
        def linearization_lambda(arg_array):
            mat = self.linearization_mat_lambda(arg_array)
            vec = self.linearization_vec_const
            linearization_epsilon = self.linearization_epsilon
            diff = np.dot(np.dot(mat,linalg.pinv(mat)),vec) - vec
            sum_squares = 0
            for i in range(0,diff.shape[0]):
                sum_squares += diff[i]**2
            return sum_squares - linearization_epsilon**2
        return {'type':'ineq','fun':linearization_lambda}


    def GetOrderedGenerators(self):
        return self.ordered_generators
