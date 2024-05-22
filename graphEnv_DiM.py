#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 22:35:26 2023

@author: 
"""
from sklearn.linear_model import LinearRegression
import numpy as np
import networkx as nx
import math

class graph_Env(object):
    def __init__(self, p = 0.5, graph= None, d=4, design=None, a = 10, b=10, c=3, outcome_mod = "noisy", n = 1000):
        
        self.p = p
        self.graph = graph
        self.d = d
        self.design = design
        self.z = None
        self.e = None
        self.y = np.zeros(n)
        self.a = a
        self.b = b
        self.c = c
        self.n = n
        self.outcome_mod = outcome_mod
    
    def outcome_model(self, z, e):
        if self.outcome_mod == "simple":

            out = self.a + self.b*z + self.c*e
            
        elif self.outcome_mod == "noisy":
            noise = np.random.normal(0,1)
            out = self.a + self.b*z + self.c*e + noise
            
        elif self.outcome_mod == "quad":
            noise = np.random.normal(0,1)
            out = self.a + self.b*z + self.c*e**2 + noise
            
            
        elif self.outcome_mod == "sigmoid":
            noise = np.random.normal(0,1)
            out = self.a + self.b*z + self.c*(1/(1+math.exp(-e))) + noise
        elif self.outcome_mod == "sine":
            noise = np.random.normal(0,1)
            out = self.a + self.b*z + self.c*(1-math.sin(math.pi*e)) + noise

            
        else:
            print("not implemented")
        return(out)
            
    
    def gen_outcome(self, zi, ei):
        #a = 20 # to be randomized
        #b = 20 # to be randomized
        #c = 2000 # to be randomized

        #c = 1 # to be randomized
        noise = np.random.normal(0,1) # need to do linear regression
        #noise =0
        #outcome = a + b*zi + c*ei**10 + noise
        #outcome = a + b*zi + (c+noise)*ei
        outcome = self.outcome_model(zi, ei)
        return outcome
    
    def gen_coefficients(self): 
        x = np.vstack((self.z,self.e))
        model = LinearRegression().fit(x.T,self.y)
        coeffs = np.append(model.intercept_, model.coef_)
        # we won't need linear regression coefficients when we're doing Lepski's method (Write out lepski's method)
        # we need linear regression coefficients when estimating the bias gain from dropping off points for more regular graphs
        return coeffs
    def gen_coefficientsH(self,h): 
        x = np.vstack((self.z,self.e))
        x_low = x[:,(self.z == 0)&(self.e <= 1-h)]

        y_low =  self.y[(self.z == 0)&(self.e <= 1-h)]

        y_low =  self.y[(self.z == 0)&(self.e <= 1-h)]
        x_up = x[:,(self.z == 1)&(self.e >= h)]
        

        y_up = self.y[(self.z ==1)&(self.e >= h)]
        if x_low.size > 0:
            lower = LinearRegression().fit(x_low.T,y_low)
            coeffs1 = np.append(lower.intercept_, lower.coef_)
        else:
            coeffs1 = np.zeros(3)
        if x_up.size > 0:
            upper = LinearRegression().fit(x_up.T,y_up)
            coeffs2 = np.append(upper.intercept_, upper.coef_)
        else:
            coeffs2 = np.zeros(3)
        return coeffs1, coeffs2
    
    def gen_graph(self, n):
        if self.graph == "erdos_renyi":
            G = nx.gnp_random_graph(self.n,self.p, seed=10)
            A = nx.to_numpy_array(G)
        elif self.graph == "cycle":
            G = nx.cycle_graph(n)
            A = nx.to_numpy_array(G)
            #k = self.d/2
            # k=1
            # A = np.zeros((n,n))
            # for i in range(n):
            #     lo = int(i-k)
            #     if lo<0:
            #         lo = 0
            #     hi = int(i+k)
            #     if hi > (n -1):
            #         hi = n-1
            #     A[i,lo:hi] = 1
        elif self.graph == "kNN":

            k = self.d/2
            A = np.zeros((n,n))
            for i in range(n):
                lo = int(i-k)
                if lo<0:
                    lo_plus = lo
                    lo = 0
                    A[i,n+lo_plus:n] = 1
                hi = int(i+k)
                if hi > (n -1):
                    hi_minus = hi-n+1
                    hi = n-1
                    A[i,0:hi_minus] = 1
                A[i,lo:i] = 1
                A[i,(i+1):(hi+1)] = 1
                #A[i,lo:hi] = 1
                            
                
        deg = A.sum(axis=1)
        D = np.diag(deg)
        exposure = np.matmul(np.linalg.inv(D), A)
        self.e = np.matmul(exposure, self.z)
        count = 0 
        for i in range(n):
            if self.e[i] >= 1:
                count += 1
        #print("count: ", count)
        return A
    
    def gen_design(self, n, prop=0.5):
        sampled = np.random.choice(np.arange(n), int(n*prop), replace=False)
        zs = np.zeros(n)
        for i in range(n):
            if i in sampled:
                zs[i] = 1
        self.z = zs

        
        return zs
    

    
    def gen_data(self,n):
        """
        returns: unit id, treatment status, exposure status, outcome
        """
        data = []
        for i in range(n):
            z_i = self.z[i]
            e_i = self.e[i]
            y_i = self.gen_outcome(z_i, e_i)
            self.y[i] = y_i
            data.append((i,z_i,e_i,y_i))
        #print("data: ", data)
        return (data)

if __name__=='__main__':
    
    Env = graph_Env(p=0.1, graph = "kNN", d =4, outcome_mod = "sine", design = "unit")

    zs = Env.gen_design(1000)
    adjmat = Env.gen_graph(1000)

    data = Env.gen_data(1000)
    Env.gen_coefficients()[2]
    import matplotlib.pyplot as plt
    
    xaxis = []
    yaxis = []
    for tup in data:
        xaxis.append(tup[2])
        yaxis.append(tup[3])
    plt.scatter(xaxis, yaxis)
    

    

    

    
    
    

