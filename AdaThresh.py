#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 11:56:39 2024

@author:
"""


from graphEnv_DiM import graph_Env
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from exposure_probability_mcmc import MC_exposure_probas
import pickle
import sys

class Bias_Var(object):
    
   
    def __init__(self,params=None):
        if params is None:
            raise Exception("Slope requires four parameters")

        self.estimator = None
        self.hyperparams = None
        self.d= None
        self.graph = None
        self.design = None
        self.n = None
        self.h = None
        self.gammaHat = None
        self.p = None
        self.graph = None
        self.kernel = None
        self.outcome_mod = None
        self.estimator_type =  None
        self.a = None
        self.b = None
        self.c = None
        self.true_ATE = None
        self.EP_treated=None
        self.EP_control=None
        self.jointEP_treated=None
        self.jointEP_control=None
        self.jointEP_inverse =None
        if 'estimator' in params.keys():
            self.estimator = params['estimator']
        if 'hyperparams' in params.keys():
            self.hyperparams = params['hyperparams']
        if 'graph_param' in params.keys():
            self.d = params['graph_param']
        if 'design_param' in params.keys():
            self.p = params['design_param']
        if 'graph' in params.keys():
            self.graph = params['graph']
        if 'design' in params.keys():
            self.design = params['design']
        if 'kernel' in params.keys():
            self.kernel = params['kernel']
        if 'env' in params.keys():
            self.env = params['env']
        if 'size' in params.keys():
            self.n = params['size']
        if 'p' in params.keys():
            self.p = params['p']
        if 'outcome_mod' in params.keys():
            self.outcome_mod = params['outcome_mod']
            
        if 'estimator_type' in params.keys():
            self.estimator_type = params['estimator_type']
        if 'a' in params.keys() and 'b' in params.keys() and 'c' in params.keys():
            self.a = params['a']
            self.b = params['b']
            self.c = params['c']
            self.true_ATE = self.b + self.c
            
        if 'EP_treated' in params.keys() and 'EP_control' in params.keys() and 'jointEP_treated' in params.keys() and 'jointEP_control' in params.keys() and 'jointEP_inverse' in params.keys():
            
            self.EP_treated = params['EP_treated']
            self.EP_control = params['EP_control']
            self.jointEP_treated = params['jointEP_treated']
            self.jointEP_control = params['jointEP_control']
            self.jointEP_inverse = params['jointEP_inverse']
        elif 'EP_treated' not in params.keys() and 'EP_control' not in params.keys() and 'jointEP_treated' not in params.keys() and 'jointEP_control' not in params.keys() and 'jointEP_inverse' not in params.keys():
            MCEP = MC_exposure_probas(params={'hyperparams': self.hyperparams, 'graph_param': self.d,'design_param': self.p,'graph':self.graph, 'design': self.design, 'size': self.n, 'kernel':self.kernel, 'env':graph_Env, 'outcome_mod': self.outcome_mod, "estimator_type": self.estimator_type, "a":self.a, "b":self.b, "c":self.c})
            self.EP_treated, self.EP_control, self.jointEP_treated, self.jointEP_control, self.jointEP_inverse = MCEP.sims_ep_call()
            
    def outcome_model(self, z, e):
        if self.outcome_mod == "simple":
            out = self.a + self.b*z + self.c*e
            
        elif self.outcome_mod == "noisy":
            noise = np.random.normal(0,1)
            out = self.a + self.b*z + self.c*e + noise
            
        elif self.outcome_mod == "quad":
            noise = np.random.normal(0,1)
            out = self.a + self.b*z + self.c*e**2 + noise
            
        else:
            print("not implemented")
        return(out)
    
    def var_estimator_DIM(self, n1, n2, data):
        n1 = int(n1)
        n2 = int(n2)
        varianceT = 0
        varianceC = 0
        var_estT =0
        var_estC = 0

        countT = 0
        countC = 0

        treated_array = []
        control_array = []
        for tup1 in data:
            (i, z_i, e_i, y_i) = tup1  # unit, unit treatment status, unit exposure level, unit outcome
            
            if z_i == 1 and e_i >= self.h:
                var_estT += y_i
                treated_array.append(y_i)
                countT += 1
            
            if z_i == 0 and e_i <= (1 - self.h):
                var_estC += y_i
                control_array.append(y_i)
                countC += 1
    
        if countT > 0:
            mean_T = var_estT / countT
            varianceT = np.mean((np.array(treated_array) - mean_T) ** 2)
        if countC > 0:
            mean_C = var_estC / countC
            varianceC = np.mean((np.array(control_array) - mean_C) ** 2)
    
        var_est = (2 / (self.n - 1)) * (varianceT + varianceC)
        return var_est
    
    def var_estimator(self, data):        
        #n = len(data)
        var_est = 0 
        epsT = 10**(-5)
        idx = np.where(self.hyperparams == self.h)[0][0]
        countjointT = 0
        countjointC = 0
        for tup1 in data:
            (i,z_i,e_i, y_i) = tup1  # unit, unit treatment status, unit exposure level, unit outcome
            
            if z_i == 1 and e_i >= self.h:
                var_est += y_i**2/(self.n**2*self.EP_treated[i,idx])*(1/self.EP_treated[i,idx] - 1)
            
            if z_i == 0 and e_i<= (1-self.h):
                var_est += y_i**2/(self.n**2*self.EP_control[i,idx])*(1/self.EP_control[i,idx] - 1)
            
            for tup2 in data:
                (j,z_j,e_j, y_j) = tup2  # unit, unit treatment status, unit exposure level, unit outcome
                if j != i:
                    if z_i == 1 and e_i >= self.h and z_j == 1 and e_j >= self.h :
                        
                        if self.jointEP_treated[i,j,idx]  > epsT and abs(self.jointEP_treated[i,j,idx] - self.EP_treated[i,idx]*self.EP_treated[j,idx]) > epsT:
                            var_est += (y_i*y_j)/(self.n**2*self.jointEP_treated[i,j,idx])*(self.jointEP_treated[i,j,idx]/(self.EP_treated[i, idx]*self.EP_treated[j, idx]) - 1)
                        elif self.jointEP_treated[i,j,idx] <= epsT and abs(self.jointEP_treated[i,j,idx] - self.EP_treated[i,idx]*self.EP_treated[j,idx]) <=epsT:
                            countjointT += 1
                        
                    
                    if z_i == 0 and e_i <= 1-self.h and z_j == 0 and e_j <= 1-self.h :
                        
                        if self.jointEP_control[i,j,idx]  > epsT and abs(self.jointEP_control[i,j,idx] - self.EP_control[i,idx]*self.EP_control[j,idx]) > epsT:
                            var_est += (y_i*y_j)/(self.n**2*self.jointEP_control[i,j,idx])*(self.jointEP_control[i,j,idx]/(self.EP_control[i, idx]*self.EP_control[j, idx]) - 1)
                        elif self.jointEP_control[i,j,idx] <= epsT and abs(self.jointEP_control[i,j,idx] - self.EP_control[i,idx]*self.EP_control[j,idx]) <=epsT:
                            countjointC += 1
                    
                    if z_i == 1 and e_i >= self.h and z_j == 0 and e_j <= 1-self.h :
                        if self.jointEP_inverse[i,j,idx]  > epsT:
                            var_est += -(2/self.n**2)*y_i*y_j*(1/(self.EP_treated[i,idx]*self.EP_control[j,idx]) - 1/self.jointEP_inverse[i,j,idx])
                            #print("self.jointEP_inverse[i,j,idx]: ", self.jointEP_inverse[i,j,idx])
                    
                    
                    if self.jointEP_inverse[i,j,idx]  <= epsT:
                        if z_i == 1 and e_i >= self.h:
                            var_est += (2/self.n**2)*(y_i**2/(2*self.EP_treated[i,idx]))
                            #print("self.EP_treated[i,idx]: ", self.EP_treated[i,idx])
                        if z_j == 0 and e_j <= 1-self.h:
                            var_est += (2/self.n**2)*(y_j**2/(2*self.EP_control[j,idx]))
                            #print("self.EP_control[j,idx] : ", self.EP_control[j,idx])
        if countjointT > 0:
            print("Count of discounted jointly treated pairs: ", countjointT)
        if countjointC > 0:
            print("Count of discounted jointly controlled pairs: ", countjointC)
        return var_est
    def point_estimate_DiM_genTruth(self, data):
        var_to_sumT = 0 
        var_to_sumC = 0 

        countT= 0
        countC = 0
        
        if self.graph == "kNN":
            if self.design == "cluster":
                clust_size = self.d + 1
                k = self.d/2
        for tup in data:
            (i,z_i,e_i, y_i) = tup  # unit, unit treatment status, unit exposure level, unit outcome
            if self.kernel == "boxcarIPW":
                if self.design == "unit":
                    
                    if z_i == 1 and e_i >= self.h:
                        #count1 += count1
                        #print(paste0("var_element: ", y[i]/probas))
                        var_to_sumT += y_i
                        countT += 1

                        
                        
                    elif z_i == 0 and e_i<= (1-self.h):
                        var_to_sumC += y_i
                        countC += 1

                
                
                    
                elif self.design == "cluster" and self.graph == "kNN":
                        if z_i == 1 and e_i >= self.h:
                            countT += 1
                            #l = (1-self.h)*self.d
                            
                            # simulations
                            plus_= 0
                            find_subt = (1-self.h)*self.d #
                            # work backward from threshold (find what we're subtracting off to get the threshold) # NOTE: this is only correct for threshold up to k
                            if (self.h < k/self.d):
                              self.h = k/self.d # redefine
                              find_subt = (1-k/self.d)*self.d # the rest do not contribute
                            
                            mid = (clust_size + 1)/2
        
        
                            if(i % clust_size == 0):
                              plus_ = clust_size
                            
                            if(((i% clust_size + plus_) <= (mid+find_subt)) and ((i%clust_size + plus_) >= (mid-find_subt))):
                              # also the other direction
                              var_to_sumT += y_i
                            else:
                              var_to_sumT += y_i
                            
                            # analytical
                            # var += ((y_i/(n))**2)*((1/self.p**2 - 1)*(3*self.d + 1 - 2*l) + (2*l+1)*(1/self.p - 1))
        
        
                        elif z_i == 0 and e_i<= (1-self.h):
                            countC += 1
                            plus_ = 0
                            find_subt = (1-self.h)*self.d #
                            # work backward from threshold (find what we're subtracting off to get the threshold) # NOTE: this is only correct for threshold up to k
                            if (self.h <= k/self.d):
                              self.h = k/self.d # redefine
                              find_subt = (1-k/self.d)*self.d # the rest do not contribute
                            
                            mid = (clust_size + 1)/2
        
        
                            if(i % clust_size == 0):
                              plus_ = clust_size
                            
                            if(((i% clust_size + plus_) <= (mid+find_subt))&((i%clust_size + plus_) >= (mid-find_subt))):
                              # also the other direction
                              var_to_sumC += y_i

                            else:
                              var_to_sumC += y_i
                
                        
            
        if countT==0:
            pointT = 0
        else:
            pointT = var_to_sumT/countT
        if countC == 0:
            pointC = 0 
        else:
            pointC = var_to_sumC/countC
              
            
     
        point_est = (pointT- pointC) # sample mean
        
            #print("var_to_subt: ", var)
        return(point_est)

   
            
    def point_estimate_HT_genTruth(self, data):
        #n = len(data)
        var_to_sumT = 0 
        var_to_sumC = 0 


        q = 1-self.p
        if self.graph == "cycle":
            probasT = self.p**3
            probasC = q**3
            
            
            if(self.h== 0.5):
              probasT = probasT + 2*(1-self.p)*self.p**(2*self.h + 1)
              probasC = q**3 + 2*self.p*q**(2*self.h + 1)
              
            elif(self.h == 0):
              probasT = probasT + 2*(1-self.p)*self.p**(2*0.5 + 1) + ((1-self.p)**2)*self.p**(2*self.h + 1)
              probasC = q**3 + 2*self.p*q**(2*0.5 + 1) + (self.p**2)*q**(2*self.h + 1)
        elif self.graph == "kNN":
            if self.design == "cluster":
                clust_size = self.d + 1
                k = self.d/2
            if self.design == "unit":
                # do it for 0 = none of my neighbors
                probasT =  self.p**(self.d + 1)
                probasC = q**(self.d+1 )
                for i in range(self.d):
                    l = (self.d - i - 1)/self.d # this goes from 1 to 0
                    if l >= self.h:
                        probasT += math.factorial(int(self.d))/(math.factorial(int(self.d*l))*math.factorial(int(self.d - self.d*l)))*(self.p**(self.d*l+1))*((1-self.p)**(self.d-self.d*l))
                        probasC += math.factorial(int(self.d))/(math.factorial(int(self.d*l))*math.factorial(int(self.d - self.d*l)))*(q**(self.d*l+1))*((1-q)**(self.d-self.d*l))
        idx = np.where(self.hyperparams == self.h)[0][0]
        for tup in data:
            (i,z_i,e_i, y_i) = tup  # unit, unit treatment status, unit exposure level, unit outcome
            if self.kernel == "boxcarIPW":
                if self.design == "unit":
                    if z_i == 1 and e_i >= self.h:
                        #count1 += count1
                        #print(paste0("var_element: ", y[i]/probas))
                        var_to_sumT += y_i/probasT

                        
                    elif z_i == 0 and e_i<= (1-self.h):
                        var_to_sumC += y_i/probasC

                
                
                    
                elif self.design == "cluster" and self.graph == "kNN":
                        if z_i == 1 and e_i >= self.h:
                            #l = (1-self.h)*self.d
                            
                            # simulations
                            plus_= 0
                            find_subt = (1-self.h)*self.d #
                            # work backward from threshold (find what we're subtracting off to get the threshold) # NOTE: this is only correct for threshold up to k
                            if (self.h < k/self.d):
                              self.h = k/self.d # redefine
                              find_subt = (1-k/self.d)*self.d # the rest do not contribute
                            
                            mid = (clust_size + 1)/2
        
        
                            if(i % clust_size == 0):
                              plus_ = clust_size
                            
                            if(((i% clust_size + plus_) <= (mid+find_subt)) and ((i%clust_size + plus_) >= (mid-find_subt))):
                              # also the other direction
                              var_to_sumT += y_i/self.EP_treated[i,idx]

                            else:
                              var_to_sumT += y_i/self.EP_treated[i,idx]

                            
                            # analytical
                            # var += ((y_i/(n))**2)*((1/self.p**2 - 1)*(3*self.d + 1 - 2*l) + (2*l+1)*(1/self.p - 1))
        
        
                        elif z_i == 0 and e_i<= (1-self.h):
                            plus_ = 0
                            find_subt = (1-self.h)*self.d #
                            # work backward from threshold (find what we're subtracting off to get the threshold) # NOTE: this is only correct for threshold up to k
                            if (self.h <= k/self.d):
                              self.h = k/self.d # redefine
                              find_subt = (1-k/self.d)*self.d # the rest do not contribute
                            
                            mid = (clust_size + 1)/2
        
        
                            if(i % clust_size == 0):
                              plus_ = clust_size
                            
                            if(((i% clust_size + plus_) <= (mid+find_subt))&((i%clust_size + plus_) >= (mid-find_subt))):
                              # also the other direction
                              var_to_sumC += y_i/self.EP_control[i,idx]

                            else:
                              var_to_sumC += y_i/self.EP_control[i,idx]

                        
    
                        
        point_est = (var_to_sumT - var_to_sumC)/self.n # sample mean -> to give true var later on
        
        return(point_est)
        
    def point_estimate_DiM(self, data):
        biasT = 0
        biasC = 0
        var_to_sumT = 0 
        var_to_sumC = 0 

        countT= 0
        countC = 0
        
        var_estimate = self.var_estimator_DIM(self.n/2, self.n/2,data)
        if self.graph == "kNN":
            if self.design == "cluster":
                clust_size = self.d + 1
                k = self.d/2
        for tup in data:
            (i,z_i,e_i, y_i) = tup  # unit, unit treatment status, unit exposure level, unit outcome
            if self.kernel == "boxcarIPW":
                if self.design == "unit":
                    
                    if z_i == 1 and e_i >= self.h:
                        #count1 += count1
                        #print(paste0("var_element: ", y[i]/probas))
                        var_to_sumT += y_i
                        biasT += (1-e_i)*self.gammaHat
                        countT += 1

                        
                        
                    elif z_i == 0 and e_i<= (1-self.h):
                        var_to_sumC += y_i
                        biasC += e_i*self.gammaHat
                        countC += 1

                
                
                    
                elif self.design == "cluster" and self.graph == "kNN":
                        if z_i == 1 and e_i >= self.h:
                            countT += 1
                            #l = (1-self.h)*self.d
                            
                            # simulations
                            plus_= 0
                            find_subt = (1-self.h)*self.d #
                            # work backward from threshold (find what we're subtracting off to get the threshold) # NOTE: this is only correct for threshold up to k
                            if (self.h < k/self.d):
                              self.h = k/self.d # redefine
                              find_subt = (1-k/self.d)*self.d # the rest do not contribute
                            
                            mid = (clust_size + 1)/2
        
        
                            if(i % clust_size == 0):
                              plus_ = clust_size
                            
                            if(((i% clust_size + plus_) <= (mid+find_subt)) and ((i%clust_size + plus_) >= (mid-find_subt))):
                              # also the other direction
                              var_to_sumT += y_i
                              biasT += (1-e_i)*self.gammaHat
                            else:
                              var_to_sumT += y_i
                              biasT += (1-e_i)*self.gammaHat
                            
                            # analytical
                            # var += ((y_i/(n))**2)*((1/self.p**2 - 1)*(3*self.d + 1 - 2*l) + (2*l+1)*(1/self.p - 1))
        
        
                        elif z_i == 0 and e_i<= (1-self.h):
                            countC += 1
                            plus_ = 0
                            find_subt = (1-self.h)*self.d #
                            # work backward from threshold (find what we're subtracting off to get the threshold) # NOTE: this is only correct for threshold up to k
                            if (self.h <= k/self.d):
                              self.h = k/self.d # redefine
                              find_subt = (1-k/self.d)*self.d # the rest do not contribute
                            
                            mid = (clust_size + 1)/2
        
        
                            if(i % clust_size == 0):
                              plus_ = clust_size
                            
                            if(((i% clust_size + plus_) <= (mid+find_subt))&((i%clust_size + plus_) >= (mid-find_subt))):
                              # also the other direction
                              var_to_sumC += y_i
                              biasC += e_i*self.gammaHat
                            else:
                              var_to_sumC += y_i
                              biasC += e_i*self.gammaHat
                        
        
        if countT==0:
            pointT = 0
        else:
            pointT = var_to_sumT/countT
            biasT_ = biasT/countT
        if countC == 0:
            pointC = 0 
        else:
            pointC = var_to_sumC/countC
            biasC_ = biasC/countC
              
            
     
        point_est = (pointT- pointC) # sample mean
        bias = biasT_ + biasC_         
            #print("var_to_subt: ", var)
        return(point_est, bias, var_estimate)

   
            
    def point_estimate_HT(self, data):
        #n = len(data)
        var_to_sumT = 0 
        var_to_sumC = 0 
        var_estimate = self.var_estimator(data)
        bias_estimate = 0
        q = 1-self.p
        if self.graph == "cycle":
            probasT = self.p**3
            probasC = q**3
            
            
            if(self.h== 0.5):
              probasT = probasT + 2*(1-self.p)*self.p**(2*self.h + 1)
              probasC = q**3 + 2*self.p*q**(2*self.h + 1)
              
            elif(self.h == 0):
              probasT = probasT + 2*(1-self.p)*self.p**(2*0.5 + 1) + ((1-self.p)**2)*self.p**(2*self.h + 1)
              probasC = q**3 + 2*self.p*q**(2*0.5 + 1) + (self.p**2)*q**(2*self.h + 1)
        elif self.graph == "kNN":
            if self.design == "cluster":
                clust_size = self.d + 1
                k = self.d/2
            if self.design == "unit":
                # do it for 0 = none of my neighbors
                probasT =  self.p**(self.d + 1)
                probasC = q**(self.d+1 )
                for i in range(self.d):
                    l = (self.d - i - 1)/self.d # this goes from 1 to 0
                    if l >= self.h:
                        probasT += math.factorial(int(self.d))/(math.factorial(int(self.d*l))*math.factorial(int(self.d - self.d*l)))*(self.p**(self.d*l+1))*((1-self.p)**(self.d-self.d*l))
                        probasC += math.factorial(int(self.d))/(math.factorial(int(self.d*l))*math.factorial(int(self.d - self.d*l)))*(q**(self.d*l+1))*((1-q)**(self.d-self.d*l))
        idx = np.where(self.hyperparams == self.h)[0][0]
        for tup in data:
            (i,z_i,e_i, y_i) = tup  # unit, unit treatment status, unit exposure level, unit outcome
            if self.kernel == "boxcarIPW":
                if self.design == "unit":
                    if z_i == 1 and e_i >= self.h:
                        #count1 += count1
                        #print(paste0("var_element: ", y[i]/probas))
                        var_to_sumT += y_i/self.EP_treated[i,idx]
                        bias_estimate += (1-e_i)*self.gammaHat/self.EP_treated[i,idx]
                        
                    elif z_i == 0 and e_i<= (1-self.h):
                        var_to_sumC += y_i/self.EP_control[i,idx]
                        bias_estimate += e_i*self.gammaHat/self.EP_control[i,idx]
                
                
                    
                elif self.design == "cluster" and self.graph == "kNN":
                        if z_i == 1 and e_i >= self.h:
                            #l = (1-self.h)*self.d
                            
                            # simulations
                            plus_= 0
                            find_subt = (1-self.h)*self.d #
                            # work backward from threshold (find what we're subtracting off to get the threshold) # NOTE: this is only correct for threshold up to k
                            if (self.h < k/self.d):
                              self.h = k/self.d # redefine
                              find_subt = (1-k/self.d)*self.d # the rest do not contribute
                            
                            mid = (clust_size + 1)/2
        
        
                            if(i % clust_size == 0):
                              plus_ = clust_size
                            
                            if(((i% clust_size + plus_) <= (mid+find_subt)) and ((i%clust_size + plus_) >= (mid-find_subt))):
                              # also the other direction
                              var_to_sumT += y_i/self.EP_treated[i,idx]
                              bias_estimate += (1-e_i)*self.gammaHat/self.EP_treated[i,idx]
                            else:
                              var_to_sumT += y_i/self.EP_treated[i,idx]
                              bias_estimate += (1-e_i)*self.gammaHat/self.EP_treated[i,idx]
                            
                            # analytical
                            # var += ((y_i/(n))**2)*((1/self.p**2 - 1)*(3*self.d + 1 - 2*l) + (2*l+1)*(1/self.p - 1))
        
        
                        elif z_i == 0 and e_i<= (1-self.h):
                            plus_ = 0
                            find_subt = (1-self.h)*self.d #
                            # work backward from threshold (find what we're subtracting off to get the threshold) # NOTE: this is only correct for threshold up to k
                            if (self.h <= k/self.d):
                              self.h = k/self.d # redefine
                              find_subt = (1-k/self.d)*self.d # the rest do not contribute
                            
                            mid = (clust_size + 1)/2
        
        
                            if(i % clust_size == 0):
                              plus_ = clust_size
                            
                            if(((i% clust_size + plus_) <= (mid+find_subt))&((i%clust_size + plus_) >= (mid-find_subt))):
                              # also the other direction
                              var_to_sumC += y_i/self.EP_control[i,idx]
                              bias_estimate += e_i*self.gammaHat/self.EP_control[i,idx]
                            else:
                              var_to_sumC += y_i/self.EP_control[i,idx]
                              bias_estimate += e_i*self.gammaHat/self.EP_control[i,idx]
                        
    
                        
        point_est = (var_to_sumT - var_to_sumC)/self.n # sample mean -> to give true var later on
        
        return(point_est, bias_estimate/self.n, var_estimate)

    def simulations(self):
        # other options for outcome mod = noisy/simple/quad
        Env = self.env(p=self.p, graph = self.graph, d =self.d, design=self.design, outcome_mod = self.outcome_mod, a = self.a, b=self.b, c=self.c, n=self.n)
        Env.gen_design(self.n)
        Env.gen_graph(self.n)
        data = Env.gen_data(self.n)
        
        point_estimateHT_h = []
        point_estimateDiM_h = []

        for h in self.hyperparams: 
            self.h = h
            #E = self.estimator(h, self.design_param ,self.graph_param, self.graph, self.design, self.soften, self.kernel) # need to fix and include h in the following as well
            
            point_estimateHT_= self.point_estimate_HT_genTruth(data)  
            
            point_estimateDiM_= self.point_estimate_DiM_genTruth(data) 
            
            
            point_estimateHT_h.append(point_estimateHT_)

            point_estimateDiM_h.append(point_estimateDiM_)


        return (point_estimateHT_h,point_estimateDiM_h)
    
    
    def simulations_var_est(self):
        # other options for outcome mod = noisy/simple/quad
        Env = self.env(p=self.p, graph = self.graph, d =self.d, design=self.design, outcome_mod = self.outcome_mod, a = self.a, b=self.b, c=self.c, n=self.n)
        Env.gen_design(self.n)
        Env.gen_graph(self.n)
        data = Env.gen_data(self.n)
        self.gammaHat = Env.gen_coefficients()[2]
        
        point_estimateHT_h = []
        biasHT_h = []
        var_estimateHT_h = []
        point_estimateDiM_h = []
        biasDiM_h = []
        var_estimateDiM_h = []
        
        for h in self.hyperparams: 
            self.h = h
            #E = self.estimator(h, self.design_param ,self.graph_param, self.graph, self.design, self.soften, self.kernel) # need to fix and include h in the following as well
            
            point_estimateHT_, biasHT_, var_estimateHT_ = self.point_estimate_HT(data)  
            
            point_estimateDiM_, biasDiM_, var_estimateDiM_ = self.point_estimate_DiM(data) 
            point_estimateHT_h.append(point_estimateHT_)
            biasHT_h.append(biasHT_)
            var_estimateHT_h.append(var_estimateHT_)
            point_estimateDiM_h.append(point_estimateDiM_)
            biasDiM_h.append(biasDiM_)
            var_estimateDiM_h.append(var_estimateDiM_)
            
        return (point_estimateHT_h, biasHT_h,var_estimateHT_h, point_estimateDiM_h, biasDiM_h, var_estimateDiM_h)
    
    
    def ground_truth(self):
        # run 1000 simulations and then average over them [truth]
        ntrials = 1000

        point_estimatesHT_array = np.zeros((ntrials, len(self.hyperparams)))

        point_estimatesDiM_array = np.zeros((ntrials, len(self.hyperparams)))

        for i in range(ntrials):

            point_estimatesHT_array[i,:], point_estimatesDiM_array[i,:] = self.simulations() # returns point-estimates
        
                
        #compute column means
        meansHT = np.mean(point_estimatesHT_array, axis=0)
        
        varHT_array = (point_estimatesHT_array -  meansHT)**2
        varHT = np.mean(varHT_array, axis = 0)
        
        #biasHT = np.mean(biasHT_array, axis=0) # mean bias across thresholds
        
        #true_bias
        bias_trueHT = meansHT - self.true_ATE
        
        mseHT = bias_trueHT**2 + varHT
        
        meansDiM = np.mean(point_estimatesDiM_array, axis=0)
        
        
        varDiM_array = (point_estimatesDiM_array -  meansDiM)**2
        varDiM = np.mean(varDiM_array, axis = 0)
        
        #biasDiM = np.mean(biasDiM_array, axis=0) # mean bias across thresholds
        
        # true_bias
        bias_trueDiM = meansDiM - self.true_ATE
        
        mseDiM = bias_trueDiM**2 + varDiM
        
        min_indHT = np.argmin(mseHT)
        min_mseHT = mseHT[min_indHT]
        
        min_indDiM = np.argmin(mseDiM)
        min_mseDiM = mseDiM[min_indDiM]
        
        return bias_trueHT, varHT, mseHT, min_indHT, min_mseHT, bias_trueDiM, varDiM, mseDiM, min_indDiM, min_mseDiM
    
    def estimates_mse(self):


        point_estimateHT = np.zeros(len(self.hyperparams))
        biasHT = np.zeros(len(self.hyperparams))
        varHT = np.zeros(len(self.hyperparams))
        point_estimateDiM = np.zeros(len(self.hyperparams))
        biasDiM = np.zeros(len(self.hyperparams))
        varDiM = np.zeros(len(self.hyperparams))

        #point_estimatesHT_array, biasHT_array, point_estimatesDiM_array, biasDiM_array = self.simulations_var_est() # returns point-estimates
        point_estimateHT, biasHT,varHT, point_estimateDiM,biasDiM, varDiM = self.simulations_var_est()
                
        minLepskiHT = self.lepski(point_estimateHT, varHT)
        minLepskiDiM = self.lepski(point_estimateDiM, varDiM)

        biasHT = np.array(biasHT)
        varHT = np.array(varHT)
        biasDiM = np.array(biasDiM)
        varDiM = np.array(varDiM)
        mseHT = biasHT**2 + varHT
        
        mseDiM = biasDiM**2 + varDiM
        
        min_indHT = np.argmin(mseHT)
        min_mseHT = mseHT[min_indHT]
        
        min_indDiM = np.argmin(mseDiM)
        min_mseDiM = mseDiM[min_indDiM]
        
        return biasHT, varHT, mseHT, min_indHT, min_mseHT, biasDiM, varDiM, mseDiM, min_indDiM, min_mseDiM, minLepskiHT, minLepskiDiM
    
    def lepski(self, point_estimates, variances):
        variances = np.array(variances)
        variances[variances<0] = 0
        widths = np.sqrt(variances)
        
        intervals = []
        # the following for-loop seems unnecessary given the monotonicity assumption
        for i in range(len(self.hyperparams)):
            if i < len(self.hyperparams)-1:
                width = max(widths[i], max(widths[i+1:]))
                
            else:
                width = widths[i]
            #intervals.append((means[i] - 2*widths[i], means[i] + 2*widths[i]))

            intervals.append((point_estimates[i] - 2*width, point_estimates[i] + 2*width))
            #print("[Slope] h = %0.2f, mean = %0.2f, low = %0.2f, high = %0.2f" % (self.hyperparams[i], point_estimates[i], intervals[-1][0], intervals[-1][1]), flush=True) 
        index = 0
        curr = [intervals[0][0], intervals[0][1]]
        for i in range(len(intervals)):
            if intervals[i][0] > curr[1] or intervals[i][1] < curr[0]:
                ### Current interval is not overlapping with previous ones, return previous index
                break
            else:
                ### Take intersection
                curr[0] = max(curr[0], intervals[i][0])
                curr[1] = min(curr[1], intervals[i][1])
                index = i
            #print("[Slope] curr_low = %0.2f, curr_high = %0.2f" % (curr[0], curr[1]))
        #print("[Slope] returning index %d" % (index), flush=True)

        return index

        
    def comparison_plot(self):
        
        biasHT, varHT, mseHT, min_indHT, min_mseHT, biasDiM, varDiM, mseDiM, min_indDiM, min_mseDiM = self.ground_truth()  
        
        # print("True min MSE HT: ",  min_mseHT)
        # print("True min MSE  index HT: ",  min_indHT)
        # print("True min MSE DiM: ",  min_mseDiM)
        # print("True min MSE  index DiM: ",  min_indDiM)
        
        # mse_comp1 = mseHT[0]

        # mse_comp2 = mseDiM[-1]
        
        # estimated bias + variance + mse
        
        biasHT_est, varHT_est, mseHT_est, min_indHT_est, min_mseHT_est, biasDiM_est, varDiM_est, mseDiM_est, min_indDiM_est, min_mseDiM_est, minLepskiHT, minLepskiDiM = self.estimates_mse()  
        mse_chosen_estHT = mseHT[min_indHT_est]
        mseLepskiHT = mseHT[minLepskiHT]
        mse_chosen_estDiM = mseDiM[min_indDiM_est]
        mseLepskiDiM = mseDiM[minLepskiDiM]

        if self.estimator_type == "HT":
            mse_comp1 = mseHT[0]

            mse_comp2 = mseHT[-1]
            mse_comp4 = mseDiM[0]

            mse_comp3 = mseDiM[-1]
            
            results1 = [mse_chosen_estHT, min_mseHT, mse_comp1, mse_comp2,mseLepskiHT]
            results2 = [mse_chosen_estDiM, min_mseDiM, mse_comp3, mse_comp4,mseLepskiDiM]
            
            
        elif self.estimator_type == 'DiM':
            mse_comp3 = mseHT[0]

            mse_comp4 = mseHT[-1]
            mse_comp2 = mseDiM[0]

            mse_comp1 = mseDiM[-1]

            results1 = [mse_chosen_estDiM, min_mseDiM, mse_comp1, mse_comp2, mseLepskiDiM]
            results2 = [mse_chosen_estHT, min_mseHT, mse_comp3, mse_comp4, mseLepskiHT]
        
        
        
        
        return biasHT, varHT, mseHT, min_indHT, min_mseHT, biasDiM, varDiM, mseDiM, min_indDiM, min_mseDiM, min_indHT_est, min_mseHT_est, min_indDiM_est, min_mseDiM_est, minLepskiHT, minLepskiDiM, results1, results2
    
if __name__=='__main__':
    n = 1000
    size = n
    #graph_param = 2
    design_param = 0.5
    kernel = 'boxcarIPW'
    graph_ = "kNN"
    graph_param = 4
    #graph_ = "cycle"
    design_ = "unit" #options: cluster or unit
    #design_ = "cluster"
    outcome_mod = "noisy"
    estimator_type = "HT" #options: HT or DiM


    # pre-load to save time 
    EP_treated = None
    EP_control = None
    jointEP_treated = None
    jointEP_control = None
    jointEP_inverse = None
    
    EP_treated = np.load('k4c20unit_EP_treated.npy')
    EP_control = np.load('k4c20unit_EP_control.npy')
    jointEP_treated = np.load('k4c20unit_jointEP_treated.npy')
    jointEP_control = np.load('k4c20unit_jointEP_control.npy')
    jointEP_inverse = np.load('k4c20unit_jointEP_inverse.npy')
    
    if graph_ == "kNN" and design_ == "cycle" and graph_param > 2:
        ls = np.arange(0, graph_param+1, 1)
    elif design_ == "unit":
        ls = np.arange(0,graph_param+1,1)
    hs = (graph_param - ls)/graph_param
    
    #nsims = 500
    
    gammaBetaRatio= np.array([0,0.25, 0.5, 1, 2, 5])
    across_simsHT = np.zeros((len(gammaBetaRatio), 5))
    across_simsDiM = np.zeros((len(gammaBetaRatio), 5))

    ns = int(sys.argv[1])
    #all_resutlsRatios = np.zeros((len(gammaBetaRatio),5))
    countgb = 0
    for gb in gammaBetaRatio:
        
        a = 10
        b = 10
        c= gb*b
    
        BV = Bias_Var(params={'hyperparams': hs, 'graph_param': graph_param,'design_param': design_param,'graph':graph_, 'design': design_, 'size': size, 'kernel':kernel, 'env':graph_Env, 'outcome_mod': outcome_mod, "estimator_type": estimator_type, "a":a, "b":b, "c":c, 'EP_treated': EP_treated, 'EP_control': EP_control, "jointEP_treated": jointEP_treated,'jointEP_control':jointEP_control, 'jointEP_inverse': jointEP_inverse})
        
        biasHT, varHT, mseHT, min_indHT, min_mseHT, biasDiM, varDiM, mseDiM, min_indDiM, min_mseDiM, min_indHT_est, min_mseHT_est, min_indDiM_est, min_mseDiM_est, minLepskiHT, minLepskiDiM, result1, result2 = BV.comparison_plot()

        across_simsHT[countgb, :] = result1
        across_simsDiM[countgb, :] = result2

        countgb += 1
    list_name = "d"+ str(graph_param) + "n"+ str(n) + "_"+ str(design_) + "_" + str(outcome_mod) + "ratio_" + str(countgb) +" "
    pickle.dump(across_simsHT, open("out/dim/array/HT" + list_name + "ns_%d.p" % ns, "wb"))
    pickle.dump(across_simsDiM, open("out/dim/array/DiM" + list_name + "ns_%d.p" % ns, "wb"))
    df0 = pd.DataFrame([biasHT, varHT, mseHT, biasDiM, varDiM, mseDiM])
    df0.to_pickle("out/dim/array/groundTruth"+ list_name + "ns_%d.p" % ns +".pkl")
    
    
    
    
    

    

    
    
    
