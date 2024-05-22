#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:38:25 2024

@author: 
"""

import numpy as np
from graphEnv import graph_Env



# number of nodes x number of thresholds



class MC_exposure_probas(object):
    
   
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
        if 'a' in params.keys():
            self.a = params['a']
        if 'b' in params.keys():
            self.b = params['b']
        if 'c' in params.keys():
            self.c = params['c']
        self.exposure_probas_all_treated = np.zeros((self.n,len(self.hyperparams)))
        self.exposure_probas_all_control= np.zeros((self.n,len(self.hyperparams)))
        self.exposure_probas_joint_treated = np.zeros((self.n,self.n,len(self.hyperparams)))
        self.exposure_probas_joint_control= np.zeros((self.n,self.n,len(self.hyperparams)))
        self.exposure_probas_joint_inverse = np.zeros((self.n,self.n,len(self.hyperparams)))


    
    def sims_ep_call(self, ntrials = 1000):
        
        for nt in range(ntrials):
            if nt % 10 == 0:
                print("ep trial: " , nt)
            self.sims_ep()
        
        return (self.exposure_probas_all_treated/ntrials, self.exposure_probas_all_control/ntrials, self.exposure_probas_joint_treated/ntrials, self.exposure_probas_joint_control/ntrials, self.exposure_probas_joint_inverse/ntrials)
    
    # calculate exposure probabilities via MC methods
    def sims_ep(self):
        # other options for outcome mod = noisy/simple/quad
        Env = self.env(p=self.p, graph = self.graph, d =self.d, design=self.design, outcome_mod = self.outcome_mod, a = self.a, b=self.b, c=self.c, n=self.n)
        Env.gen_design(self.n)
        Env.gen_graph(self.n)
        data = Env.gen_data(self.n)
        
        counth = 0
        for h in self.hyperparams: 
            
            for tup1 in data:
                (i,z_i,e_i, y_i) = tup1
                if z_i == 1 and e_i >= h:
                    self.exposure_probas_all_treated[i, counth] += 1
    
                    
                    
                elif z_i == 0 and e_i<= (1-h):
                    self.exposure_probas_all_control[i, counth] += 1
                    
                for tup2 in data:
                    (j,z_j,e_j, y_j) = tup2
                    
                    
                    if z_i == 1 and e_i >= h and z_j == 1 and e_j >= h :
                        self.exposure_probas_joint_treated[i, j, counth] += 1
                    
                    if z_i == 0 and e_i <= 1-h and z_j == 0 and e_j <= 1-h :
                        self.exposure_probas_joint_control[i, j, counth] += 1
                    
                    if z_i == 1 and e_i >= h and z_j == 0 and e_j <= 1-h :
                        self.exposure_probas_joint_inverse[i, j, counth] += 1
                    
            counth += 1
                    

    
if __name__=='__main__':
    n = 1000
    size = n
    #graph_param = 2
    design_param = 0.5
    kernel = 'boxcarIPW'
    #graph_ = "kNN"
    graph_param = 2
    graph_ = "cycle"
    design_ = "unit" #options: cluster or unit
    #design_ = "cluster"
    outcome_mod = "simple"
    estimator_type = "HT" #options: HT or DiM
    a = 10
    b = 10
    c= 10

    
    ls = np.arange(0,graph_param+1,1)
    hs = (graph_param - ls)/graph_param
    
    MCEP = MC_exposure_probas(params={'hyperparams': hs, 'graph_param': graph_param,'design_param': design_param,'graph':graph_, 'design': design_, 'size': size, 'kernel':kernel, 'env':graph_Env, 'outcome_mod': outcome_mod, "estimator_type": estimator_type, "a":a, "b":b, "c":c})
    
    

    
    EP_treated, EP_control, jointEP_treated, jointEP_control, jointEP_inverse = MCEP.sims_ep_call()
    
    # np.save('k4c20unit_EP_treated.npy', EP_treated)
    # np.save('k4c20unit_EP_control.npy', EP_control)
    # np.save('k4c20unit_jointEP_treated.npy', jointEP_treated)
    # np.save('k4c20unit_jointEP_control.npy', jointEP_control)
    # np.save('k4c20unit_jointEP_inverse.npy', jointEP_inverse)
    
        
    np.save('d2c10unit_EP_treated.npy', EP_treated)
    np.save('d2c10unit_EP_control.npy', EP_control)
    np.save('d2c10unit_jointEP_treated.npy', jointEP_treated)
    np.save('d2c10unit_jointEP_control.npy', jointEP_control)
    np.save('d2c10unit_jointEP_inverse.npy', jointEP_inverse)
    
    



    


