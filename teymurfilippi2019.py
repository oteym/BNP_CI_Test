## BAYESIAN NONPARAMETRIC CONDITIONAL INDEPENDENCE TEST; TEYMUR & FILIPPI 2019
##
## README: 
## To run F&H independence test call function indep_test with 2D input array (X,Y) having defined an object of class 'inputs' for second argument
## To run T&F conditional independence test call function cond_indep_test with 2D input array (X,Y) and 1D input array Z (np.newaxis must be used) having defined an object of class 'inputs' for third argument
## Support of (X,Y,Z) required to be [0,1]^3. If not, call scale_data_squeeze (or use some other transformation more appropriate to your dataset)
## Simple single-call examples given near bottom of file in Section WALK-THROUGH EXAMPLE
## Plots require plotnine package (a Python port of R's ggplot2)
## NOTE: indep_test and cond_indep_test use opposite conventions for which way up the Bayes Factor is (this reflects a difference in the papers themselves)


#############################################################################


## PREAMBLE

# import os
# os.chdir('<directory>') #set current directory (if required)

import numpy as np
import pandas as pd
import copy
from scipy.special import logsumexp


## CLASS DEFINITIONS
        
class inputs:
    def __init__(self,max_depth_x = 20,max_depth_z = 20,c_par = 1,c_multiplier = 2,n_samps = 0,rho_z = 0):
        self.max_depth_x = max_depth_x
        self.max_depth_z = max_depth_z
        self.c_par = c_par
        self.c_multiplier = c_multiplier
        self.n_samps = n_samps
        self.rho_z = rho_z

class node_class:
    def __init__(self,dim,coords,address,dataindices,Phi_0,Phi,leaf):
        self.dim = dim
        self.coords = coords
        self.address = address
        self.level = len(self.address[0])
        self.leb = np.prod(self.coords[:,1]-self.coords[:,0])
        self.dataindices = dataindices
        self.data_n = self.dataindices.shape[0]
        self.Phi_0 = Phi_0
        self.Phi = Phi
        self.leaf = leaf


## SUBROUTINES

def scale_data_squeeze(x):
    for i in range(0,x.shape[1]):
        xmin = np.min(x[:,i])
        xmax = np.max(x[:,i])
        x[:,i] = np.copy((x[:,i]-xmin)/(xmax-xmin))
    return x


## TEST FUNCTIONS

def generate_uniform(n):
  x = np.random.uniform(0,1,n)
  y = np.random.uniform(0,1,n)
  z = np.random.uniform(0,1,n)
  data = np.array([x,y,z]).transpose()
  return data

def generate_sum_dependence(n):
  a = np.random.uniform(0,1,n)
  b = np.random.uniform(0,1,n)
  z = np.random.uniform(0,1,n)
  x = 0.5*(a+z)
  y = 0.5*(b+z)
  data = np.array([x,y,z]).transpose()
  return data

def generate_uniform_spiral_with_noise(n):
  x = np.random.uniform(0,1,n)
  y = np.random.uniform(0,1,n)
  eps = np.random.normal(0,0.5,n)
  z = (1/np.pi)*np.arctan(y/x + eps) + 0.5
  data = np.array([x,y,z]).transpose()
  return data

def generate_partial(n):
  a = np.random.multivariate_normal([0.5,0.5],[[0.1, 0.05],[0.05,0.1]],n*50)
  a = a[np.all([ a[:,0]<1 , a[:,0]>0, a[:,1]<1 , a[:,0]>0 ],axis=0),]
  b = np.random.normal(0.5,0.5,n*50)
  b = b[np.all([ b<1 , b>0],axis=0)]
  x = a[0:n,0]
  y = a[0:n,1]
  c = np.random.binomial(1,0.9,n)
  z = c*b[0:n] + (1-c)*b[0:n]*x*y
  data = np.array([x,y,z]).transpose()
  return data


## UNCONDITIONAL MARGINAL LIKELIHOOD CALC AND TEST AS IN FILIPPI&HOLMES

def marginal_likelihood(data,ins):
    if data.shape[0] <= 1:
        return 0
    ml = 0
    t_mat = np.zeros((data.shape[0]-1,ins.max_depth_x))
    indices_matrix = np.zeros((data.shape[0],ins.max_depth_x,data.shape[1]),dtype=int)
    for i in range(0,data.shape[0]):
        for d in range(0,data.shape[1]): 
            indices_matrix[i,0:ins.max_depth_x,d] = np.floor(np.power(2,np.arange(0, ins.max_depth_x, 1))*data[i,d])
    
    for i in range(1,data.shape[0]):        
        data_left = np.arange(0,i)
      
        for j in range(0,ins.max_depth_x):
            if len(data_left)==0: break
            data_left = data_left[np.all(indices_matrix[data_left,j,]==indices_matrix[i,j,],axis=1)]
            t_mat[i-1,j] = len(data_left)
            if j>0:
                ml = ml + np.log(np.power(2,data.shape[1]) * ins.c_par*(j)**2 + np.power(2,data.shape[1]) * t_mat[i-1,j]) \
                    - np.log(np.power(2,data.shape[1]) * ins.c_par*(j)**2 + t_mat[i-1,j-1]) 
    return ml


def indep_test(data,ins):
    if data.shape[1] != 2:
        print("Dataset must be two-dimensional")
        return    
    data = np.copy(scale_data_squeeze(data))   
    H1 = marginal_likelihood(data,ins)
    ins2 = copy.deepcopy(ins)
    ins2.c_par = ins.c_multiplier*ins.c_par
    H0_A = marginal_likelihood(data[:,0,np.newaxis],ins2)
    H0_B = marginal_likelihood(data[:,1,np.newaxis],ins2)
    bf = np.exp(H1 - H0_A - H0_B) # note order of hypotheses! opposite to cond_indep_test
    return(1/(1+bf))


## CONDITIONAL MARGINAL LIKELIHOOD CALC AND TEST AS IN TEYMUR&FILIPPI

def conditional_marginal_likelihood(data_x,data_z,ins,output_toggle):
    if data_z.shape[0] <= 1:
        return 0  
    root_coords = np.stack((np.zeros(data_z.shape[1]),np.ones(data_z.shape[1])),axis=-1)       
    root = node_class(dim = data_z.shape[1], \
                      coords = root_coords, \
                      address = np.repeat("",data_z.shape[1]), \
                      dataindices = np.where(np.prod([np.asarray((data_z[:,x] > root_coords[x,0]) & (data_z[:,x] < root_coords[x,1])) for x in range(0,data_z.shape[1])],axis=0) == 1)[0], \
                      Phi_0 = marginal_likelihood(data_x,ins), \
                      Phi = 0, \
                      leaf = 0 )
    global output_z    
    output_z = pd.DataFrame(columns = ['Level' , 'Address', 'n' , 'Phi_0' , 'Phi' , 'leaf' , 'leb'])
                              
    def recurs_func(node,ins):
        global output_z
        if node.level >= ins.max_depth_z:
            node.Phi = node.Phi_0
            node.leaf = 2
            output_z = output_z.append({'Level' : node.level , 'Address' : node.address , 'n' : node.data_n , 'Phi_0' : node.Phi_0 , 'Phi' : node.Phi , 'leaf' : node.leaf , 'leb' : node.leb},ignore_index=True)            
            return node.Phi
        elif node.data_n < 2:
            node.Phi = node.Phi_0
            node.leaf = 1
            output_z = output_z.append({'Level' : node.level , 'Address' : node.address , 'n' : node.data_n , 'Phi_0' : node.Phi_0 , 'Phi' : node.Phi , 'leaf' : node.leaf , 'leb' : node.leb},ignore_index=True)
            return node.Phi
        else:
            children = []
            for i in range(0,2^node.dim - 1):
                which_child = np.array(list(format(i,'0{a}b'.format(a=node.dim))))  # produces array of binary indices
                coords1 = node.coords[:,0] + 0.5*which_child.astype(np.float)*(node.coords[:,1]-node.coords[:,0])
                coords2 = coords1 + 0.5*(node.coords[:,1]-node.coords[:,0])
                child_node = node_class(dim = node.dim, \
                                        coords = np.stack((coords1,coords2),axis=-1), \
                                        address = np.core.defchararray.add(node.address,which_child), \
                                        dataindices = node.dataindices[ np.where(np.prod([np.asarray((data_z[node.dataindices,x] > coords1[x]) & (data_z[node.dataindices,x] < coords2[x])) for x in range(0,data_z.shape[1])],axis=0) == 1)[0] ], \
                                        Phi_0 = [], \
                                        Phi = node.Phi, \
                                        leaf = node.leaf)
                child_node.Phi_0 = marginal_likelihood(data_x[child_node.dataindices,],ins)
                children.append(child_node)
            node.Phi = logsumexp([ np.log(ins.rho_z) + node.Phi_0 , np.log(1 - ins.rho_z) + np.sum([recurs_func(y,ins) for y in children])])
            output_z = output_z.append({'Level' : node.level , 'Address' : node.address , 'n' : node.data_n , 'Phi_0' : node.Phi_0 , 'Phi' : node.Phi , 'leaf' : node.leaf , 'leb' : node.leb},ignore_index=True)
            return node.Phi
        
    cml = recurs_func(root,ins)
    if output_toggle == 1:
        with pd.option_context('display.max_rows', 1000):
            print(output_z)
    return cml


def cond_indep_test(data_x,data_z,ins):
    if data_x.shape[1] != 2:
        print("Dataset must be two-dimensional")
        return
    if data_x.shape[0] == 1:
        return 0.5  
    data_x = np.copy(scale_data_squeeze(data_x))
    data_z = np.copy(scale_data_squeeze(data_z)) 
    H1 = conditional_marginal_likelihood(data_x,data_z,ins,0)
    ins2 = copy.deepcopy(ins)
    ins2.c_par = ins.c_multiplier*ins.c_par
    H0_A = conditional_marginal_likelihood(data_x[:,0,np.newaxis],data_z,ins2,0)
    H0_B = conditional_marginal_likelihood(data_x[:,1,np.newaxis],data_z,ins2,0)
    bf = np.exp(-H1 + H0_A + H0_B) # NOTE: Bayes Factor is opposite way round compared to filippi & holmes
    return(1/(1+bf))


## WALK-THROUGH EXAMPLE

# # set algorithm parameters using object of class 'inputs' (defaults provided in class definition for unassigned parameters) 
# ins = inputs(max_depth_x = 5,max_depth_z = 5,c_par = 1,c_multiplier = 2,n_samps = 0, rho_z = 0.5)

# # example dataset
# data = generate_uniform(1000)

# # independence test (outputs probability of INDEPENDENCE)
# print(indep_test(data[:,0:2],ins))

# # conditional independence test (outputs probability of (conditional) DEPENDENCE)
# print(cond_indep_test(data[:,0:2],data[:,2,np.newaxis],ins))
    

## RUN EXPERIMENTS ON FOUR TEST PROBLEMS
    
# def call_tests(output_toggle):
#     global four_test_problems_results    
#     four_test_problems_results = pd.DataFrame(columns = ['Model' , 'Seed', 'n' , 'c_par', 'c_multiplier', 'max_depth_x' , 'max_depth_z', 'rho_z' , 'probability'])  
#     nvec = np.array([1,2,5,10,21,46,100,215,464,1000])
#     repetitions = 10 # Figure 3 in T&F has 100
#     for k in range(4):
#         for i in range(nvec.size):
#             print([k+1,nvec[i]])
#             for j in range(repetitions):
#                 np.random.seed(j)
#                 if k == 0: data = generate_uniform(nvec[i])
#                 if k == 1: data = generate_sum_dependence(nvec[i])
#                 if k == 2: data = generate_uniform_spiral_with_noise(nvec[i])
#                 if k == 3: data = generate_partial(nvec[i])
#                 # md = max(int(np.ceil(3/2*np.log2(nvec[i]))),8)
#                 md = 3 # Figure 3 in T&F uses line above
#                 ins = inputs(max_depth_x = md,max_depth_z = md,c_par = 1,c_multiplier = 2,n_samps = 0, rho_z = 0.5)
#                 test_outcome = cond_indep_test(data[:,0:2],data[:,2,np.newaxis],ins)
#                 four_test_problems_results = four_test_problems_results.append({'Model' : k+1, 'Seed' : j, 'n' : nvec[i], 'c_par' : ins.c_par, 'c_multiplier' : ins.c_multiplier, 'max_depth_x' : ins.max_depth_x , 'max_depth_z' : ins.max_depth_z, 'rho_z' : ins.rho_z , 'probability' : test_outcome},ignore_index=True)
#     if output_toggle == 1:
#         with pd.option_context('display.max_rows', 1000):
#             print(four_test_problems_results)
#     return(four_test_problems_results)


## CALL EXPERIMENTS (first line reproduces data from Figure 3 in T&F)

# four_tests = call_tests(0)
# four_tests.to_csv('out.csv',index=False)


## PRODUCE PLOTS (requires plotnine package)

# import plotnine as p9

# data_summary = four_tests.groupby(['n','Model']).agg(q1=('probability' , lambda x: x.quantile(0.05)), q2=('probability' , lambda x: x.quantile(0.25)), q3=('probability' , lambda x: x.quantile(0.5)), q4=('probability' , lambda x: x.quantile(0.75)), q5=('probability' , lambda x: x.quantile(0.95))).reset_index()
# (p9.ggplot(data_summary) +
#   p9.geom_ribbon(p9.aes(x='n',ymin = 'q1', ymax = 'q5'), fill ="#175EAD", alpha = 0.2) +
#   p9.geom_ribbon(p9.aes(x='n',ymin = 'q2', ymax = 'q4'), fill ="#175EAD", alpha = 0.4) +
#   p9.geom_line(p9.aes(x='n',y='q3'),size = 1, color = "#318AED") +
#   p9.facet_grid('.~Model') +
#   p9.scale_x_log10() +
#   p9.xlab("n") +
#   p9.ylab("p(H1|W)")
# )

