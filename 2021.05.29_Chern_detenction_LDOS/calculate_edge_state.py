#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:11:43 2019

@author: mplodzien
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 17:48:09 2019

@author: mplodzien
"""


from mpl_toolkits import mplot3d


import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy.sparse import block_diag
from scipy.optimize import differential_evolution
from sympy import ordered, Matrix, hessian
from sympy import *
from sympy import lambdify
from sympy.tensor.array import derive_by_array
import sympy as sp
init_printing(use_unicode=True)
from sympy.functions.elementary.exponential import exp
from sympy import I
#%%
kx_s, ky_s, k2_s, k4_s, c_s, M, g1_s, g2_s, parameters_s, eta_s, delta_s, m_s, phi_s, s_s =symbols('kx_s ky_s k2_s k4_s c_s M g1_s g2_s parameters_s eta_s delta_s m_s phi_s s_s',real=True)


#
#tmp1 =   cos(k1_s) +  cos(k3_s) +  cos(k4_s) + m_s
#tmp2 =  c_s*(sin(k3_s) + 1j*sin(k4_s))
#tmp3 =  c_s*(sin(k3_s) - 1j*sin(k4_s))
#row_1 =  [ 1j*g1_s            , tmp1 + c_s*1j*sin(k1_s),    0                ,  tmp3] 
#row_2 =  [ tmp1 - c_s*1j*sin(k1_s), -1j*g2_s           ,    tmp3             ,  0   ] 
#row_3 =  [ 0                  , tmp2               , -1j*g1_s            , -tmp1 + c_s*1j*sin(k1_s) ] 
#row_4 =  [ tmp2               , 0                  , -tmp1 - c_s*1j*sin(k1_s), 1j*g2_s  ] 


#%%



#%%
row_1 = [m_s*exp(I*phi_s) + cos(kx_s) + cos(ky_s)   , s_s*(sin(kx_s) - I*sin(ky_s))];
row_2 = [s_s*(sin(kx_s) + I*sin(ky_s))              , -m_s*exp(I*phi_s) - cos(kx_s) - cos(ky_s) ];



H = Matrix([row_1,row_2])
#eval_up, eval_down = H.eigenvals()
#
#E_up = lambdify((kx, ky, M, delta_s), eval_up, modules='numpy') 
#E_down = lambdify((kx, ky, M, delta_s), eval_down, modules='numpy') 
#
#Sigma = Matrix([ [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1] ])
#
#H_eff = Sigma*(eta_s*sp.eye(4) - 1j*H) 

#%%
#get_H_numeric = lambdify((kx_s, ky_s, m_s, phi_s, s_s), H, modules='numpy') 
 

def get_H_numeric(kx,ky,alpha,mu,N):
    
    xi_nn = -2*(np.cos(kx) + np.cos(ky)) - mu
    xi_nnn = -2*(np.cos(kx+ky) + np.cos(kx-ky)) - mu
    
    xi_alpha = (1-alpha)*xi_nn + alpha*xi_nnn
    
    Delta_k = np.sin(kx) + 1j*np.cos(ky)
    Delta_N = np.abs(Delta_k)*np.exp(1j*N*np.angle(Delta_k))
    
    H = np.array([[xi_alpha, Delta_N],[np.conjugate(Delta_N),-xi_alpha]])
    
    
    return H


#%%

def get_H_in_k1_k2(k1,k2,parameters):
    alpha, mu, N = parameters

    H = get_H_numeric(k1,k2,alpha,mu,N)
    return H




#%%
def H_n_fourier_component_of_k1(n,k2,parameters): #n-th Fourier component of the Hamiltonian H(kx,ky) in y-direction, i.e. H(kx,ky) = sum_{n} H_tilda(kx,n)*Exp[-1j*n*ky] + h.c.
    m, phi,s  = parameters
    N_modes = 11 #arbitrary odd integer - rather much greater than 1
    N_band = 2
    h_n_mode = np.zeros([N_band,N_band])
    dk = 2.0*np.pi/N_modes
    if(n==0):
        for l in range(0,N_modes):
            k_l = dk*l    #%%
            h_n_mode = h_n_mode + get_H_in_k1_k2(k_l,k2,parameters)
    if(n>0):
        l_min = int(-(N_modes-1)/2)
        l_max = int((N_modes-1)/2)
        for l in range(l_min,l_max+1):
            k_l = dk*l
            h_n_mode = h_n_mode + np.exp(-1j*k_l*n)*get_H_in_k1_k2(k_l,k2,parameters)
    return h_n_mode/N_modes 

#%%
def get_H_N1_k2(N1,k2,hopping_range,parameters): #Hamiltonian in kx with finite (Ny) slices in y-direction
    N_band = 2
    H_on_diagonal   =   np.zeros([N_band*N1,N_band*N1])
    H_off_diagonal  =   np.zeros([N_band*N1,N_band*N1])
    n = 0
    H_on_diagonal   =   np.kron(np.eye(N1,k=0),H_n_fourier_component_of_k1(n,k2,parameters)) 
    for n in np.arange(1,hopping_range+1): 
        H_off_diagonal = H_off_diagonal + np.kron(np.eye(N1,k=n),H_n_fourier_component_of_k1(n,k2,parameters))         
    H = H_on_diagonal + H_off_diagonal + np.transpose(np.conjugate(H_off_diagonal))
    
    return H    






#%%
N1_vec = np.array([100])
hopping_range = 5

N_Brillouin_grid_k2 = 100
 

N_Brillouin_grid = N_Brillouin_grid_k2
k_min = -np.pi 
k_max = np.pi
 
k2_vec = np.linspace(k_min,k_max,N_Brillouin_grid_k2)
 
 
tuples_vec = np.array([[0.2,0,1]])


for N1 in N1_vec: 
    no_of_bands = 2
    DimH = no_of_bands*N1
    for tuples in tuples_vec:
        [alpha,mu,N] = tuples           
  
 
        string_parameters = "_model_1" + "_N1." + str(N1) + "_N_BZ." + str(N_Brillouin_grid) +  "_alpha." + "{:2.2f}".format(alpha) + "_mu." + "{:2.2f}".format(mu) + "_N." + "{:2.2f}".format(N)
        fileID = open("spectrum_system" + string_parameters + ".dat","w")            
        parameters = np.array([alpha,mu,N])
        for k2 in k2_vec:
 
            H_opened_in_k1 = get_H_N1_k2(N1,k2,hopping_range,parameters)
            Eigenenergies, P  = LA.eig(H_opened_in_k1)
            for ii in range(Eigenenergies.shape[0]):
 
                stringA = "{:2.4f}".format(k2) + " " + str(ii) + " " + "{:2.4f}".format(Eigenenergies[ii].real) + "\n"
#                Psi = P[:,ii]
#                
#                psi_up_band = Psi[np.arange(0,DimH,no_of_bands)]
#                psi_bottom_band = Psi[np.arange(1,DimH,no_of_bands)]
#
#
#                
#                rho = np.abs(psi_up_band)**2 + np.abs(psi_bottom_band)**2  
#                rho_left_edge = 0
#                rho_right_edge = 0                                
#                for j in range(0,10):
#                    rho_left_edge  = rho_left_edge + rho[j]
#                    rho_right_edge = rho_right_edge + rho[N1-1-j]
#                
#                rho_edges = rho_left_edge + rho_right_edge
#                rho_bulk = 1 - rho_edges
#
#
#                
#                stringB = " " + "{:2.4f}".format(rho_left_edge) +  " " + "{:2.4f}".format(rho_right_edge) + " " + "{:2.4f}".format(rho_bulk) + "\n"
                print(alpha,mu,N)
#                print(stringA + stringB)
#                fileID.write(stringA + stringB)
                print(stringA)
                fileID.write(stringA)
            fileID.write("\n")
        fileID.close()
                
            