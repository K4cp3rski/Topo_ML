#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:53:35 2019

@author: marcin
"""
import time
import numpy as np
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy.matlib
import h5py
#import theano
import os
from keras.layers import Dropout
import random
#%%
def if_file_exists(filename):
    
    return os.path.isfile('./'+filename)
#%%
    
def get_H0_bulk(kx,ky,N,mu,alfa,Delta0):
    
    H0_bulk = np.array([[0,0],[0,0]],dtype="complex128")

    xi_nn = -2.0*(np.cos(kx) + np.cos(ky)) - mu
    xi_nnn = -2.0*(np.cos(kx+ky) + np.cos(kx-ky)) - mu    
    
    delta = Delta0*(np.sin(kx) + 1j*np.sin(ky))
    DELTA = np.abs(delta)*np.exp(1j*N*np.angle(delta))
    H0_bulk[0,0] = (1-alpha)*xi_nn + alpha*xi_nnn
    H0_bulk[0,1] = DELTA
    H0_bulk[1,0] = np.conjugate(DELTA)
    H0_bulk[1,1] = -H0_bulk[0,0]
    
    return H0_bulk

#%%

def get_gap_bulk(N,mu,alfa,Delta0,on_off_flat):
    
    gap = 100000
    Nk = 50
    kx_vec = np.linspace(-np.pi,np.pi,Nk)
    ky_vec = kx_vec
    if(on_off_flat==0):
        for kx in kx_vec:
            for ky in ky_vec:
                H0_bulk = get_H0_bulk(kx,ky,N,mu,alpha,Delta0)
                evals = LA.eigvalsh(H0_bulk)
                gap_new = np.abs(evals[0]-evals[1])
                if(gap_new<gap):
                    gap = gap_new    
    if(on_off_flat==1):
        gap = 2.0
    return gap



#%%
def get_H0(Nx,Ny,N,nc,mu,alfa,Delta0,on_off_flat):
#    parameters_str = "Nx." + str(Nx) + "_Ny." + str(Ny) + "_mu." + "{:2.2f}".format(mu) + "_alfa." + "{:2.2f}".format(alfa) + "_v." + "{:2.2f}".format(v)
 
    #Duration of clean system Hamiltonian generation
#    start_time = timeit.default_timer()

    id = np.array([[1,0],[0,1]])
    sz = np.array([[1,0],[0,-1]])
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    
    dkx = 2.0*np.pi/(Nx)
    dky = 2.0*np.pi/(Ny)
    
    # create (1*NxNy) Brillouin zone coordinate vectors kx,ky 
    KX = np.arange(-np.pi,np.pi,dkx)
    KY = np.arange(-np.pi,np.pi,dky)
    kx = np.kron(KX,np.ones([Nx]))
    ky = np.kron(np.ones([Nx]),KY)
    
 
    # create d(k) vector, size (1*NxNy)
    dx = Delta0*np.sin(kx)
    dy = Delta0*np.sin(ky)
    xi_nn = -2.0*(np.cos(kx) + np.cos(ky)) - mu
    xi_nnn = -2.0*(np.cos(kx+ky) + np.cos(kx-ky)) - mu
    #dz is equivalent to xi_alfa(kx,ky) in Chern-ML Slack notes
    dz = (1.0-alfa)*xi_nn + alfa*xi_nnn
    d =  (dx**2 + dy**2 + dz**2)**(0.5)
    
    #  
    a = np.where(dy>=0)
    b = np.where(dy<0)
    phi = np.zeros([Nx*Ny])
    
    ##Careful here: arccos has different branches! Use different expressions for
    ##negative and positive dy values. Also avoid division by zero.
    phi[a] =  np.arccos(dx[a] / (dx[a]**2.0 + dy[a]**2.0 + 10**(-10))**0.5)
    phi[b] = -np.arccos(dx[b] / (dx[b]**2.0 + dy[b]**2.0 + 10**(-10))**0.5)
 
    #
    #
    ##create the d(k)^N vector, denoted by D(k). Two choices: flat band version
    ##given by the first set of Dx,Dy,Dz or the non-flat band given by the
    ##second set
    if(on_off_flat == 1):
        Dx=(1.0 - (dz/d)**2)**(0.5)*np.cos(N*phi);
        Dy=(1.0 - (dz/d)**2)**(0.5)*np.sin(N*phi);
        Dz=dz/d;
    
#    stop
#    print("band gap width = ", band_gap_width)

    #
    if(on_off_flat == 0):
        Dx = (dx**2.0 + dy**2.0)**0.5*np.cos(N*phi)
        Dy = (dx**2.0 + dy**2.0)**0.5*np.sin(N*phi)
        Dz = dz
 
    # (nc+1)*(nc+1) hopping ampliutde matrices Tx,Ty,Tz for nc nearest neighbor sites in the positive x and y direction
    # Nx, Ny shoul be larger than nc
    
    Tx = np.zeros([nc+1,nc+1],dtype=np.complex_)
    Ty = np.zeros([nc+1,nc+1],dtype=np.complex_)
    Tz = np.zeros([nc+1,nc+1],dtype=np.complex_)
    
    for nx in range(0,nc+1):
        for ny in range(0,nc+1):
            Tx[nx,ny] = sum(Dx*np.exp(1j*nx*kx+1j*ny*ky))
            Ty[nx,ny] = sum(Dy*np.exp(1j*nx*kx+1j*ny*ky))
            Tz[nx,ny] = sum(Dz*np.exp(1j*nx*kx+1j*ny*ky))
            
    Tx = Tx/(Nx*Ny)
    Ty = Ty/(Nx*Ny)
    Tz = Tz/(Nx*Ny)
        

    # form the 2d Hamiltonian from Nx different 2Ny*2Ny blocks describing 1d chains in y direction 
    # Let's assume hopping between NC neighbours in x and y direction    
        
    
        
    NC = nc
    H0 = np.zeros([2*Nx*Ny,2*Nx*Ny])
    
    for nx in range(0,NC+1):
     
        h = np.zeros([2*Ny,2*Ny])
        if nx == 0:
            for ny in range(1,NC+1):
                T = Tx[nx,ny]*sx + Ty[nx,ny]*sy + Tz[nx,ny]*sz
                h = h + np.kron( np.diag(np.ones(Ny-ny),-ny) , T ) 
    
          
        else:
            for ny in range(0,NC+1):
                # hopping amplitudes in positive y direction
                T1 = Tx[nx,ny]*sx + Ty[nx,ny]*sy + Tz[nx,ny]*sz
              
                # hopping amplitudes in negative y direction
                T2 = Tx[nx,ny]*sx - Ty[nx,ny]*sy + Tz[nx,ny]*sz
     
                if ny==0:
                    h = h + np.kron(   np.diag( np.ones(Ny-ny) ,ny)  ,T1)
                else:
                    h = h + np.kron(np.diag( np.ones(Ny-ny) , ny) ,T2);
                    h = h + np.kron(np.diag( np.ones(Ny-ny) ,-ny) ,T1);
        hT = h.T
        H0 = H0 + np.kron( np.diag(np.ones(Nx-nx),nx) ,hT)
        
    
    ### Treat diagonal elements of H separately
    T0 = Tx[0,0]*sx + Ty[0,0]*sy + Tz[0,0]*sz
    h0 = np.kron(np.eye(Ny),T0)
    H0 = H0 + H0.T + np.kron(np.eye(Nx),h0)


    return H0   

def get_Chern_marker(seed,H0,Nx,Ny,v_max,gap_clean,energy_window_vec):

   # x and y operator prepared for projection in (1)
    x1 = np.arange(1,Nx+1,1)
    y1 = np.arange(1,Ny+1,1)    
    
    x = np.kron(np.transpose(x1),np.ones([1,Ny]))
    x = x.T
    x = np.kron(x,np.vstack((1,1)))
    x = np.diag(x[:,0])
#         
    
    y = np.kron(np.ones([1,Nx]),y1.T)
    y = y.T
    y = np.kron(y, np.vstack((1,1)))
    y = np.diag(y[:,0])  
    # Ranges prepared for calculating Chern marker in (2)
    NE_range = np.arange(0,Nx*Nx)

    R = np.floor(0.3*Nx)
    bulk_X = np.arange(int(Nx/2-R),int(Nx/2+R),1)
    bulk_Y = np.arange(int(Ny/2-R),int(Ny/2+R),1)    
  
    # Add disorder, V is the strength of spatially uncorrelated "nonmangetic" disorder
    # potential characterized by magniteude V
#    C_marker_aver = 0
#    C_square_aver = 0

    # seed = int(time.time())    
    rng = np.random.RandomState(seed)
    
#    v_sample = rng.uniform(0,v_max)
    v_sample = v_max
    V_disorder = v_sample*gap_clean*rng.uniform(-1,1,Nx*Ny)

#    mean_normal_distribution = 0     # mean of normal distribution
#    std_normal_distribution = 0.3                # standard deviation
#    V_disorder = np.random.normal(mean_normal_distribution, std_normal_distribution, Nx*Ny)


    V = np.kron(V_disorder,np.array([1,-1]))
    H_diss = np.diag(V)
    H = H0 + H_diss
        
        #Spectrum and wavefunctions       

   
    [E,P] = LA.eigh(H)    
    

    
    N_window = energy_window_vec.shape[0]
    counter = np.zeros(N_window,)
    LDOS = np.zeros([N_window,Nx*Nx])
    for energy_window_i in range(0,N_window):
        energy_window = energy_window_vec[energy_window_i]
        for i in range(0,E.shape[0]):
            if(np.abs(E[i])<=energy_window*gap_clean):
                LDOS[energy_window_i,:] = LDOS[energy_window_i,:] + np.abs(P[np.arange(0,2*Nx*Ny,2),i])**2.0 + np.abs(P[np.arange(1,2*Nx*Ny,2),i])**2.0
                counter[energy_window_i] = counter[energy_window_i] + 1
        # no_states_check = np.where(np.abs(E)<=energy_window*gap_clean)[0]
        
        # print("Check = ",counter,no_states_check.shape[0])
#    print("LDOS: %s s"  % (time.time() - start_time) )
#         Chern marker
#         (1) PP - projection operator to negative energy states
 
 
    NE = P[:,NE_range]
    PP = np.dot(np.conj(NE),NE.T)
    # Px is projected x coordinate PxP
    Px = np.dot(PP,np.dot(x,PP))
    # Py is projected y coordinate PyP
    Py = np.dot(PP,np.dot(y,PP))
    
    # (2) C = Chern marker matrix
    C = -2*np.pi*1j*(np.dot(Px,Py)-np.dot(Py,Px))
    # diagonal elements give the desired info
    c = np.diag(np.real(C))
    # Sum different spin components corresponding to the same spatial coordinate
    c_up = c[np.arange(0,2*Nx*Ny,2)]
    c_down = c[np.arange(1,2*Nx*Ny,2)]
    c = c_up + c_down
    # reshape c to Nx*Ny array
 
    X,Y = np.meshgrid(np.arange(0,Nx,1),np.arange(0,Ny,1))
    c = np.reshape(c,[Nx,Ny])
    C_marker = np.mean(c[bulk_X,bulk_Y])
#    print("Chern marker: %s"  % (time.time() - start_time) )
 

    
    # Shift data
    for energy_window_i in range(0,N_window):
        if(counter[energy_window_i]>0):
                # mean = np.mean(LDOS[energy_window_i,:])
                norm = np.sum(LDOS[energy_window_i,:])
                # std = np.std(LDOS[energy_window_i,:])
                # LDOS[energy_window_i,:] = (LDOS[energy_window_i,:] - mean)/std
                LDOS[energy_window_i,:] = LDOS[energy_window_i,:]/norm
                # print("Norm = ", norm)
 
    return C_marker ,  LDOS , counter,  v_sample, seed 


#%%
def bulk_Chern(N,alpha,mu):
    
    mu_1 = 4*alpha
    mu_2 = -8*alpha + 4
    
    if(mu <= mu_1 and mu<=mu_2):
        Chern_bulk = N
    if(mu <= mu_1 and mu>=mu_2):
        Chern_bulk = 2*N
    if(mu >= mu_1 and mu<=mu_2):
        Chern_bulk = -N
    if(mu >= mu_1 and mu>=mu_2):
        Chern_bulk = 0
 
    return Chern_bulk

    
 
    #%%
    

 
###############################################################################################################################
#############################    Generate training and testing data                     ####################################### 
###############################################################################################################################
#%% Parameters of the generated data sets. All parameters are used to contruct filename for data sets.
## Generate training and testing data for different disorder amplitude
import time
import random
from datetime import datetime

Nx_vec = np.array([24])                         # System size : square (Nx,Nx)
 
nc = 5                         # Hopping radius 
N_mu = 25
N_alpha = 25

on_off_flat = 0

for Nx in Nx_vec:
    Ny = Nx
    
    mu_vec = np.linspace(-4,4,N_mu)
      
    class_type = "D"
    energy_window_vec = np.array([0.1])
    energy_window = energy_window_vec[0]
    Delta0_vec = np.array([1.0])    
    v_max_vec = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])
    N_v = v_max_vec.shape[0]
      
    for Delta0 in Delta0_vec:
        if(class_type == "A"):
 
            N_vec = np.array([3,1]) #integers
            alpha_vec = np.linspace(0.05,1,N_alpha)
            filename_data = "raw_A_flat." + str(on_off_flat) + "_Nx." + str(Nx) + "_Delta." + "{:2.2f}".format(Delta0)  + ".data_raw" 
        
        if(class_type == "D"):
            N_vec = np.array([1,3])  #odd integers
            alpha_vec = np.linspace(0.05,1,N_alpha)

 
        
        # Select proper points on phase diagram
        proper_points_on_phase_diagram = np.empty((1,10))
        seed_integer = 0
        for alpha in alpha_vec:
            for mu in mu_vec:
                for N in N_vec:
                    seed_integer = seed_integer + 1
                    H0 = get_H0(Nx,Ny,N,nc,mu,alpha,Delta0,on_off_flat)
                    gap_clean = get_gap_bulk(N,mu,alpha,Delta0,on_off_flat)
                    C_bulk = bulk_Chern(N,alpha,mu)
                    # if(band_gap_clean > gap_minimal):
                        # gap_aver_clean = gap_aver_clean + band_gap_clean
                        # counter_gap = counter_gap + 1                                                
                    idx_NaN = np.argwhere(np.isnan(H0))
                    
                    if(idx_NaN.shape[0]==0 and np.abs(C_bulk)<4.0):
                        v_max = 0
                        C_marker_clean,  LDOS, no_states,  v_sample, seed = get_Chern_marker(seed_integer,H0,Nx,Ny,v_max,gap_clean,energy_window_vec)
                        data_row = np.array([Nx, N, seed, energy_window, mu, alpha, gap_clean, v_sample, C_bulk, C_marker_clean])
                        proper_points_on_phase_diagram = np.vstack((proper_points_on_phase_diagram,data_row))
                        string_data_row = "{:2.3f}".format(N) + " " + "{:d}".format(seed) + " " + "{:2.3f}".format(energy_window) + " "  
                        string_data_row = string_data_row + "{:2.3f}".format(mu) + " " + "{:2.3f}".format(alpha)  + " " 
                        string_data_row = string_data_row + "{:2.3f}".format(gap_clean)   + " " + "{:2.3f}".format(v_max) + " "
                        string_data_row = string_data_row + "{:2.3f}".format(C_bulk) + " " + "{:2.3f}".format(C_marker_clean)  
                     
                        print(string_data_row)
        
        proper_points_on_phase_diagram = np.delete(proper_points_on_phase_diagram,0,0)
        
        np.savetxt('proper_points.dat',proper_points_on_phase_diagram)
 
#%%
        
        from numpy.random import default_rng
        
        rng = default_rng()
             
        idx_C_0 = np.where(np.abs(np.rint(proper_points_on_phase_diagram[:,8])) == 0)[0]
        amount_C_0 = idx_C_0.shape[0]

        idx_C_1 = np.where(np.abs(np.rint(proper_points_on_phase_diagram[:,8])) == 1)[0]
        amount_C_1 = idx_C_1.shape[0]

        idx_C_2 = np.where(np.abs(np.rint(proper_points_on_phase_diagram[:,8])) == 2)[0]
        amount_C_2 = idx_C_2.shape[0]

        idx_C_3 = np.where(np.abs(np.rint(proper_points_on_phase_diagram[:,8])) == 3)[0]
        amount_C_3 = idx_C_3.shape[0]                     

        row_C_3 = proper_points_on_phase_diagram[idx_C_3,:]
        row_C_2 = proper_points_on_phase_diagram[idx_C_2,:]
        row_C_1 = proper_points_on_phase_diagram[idx_C_1,:]
        row_C_0 = proper_points_on_phase_diagram[idx_C_0,:]


            
#
        amount_minimal = 99;
        
        amount_minimal = np.min([amount_C_0, amount_C_1, amount_C_2, amount_C_3])

        idx_random_minimal_C_3 = rng.choice(row_C_3.shape[0], size=amount_minimal, replace=False)
        idx_random_minimal_C_2 = rng.choice(row_C_2.shape[0], size=amount_minimal, replace=False)
        idx_random_minimal_C_1 = rng.choice(row_C_1.shape[0], size=amount_minimal, replace=False)
        idx_random_minimal_C_0 = rng.choice(row_C_0.shape[0], size=amount_minimal, replace=False)
#
#        
        proper_points_on_phase_diagram_new = row_C_3[idx_random_minimal_C_3,:]
        proper_points_on_phase_diagram_new = np.vstack((proper_points_on_phase_diagram_new,row_C_2[idx_random_minimal_C_2,:]))
        proper_points_on_phase_diagram_new = np.vstack((proper_points_on_phase_diagram_new,row_C_1[idx_random_minimal_C_1,:]))
        proper_points_on_phase_diagram_new = np.vstack((proper_points_on_phase_diagram_new,row_C_0[idx_random_minimal_C_0,:]))
#        
 
        N_disorder_realizations = 10
        
        #shuffle
        numpy.random.shuffle(proper_points_on_phase_diagram_new)
        
        N_samples = proper_points_on_phase_diagram_new.shape[0]*N_v*N_disorder_realizations
#%%         
        
        
        filename_data = "training_data_unique_seed_LDOS_normalized_to_1_raw_" + class_type  + "_flat." + str(on_off_flat) + "_Nx." + str(int(Nx)) + "_Delta." + "{:2.2f}".format(Delta0) + "_balanced_set.data_raw"         
        file_data = open(filename_data,"a+")        
        index = 0
        N_row = proper_points_on_phase_diagram_new.shape[0]
        seed_integer = 0
        for disorder_realization in range(0,N_disorder_realizations):
            for row_i in range(0,proper_points_on_phase_diagram_new.shape[0]):
     
                data_row = proper_points_on_phase_diagram_new[row_i,:]
                Nx, N, seed, energy_window, mu, alpha, gap_clean, v_sample, C_bulk, C_marker_clean = data_row
                
                Nx = int(Nx)
                H0 = get_H0(Nx,Ny,N,nc,mu,alpha,Delta0,on_off_flat)
            

                for v_max in v_max_vec:
                    seed_integer = seed_integer + 1
                                
                    
                    start_time = time.time()                               
                    C_marker_disorder,  LDOS, no_states,  v_sample, seed = get_Chern_marker(seed_integer,H0,Nx,Ny,v_max,gap_clean,energy_window_vec)
                    string_to_print = "{:2.2f}".format(alpha) + " " + "{:2.2f}".format(mu) + " " + "{:2.2f}".format(C_bulk) + " " + "{:2.2f}".format(C_marker_disorder) + " t = " + "{:2.2f}".format((time.time()-start_time))
                    print(string_to_print)
    
                    for energy_window_i in range(0,energy_window_vec.shape[0]):
                        energy_window = energy_window_vec[energy_window_i]        
                        string_to_file = str(Nx) + " " + str(N)  + " " + str(seed) + " " + "{:2.2f}".format(energy_window) + " "
                        string_to_file = string_to_file + "{:2.3f}".format(mu) + " " + "{:2.3f}".format(alpha) + " " 
                        string_to_file = string_to_file + "{:2.4f}".format(gap_clean) + " " + "{:2.2f}".format(v_sample) + " "
                        string_to_file = string_to_file + "{:2.3f}".format(C_bulk) + " " + "{:2.3f}".format(C_marker_clean) + " " + "{:2.3f}".format(C_marker_disorder) + " " + str(no_states[energy_window_i]) + " "
                    
                    
                        LDOS_to_file = " "
                        for i in range(0,Nx*Ny):
                            LDOS_to_file = LDOS_to_file + " " + str(LDOS[energy_window_i,i])    
                        LDOS_to_file   = LDOS_to_file           
                        
                        file_data.write(string_to_file + " " + LDOS_to_file + " " + "\n")
     
                        string = str(row_i) + " / " + str(N_row)  + " | window = " + str(energy_window) 
                        string = string + " | N = " + str(N) + " | alpha = " + "{:1.3f}".format(alpha)  + " | mu = " + "{:1.3f}".format(mu) +  " | v_max = " + "{:1.2f}".format(v_max) +  " | #  = " + str(disorder_realization)
                        string = string + " | C_b = " + "{:1.0f}".format(C_bulk)  + " | C_0 = " + "{:1.3f}".format(C_marker_clean)  + " | C_m = " + "{:1.3f}".format(C_marker_disorder) 
                        string = string + " | gap = " + "{:2.3f}".format(gap_clean)  +  " | #states = " + str(no_states[energy_window_i])
                        print(string)    
                        index = index + 1
        file_data.close()
