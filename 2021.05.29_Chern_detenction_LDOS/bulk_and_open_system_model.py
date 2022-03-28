#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:16:01 2019

@author: mplodzien
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

##%% Testing procedures for obtaining n-th Fourier component
#def hamiltonian_test(kx):
#    fourier_components = np.array([1.5, 2, 3, 4, 7])    
#    sz = np.array([[1,0],[0,-1]])
#    sx = np.array([[0,1],[1,0]])
#    sy = np.array([[0,-1j],[1j,0]])    
#    h = np.zeros([2,2])
#    s = sx
#    for n in range(1,fourier_components.shape[0]):
#        h = h + fourier_components[n]*s*np.exp(1j*n*kx) + fourier_components[n]*s*np.exp(-1j*n*kx)
#    h = h +  fourier_components[0]*s
#    return h
#
#def hamiltonian_fourier_component_test(kx,n):
#    N_modes = 51
#    h_n_mode = np.zeros([2,2])
#    dk = 2.0*np.pi/N_modes
#    if(n==0):
#        for l in range(0,N_modes):
#            h_n_mode = h_n_mode + hamiltonian_test(dk*l)
#    if(n>0):
#        l_min = int(-(N_modes-1)/2)
#        l_max = int((N_modes-1)/2)
#        for l in range(l_min,l_max+1):
#            k_l = dk*l
#            h_n_mode = h_n_mode + np.exp(-1j*k_l*n)*hamiltonian_test(k_l)
#    return h_n_mode/N_modes          
#    
#
#
#
#kx = np.pi
#h = hamiltonian_test(kx)
#print(h)
#print("==========")
#for n in range(0,10):
#    print(n," Fourier component")
#    h_n = hamiltonian_fourier_component_test(kx,n)
#    print(h_n)
#    print("==========")
#
#
#
#
#def function_test(kx):
#    fourier_components = np.array([1,2,3])
#    val = 0
# 
#    for n in range(0,fourier_components.shape[0]):
##        print(i,fourier_components[i])
#        val = val + fourier_components[n]*np.exp(1j*n*kx)  
#    return val
#
#
#def fourier_component(n,N_components):
# 
# 
#    i_min = -int((N_components-1)/2)
#    i_max = int((N_components-1)/2)  
#
#    h_n = 0
#    for i in range(i_min,i_max + 1):
#        k_i = 2.0*np.pi/N_components*i
#        h_n = h_n +  1./(N_components*1.0)*np.exp(-1j*k_i*n)*function_test(k_i)  
#        
#    return h_n
#
#N = 5
#n = 1
#kx = 2.0*np.pi/N*n
#c = function_test(kx) 
#print(c)
#N_components = 15
#for n in range(0,15):
#    h_n = fourier_component(n,N_components)
#    print(n,h_n.real )
 
#%%
def is_hermitian(H):
    return np.all(H==np.conjugate(np.transpose(H)))

def Hamiltonian_kx_ky(kx,ky,parameters): #system Hamiltonian

###    BHZ model   
#    id = np.array([[1,0],[0,1]])
#    sz = np.array([[1,0],[0,-1]])
#    sx = np.array([[0,1],[1,0]])
#    sy = np.array([[0,-1j],[1j,0]])


#    A = parameters[0]
#    B = parameters[1]
#    M = parameters[2]    
#    m = M - 2*B*(2 - np.cos(kx) - np.cos(ky))
#    H = A*np.sin(kx)*sx + A*np.sin(ky)*sy + m*sz

#   Flat-band model
    N = parameters[0]
    mu = parameters[1]
    alfa = parameters[2]
    flat_band_on_off = parameters[3]
    xi_nn = -2.0*(np.cos(kx) + np.cos(ky)) - mu
    xi_nnn = -2.0*(np.cos(kx+ky) + np.cos(kx-ky)) - mu
    xi_alfa = (1.0-alfa)*xi_nn + alfa*xi_nnn


    delta_x = np.sin(kx)
    delta_y = np.sin(ky)
    delta_radius = np.sqrt(delta_x**2.0 + delta_y**2.0)
    if(flat_band_on_off == 1):
        denominator = np.sqrt(xi_alfa**2.0 + delta_x**2.0 + delta_y**2.0)   # flat band model
    if(flat_band_on_off == 0):
        denominator = 1                                                     # non-flat band model
    
    if(delta_y>=0):
        phi =  np.arccos(delta_x / (delta_x**2.0 + delta_y**2.0 + 10**(-10))**0.5)
    if(delta_y<0):
        phi = -np.arccos(delta_x / (delta_x**2.0 + delta_y**2.0 + 10**(-10))**0.5)    
        
    h11 = xi_alfa/denominator
    h22 = -xi_alfa/denominator
    h12 = delta_radius*np.exp(1j*N*phi)/denominator
    h21 = np.conjugate(h12)

    H = np.array([[h11,h12],[h21,h22]])
    return H         


def H_n_fourier_component_of_ky(kx,n,parameters): #n-th Fourier component of the Hamiltonian H(kx,ky) in y-direction, i.e. H(kx,ky) = sum_{n} H_tilda(kx,n)*Exp[-1j*n*ky] + h.c.
    N_modes = 51 #arbitrary odd integer - rather much greater than 1
    N_band = 2
    h_n_mode = np.zeros([N_band,N_band])
    dk = 2.0*np.pi/N_modes
    if(n==0):
        for l in range(0,N_modes):
            k_l = dk*l
            h_n_mode = h_n_mode + Hamiltonian_kx_ky(kx,k_l,parameters)
    if(n>0):
        l_min = int(-(N_modes-1)/2)
        l_max = int((N_modes-1)/2)
        for l in range(l_min,l_max+1):
            k_l = dk*l
            h_n_mode = h_n_mode + np.exp(-1j*k_l*n)*Hamiltonian_kx_ky(kx,k_l,parameters)
    return h_n_mode/N_modes  

def Hamiltonian_kx_Ny(kx,Ny,hopping_range,parameters): #Hamiltonian in kx with finite (Ny) slices in y-direction
    H_on_diagonal   =   np.zeros([2*Ny,2*Ny])
    H_off_diagonal  =   np.zeros([2*Ny,2*Ny])
    H_on_diagonal   =   np.kron(np.eye(Ny,k=0),H_n_fourier_component_of_ky(kx,0,parameters)) 
    for n in np.arange(1,hopping_range+1): 
        H_off_diagonal = H_off_diagonal + np.kron(np.eye(Ny,k=n),H_n_fourier_component_of_ky(kx,n,parameters))         
    H = H_on_diagonal + H_off_diagonal + np.transpose(np.conjugate(H_off_diagonal))    
    return H    



#%%
def H_m_fourier_component_of_kx(Ny,hopping_range,m,parameters):
    N_modes = 51
    N_band = 2
    h_m_mode = np.zeros([N_band*Ny,N_band*Ny])
    dk = 2.0*np.pi/N_modes
    if(m==0):
        for l in range(0,N_modes):
            k_l = dk*l
            h_m_mode = h_m_mode + Hamiltonian_kx_Ny(k_l,Ny,hopping_range,parameters)
    if(m>0):
        l_min = int(-(N_modes-1)/2)
        l_max = int((N_modes-1)/2)
        for l in range(l_min,l_max+1):
            k_l = dk*l
            h_m_mode = h_m_mode + np.exp(-1j*k_l*m)*Hamiltonian_kx_Ny(k_l,Ny,hopping_range,parameters)
    
    return h_m_mode/N_modes

def Hamiltonian_Nx_Ny(Nx,Ny,hopping_range,parameters): #Hamiltonian in kx with finite (Ny) slices in y-direction
    N_band = 2
    H_size_x = N_band*Ny*Nx
    print("H_size_x = ", H_size_x)
    H_on_diagonal   =   np.zeros([H_size_x, H_size_x])
    H_off_diagonal  =   np.zeros([H_size_x, H_size_x])
    H_on_diagonal   =   np.kron(np.eye(Nx,k=0),H_m_fourier_component_of_kx(Ny,hopping_range,0,parameters)) 
    for n in np.arange(1,hopping_range+1): 
        H_off_diagonal = H_off_diagonal + np.kron(np.eye(Nx,k=n),H_m_fourier_component_of_kx(Ny,hopping_range,n,parameters))         
    H = H_on_diagonal + H_off_diagonal + np.transpose(np.conjugate(H_off_diagonal))    
    return H    

#%% full open Hamiltonian_Nx_Ny
    
flat_band_on_off = 0
N_band = 2
N = 1
mu =  -2
#mu = 3            
alfa = 0.6

Nx = 24
Ny = Nx
X,Y = np.meshgrid(np.arange(0,Nx,1),np.arange(0,Ny,1))
hopping_range = 4
parameters  = np.array([N,mu,alfa,flat_band_on_off])
parameters_str = "flat_band_on_off." + str(flat_band_on_off) + "_N." + "{:02d}".format(N) + "_mu." + "{:2.2f}".format(mu) + "_alfa." + "{:2.2f}".format(alfa)



#%%
##  BHZ model
##  system parameters
#N_band = 2
#A = 1
#B = 2
#M = 1
#parameters = np.array([A,B,M])
#parameters_str = "_A." + "{:2.2f}".format(A) + "_B." + "{:2.2f}".format(B) + "_M." + "{:2.2f}".format(M)




    

 
#%% Bulk Hamiltonian H_kx_ky
dk = 0.1
kx_vec = np.arange(-np.pi,np.pi,dk)
N_kx = kx_vec.shape[0]
filename = "H_kx_ky_dispersion_relation" + parameters_str 
file = open(filename  + ".dat","w")
N_band = 2
E_kx_ky_mat = np.zeros([N_kx,2])
for kx_i in range(0,N_kx):
    kx = kx_vec[kx_i]
    ky = kx
    H = Hamiltonian_kx_ky(kx,ky,parameters)
    E_vec,P_mat =  LA.eigh(H)
    E_kx_ky_mat[kx_i,:] = E_vec
    for E_i in range(0,E_vec.shape[0]):
        string = str(kx) + " " + str(E_i) + " " + str(E_vec[E_i]) + "\n"
        file.write(string)
    file.write("\n")  
file.close()
 

        
#%% half open hamiltonian: H_kx_Ny


 
N_kx = kx_vec.shape[0]
filename = "H_kx_Ny_dispersion_relation" + parameters_str 
file = open(filename + ".dat","w")
E_kx_Ny_mat = np.zeros([N_kx,N_band*Ny])
for kx_i in range(0,N_kx):
    kx = kx_vec[kx_i]
    print("kx = ",kx)
    H = Hamiltonian_kx_Ny(kx,Ny,hopping_range,parameters)
    E_vec,P_mat =  LA.eigh(H)
    E_kx_Ny_mat[kx_i,:] = E_vec
    for E_i in range(0,Ny):
        string = str(kx) + " " + str(E_i) + " " + str(E_vec[E_i]) + "\n"
        file.write(string)
    file.write("\n")
file.close()        


#%%Full open hamiltonain: H_Nx_Ny
H_Nx_Ny = Hamiltonian_Nx_Ny(Nx,Ny,hopping_range,parameters)
eigenvalues_Nx_Ny, P = LA.eigh(H_Nx_Ny)


#%%
DimH = H_Nx_Ny.shape[0]
fig, ax = plt.subplots(1,3)
for y_i in range(0,N_band):
    ax[0].plot(kx_vec,E_kx_ky_mat[:,y_i])  
ax[0].set_xlabel("momentum $k_x = k_y$")
ax[0].set_ylabel("Energy")
ax[0].set_yticks(np.arange(-3,4,2))
ax[0].text(-0.2,-0.2,"$\delta_0$")
ax[0].arrow(0.8,-1.1,0,2.1)

for y_i in range(0,N_band*Ny):
    ax[1].plot(kx_vec,E_kx_Ny_mat[:,y_i])    
ax[1].set_xlabel("momentum $k_x$") 


ax[2].plot(eigenvalues_Nx_Ny,'o')
#ax[2].add_patch(circle)
ax[2].set_xlabel("eigenstate index i")

ax[1].set_ylim([-1.5,1.5])   
#ax[2].set_ylim([-1.5,1.5])
ax[2].set_ylim([-1.5,1.5])
ax[1].set_yticks(np.arange(-1.5,1.6,.5))
ax[2].set_yticks(np.arange(-1.5,1.6,.5))

left = int(DimH/2*(1-0.1))
right = int(DimH/2*(1+0.1))
ax[2].set_xlim([left, right])
ax[2].set_xticks(np.arange(left,right,30))

ax[0].set_title("Closed \n geometry")
ax[1].set_title("Half-opened \n geometry")
ax[2].set_title("Fully-opened \n geometry")

ax[0].set_aspect(1)
ax[1].set_aspect(2.)
ax[2].set_aspect(35)
fig.subplots_adjust(wspace=0.4)
plt.show()



#%%
fig = plt.figure()
 

index_left = 548
E_left = eigenvalues_Nx_Ny[index_left]
psi_upper_band_left = P[np.arange(0,2*Nx*Ny-1,2), index_left];
psi_lower_band_left = P[np.arange(1,2*Nx*Ny,2)  , index_left];
rho_left = np.abs(psi_upper_band_left[:])**2.0 + np.abs(psi_lower_band_left[:])**2.0
#rho_left = np.abs(psi_lower_band_left)**2.0
rho_left = np.reshape(rho_left,[Nx,Ny])
E_given_left = eigenvalues_Nx_Ny[index_left]
#
index_centre = int(578)
E_centre = eigenvalues_Nx_Ny[index_centre]
psi_upper_band_centre = P[np.arange(0,2*Nx*Ny-1,2), index_centre];
psi_lower_band_centre = P[np.arange(1,2*Nx*Ny,2)  , index_centre];
rho_centre = np.abs(psi_upper_band_centre[:])**2.0 + np.abs(psi_lower_band_centre[:])**2.0
#rho_centre = np.abs(psi_lower_band_centre[:])**2.0
rho_centre = np.reshape(rho_centre,[Nx,Ny])
E_given_centre = eigenvalues_Nx_Ny[index_centre]
#
#index_right = -100
#E_right = eigenvalues_Nx_Ny[index_right]
#psi_upper_band_right = P[np.arange(0,2*Nx*Ny-1,2), index_right];
#psi_lower_band_right = P[np.arange(1,2*Nx*Ny,2)  , index_right];
#rho_right = np.abs(psi_upper_band_right[:])**2.0 + np.abs(psi_lower_band_right[:])**2.0
#rho_right= np.reshape(rho_right,[Nx,Ny])
#E_given_right = eigenvalues_Nx_Ny[index_right]
#
#plt.show()

ax_left = fig.add_subplot(111, projection='3d')
surf = ax_left.plot_surface(X, Y, rho_left.T, cmap=cm.coolwarm,linewidth=2, antialiased=False)
ax_left.set_xlabel('x')
ax_left.set_ylabel('y')
ax_left.title.set_text('$E_{%d}$ = %2.2f ' % (index_left, E_left))
#ax_centre.title.set_text('$E_{%d}$ = %2.2f ' % (index_centre, E_centre))
#ax_left.set_zlabel('$|\psi(x,y)|^2$')
#ax_left.set_zlabel('probability density')
plt.show()
#ax_left.set_zlabel('probability density')

 
fig = plt.figure()
ax_centre = fig.add_subplot(111, projection='3d')
surf = ax_centre.plot_surface(X, Y, rho_centre.T, cmap=cm.coolwarm,linewidth=0, antialiased=False)
ax_centre.set_xlabel('x')
ax_centre.set_ylabel('y')
#ax_centre.set_zlabel('probability density')

#ax_right = fig.add_subplot(236, projection='3d')
#surf = ax_right.plot_surface(X, Y, rho_right.T, cmap=cm.coolwarm,linewidth=0, antialiased=False)
#ax_right.set_xlabel('x')
#ax_right.set_ylabel('y')
#ax_right.set_zlabel('probability density')

#ax1 = fig.add_subplot(231)
#ax2 = fig.add_subplot(232)
#ax3 = fig.add_subplot(233)
#ax4 = fig.add_subplot(234)
#ax1.title.set_text('Closed system spectrum')
#ax2.title.set_text('Half-open system in y direction')
#ax3.title.set_text('Full-open system in x and y direction')
#ax_left.title.set_text('$E_{%d}$ = %2.2f ' % (index_left, E_left))
ax_centre.title.set_text('$E_{%d}$ = %2.2f ' % (index_centre, E_centre))
#ax_right.title.set_text('State with energy E = %2.2f ' % E_right)

plt.show()
