#!/usr/bin/env python3
"""
Created on Wed Jun 19 13:16:01 2019

@author: mplodzien
@editor: kcybinski

Edited on Mon Mar 28.03.2022
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from numpy import linalg as LA

#%%
def is_hermitian(H):
    return np.all(H == np.conjugate(np.transpose(H)))


def Hamiltonian_kx_ky(kx, ky, parameters):  # system Hamiltonian

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
    xi_nn = -2.0 * (np.cos(kx) + np.cos(ky)) - mu
    xi_nnn = -2.0 * (np.cos(kx + ky) + np.cos(kx - ky)) - mu
    xi_alfa = (1.0 - alfa) * xi_nn + alfa * xi_nnn

    delta_x = np.sin(kx)
    delta_y = np.sin(ky)
    delta_radius = np.sqrt(delta_x**2.0 + delta_y**2.0)
    if flat_band_on_off == 1:
        denominator = np.sqrt(
            xi_alfa**2.0 + delta_x**2.0 + delta_y**2.0
        )  # flat band model
    if flat_band_on_off == 0:
        denominator = 1  # non-flat band model

    if delta_y >= 0:
        phi = np.arccos(
            delta_x / (delta_x**2.0 + delta_y**2.0 + 10 ** (-10)) ** 0.5
        )
    if delta_y < 0:
        phi = -np.arccos(
            delta_x / (delta_x**2.0 + delta_y**2.0 + 10 ** (-10)) ** 0.5
        )

    h11 = xi_alfa / denominator
    h22 = -xi_alfa / denominator
    h12 = delta_radius * np.exp(1j * N * phi) / denominator
    h21 = np.conjugate(h12)

    H = np.array([[h11, h12], [h21, h22]])
    return H


def H_n_fourier_component_of_ky(
    kx, n, parameters
):  # n-th Fourier component of the Hamiltonian H(kx,ky) in y-direction, i.e. H(kx,ky) = sum_{n} H_tilda(kx,n)*Exp[-1j*n*ky] + h.c.
    N_modes = 51  # arbitrary odd integer - rather much greater than 1
    N_band = 2
    h_n_mode = np.zeros([N_band, N_band])
    dk = 2.0 * np.pi / N_modes
    if n == 0:
        for l in range(0, N_modes):
            k_l = dk * l
            h_n_mode = h_n_mode + Hamiltonian_kx_ky(kx, k_l, parameters)
    if n > 0:
        l_min = int(-(N_modes - 1) / 2)
        l_max = int((N_modes - 1) / 2)
        for l in range(l_min, l_max + 1):
            k_l = dk * l
            h_n_mode = h_n_mode + np.exp(-1j * k_l * n) * Hamiltonian_kx_ky(
                kx, k_l, parameters
            )
    return h_n_mode / N_modes


def Hamiltonian_kx_Ny(
    kx, Ny, hopping_range, parameters
):  # Hamiltonian in kx with finite (Ny) slices in y-direction
    H_on_diagonal = np.zeros([2 * Ny, 2 * Ny])
    H_off_diagonal = np.zeros([2 * Ny, 2 * Ny])
    H_on_diagonal = np.kron(
        np.eye(Ny, k=0), H_n_fourier_component_of_ky(kx, 0, parameters)
    )
    for n in np.arange(1, hopping_range + 1):
        H_off_diagonal = H_off_diagonal + np.kron(
            np.eye(Ny, k=n), H_n_fourier_component_of_ky(kx, n, parameters)
        )
    H = H_on_diagonal + H_off_diagonal + np.transpose(np.conjugate(H_off_diagonal))
    return H


#%%
def H_m_fourier_component_of_kx(Ny, hopping_range, m, parameters):
    N_modes = 51
    N_band = 2
    h_m_mode = np.zeros([N_band * Ny, N_band * Ny])
    dk = 2.0 * np.pi / N_modes
    if m == 0:
        for l in range(0, N_modes):
            k_l = dk * l
            h_m_mode = h_m_mode + Hamiltonian_kx_Ny(k_l, Ny, hopping_range, parameters)
    if m > 0:
        l_min = int(-(N_modes - 1) / 2)
        l_max = int((N_modes - 1) / 2)
        for l in range(l_min, l_max + 1):
            k_l = dk * l
            h_m_mode = h_m_mode + np.exp(-1j * k_l * m) * Hamiltonian_kx_Ny(
                k_l, Ny, hopping_range, parameters
            )

    return h_m_mode / N_modes


def Hamiltonian_Nx_Ny(
    Nx, Ny, hopping_range, parameters
):  # Hamiltonian in kx with finite (Ny) slices in y-direction
    N_band = 2
    H_size_x = N_band * Ny * Nx
    # print("H_size_x = ", H_size_x)
    H_on_diagonal = np.zeros([H_size_x, H_size_x])
    H_off_diagonal = np.zeros([H_size_x, H_size_x])
    H_on_diagonal = np.kron(
        np.eye(Nx, k=0), H_m_fourier_component_of_kx(Ny, hopping_range, 0, parameters)
    )
    for n in np.arange(1, hopping_range + 1):
        H_off_diagonal = H_off_diagonal + np.kron(
            np.eye(Nx, k=n),
            H_m_fourier_component_of_kx(Ny, hopping_range, n, parameters),
        )
    H = H_on_diagonal + H_off_diagonal + np.transpose(np.conjugate(H_off_diagonal))
    return H


# Here end the auxiliary functions

# -------------------------------------------------------------------------------------
#%% full open Hamiltonian_Nx_Ny


def get_full_open_Hamiltonian_Nx_Ny(N, flat_band_on_off, mu, alfa):
    parameters_str = (
        "flat_band_on_off."
        + str(flat_band_on_off)
        + "_N."
        + f"{N:02d}"
        + "_mu."
        + f"{mu:2.2f}"
        + "_alfa."
        + f"{alfa:2.2f}"
    )
    return parameters_str


#%% Bulk Hamiltonian H_kx_ky


def get_bulk_H_kx_ky(parameters, parameters_str, dk, kx_vec):
    N_kx = kx_vec.shape[0]
    E_kx_ky_mat = np.zeros([N_kx, 2])
    filename = "H_kx_ky_dispersion_relation" + parameters_str
    file = open(filename + ".dat", "w")
    for kx_i in range(0, N_kx):
        kx = kx_vec[kx_i]
        ky = kx
        H = Hamiltonian_kx_ky(kx, ky, parameters)
        E_vec, P_mat = LA.eigh(H)
        E_kx_ky_mat[kx_i, :] = E_vec
        for E_i in range(0, E_vec.shape[0]):
            string = str(kx) + " " + str(E_i) + " " + str(E_vec[E_i]) + "\n"
            file.write(string)
        file.write("\n")
    file.close()
    return E_kx_ky_mat


#%% half open hamiltonian: H_kx_Ny


def get_half_open_H_kx_Ny(
    kx_vec, parameters_str, N_band, Ny, hopping_range, parameters
):
    N_kx = kx_vec.shape[0]
    filename = "H_kx_Ny_dispersion_relation" + parameters_str
    file = open(filename + ".dat", "w")
    E_kx_Ny_mat = np.zeros([N_kx, N_band * Ny])
    for kx_i in range(0, N_kx):
        kx = kx_vec[kx_i]
        # print("kx = ", kx)
        H = Hamiltonian_kx_Ny(kx, Ny, hopping_range, parameters)
        E_vec, P_mat = LA.eigh(H)
        E_kx_Ny_mat[kx_i, :] = E_vec
        for E_i in range(0, Ny):
            string = str(kx) + " " + str(E_i) + " " + str(E_vec[E_i]) + "\n"
            file.write(string)
        file.write("\n")
    file.close()
    return E_kx_Ny_mat


#%%Full open hamiltonain: H_Nx_Ny
def get_full_open_H_Nx_Ny(Nx, Ny, hopping_range, parameters):
    H_Nx_Ny = Hamiltonian_Nx_Ny(Nx, Ny, hopping_range, parameters)
    eigenvalues_Nx_Ny, P = LA.eigh(H_Nx_Ny)
    return H_Nx_Ny, eigenvalues_Nx_Ny, P


#%%
def get_plots_geometries(
    Nx,
    Ny,
    hopping_range,
    N,
    flat_band_on_off,
    mu,
    alfa,
    line_len,
    parameters,
    dk,
    N_band,
):
    kx_vec = np.arange(-np.pi, np.pi, dk)
    parameters_str = get_full_open_Hamiltonian_Nx_Ny(N, flat_band_on_off, mu, alfa)
    E_kx_ky_mat = get_bulk_H_kx_ky(parameters, parameters_str, dk, kx_vec)
    H_Nx_Ny, eigenvalues_Nx_Ny, P = get_full_open_H_Nx_Ny(
        Nx, Ny, hopping_range, parameters
    )
    E_kx_Ny_mat = get_half_open_H_kx_Ny(
        kx_vec, parameters_str, N_band, Ny, hopping_range, parameters
    )
    DimH = H_Nx_Ny.shape[0]
    fig, ax = plt.subplots(1, 3, figsize=(9, 2))
    for y_i in range(0, N_band):
        ax[0].plot(kx_vec, E_kx_ky_mat[:, y_i])
    ax[0].set_xlabel("momentum $k_x = k_y$")
    ax[0].set_ylabel("Energy")
    ax[0].set_yticks(np.arange(-3, 4, 2))
    ax[0].text(-0.2, -0.2, r"$\delta_0$")
    ax[0].arrow(line_len[0], line_len[1], line_len[2], line_len[3])

    for y_i in range(0, N_band * Ny):
        ax[1].plot(kx_vec, E_kx_Ny_mat[:, y_i])
    ax[1].set_xlabel("momentum $k_x$")

    ax[2].plot(eigenvalues_Nx_Ny, "o")
    # ax[2].add_patch(circle)
    ax[2].set_xlabel("eigenstate index i")

    ax[1].set_ylim([-1.5, 1.5])
    # ax[2].set_ylim([-1.5,1.5])
    ax[2].set_ylim([-1.5, 1.5])
    ax[1].set_yticks(np.arange(-1.5, 1.6, 0.5))
    ax[2].set_yticks(np.arange(-1.5, 1.6, 0.5))

    left = int(DimH / 2 * (1 - 0.1))
    right = int(DimH / 2 * (1 + 0.1))
    ax[2].set_xlim([left, right])
    ax[2].set_xticks(np.arange(left, right, 30))

    ax[0].set_title("Closed \n geometry")
    ax[1].set_title("Half-opened \n geometry")
    ax[2].set_title("Fully-opened \n geometry")

    fig.subplots_adjust(wspace=0.4)
    plt.show()


#%%


def get_plots_3d(Nx, Ny, hopping_range, parameters, X, Y):
    H_Nx_Ny, eigenvalues_Nx_Ny, P = get_full_open_H_Nx_Ny(
        Nx, Ny, hopping_range, parameters
    )
    fig = plt.figure()
    index_left = 548
    E_left = eigenvalues_Nx_Ny[index_left]
    psi_upper_band_left = P[np.arange(0, 2 * Nx * Ny - 1, 2), index_left]
    psi_lower_band_left = P[np.arange(1, 2 * Nx * Ny, 2), index_left]
    rho_left = (
        np.abs(psi_upper_band_left[:]) ** 2.0 + np.abs(psi_lower_band_left[:]) ** 2.0
    )
    # rho_left = np.abs(psi_lower_band_left)**2.0
    rho_left = np.reshape(rho_left, [Nx, Ny])
    E_given_left = eigenvalues_Nx_Ny[index_left]
    #
    index_centre = int(578)
    E_centre = eigenvalues_Nx_Ny[index_centre]
    psi_upper_band_centre = P[np.arange(0, 2 * Nx * Ny - 1, 2), index_centre]
    psi_lower_band_centre = P[np.arange(1, 2 * Nx * Ny, 2), index_centre]
    rho_centre = (
        np.abs(psi_upper_band_centre[:]) ** 2.0
        + np.abs(psi_lower_band_centre[:]) ** 2.0
    )
    # rho_centre = np.abs(psi_lower_band_centre[:])**2.0
    rho_centre = np.reshape(rho_centre, [Nx, Ny])
    E_given_centre = eigenvalues_Nx_Ny[index_centre]

    ax_left = fig.add_subplot(111, projection="3d")
    surf = ax_left.plot_surface(
        X, Y, rho_left.T, cmap=cm.coolwarm, linewidth=2, antialiased=False
    )
    ax_left.set_xlabel("x")
    ax_left.set_ylabel("y")
    ax_left.title.set_text("$E_{%d}$ = %2.2f " % (index_left, E_left))
    plt.show()

    fig = plt.figure()
    ax_centre = fig.add_subplot(111, projection="3d")
    surf = ax_centre.plot_surface(
        X, Y, rho_centre.T, cmap=cm.coolwarm, linewidth=0, antialiased=False
    )
    ax_centre.set_xlabel("x")
    ax_centre.set_ylabel("y")
    ax_centre.title.set_text("$E_{%d}$ = %2.2f " % (index_centre, E_centre))
    plt.show()


if __name__ == "__main__":
    # Constants
    flat_band_on_off = 0
    N_band = 2
    N = 1
    alfa = 0.6
    Nx = 24
    Ny = Nx
    # mu and line_len can be modified to get figures from the tutorial
    mu = -2
    line_len = [0.8, -1.1, 0, 2.1]

    X, Y = np.meshgrid(np.arange(0, Nx, 1), np.arange(0, Ny, 1))
    hopping_range = 4
    parameters = np.array([N, mu, alfa, flat_band_on_off])
    dk = 0.1
    kx_vec = np.arange(-np.pi, np.pi, dk)

    #     File run procedure
    get_plots_3d(Nx, Ny, hopping_range, parameters, X, Y)
    get_plots_geometries(Nx, Ny, hopping_range, N, flat_band_on_off, mu, alfa, line_len)
