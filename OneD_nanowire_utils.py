import numpy as np
from tqdm import tqdm
from scipy import sparse
from config_1D_nanowire import ModelParams
from pfapack.ctypes import pfaffian as cpf
import scipy.linalg as la
import random
from joblib import Parallel, delayed
from copy import deepcopy

def Gen_disorder(nd, sigma, Length, A_max=3):
    x_n = np.array([random.randint(0,Length) for _ in range(nd)])
    wire = np.arange(0,Length,1)
    Disorder = np.zeros(Length)
    for i in range(nd):
        A_n = random.uniform(-A_max, A_max)
        Disorder += A_n*np.exp(-1.0*((wire-x_n[i])**2.0)/(2.0*(sigma**2.0)))
    Disorder -= np.mean(Disorder)
    Disorder /= np.sqrt(np.mean(Disorder**2.0))
    Disorder -= np.mean(Disorder)
    return Disorder

def onsite_h(i: int, H: np.ndarray, mu: float, Ez: float, params: ModelParams, No_QD=True):
    Length = params.Length
    t = params.t
    V_QD = params.V_QD
    V = params.Disorder
    ##Onsite Chemical and Disorder Potential terms
    if (No_QD or (i != 4*Length)):
        H[i][i] =  (2*t - mu) + V[int(i/4)]
        H[i + 1][i + 1] = (2*t - mu) + V[int(i/4)]
        H[i + 2][i + 2] = -H[i][i]
        H[i + 3][i + 3] = -H[i][i]
        ##Zeeman term
        H[i][i+1] = H[i+1][i] = Ez
        H[i+2][i+3] = H[i+3][i+2] = -Ez
    else:
        H[i][i] =  (0. - mu) + V_QD
        H[i + 1][i + 1] = (0. - mu) + V_QD
        H[i + 2][i + 2] = -H[i][i]
        H[i + 3][i + 3] = -H[i][i]
    return H

def H_delta(i: int, H: np.ndarray, Ez: float, params: ModelParams, No_QD=True):
    Length = params.Length
    delta_parent = params.Delta0
    gamma = params.smsc_coupling
    delta_0 = lambda B: delta_parent - 0.17173*(B**2.5)
    delta = delta_0(Ez)*gamma/(delta_0(Ez)+gamma)
    Z = delta_0(Ez)/(delta_0(Ez)+gamma)
    ##Superconducting parameter
    if (No_QD or (i != 4*Length)):
        H[i][i+3] = H[i+3][i] = delta*(1/Z)
        H[i+1][i+2] = H[i+2][i+1] = -delta*(1/Z)
        if (No_QD and (i == 4*(Length-1))):
            H = Z*H
    if (i == 4 * Length):
        H[0:4*Length,0:4*Length] = Z*H[0:4*Length,0:4*Length]
    return H

def hopping_h(i: int, H: np.ndarray, phi: float, params: ModelParams, No_QD=True, calc_pfaffian=False, end_hopping=False):
    Length = params.Length
    t = params.t
    alpha = params.alpha
    w_left = params.w_left
    w_right = params.w_right
    ##Hopping and Rashba Spin orbit coupling terms
    if No_QD:
        if (i < 4 * (Length - 1)):
            ##Hopping term
            H[i][i + 4] = H[i + 4][i] = -t
            H[i + 1][i + 5] = H[i + 5][i + 1] = -t
            H[i + 2][i + 6] = H[i + 6][i + 2] = t
            H[i + 3][i + 7] = H[i + 7][i + 3] = t
            ##Rashba Spin orbit coupling term
            H[i][i + 5] = H[i + 5][i] = alpha/2.0 
            H[i + 1][i + 4] = H[i + 4][i + 1] = -alpha/2.0
            H[i + 2][i + 7] = H[i + 7][i + 2] = -alpha/2.0
            H[i + 3][i + 6] = H[i + 6][i + 3] = alpha/2.0
        if calc_pfaffian or end_hopping:
            w_avg = 1.0
            if (i == 4*(Length - 1)):
                ##Phased Hopping term from N-1th site to 0th site
                H[i][0] = -w_avg*t*np.exp(-1.0j*phi)
                H[i+1][1] = -w_avg*t*np.exp(-1.0j*phi)
                H[i+2][2] = w_avg*t*np.exp(1.0j*phi)
                H[i+3][3] = w_avg*t*np.exp(1.0j*phi)
                ##Phased Hopping term from 0th site to N-1th site
                H[0][i] = -w_avg*t*np.exp(1.0j*phi)
                H[1][i+1] = -w_avg*t*np.exp(1.0j*phi)
                H[2][i+2] = w_avg*t*np.exp(-1.0j*phi)
                H[3][i+3] = w_avg*t*np.exp(-1.0j*phi)
                #"""
                ##Phased Rashba Spin orbit coupling term from N-1th site to 0th site
                H[i][1] = w_avg*(alpha/2.0)*np.exp(-1.0j*phi)
                H[i+1][0] = -w_avg*(alpha/2.0)*np.exp(-1.0j*phi)
                H[i+2][3] = -w_avg*(alpha/2.0)*np.exp(1.0j*phi)
                H[i+3][2] = w_avg*(alpha/2.0)*np.exp(1.0j*phi)
                ##Phased Rashba Spin orbit coupling term from 0th site to N-1th site
                H[1][i] = w_avg*(alpha/2.0)*np.exp(1.0j*phi)
                H[0][i+1] = -w_avg*(alpha/2.0)*np.exp(1.0j*phi)
                H[3][i+2] = -w_avg*(alpha/2.0)*np.exp(-1.0j*phi)
                H[2][i+3] = w_avg*(alpha/2.0)*np.exp(-1.0j*phi)
                #"""

    else:
        if ((i < 4 * (Length - 1))):
            ##Hopping term
            H[i][i + 4] = H[i + 4][i] = -t
            H[i + 1][i + 5] = H[i + 5][i + 1] = -t
            H[i + 2][i + 6] = H[i + 6][i + 2] = t
            H[i + 3][i + 7] = H[i + 7][i + 3] = t
            ##Rashba Spin orbit coupling term
            H[i][i + 5] = H[i + 5][i] = alpha/2.0 
            H[i + 1][i + 4] = H[i + 4][i + 1] = -alpha/2.0
            H[i + 2][i + 7] = H[i + 7][i + 2] = -alpha/2.0
            H[i + 3][i + 6] = H[i + 6][i + 3] = alpha/2.0

        if (i == 4*(Length - 1)):
            ##Phased Hopping term from N-1th site to Nth site
            H[i][i + 4] = -w_left*t
            H[i + 1][i + 5] = -w_left*t
            H[i + 2][i + 6] = w_left*t
            H[i + 3][i + 7] = w_left*t
            ##Phased Hopping term from Nth site to N-1th site
            H[i + 4][i] = -w_left*t
            H[i + 5][i + 1] = -w_left*t
            H[i + 6][i + 2] = w_left*t
            H[i + 7][i + 3] = w_left*t
            ##Phased Rashba Spin orbit coupling term from N-1th site to Nth site
            H[i][i + 5] = w_left*(alpha/2.0)
            H[i + 1][i + 4] = -w_left*(alpha/2.0)
            H[i + 2][i + 7] = -w_left*(alpha/2.0)
            H[i + 3][i + 6] = w_left*(alpha/2.0)
            ##Phased Rashba Spin orbit coupling term from Nth site to N-1th site
            H[i + 5][i] = w_left*(alpha/2.0)
            H[i + 4][i + 1] = -w_left*(alpha/2.0)
            H[i + 7][i + 2] = -w_left*(alpha/2.0)
            H[i + 6][i + 3] = w_left*(alpha/2.0)
    
        if (i == 4*(Length)):
            ##Phased hopping term from Nth site to 0th site
            H[i][0] = -w_right*t*np.exp(-1.0j*phi)
            H[i + 1][1] = -w_right*t*np.exp(-1.0j*phi)
            H[i + 2][2] = w_right*t*np.exp(1.0j*phi)
            H[i + 3][3] = w_right*t*np.exp(1.0j*phi)
            ##Phased hopping term from 0th site to Nth site
            H[0][i] = -w_right*t*np.exp(1.0j*phi)
            H[1][i + 1] = -w_right*t*np.exp(1.0j*phi)
            H[2][i + 2] = w_right*t*np.exp(-1.0j*phi)
            H[3][i + 3] = w_right*t*np.exp(-1.0j*phi)
            ##Phased Rashba Spin orbit coupling term from Nth site to 0th site
            H[i][1] = w_right*(alpha/2.0)*np.exp(-1.0j*phi)
            H[i + 1][0] = -w_right*(alpha/2.0)*np.exp(-1.0j*phi)
            H[i + 2][3] = -w_right*(alpha/2.0)*np.exp(1.0j*phi)
            H[i + 3][2] = w_right*(alpha/2.0)*np.exp(1.0j*phi)
            ##Phased Rashba Spin orbit coupling term from 0th site to Nth site
            H[1][i] = w_right*(alpha/2.0)*np.exp(1.0j*phi)
            H[0][i + 1] = -w_right*(alpha/2.0)*np.exp(1.0j*phi)
            H[3][i + 2] = -w_right*(alpha/2.0)*np.exp(-1.0j*phi)
            H[2][i + 3] = w_right*(alpha/2.0)*np.exp(-1.0j*phi)
        
    return H

def construct_Hamil(mu: float, Ez: float, phi: float, params: ModelParams, calc_pfaffian=False, No_QD=True, end_hopping=False):
    Length = params.Length
    S = params.chirality_op
    if No_QD:
        H_0 = np.zeros((4* (Length), 4 * (Length)), dtype=complex)
        Hamiltonian = []
        for i in range(0, 4 * (Length), 4):
            H_0 = onsite_h(i, H_0, mu, Ez, params, No_QD)
            H_0 = hopping_h(i, H_0, phi, params, No_QD, calc_pfaffian, end_hopping)
            H_0 = H_delta(i, H_0, Ez, params, No_QD) ## Call superconducting term 
            #last since the renormalization is happening in this function
    if not No_QD:
        H_0 = np.zeros((4* (Length+1), 4 * (Length+1)), dtype=complex)
        Hamiltonian = []
        for i in range(0, 4 * (Length+1), 4):
            H_0 = onsite_h(i, H_0, mu, Ez, params, No_QD)
            H_0 = hopping_h(i, H_0, phi, params, No_QD)
            H_0 = H_delta(i, H_0, Ez, params, No_QD) ## Call superconducting term 
            #last since the renormalization is happening in this function
    Hamiltonian = np.array(H_0)
    if ((calc_pfaffian) and (No_QD)):
        Hamiltonian = np.real(np.matmul(H_0,np.kron(np.eye(Length),S)))
        Hamiltonian = np.ascontiguousarray(Hamiltonian, dtype=np.float64)
    if ((calc_pfaffian) and (not No_QD)):
        Hamiltonian = np.real(np.matmul(H_0,np.kron(np.eye(Length+1),S)))
        Hamiltonian = np.ascontiguousarray(Hamiltonian, dtype=np.float64)
    return Hamiltonian

def Diagonalize(H: np.ndarray, extract_exact = True, num_eigvals=30):
    if extract_exact:
        eneg, eigen_vec = np.linalg.eigh(H)
        ordering = np.argsort(eneg)
        return eneg[ordering], eigen_vec[:,ordering]
    eneg, eigen_vec = sparse.linalg.eigsh(sparse.csr_matrix(H), k=num_eigvals, sigma=0, which='LM')
    ordering = np.argsort(eneg)
    eneg, eigen_vec = eneg[ordering], eigen_vec[:,ordering]
    return eneg, eigen_vec

def Energy_spectrum(mu: float, params: ModelParams, No_QD=True, num_eigvals=30, n_jobs=-1):
    Ez_low = params.Ez_low
    Ez_high = params.Ez_high
    Ez_points = params.Ez_points
    Ez_sweep = np.linspace(Ez_low, Ez_high, Ez_points)
    Energy_eigenvals = []

    def _worker_energy_spectrum(Ez0, mu, params, No_QD, num_eigvals):
        H = construct_Hamil(mu, Ez0, phi=0.0, params=deepcopy(params), No_QD=No_QD)
        eigenvals, _ = Diagonalize(H, extract_exact=False, num_eigvals=num_eigvals)
        return eigenvals
    
    tasks = [delayed(_worker_energy_spectrum)(Ez0, mu, deepcopy(params), No_QD, num_eigvals)
             for Ez0 in tqdm(Ez_sweep, desc="Ez sweep")]
    results = Parallel(n_jobs=n_jobs, backend="loky")(tasks)

    # results is list of eigenvalue arrays per Ez
    Energy_eigenvals = np.array(results)
    return Energy_eigenvals

def wavefunc(mu: float, Ez: float, phi0: float, No_QD: bool, params: ModelParams):
    eneg_vals, eigen_vecs = [], []
    Hamiltonian = construct_Hamil(mu, Ez, phi=phi0, params=params, No_QD=No_QD)
    eneg_vals, eigen_vecs = np.linalg.eigh(Hamiltonian)
    order = np.argsort(np.abs(eneg_vals))
    ordered_eigenvals = eneg_vals[order]
    ordered_eigenfunctions = eigen_vecs[:,order]
    wf1 = np.abs((1.0/2.0**0.5)*(ordered_eigenfunctions[:,0] + np.conj(ordered_eigenfunctions[:,1])))
    wf2 = np.abs((-1.0j/2.0**0.5)*(ordered_eigenfunctions[:,0] - np.conj(ordered_eigenfunctions[:,1])))
    quasi_wf1 = np.abs((1.0/2.0**0.5)*(ordered_eigenfunctions[:,2] + np.conj(ordered_eigenfunctions[:,3])))
    quasi_wf1 = np.abs((-1.0j/2.0**0.5)*(ordered_eigenfunctions[:,2] - np.conj(ordered_eigenfunctions[:,3])))
    wavefunc_length = params.Length
    if not No_QD:
        wavefunc_length = params.Length + 1
    mf1 = wf1[0:4*wavefunc_length:4]**2.0 + wf1[1:4*wavefunc_length:4]**2.0 + wf1[2:4*wavefunc_length:4]**2.0 + wf1[3:4*wavefunc_length:4]**2.0
    mf2 = wf2[0:4*wavefunc_length:4]**2.0 + wf2[1:4*wavefunc_length:4]**2.0 + wf2[2:4*wavefunc_length:4]**2.0 + wf2[3:4*wavefunc_length:4]**2.0
    qmf1 = quasi_wf1[0:4*wavefunc_length:4]**2.0 + quasi_wf1[1:4*wavefunc_length:4]**2.0 + quasi_wf1[2:4*wavefunc_length:4]**2.0 + quasi_wf1[3:4*wavefunc_length:4]**2.0
    qmf2 = quasi_wf1[0:4*wavefunc_length:4]**2.0 + quasi_wf1[1:4*wavefunc_length:4]**2.0 + quasi_wf1[2:4*wavefunc_length:4]**2.0 + quasi_wf1[3:4*wavefunc_length:4]**2.0
    return mf1, mf2, qmf1, qmf2, ordered_eigenvals[0:4] 

def Energy_tracker(energies, wavefunctions, initial_wf):
    ordering = np.argsort(abs(energies))
    eneg_vals, eigen_vec = energies[ordering], wavefunctions[:, ordering]
    wf0 = initial_wf
    overlaps = np.abs(np.matmul(np.conj(wf0).T,eigen_vec))
    idx = np.argmax(overlaps)
    new_E0 = eneg_vals[idx]
    phase = np.vdot(np.conj(wf0).T, eigen_vec[:, idx])
    new_wf0 = -eigen_vec[:, idx] if phase<0.0 else eigen_vec[:, idx]
    return new_E0, new_wf0

def Energy_tracker_phi(mu: float, Ez: float, params: ModelParams, No_QD=False, end_hopping=False, n_jobs=-1):
    phi_sweep = np.linspace(params.phi_low, params.phi_high, params.phi_points)
    lowest_pos_E0, lowest_neg_E0, wf0, wf0_n = [], [], [], []

    def _worker_eigpair(mu, Ez, phi0, params, No_QD, end_hopping):
        H = construct_Hamil(mu, Ez, phi=phi0, params=deepcopy(params), No_QD=No_QD, end_hopping=end_hopping)
        eigvals, eigvecs = Diagonalize(H, extract_exact=False, num_eigvals=5)
        order = np.argsort(np.abs(eigvals))
        return eigvals[order], eigvecs[:, order]
    
    tasks = [delayed(_worker_eigpair)(mu, Ez, phi0, deepcopy(params), No_QD, end_hopping) for phi0 in tqdm(phi_sweep, desc="phi eigs")]
    results = Parallel(n_jobs=n_jobs, backend="loky")(tasks)

    # results[i] = (ordered_eigvals, ordered_eigvecs)
    ordered_vals = [res[0] for res in results]
    ordered_vecs = [res[1] for res in results]

    # sequential tracking using the precomputed ordered_vals/ordered_vecs (same as original loop)
    lowest_pos_E0, lowest_neg_E0 = [], []
    # determine initial wf0/wf0_n from the phi=0 point (results[0])
    eigvals0, eigvec0 = ordered_vals[0], ordered_vecs[0]
    if eigvals0[0] > 0.0:
        wf0 = eigvec0[:, 0]; wf0_n = eigvec0[:, 1]
    else:
        wf0 = eigvec0[:, 1]; wf0_n = eigvec0[:, 0]

    for eneg_vals, eigen_vec in zip(ordered_vals, ordered_vecs):
        overlaps = np.abs(np.matmul(np.conj(wf0).T, eigen_vec))
        overlaps_n = np.abs(np.matmul(np.conj(wf0_n).T, eigen_vec))
        idx = np.argmax(overlaps); idx_n = np.argmax(overlaps_n)
        lowest_pos_E0.append(eneg_vals[idx])
        lowest_neg_E0.append(eneg_vals[idx_n])
        phase = np.vdot(np.conj(wf0).T, eigen_vec[:, idx])
        phase_n = np.vdot(np.conj(wf0_n).T, eigen_vec[:, idx_n])
        wf0 = -eigen_vec[:, idx] if phase < 0.0 else eigen_vec[:, idx]
        wf0_n = -eigen_vec[:, idx_n] if phase_n < 0.0 else eigen_vec[:, idx_n]

    return np.array(lowest_pos_E0), np.array(lowest_neg_E0)

def Energy_tracker_VQD(mu: float, Ez: float, phi0: float, params: ModelParams, No_QD=False, end_hopping=False, n_jobs=-1):
    vqd_sweep = np.linspace(params.V_QD_low, params.V_QD_high, params.V_QD_points)
    lowest_pos_E0, lowest_neg_E0, wf0, wf0_n = [], [], [], []

    def _worker_eigpair_vqd(mu, Ez, phi0, vqd0, params, No_QD, end_hopping):
        p = deepcopy(params)
        p.V_QD = vqd0
        H = construct_Hamil(mu, Ez, phi=phi0, params=p, No_QD=No_QD, end_hopping=end_hopping)
        eigvals, eigvecs = Diagonalize(H, extract_exact=False, num_eigvals=5)
        order = np.argsort(np.abs(eigvals))
        return eigvals[order], eigvecs[:, order]
    
    tasks = [
        delayed(_worker_eigpair_vqd)(mu, Ez, phi0, vqd0, deepcopy(params), No_QD, end_hopping)
        for vqd0 in tqdm(vqd_sweep, desc="precomputing V_QD eigs")
    ]
    results = Parallel(n_jobs=n_jobs, backend="loky")(tasks)

    # 2) collect ordered eigenvals/eigenvecs
    ordered_vals = [res[0] for res in results]
    ordered_vecs = [res[1] for res in results]

    # 3) sequential tracking (same logic as original function)
    lowest_pos_E0, lowest_neg_E0 = [], []

    # initialize wf0 and wf0_n from the first point (vqd_sweep[0])
    eigvals0, eigen_vec0 = ordered_vals[0], ordered_vecs[0]
    if eigvals0[0] > 0.0:
        wf0 = eigen_vec0[:, 0]
        wf0_n = eigen_vec0[:, 1]
    else:
        wf0 = eigen_vec0[:, 1]
        wf0_n = eigen_vec0[:, 0]

    for eneg_vals, eigen_vec in zip(ordered_vals, ordered_vecs):
        overlaps = np.abs(np.matmul(np.conj(wf0).T, eigen_vec))
        overlaps_n = np.abs(np.matmul(np.conj(wf0_n).T, eigen_vec))
        idx = np.argmax(overlaps)
        idx_n = np.argmax(overlaps_n)
        lowest_pos_E0.append(eneg_vals[idx])
        lowest_neg_E0.append(eneg_vals[idx_n])
        phase = np.vdot(np.conj(wf0).T, eigen_vec[:, idx])
        phase_n = np.vdot(np.conj(wf0_n).T, eigen_vec[:, idx_n])
        wf0 = -eigen_vec[:, idx] if phase < 0.0 else eigen_vec[:, idx]
        wf0_n = -eigen_vec[:, idx_n] if phase_n < 0.0 else eigen_vec[:, idx_n]

    return np.array(lowest_pos_E0), np.array(lowest_neg_E0)

def Energy_spectrum_phi(mu: float, Ez: float, params: ModelParams, consider_all_states=False, No_QD=False, end_hopping=False, num_eigvals=30, n_jobs=-1):
    phi_low = params.phi_low
    phi_high = params.phi_high
    phi_points = params.phi_points
    phi_sweep = np.linspace(phi_low, phi_high, phi_points)
    Energy_eigenvals_v_phi = []
    Energy_even_parity, Energy_odd_parity = [], []

    def _worker_eigpair_phi(phi0, mu, Ez, params, extract_exact, No_QD, end_hopping, num_eigvals):
        H = construct_Hamil(mu, Ez, phi0, params=deepcopy(params), No_QD=No_QD, end_hopping=end_hopping)
        eigvals, eigvecs = Diagonalize(H, extract_exact=extract_exact, num_eigvals=num_eigvals)
        return phi0, eigvals, eigvecs
    
    # parallel compute eigenpairs for each phi
    tasks = [delayed(_worker_eigpair_phi)(phi0, mu, Ez, deepcopy(params), consider_all_states, No_QD, end_hopping, num_eigvals)
             for phi0 in phi_sweep]
    results = Parallel(n_jobs=n_jobs, backend="loky")(tasks)

    # sort results back in phi order (in case Parallel shuffles)
    results_sorted = sorted(results, key=lambda x: x[0])
    eigenvals_list = [res[1] for res in results_sorted]
    eigenvecs_list = [res[2] for res in results_sorted]

    # sequential post-processing (parity lists, tracking etc.) using eigenvals_list/eigenvecs_list
    Energy_even_parity, Energy_odd_parity = [], []
    Energy_eigenvals_v_phi = []
    for eigvals, eigvecs in zip(eigenvals_list, eigenvecs_list):
        order = np.argsort(np.abs(eigvals))
        ordered_eigenvals = eigvals[order]
        # same logic as original: build parity lists etc.
        Energy_eigenvals_v_phi.append(eigvals)
        Energy_even_temp, Energy_odd_temp = [], []
        if ordered_eigenvals[0] < 0.:
            Energy_even_temp.append(ordered_eigenvals[0]); Energy_odd_temp.append(ordered_eigenvals[1])
        else:
            Energy_even_temp.append(ordered_eigenvals[1]); Energy_odd_temp.append(ordered_eigenvals[0])
        for ord_energy in ordered_eigenvals[2:]:
            if ord_energy < 0.0:
                Energy_even_temp.append(ord_energy)
                Energy_odd_temp.append(ord_energy)
        Energy_even_parity.append(Energy_even_temp)
        Energy_odd_parity.append(Energy_odd_temp)

    # normalize lengths as in original function
    even_lengths = [len(x) for x in Energy_even_parity]
    odd_lengths = [len(x) for x in Energy_odd_parity]
    min_len = min(min(even_lengths), min(odd_lengths))
    Energy_even_parity = np.array([lst[:min_len] for lst in Energy_even_parity])
    Energy_odd_parity = np.array([lst[:min_len] for lst in Energy_odd_parity])
    Energy_eigenvals_v_phi = np.array(Energy_eigenvals_v_phi)

    return Energy_eigenvals_v_phi, Energy_even_parity, Energy_odd_parity

def Energy_spectrum_v_VQD(mu: float, Ez: float, phi: float, params: ModelParams, consider_all_states=False, num_eigvals=30, n_jobs=-1):
    V_QD_low = params.V_QD_low
    V_QD_high = params.V_QD_high
    V_QD_points = params.V_QD_points
    V_QD_sweep = np.linspace(V_QD_low, V_QD_high, V_QD_points)
    Energy_eigenvals_v_VQD = []
    Energy_even_parity, Energy_odd_parity = [], []

    def _worker_eigpair_vqd(V_QD0, mu, Ez, phi, params, num_eigvals, extract_exact):
        p = deepcopy(params); p.V_QD = V_QD0
        H = construct_Hamil(mu, Ez, phi=phi, params=p, No_QD=False)
        eigvals, eigvecs = Diagonalize(H, extract_exact=extract_exact, num_eigvals=num_eigvals)
        return eigvals, eigvecs
    
    tasks = [delayed(_worker_eigpair_vqd)(vqd, mu, Ez, phi, deepcopy(params), num_eigvals, consider_all_states)
             for vqd in tqdm(V_QD_sweep, desc="V_QD eigs")]
    results = Parallel(n_jobs=n_jobs, backend="loky")(tasks)
    # results -> list of (eigvals_ordered, eigvecs_ordered) in same V_QD order
    eigenvals_list = [res[0] for res in results]
    eigenvecs_list = [res[1] for res in results]
    Energy_eigenvals_v_VQD = np.array(eigenvals_list)
    # now sequentially perform parity extraction / building Energy_even_parity, Energy_odd_parity
    Energy_even_parity, Energy_odd_parity = [], []
    for eigvals in eigenvals_list:
        order = np.argsort(np.abs(eigvals))
        ordered_eigenvals = eigvals[order]
        Energy_even_temp, Energy_odd_temp = [], []
        if ordered_eigenvals[0] < 0.:
            Energy_even_temp.append(ordered_eigenvals[0]); Energy_odd_temp.append(ordered_eigenvals[1])
        else:
            Energy_even_temp.append(ordered_eigenvals[1]); Energy_odd_temp.append(ordered_eigenvals[0])
        for ord_energy in ordered_eigenvals[2:]:
            if ord_energy < 0.0:
                Energy_even_temp.append(ord_energy); Energy_odd_temp.append(ord_energy)
        Energy_even_parity.append(Energy_even_temp); Energy_odd_parity.append(Energy_odd_temp)

    # normalize lengths to min_len as in original function
    even_lengths = [len(x) for x in Energy_even_parity]
    odd_lengths = [len(x) for x in Energy_odd_parity]
    min_len = min(min(even_lengths), min(odd_lengths))
    Energy_even_parity = np.array([lst[:min_len] for lst in Energy_even_parity])
    Energy_odd_parity = np.array([lst[:min_len] for lst in Energy_odd_parity])
    return Energy_eigenvals_v_VQD, Energy_even_parity, Energy_odd_parity

def sign(x):
    signed_x = np.zeros_like(x)
    for x0 in x:
        if x0 > 0:
            signed_x[x == x0] = 1
        elif x0 < 0:
            signed_x[x == x0] = -1
        else:
            signed_x[x == x0] = 0
    return signed_x

def pfaffian_schur(A, overwrite_a=False):
    """Calculate Pfaffian of a real antisymmetric matrix using
    the Schur decomposition. (Hessenberg would in principle be faster,
    but scipy-0.8 messed up the performance for scipy.linalg.hessenberg()).

    This function does not make use of the skew-symmetry of the matrix A,
    but uses a LAPACK routine that is coded in FORTRAN and hence faster
    than python. As a consequence, pfaffian_schur is only slightly slower
    than pfaffian().
    """

    assert np.issubdtype(A.dtype, np.number) and not np.issubdtype(
        A.dtype, np.complexfloating
    )

    assert A.shape[0] == A.shape[1] > 0

    assert abs(A + A.T).max() < 1e-14

    # Quick return if possible
    if A.shape[0] % 2 == 1:
        return 0

    (t, z) = la.schur(A, output="real", overwrite_a=overwrite_a)
    l = np.diag(t, 1)  # noqa: E741
    return np.prod(sign(l[::2])) * la.det(z)

def pfaffian_calculation(mu: float, Ez: float, No_QD: bool, params: ModelParams):
    H_pfaffian_0 = construct_Hamil(mu, Ez, phi=0.0, params=params, calc_pfaffian=True, No_QD=No_QD)
    pfaffian_0 = 1 if np.real(cpf(H_pfaffian_0)) > 1e-9 else -1
    H_pfaffian_pi = construct_Hamil(mu, Ez, phi=np.pi, params=params, calc_pfaffian=True, No_QD=No_QD)
    pfaffian_pi = 1 if np.real(cpf(H_pfaffian_pi)) > 1e-9 else -1
    pfaffian = pfaffian_0/pfaffian_pi
    return pfaffian

def Topological_stability(mu: float, Ez: float, No_QD: bool, params: ModelParams):
    H_pfaffian_0 = construct_Hamil(mu, Ez, phi=0.0, params=params, calc_pfaffian=True, No_QD=No_QD)
    E_vals0, _ = Diagonalize(H_pfaffian_0, extract_exact=False, num_eigvals=5)
    E_vals0 = E_vals0[np.argsort(abs(E_vals0))]
    Eg0 = abs(E_vals0[0]-E_vals0[1])
    H_pfaffian_pi = construct_Hamil(mu, Ez, phi=np.pi, params=params, calc_pfaffian=True, No_QD=No_QD)
    E_valspi, _ = Diagonalize(H_pfaffian_pi, extract_exact=False, num_eigvals=5)
    E_valspi = E_valspi[np.argsort(abs(E_valspi))]
    Egpi = abs(E_valspi[0]-E_valspi[1])
    pfaffian_0 = 1 if np.real(cpf(H_pfaffian_0)) > 1e-9 else -1
    pfaffian_pi = 1 if np.real(cpf(H_pfaffian_pi)) > 1e-9 else -1
    pfaffian = pfaffian_0/pfaffian_pi
    Ts = pfaffian*min(Eg0,Egpi)/max(Eg0,Egpi)
    return Ts

def Energy_GS_curvature_phi(Energies: np.ndarray, params: ModelParams):
    phi_low = params.phi_low
    phi_high = params.phi_high
    phi_points = params.phi_points
    phi_sweep = np.linspace(phi_low, phi_high, phi_points)
    dphi = phi_sweep[1] - phi_sweep[0]
    current, Inductance = np.zeros(phi_points), np.zeros(phi_points)
    for i in range(len(phi_sweep)-1):
        current[i] = np.sum((Energies[i+1]-Energies[i])/(dphi))
        if i>0:
            Inductance[i] = np.sum((Energies[i+1]-2*Energies[i]+Energies[i-1])/(dphi**2))
    current[-1] = np.sum((Energies[-1]-Energies[-2])/(dphi))
    Inductance[0] = np.sum((Energies[2]-2*Energies[1]+Energies[0])/(dphi**2))
    Inductance[-1] = np.sum((Energies[-1]-2*Energies[-2]+Energies[-3])/(dphi**2))
    return current, Inductance

def Energy_GS_curvature_VQD(Energies: np.ndarray, params: ModelParams):
    V_QD_low = params.V_QD_low
    V_QD_high = params.V_QD_high
    V_QD_points = params.V_QD_points
    V_QD_sweep = np.linspace(V_QD_low, V_QD_high, V_QD_points)
    dV_QD = V_QD_sweep[1] - V_QD_sweep[0]
    d2E_dVQD2 = np.zeros(V_QD_points)
    for i in range(len(V_QD_sweep)-1):
        if i>0:
            d2E_dVQD2[i] = np.sum((Energies[i+1]-2*Energies[i]+Energies[i-1])/(dV_QD**2))
    d2E_dVQD2[0] = np.sum((Energies[2]-2*Energies[1]+Energies[0])/(dV_QD**2))
    d2E_dVQD2[-1] = np.sum((Energies[-1]-2*Energies[-2]+Energies[-3])/(dV_QD**2))
    return d2E_dVQD2

            
        