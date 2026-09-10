import numpy as np
from tqdm import tqdm
from config_1D_nanowire import ModelParams
from OneD_nanowire_utils import *
from joblib import Parallel, delayed
from copy import deepcopy

sx = np.array([[0, 1],[1, 0]])
sy = np.array([[0, -1j],[1j, 0]])
sz = np.array([[1, 0],[0, -1]])
s0 = np.array([[1, 0],[0, 1]])

def capacitance(energies: np.ndarray, eigenfunctions: np.ndarray, params: ModelParams):
    
    #Calculate the capacitance based on the given energies and eigenfunctions.
    #energies: array of energy levels in the form of [E1, E2, E3, ...]
    #eigenfunctions: corresponding eigenfunctions in the form of a 2D array where each column corresponds to the particular energy level 
    
    Length = params.Length
    w = params.w
    eta = params.eta
    kappa = 0.1602 #converts to femtofarads
    w0 = w + (1.0j*eta)
    Eneg_pos, Eneg_neg = [],[]
    qd_wfs_pos, qd_wfs_neg = [],[]
    cap_obs_op = np.kron(sz,s0)
    order = np.argsort(np.abs(energies))
    ordered_energies = energies[order]
    ordered_eigenfunctions = eigenfunctions[:,order]
    for i in range(len(ordered_energies)):
        # separate sorted positive and negative energy states and their corresponding eigenfunctions at the QD site
        if ordered_energies[i]>0.0:
            Eneg_pos.append(ordered_energies[i])
            qd_wfs_pos.append(ordered_eigenfunctions[4*Length:4*Length+4,i])
        else:
            Eneg_neg.append(ordered_energies[i])
            qd_wfs_neg.append(ordered_eigenfunctions[4*Length:4*Length+4,i])

    Eneg_pos = np.array(Eneg_pos)
    Eneg_neg = np.array(Eneg_neg)
    qd_wfs_pos = np.array(qd_wfs_pos)
    qd_wfs_neg = np.array(qd_wfs_neg)

    min_len = min(len(Eneg_pos), len(Eneg_neg))
    Eneg_pos = Eneg_pos[:min_len]
    Eneg_neg = Eneg_neg[:min_len]
    qd_wfs_pos = qd_wfs_pos[:,:min_len]
    qd_wfs_neg = qd_wfs_neg[:,:min_len]

    C_const = np.matmul(np.conj(qd_wfs_pos[1:,:]), np.matmul(cap_obs_op, qd_wfs_neg[1:,:].T))
    C_even = np.matmul(np.conj(qd_wfs_pos[0,:]), np.matmul(cap_obs_op, qd_wfs_neg[1:,:].T))
    C_odd = np.matmul(np.conj(qd_wfs_neg[0,:]), np.matmul(cap_obs_op, qd_wfs_neg[1:,:].T))
    C_even_1 = (np.abs(np.matmul(np.conj(qd_wfs_pos[0,:]), np.matmul(cap_obs_op, qd_wfs_neg[0,:].T)))**2.0)*(Eneg_pos[0]-Eneg_neg[0])/((Eneg_pos[0]-Eneg_neg[0])**2.0 - w0**2.0)
    C_odd_1 = (np.abs(np.matmul(np.conj(qd_wfs_neg[0,:]), np.matmul(cap_obs_op, qd_wfs_pos[0,:].T)))**2.0)*(Eneg_neg[0]-Eneg_pos[0])/((Eneg_neg[0]-Eneg_pos[0])**2.0 - w0**2.0)
    C_even_2 = np.matmul(np.conj(qd_wfs_pos[1:,:]), np.matmul(cap_obs_op, qd_wfs_neg[0,:].T))
    C_odd_2 = np.matmul(np.conj(qd_wfs_pos[1:,:]), np.matmul(cap_obs_op, qd_wfs_pos[0,:].T))
    for i in range(len(Eneg_pos)-1):
        for j in range(len(Eneg_pos)-1):
            C_const[i][j] = (np.abs(C_const[i][j])**2.0)*(Eneg_pos[i+1]-Eneg_neg[j+1])/((Eneg_pos[i+1]-Eneg_neg[j+1])**2.0 - w0**2.0)
        C_even[i] = (np.abs(C_even[i])**2.0)*(Eneg_pos[0]-Eneg_neg[i+1])/((Eneg_pos[0]-Eneg_neg[i+1])**2.0 - w0**2.0)
        C_odd[i] = (np.abs(C_odd[i])**2.0)*(Eneg_neg[0]-Eneg_neg[i+1])/((Eneg_neg[0]-Eneg_neg[i+1])**2.0 - w0**2.0)
        C_even_2[i] = (np.abs(C_even_2[i])**2.0)*(Eneg_pos[i+1]-Eneg_neg[0])/((Eneg_pos[i+1]-Eneg_neg[0])**2.0 - w0**2.0)
        C_odd_2[i] = (np.abs(C_odd_2[i])**2.0)*(Eneg_pos[i+1]-Eneg_pos[0])/((Eneg_pos[i+1]-Eneg_pos[0])**2.0 - w0**2.0)
    C_value_const = np.sum(C_const)
    C_value_even, C_value_odd = np.sum(C_even), np.sum(C_odd)
    C_value_even += C_even_1 + np.sum(C_even_2)
    C_value_odd += C_odd_1 + np.sum(C_odd_2)
    return np.real(kappa*(C_value_const+C_value_even)), np.real(kappa*(C_value_const+C_value_odd))

def Capacitance_vs_phi(
    mu: float,
    Ez: float,
    params: ModelParams,
    consider_all_states=False,
    number_of_states=30,
    n_jobs=6,
    n_return=2,
    return_parity_energies=False
):

    phi_low = params.phi_low
    phi_high = params.phi_high
    phi_points = params.phi_points

    phi_sweep = np.linspace(
        phi_low,
        phi_high,
        phi_points
    )

    def _worker_cap_vs_phi(
        phi0,
        mu,
        Ez,
        params,
        consider_all_states,
        number_of_states,
        n_return
    ):

        p = deepcopy(params)

        H_ang = construct_Hamil(
            mu,
            Ez,
            phi0,
            params=p,
            calc_pfaffian=False,
            No_QD=False
        )

        eneg, wfs = Diagonalize(
            H_ang,
            extract_exact=consider_all_states,
            num_eigvals=number_of_states
        )

        order = np.argsort(np.abs(eneg))

        eneg_ord = eneg[order]
        wfs_ord = wfs[:, order]

        Ce_raw, Co_raw = capacitance(
            eneg_ord,
            wfs_ord,
            params=p
        )

        ncols = min(
            n_return,
            wfs_ord.shape[1]
        )
        negative_energies = eneg_ord[
            eneg_ord <= 0.0
        ]

        positive_energies = eneg_ord[
            eneg_ord > 0.0
        ]

        E_even_raw = negative_energies[0]
        E_odd_raw = positive_energies[0]

        return (
            eneg_ord[:ncols],
            wfs_ord[:, :ncols],
            Ce_raw,
            Co_raw,
            E_even_raw,
            E_odd_raw
        )

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky"
    )(
        delayed(_worker_cap_vs_phi)(
            phi0,
            mu,
            Ez,
            deepcopy(params),
            consider_all_states,
            number_of_states,
            n_return
        )
        for phi0 in tqdm(
            phi_sweep,
            desc="capacitance phi sweep"
        )
    )

    Ce_ar = []
    Co_ar = []

    Ee_ar = []
    Eo_ar = []

    E0_neg = 0.0
    current_parity = "even"
    wf0 = None

    for idx, result in enumerate(results):

        (
            eneg_small,
            wfs_small,
            Ce_raw,
            Co_raw,
            Ee_raw,
            Eo_raw
        ) = result

        if idx == 0:

            ordering0 = np.argsort(
                np.abs(eneg_small)
            )

            eneg_ord0 = eneg_small[ordering0]

            if (
                eneg_ord0.shape[0] > 1
                and eneg_ord0[0] > 0.0
            ):
                wf0 = wfs_small[:, 1]
                E0_neg = eneg_ord0[1]

            else:
                wf0 = wfs_small[:, 0]
                E0_neg = eneg_ord0[0]

        else:

            Ep_neg, wfp = Energy_tracker(
                eneg_small,
                wfs_small,
                wf0
            )

            if (E0_neg * Ep_neg) < 0.0:
                current_parity = (
                    "odd"
                    if current_parity == "even"
                    else "even"
                )

            E0_neg = Ep_neg
            wf0 = wfp

        # Apply the same fixed-parity assignment to both
        # the capacitance and the corresponding energy.
        if current_parity == "odd":

            Ce = Co_raw
            Co = Ce_raw

            Ee = Eo_raw
            Eo = Ee_raw

        else:

            Ce = Ce_raw
            Co = Co_raw

            Ee = Ee_raw
            Eo = Eo_raw

        Ce_ar.append(Ce)
        Co_ar.append(Co)

        Ee_ar.append(Ee)
        Eo_ar.append(Eo)

    C_even = np.array(Ce_ar)
    C_odd = np.array(Co_ar)

    if not return_parity_energies:
        return C_even, C_odd

    E_even = np.array(Ee_ar)
    E_odd = np.array(Eo_ar)

    return C_even, C_odd, E_even, E_odd

def Capacitance_vs_VQD(mu: float, Ez: float, params: ModelParams, consider_all_states=False, number_of_states=30, n_jobs=6, n_return=2):
    V_QD_low = params.V_QD_low
    V_QD_high = params.V_QD_high
    V_QD_points = params.V_QD_points
    temp_store_initial_V_QD_val = params.V_QD
    V_QD_sweep = np.linspace(V_QD_low, V_QD_high, V_QD_points)

    def _worker_cap_vqd(vqd0, mu, Ez, params, consider_all_states, number_of_states, n_return):
        p = deepcopy(params)
        p.V_QD = vqd0

        # phi = 0
        H0 = construct_Hamil(mu, Ez, phi=0.0, params=p, calc_pfaffian=False, No_QD=False)
        eneg0, wfs0 = Diagonalize(H0, extract_exact=consider_all_states, num_eigvals=number_of_states)
        order0 = np.argsort(np.abs(eneg0))
        eneg0_ord = eneg0[order0]
        wfs0_ord = wfs0[:, order0]
        # compute capacitances using ordered arrays (keeps consistency)
        Ce0, Co0 = capacitance(eneg0_ord, wfs0_ord, params=p)
        wfs0_small = wfs0_ord[:, :min(n_return, wfs0_ord.shape[1])]

        # phi = pi
        Hpi = construct_Hamil(mu, Ez, phi=np.pi, params=p, calc_pfaffian=False, No_QD=False)
        enegpi, wfspi = Diagonalize(Hpi, extract_exact=consider_all_states, num_eigvals=number_of_states)
        order_pi = np.argsort(np.abs(enegpi))
        enegpi_ord = enegpi[order_pi]
        wfspi_ord = wfspi[:, order_pi]
        Cepi, Copi = capacitance(enegpi_ord, wfspi_ord, params=p)
        wfspi_small = wfspi_ord[:, :min(n_return, wfspi_ord.shape[1])]

        return (eneg0_ord[:min(n_return, wfs0_ord.shape[1])], wfs0_small, Ce0, Co0,
                enegpi_ord[:min(n_return, wfspi_ord.shape[1])], wfspi_small, Cepi, Copi)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_worker_cap_vqd)(vqd0, mu, Ez, deepcopy(params), consider_all_states, number_of_states, n_return)
        for vqd0 in tqdm(V_QD_sweep)
    )

    delta_C_0, delta_C_pi = [], []
    E0_neg_0, E0_neg_pi = 0.0, 0.0
    current_parity_0, current_parity_pi = "even", "even"
    wf0_0, wf0_pi = None, None

    for idx, res in enumerate(results):
        eneg0, wfs0_small, Ce0_raw, Co0_raw, enegpi, wfspi_small, Cepi_raw, Copi_raw = res

        # phi=0 tracking (same as original)
        if idx == 0:
            ordering = np.argsort(np.abs(eneg0))
            eneg_ord0 = eneg0[ordering]
            wf0_0 = wfs0_small[:, 1] if (eneg_ord0[0] > 0.0 and wfs0_small.shape[1] > 1) else wfs0_small[:, 0]
            E0_neg_0 = eneg_ord0[1] if (eneg_ord0[0] > 0.0 and eneg_ord0.shape[0] > 1) else eneg_ord0[0]

            # initialize wf0_pi using wf0_0 (match your original logic)
            wf0_pi = wf0_0
            E0_neg_pi, wf0_pi = Energy_tracker(enegpi, wfspi_small, wf0_pi)
            if E0_neg_pi * E0_neg_0 < 0.0:
                current_parity_pi = "odd"
        if idx != 0:
            Ep_neg_0, wfp_0 = Energy_tracker(eneg0, wfs0_small, wf0_0)
            if (E0_neg_0 * Ep_neg_0) < 0.0:
                current_parity_0 = "odd" if current_parity_0 == "even" else "even"
            E0_neg_0, wf0_0 = Ep_neg_0, wfp_0

        Ce0, Co0 = (Co0_raw, Ce0_raw) if current_parity_0 == "odd" else (Ce0_raw, Co0_raw)
        delta_C_0.append(Ce0 - Co0)

        # phi=pi tracking (continuation after initialization)
        if idx != 0:
            Ep_neg_pi, wfp_pi = Energy_tracker(enegpi, wfspi_small, wf0_pi)
            if (E0_neg_pi * Ep_neg_pi) < 0.0:
                current_parity_pi = "odd" if current_parity_pi == "even" else "even"
            E0_neg_pi, wf0_pi = Ep_neg_pi, wfp_pi

        Cepi, Copi = (Copi_raw, Cepi_raw) if current_parity_pi == "odd" else (Cepi_raw, Copi_raw)
        delta_C_pi.append(Cepi - Copi)

    params.V_QD = temp_store_initial_V_QD_val
    return np.array(delta_C_0), np.array(delta_C_pi)

def expectation_value(wf1: np.ndarray, operator: np.ndarray, wf2: np.ndarray):
    return np.matmul(np.conj(wf1), np.matmul(operator, wf2.T))

def inductance(phi: float, energies: np.ndarray, wavefunctions: np.ndarray, params: ModelParams, include_diamagnetic_term=True):
    Length = params.Length
    t = params.t
    alpha = params.alpha
    w_right = params.w_right
    w = params.w
    eta = params.eta
    kappa = 0.3698 #Converts to nano Henry inverse
    
    ordered_indices = np.argsort(np.abs(energies))
    ordered_energies = energies[ordered_indices]
    ordered_wavefunctions = wavefunctions[:, ordered_indices]
    eneg_pos, eneg_neg = [], []
    positive_wavefunctions, negative_wavefunctions = [], []
    w0 = w + 1.0j*eta

    hopping_particle_1 = np.array([[-t, alpha/2.0],[-alpha/2.0, -t]])
    hopping_hole_1 = np.array([[t, -alpha/2.0],[alpha/2.0, t]])
    hopping_particle_2 = np.array([[-t, -alpha/2.0], [alpha/2.0, -t]])
    hopping_hole_2 = np.array([[t, alpha/2.0], [-alpha/2.0, t]])

    #Defining J (current operator) and D (diamagnetic curvature operator)

    H_op_1 = np.zeros((4,4), dtype=complex)
    H_op_1[0:2,0:2] = -1.0j*hopping_particle_1*np.exp(-1.0j*phi)
    H_op_1[2:4,2:4] = 1.0j*hopping_hole_1*np.exp(1.0j*phi)

    H_op_2 = np.zeros((4,4), dtype=complex)
    H_op_2[0:2,0:2] = 1.0j*hopping_particle_2*np.exp(1.0j*phi)
    H_op_2[2:4,2:4] = -1.0j*hopping_hole_2*np.exp(-1.0j*phi)

    J_operator = np.zeros((8,8), dtype=complex)
    J_operator[0:4, 4:8] = w_right*H_op_2
    J_operator[4:8, 0:4] = w_right*H_op_1

    H_op_3 = np.zeros((4,4), dtype=complex)
    H_op_3[0:2,0:2] = -1.0*hopping_particle_1*np.exp(-1.0j*phi)
    H_op_3[2:4,2:4] = -1.0*hopping_hole_1*np.exp(1.0j*phi)

    H_op_4 = np.zeros((4,4), dtype=complex)
    H_op_4[0:2,0:2] = -1.0*hopping_particle_2*np.exp(1.0j*phi)
    H_op_4[2:4,2:4] = -1.0*hopping_hole_2*np.exp(-1.0j*phi)

    D_operator = np.zeros((8,8), dtype=complex)
    D_operator[0:4, 4:8] = w_right*H_op_4
    D_operator[4:8, 0:4] = w_right*H_op_3

    for od_energies_idx in range(len(ordered_energies)):
        if ordered_energies[od_energies_idx] > 0.0:
            positive_wavefunctions.append(np.concatenate((ordered_wavefunctions[0:4, od_energies_idx], ordered_wavefunctions[4*Length:4*Length+4, od_energies_idx])))
            eneg_pos.append(ordered_energies[od_energies_idx])
        else:
            negative_wavefunctions.append(np.concatenate((ordered_wavefunctions[0:4, od_energies_idx], ordered_wavefunctions[4*Length:4*Length+4, od_energies_idx])))
            eneg_neg.append(ordered_energies[od_energies_idx])
    positive_wavefunctions = np.array(positive_wavefunctions)
    negative_wavefunctions = np.array(negative_wavefunctions)
    eneg_pos = np.array(eneg_pos)
    eneg_neg = np.array(eneg_neg)

    min_len = min(len(eneg_pos), len(eneg_neg))
    eneg_pos = eneg_pos[:min_len]
    eneg_neg = eneg_neg[:min_len]
    positive_wavefunctions = positive_wavefunctions[:,:min_len]
    negative_wavefunctions = negative_wavefunctions[:,:min_len]
    
    L_const = expectation_value(positive_wavefunctions[1:, :], J_operator, negative_wavefunctions[1:, :])
    L_odd = expectation_value(negative_wavefunctions[0, :], J_operator, negative_wavefunctions[1:, :])
    L_odd_1 = expectation_value(positive_wavefunctions[1:, :], J_operator, positive_wavefunctions[0, :])
    L_odd_2 = expectation_value(negative_wavefunctions[0, :], J_operator, positive_wavefunctions[0, :])
    L_odd_2 = (np.abs(L_odd_2)**2.0)*(eneg_neg[0]-eneg_pos[0])/((eneg_neg[0]-eneg_pos[0])**2.0 - w0**2.0)


    L_even = expectation_value(positive_wavefunctions[0, :], J_operator, negative_wavefunctions[1:, :])
    L_even_1 = expectation_value(positive_wavefunctions[1:, :], J_operator, negative_wavefunctions[0, :])
    L_even_2 = expectation_value(positive_wavefunctions[0, :], J_operator, negative_wavefunctions[0, :])
    L_even_2 = (np.abs(L_even_2)**2.0)*(eneg_pos[0]-eneg_neg[0])/((eneg_pos[0]-eneg_neg[0])**2.0 - w0**2.0)

    L_D_even, L_D_odd = 0.0 + 0.0j, 0.0 + 0.0j

    if include_diamagnetic_term:

        ## Diamagnetic curvature term

        L_D_even = expectation_value(negative_wavefunctions[: ,:], D_operator, negative_wavefunctions[:, :])
        L_D_odd = expectation_value(negative_wavefunctions[1:, :], D_operator, negative_wavefunctions[1:, :])
        L_D_odd_1 = expectation_value(positive_wavefunctions[0, :], D_operator, positive_wavefunctions[0, :])

        L_D_odd = np.trace(L_D_odd) + L_D_odd_1
        L_D_even = np.trace(L_D_even)

    for i in range(len(eneg_pos)-1):
        for j in range(len(eneg_pos)-1):
            L_const[i][j] = (np.abs(L_const[i][j])**2.0)*(eneg_pos[i+1]-eneg_neg[j+1]) / ((eneg_pos[i+1]-eneg_neg[j+1])**2.0 - w0**2.0)
        L_odd[i] = (np.abs(L_odd[i])**2.0)*(eneg_neg[0]-eneg_neg[i+1]) / ((eneg_neg[0]-eneg_neg[i+1])**2.0 - w0**2.0)
        L_odd_1[i] = (np.abs(L_odd_1[i])**2.0)*(eneg_pos[i+1]-eneg_pos[0]) / ((eneg_pos[i+1]-eneg_pos[0])**2.0 - w0**2.0)
        L_even[i] = (np.abs(L_even[i])**2.0)*(eneg_pos[0]-eneg_neg[i+1]) / ((eneg_pos[0]-eneg_neg[i+1])**2.0 - w0**2.0)
        L_even_1[i] = (np.abs(L_even_1[i])**2.0)*(eneg_pos[i+1]-eneg_neg[0]) / ((eneg_pos[i+1]-eneg_neg[0])**2.0 - w0**2.0)
    
    L_value_const = np.sum(L_const)
    L_value_even_P = -2*(L_value_const + L_even_2 + np.sum(L_even_1) + np.sum(L_even))
    L_value_odd_P = -2*(L_value_const + L_odd_2 + np.sum(L_odd_1) + np.sum(L_odd))
    return np.real(kappa*L_value_even_P), np.real(kappa*L_D_even), np.real(kappa*L_value_odd_P), np.real(kappa*L_D_odd)


def inductance_vs_phi(mu: float, Ez: float, params: ModelParams,
                      include_diamagnetic_term=True, consider_all_states=False,
                      number_of_states=30, n_jobs=6, n_return=2):
    phi_low = params.phi_low
    phi_high = params.phi_high
    phi_points = params.phi_points
    phi_sweep = np.linspace(phi_low, phi_high, phi_points)

    def _worker_ind_vs_phi(phi0, mu, Ez, params, include_diamagnetic_term, consider_all_states, number_of_states, n_return):
        p = deepcopy(params)
        H_ang = construct_Hamil(mu, Ez, phi=phi0, params=p, calc_pfaffian=False, No_QD=False)
        eneg, wfs = Diagonalize(H_ang, extract_exact=consider_all_states, num_eigvals=number_of_states)
        order = np.argsort(np.abs(eneg))
        eneg_ord = eneg[order]
        wfs_ord = wfs[:, order]
        Le_P_raw, Le_D_raw, Lo_P_raw, Lo_D_raw = inductance(phi0, eneg_ord, wfs_ord, params=p, include_diamagnetic_term=include_diamagnetic_term)
        ncols = min(n_return, wfs_ord.shape[1])
        return eneg_ord[:ncols], wfs_ord[:, :ncols], Le_P_raw, Le_D_raw, Lo_P_raw, Lo_D_raw

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_worker_ind_vs_phi)(phi0, mu, Ez, deepcopy(params), include_diamagnetic_term, consider_all_states, number_of_states, n_return)
        for phi0 in tqdm(phi_sweep)
    )

    Le_P_ar, Lo_P_ar, Le_D_ar, Lo_D_ar = [], [], [], []
    E0_neg = 0.0
    current_parity = "even"
    wf0 = None
    for idx, (eneg_small, wfs_small, Le_P_raw, Le_D_raw, Lo_P_raw, Lo_D_raw) in enumerate(results):
        if idx == 0:
            ordering0 = np.argsort(np.abs(eneg_small))
            eneg_ord0 = eneg_small[ordering0]
            wf0 = wfs_small[:, 1] if (eneg_ord0.shape[0] > 1 and eneg_ord0[0] > 0.0) else wfs_small[:, 0]
            E0_neg = eneg_ord0[1] if (eneg_ord0.shape[0] > 1 and eneg_ord0[0] > 0.0) else eneg_ord0[0]
        if idx != 0:
            Ep_neg, wfp = Energy_tracker(eneg_small, wfs_small, wf0)
            if (E0_neg * Ep_neg) < 0.0:
                current_parity = "odd" if current_parity == "even" else "even"
            E0_neg, wf0 = Ep_neg, wfp

        if current_parity == "odd":
            Le_P, Lo_P = Lo_P_raw, Le_P_raw
            Le_D, Lo_D = Lo_D_raw, Le_D_raw
        if current_parity == "even":
            Le_P, Lo_P = Le_P_raw, Lo_P_raw
            Le_D, Lo_D = Le_D_raw, Lo_D_raw

        Le_P_ar.append(Le_P); Lo_P_ar.append(Lo_P)
        Le_D_ar.append(Le_D); Lo_D_ar.append(Lo_D)
    return np.array(Le_P_ar), np.array(Le_D_ar), np.array(Lo_P_ar), np.array(Lo_D_ar)

def inductance_vs_VQD(mu: float, Ez: float, params: ModelParams, include_diamagnetic_term=True, consider_all_states=False, number_of_states=30, n_jobs=6, n_return=2):
    V_QD_low = params.V_QD_low
    V_QD_high = params.V_QD_high
    V_QD_points = params.V_QD_points
    temp_store_initial_V_QD_val = params.V_QD
    V_QD_sweep = np.linspace(V_QD_low, V_QD_high, V_QD_points)

    def _worker_ind_vqd(vqd0, mu, Ez, params, include_diamagnetic_term, consider_all_states, number_of_states, n_return):
        p = deepcopy(params)
        p.V_QD = vqd0

        # phi = 0
        H0 = construct_Hamil(mu, Ez, phi=0.0, params=p, calc_pfaffian=False, No_QD=False)
        eneg0, wfs0 = Diagonalize(H0, extract_exact=consider_all_states, num_eigvals=number_of_states)
        order0 = np.argsort(np.abs(eneg0))
        eneg0_ord = eneg0[order0]
        wfs0_ord = wfs0[:, order0]
        Le_P_0, Le_D_0, Lo_P_0, Lo_D_0 = inductance(0.0, eneg0_ord, wfs0_ord, params=p, include_diamagnetic_term=include_diamagnetic_term)
        wfs0_small = wfs0_ord[:, :min(n_return, wfs0_ord.shape[1])]

        # phi = pi
        Hpi = construct_Hamil(mu, Ez, phi=np.pi, params=p, calc_pfaffian=False, No_QD=False)
        enegpi, wfspi = Diagonalize(Hpi, extract_exact=consider_all_states, num_eigvals=number_of_states)
        order_pi = np.argsort(np.abs(enegpi))
        enegpi_ord = enegpi[order_pi]
        wfspi_ord = wfspi[:, order_pi]
        Le_P_pi, Le_D_pi, Lo_P_pi, Lo_D_pi = inductance(np.pi, enegpi_ord, wfspi_ord, params=p, include_diamagnetic_term=include_diamagnetic_term)
        wfspi_small = wfspi_ord[:, :min(n_return, wfspi_ord.shape[1])]

        return (eneg0_ord[:min(n_return, wfs0_ord.shape[1])], wfs0_small, Le_P_0, Le_D_0, Lo_P_0, Lo_D_0,
                enegpi_ord[:min(n_return, wfspi_ord.shape[1])], wfspi_small, Le_P_pi, Le_D_pi, Lo_P_pi, Lo_D_pi)

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_worker_ind_vqd)(vqd0, mu, Ez, deepcopy(params), include_diamagnetic_term, consider_all_states, number_of_states, n_return)
        for vqd0 in tqdm(V_QD_sweep)
    )

    delta_L_P_0, delta_L_D_0, delta_L_P_pi, delta_L_D_pi = [], [], [], []
    E0_neg_0, E0_neg_pi = 0.0, 0.0
    current_parity_0, current_parity_pi = "even", "even"
    wf0_0, wf0_pi = None, None

    for idx, res in enumerate(results):
        (eneg0, wfs0_small, Le_P_0_raw, Le_D_0_raw, Lo_P_0_raw, Lo_D_0_raw,
         enegpi, wfspi_small, Le_P_pi_raw, Le_D_pi_raw, Lo_P_pi_raw, Lo_D_pi_raw) = res

        # phi=0 tracking
        if idx == 0:
            ordering = np.argsort(np.abs(eneg0))
            eneg_ord0 = eneg0[ordering]
            wf0_0 = wfs0_small[:, 1] if (eneg_ord0[0] > 0.0 and wfs0_small.shape[1] > 1) else wfs0_small[:, 0]
            E0_neg_0 = eneg_ord0[1] if (eneg_ord0[0] > 0.0 and eneg_ord0.shape[0] > 1) else eneg_ord0[0]

            wf0_pi = wf0_0
            E0_neg_pi, wf0_pi = Energy_tracker(enegpi, wfspi_small, wf0_pi)
            if E0_neg_pi * E0_neg_0 < 0.0:
                current_parity_pi = "odd"
        if idx != 0:
            Ep_neg_0, wfp_0 = Energy_tracker(eneg0, wfs0_small, wf0_0)
            if (E0_neg_0 * Ep_neg_0) < 0.0:
                current_parity_0 = "odd" if current_parity_0 == "even" else "even"
            E0_neg_0, wf0_0 = Ep_neg_0, wfp_0

        if current_parity_0 == "odd":
            Le_P_0, Lo_P_0 = Lo_P_0_raw, Le_P_0_raw
            Le_D_0, Lo_D_0 = Lo_D_0_raw, Le_D_0_raw
        if current_parity_0 == "even":
            Le_P_0, Lo_P_0 = Le_P_0_raw, Lo_P_0_raw
            Le_D_0, Lo_D_0 = Le_D_0_raw, Lo_D_0_raw

        delta_L_P_0.append(Le_P_0 - Lo_P_0)
        delta_L_D_0.append(Le_D_0 - Lo_D_0)

        # phi=pi tracking (continue after initialization)
        if idx != 0:
            Ep_neg_pi, wfp_pi = Energy_tracker(enegpi, wfspi_small, wf0_pi)
            if (E0_neg_pi * Ep_neg_pi) < 0.0:
                current_parity_pi = "odd" if current_parity_pi == "even" else "even"
            E0_neg_pi, wf0_pi = Ep_neg_pi, wfp_pi

        if current_parity_pi == "odd":
            Le_P_pi, Lo_P_pi = Lo_P_pi_raw, Le_P_pi_raw
            Le_D_pi, Lo_D_pi = Lo_D_pi_raw, Le_D_pi_raw
        if current_parity_pi == "even":
            Le_P_pi, Lo_P_pi = Le_P_pi_raw, Lo_P_pi_raw
            Le_D_pi, Lo_D_pi = Le_D_pi_raw, Lo_D_pi_raw

        delta_L_P_pi.append(Le_P_pi - Lo_P_pi)
        delta_L_D_pi.append(Le_D_pi - Lo_D_pi)

    params.V_QD = temp_store_initial_V_QD_val
    return (np.array(delta_L_P_0), np.array(delta_L_D_0),
            np.array(delta_L_P_pi), np.array(delta_L_D_pi))