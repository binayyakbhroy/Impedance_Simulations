import numpy as np
from dataclasses import dataclass

@dataclass
class ModelParams:
    #Wire geometry
    Length: int = 300
    
    #Physical parameters
    t: float = 16.56
    alpha: float = 1.4
    mu: float = 0.0
    Delta0: float = 0.3
    Ez: float = 0.5
    smsc_coupling: float = 0.2
    
    #Quantum dot parameters
    w_left: float = 0.025
    w_right: float = 0.025
    V_QD: float = 0.0

    #Parameter Sweeps
    Ez_low: float = 0.1
    Ez_high: float = 1.25
    Ez_points: int = 150
    mu_low: float = -0.3
    mu_high: float = 0.5
    mu_points: int = 150
    phi_low: float = 0.0
    phi_high: float = 2.0*np.pi
    phi_points: int = 150
    V_QD_low: float = -0.3
    V_QD_high: float = 0.5
    V_QD_points: int = 150

    #Transport calculation parameters
    w: float = 0.0
    eta: float = 0.002

    Disorder: np.ndarray | None = None
    chirality_op: np.ndarray | None = None