import numpy as np
from typing import Optional, Sequence, Dict, Any, Tuple

class CANNetwork:
    """
    1D Continuous Attractor Network (single module, two populations: Right & Left)
    Defines the architecture and the static parameters

    Attributes
    ----------
    K : int
        Number of neurons per ring (Right and Left populations each have N neurons).
        Total neurons = 2K.
    dt : float
        Integration time step (seconds) -> discretize time in steps so that each step lasts dt seconds
    tau_s : float
        Synaptic time constant (seconds).
    beta_0 : float
        Global feedforward excitatory drive amplitude.
    beta_vel : float
        Gain on velocity input.
    
        
    mex_hat : np.ndarray, shape (N,)
        Base mexican-hat connectivity kernel on the ring.
    W_RR, W_LL, W_RL, W_LR : np.ndarray, shape (N, N)
        Synaptic weight matrices between populations:
        - W_RR: Right -> Right (shifted kernel, one direction)
        - W_LL: Left  -> Left  (shifted kernel, opposite direction)
        - W_RL: Left  -> Right (no shift)
        - W_LR: Right -> Left  (no shift)
    FF_global : np.ndarray, shape (N,)
        Constant feedforward input to each neuron in each population.
    """

    def __init__(
        self, 
        K: int,
        dt: float = 1.0 / 2000.0,
        tau_s: float = 40.0 / 1000.0, #(40 ms)
        beta_vel: float = 1.0, 
        beta_0: float = 100.0,
        ):

        self.K = K
        self.dt = dt
        self.tau_s = tau_s
        self.beta_vel = beta_vel
        self.beta_0 = beta_0

        # Global feedforward excitation
        self.FF_global = self.beta_0 * np.ones(self.K, dtype= float)

        # Build connectivity
        self.mexhat = self.build_mexhat()
        self.W_RR, self.W_LL, self.W_RL, self.W_LR = self.build_synapses()

    
    def build_mexhat(
            self,
            A_exc: float = 1000.0,
            s_exc: float = 1.05 / 100.0,
            A_inh: float = 1000.0,
            s_inh: float = 1.00 / 100.0
            ) -> np.array:
        z = np.arange(-self.K/2, self.K/2, 1.0)
        mex = A_exc * np.exp(-s_exc * z**2) - A_inh * np.exp(-s_inh * z**2)

        #circshitft by N/2 -1
        shift = int(self.K/2 -1)
        mex = np.roll(mex, shift)
        return mex.astype(float)
    

    def build_synapses(self) -> Tuple[np.array, np.array, np.array, np.array]:
        # returns (W_RR, W_LL, W_RL, W_LL)
        W_RR = np.zeros((self.K, self.K), dtype=float)
        W_LL = np.zeros((self.K, self.K), dtype=float)
        W_RL = np.zeros((self.K, self.K), dtype=float)
        W_LR = np.zeros((self.K, self.K), dtype=float)

        for i in range(self.K):
            # indexing 0, ..., K-1
            
            W_RR[i, :] = np.roll(self.mexhat, i)        
            W_LL[i, :] = np.roll(self.mexhat, i + 2)
            W_RL[i, :] = np.roll(self.mexhat, i + 1)
            W_LR[i, :] = np.roll(self.mexhat, i + 1)

            # earlier version
            #W_RR[i, :] = np.roll(self.mexhat, i-1)
            #W_LL[i, :] = np.roll(self.mexhat, i+1)
            #W_RL[i, :] = np.roll(self.mexhat, i)
            #W_LR[i, :] = np.roll(self.mexhat, i)
        
        return W_RR, W_LL, W_RL, W_LR




