from .CAN_network import CANNetwork
import numpy as np
from typing import Tuple

class CANSimulator:
    """
    Simulator class for CAN dynamics

    - Based on Burak & Fiete 2009 for the continuous attractor [BF09-like]
    - With internal landmark-driven inputs as in Neupane et al. 2024 [NFJ24 new]

    Contains:
        - init_state:
            Classic ring attractor with asymmetric connectivity. Ceates a moving bump of grid-cell activity that represents elapsed path distance.
        - run_trial
            The bump moves clockwise or counterclockwise according to the noisy velocity input and its direction
            Can include internal triggers after learning internal landmarks, or can run witout landmarks (a bump forms spontaneously and drifts according to v(t))
    """

    def __init__(self, network:CANNetwork):
        self.network = network
    

    def generate_landmark_input(self, centers, std, ampl_scaling, normalize_single=False):
        """
        Produces the total landmark input across K ring neurons
        Later, it will be added to the neural input to stabilize the bump

       Inputs:
        -----
        - centers: list or array of length L
            Landmark centers, expressed as neuron indices in 1, ..., K (the phases at which landmarks are shown)
        - std:
            standard dev of each Gaussian bump
        - ampl_scaling:
            amplitude of an extra excitatory Gaussian input injected when you’re near a landmark phase. used to model internal landmark 'kicks'
            (the strength of the landmark correction)
        - normalize_single: bool
            if True and L == 1, normalize max to 1 before scaling
        
            
        Returns:
        -------
        landmark_input: np.array of shape (K,)
            The total landmark drive to each of the K neurons
        
        """
        K = self.network.K
        centers = np.atleast_1d(centers).astype(float)
        L = centers.shape[0]

        # row vector of L centers
        centers_row = centers.reshape(1, L)    #(1,L)
        
        # column vector of K indices (from 1 to K)
        x_col = np.arange(1, K+1, dtype=float).reshape(K,1)     #(K,1)

        # For each neuron i and each landmark l:
        #   gauss[i, l] = exp( - (i - center_l)^2 / (2 * std^2) )
        #   gaussian_matr has shape (K, L)
        gaussian_matr = np.exp(-(x_col-centers_row)**2 / (2.0 * std**2))

        if normalize_single and L==1:
            gaussian_matr = gaussian_matr / gaussian_matr.max()
        
        landmark_input = ampl_scaling * gaussian_matr.sum(axis=1)   #(K,1)
        return landmark_input
    

    def init_state(self,
                   T=10.0,
                   v_base=0.0,
                   weber_frac=0.01,
                   landmark_mean= 0.0,
                   landmark_std = 30.0,
                   landmark_contribution=3.0) -> np.ndarray:
        """
        Run a short initialization to let the network settle into a bump attractor

        - uses low-amplitude noisy velocity input
        - adds a static gaussian landmark bump

        Return population activity at the final time point (shape (2K, ))
        """
        net = self.network
        n_steps = int(T / net.dt)

        # noisy speed input
        v_noise = weber_frac * np.random.randn()
        v = (v_base + v_noise) * np.ones(n_steps) # scalar velocity input at time steps

        # Generate a soft static landmark bump (normalized and scaled)
        landmark = self.generate_landmark_input(
            centers=[landmark_mean],
            std=landmark_std,
            ampl_scaling=landmark_contribution,
            normalize_single=True,
        )

        # update activity during time steps
        activity = np.zeros((2 * net.K, n_steps))

        for t in range(1, n_steps+1):
            prev = activity[:, t-1]
            s_L = prev[:net.K]      # (K,)
            s_R = prev[net.K:]      # (K,)

            # velocity gains
            v_L, v_R = self.velocity_bias(v[t])     # scalars

            # recurrent synaptic contributions
            g_RR, g_LL, g_RL, g_LR = self.synaptic_inp_contrib(net, s_L, s_R)   # each (K,)

            # total pre-activation current
            G = self.pre_activation_current(v_L, v_R, g_RR, g_LL, g_RL, g_LR)

            # threshold-linear activation (from B&F)
            #   negative input -> no firing / positive input -> proportional firing
            F = np.maximum(G, 0.0)

            activity[:, t] = prev + (F - prev) * net.dt / net.tau_s
        
        return activity[:, -1].copy()





    def velocity_bias(self, net:CANNetwork, v_t:np.array)-> Tuple[float, float]:
        """
        They modulate how strongly the recurrent input moves the bump left / right
        v[t] > 0 -> right population strengthened, left weakened -> bump drifts in right direction
        v[t] < 0 -> left population strengthened, right weakened -> bump drifts in the opposite direction
        """
        v_L = (1.0 - net.beta_vel * v_t)
        v_R = (1.0 + net.beta_vel * v_t)
        return v_L, v_R # both scalars
    
    def synaptic_inp_contrib(self, net:CANNetwork, s_L:np.ndarry, s_R:np.ndarray)->Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        g_LL = net.W_LL @ s_L
        g_LR = net.W_LR @ s_R
        g_RL = net.W_RL @ s_L
        g_RR = net.W_RR @ s_R
        return g_RR, g_LL, g_RL, g_LR

    def pre_activation_current(net:CANNetwork, v_L:float, v_R:float, g_RR, g_LL, g_RL, g_LR)-> np.ndarray:
        """
        Input current into each neuron of both populations.

        Recurrent input + constant global drive
        Multiplied by global velocity modulation parameter
        """
        G_L = v_L * (g_LL + g_LR + net.FF_global)
        G_R = v_R * (g_RL + g_RR + net.FF_global)
        return np.concatenate([G_L, G_R])
    



    def run_trial(self,
                  init_state,
                  initial_state = 30.0,
                  end_state=360.0,
                  landmarkpresent=True,
                  landmark_input_loc=(60, 120, 180, 240, 300),
                  wolm_speed=0.35,
                  wlm_speed=0.42,
                  wm=0.05,
                  T_max=60.0,
                  landmark_onset_steps=500,
                  landmark_tau_steps=1500):
        """
        Run one CAN trial with or without internal landmarks.

        - init_state: (2K,) array from init_state()
        - inital_state, end_state: initial/final phase (treated as ~neuron index)
        - landmarkpresent: True → internal landmarks active (NFJ24); False → no LM
        - landmark_input_loc: phases (neuron indices) where internal landmarks are stored
        - wolm_speed, wlm_speed, wm: speed and Weber fraction parameters
        """

        net = self.network
        max_steps = int(T_max / net.dt)

        s = np.zeros((2*net.K, max_steps))
        s[:, 0] = init_state.copy()

        # track phase on ring where the bump center is
        nn_state = [initial_state]

        # velocity input
        v_base = wlm_speed if landmarkpresent else wolm_speed
        v_noise = v_base * wm * np.random.randn()
        noisy_vel_input = v_base + v_noise
        v = np.full(max_steps, noisy_vel_input)

        # landmark locations
        lm_locs = np.array(landmark_input_loc, dtype= float)
        L = lm_locs.shape[0]
        lm_flag = 0
        lm_times = np.full(L, np.nan)


        t=0
        while nn_state[-1] < end_state and t < max_steps -1:
            t += 1
            prev = s[:, t-1]
            s_L = prev[:net.K]
            s_R = prev[net.K:]
            v_L, v_R = self.velocity_bias(net, v[t])
            g_RR, g_LL, g_RL, g_LR = self.synaptic_inp_contrib(net, s_L, s_R)
            

