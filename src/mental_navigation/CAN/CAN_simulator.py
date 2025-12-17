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
        K = net.K
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
        ) #shape (K,)

        # Activity over time: 2 populations (L,R)-> total shape (2K,)
        activity = np.zeros((2 * K, n_steps))

        for t in range(1, n_steps):
            prev = activity[:, t-1]
            s_L = prev[:K]      # (K,)
            s_R = prev[K:]      # (K,)

            # Step 1. velocity gains
            v_L, v_R = self.velocity_bias(v[t])     # scalars

            # Step 2. recurrent synaptic contributions
            g_RR, g_LL, g_RL, g_LR = self.synaptic_inp_contrib(s_L, s_R)   # each (K,)

            # Step 3. total input current (including static landmark bump)
            G = self.pre_activation_current(v_L, v_R,
                                            g_RR, g_LL, g_RL, g_LR,
                                            landmark_input = landmark)

            # Step 4. Threshold-linear activation (ReLU from B&F)
            F = np.maximum(G, 0.0)

            # Step 5. state update (first-order low-pass)
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
    
    def synaptic_inp_contrib(self, s_L:np.ndarry, s_R:np.ndarray)->Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        net = self.network
        g_LL = net.W_LL @ s_L
        g_LR = net.W_LR @ s_R
        g_RL = net.W_RL @ s_L
        g_RR = net.W_RR @ s_R
        return g_RR, g_LL, g_RL, g_LR


    def pre_activation_current(
            self,
            v_L:float, v_R:float,
            g_RR: np.ndarray, g_LL: np.ndarray, g_RL:np.ndarray, g_LR:np.ndarray,
            landmark_input: np.ndarray | None = None
            )-> np.ndarray:
        """
        Input current into each neuron of both populations.

        Recurrent input + constant global drive
        Multiplied by global velocity modulation parameter
        Optionally plus landmark input
        """
        net = self.network
        if landmark_input is None:
            landmark_input = np.zeros(self.network.K)
        G_L = v_L * (g_LL + g_LR + net.FF_global) + landmark_input
        G_R = v_R * (g_RL + g_RR + net.FF_global) + landmark_input
        return np.concatenate([G_L, G_R])
    



    def run_trial(self,
                  init_condition:np.array,
                  initial_phase: float = 30.0,
                  end_phase: float =360.0,
                  landmarkpresent: bool =True,
                  landmark_input_loc=(60, 120, 180, 240, 300),
                  wolm_speed: float =0.35,
                  wlm_speed: float =0.42,
                  wm: float=0.05,
                  T_max: float =60.0,
                  landmark_onset_steps: int =500,
                  landmark_tau_steps: int =1500):
        """
        Run one CAN trial with or without internal landmarks.

        Parameters:
        ------------
        - init_state:
            Initial populations activity (2K,) typically from init_state()
        - inital_phase
            Initial phase on the ring (approximate neuron index)
        - end_phase:
            Phase at which to stop the simulation
        - landmarkpresent: bool
            if True -> internal landmark corrections are active (NFJ24)
            if False -> no internal landmark inputs are added
        - landmark_input_loc: sequence of floats
            phases (neuron indices) at which internal landmarks are stored
        - wolm_speed, wlm_speed: float params
            Baseline speeds without/with landmarks
        - wm: float
            Weber fraction controlling speed noise

        - T_max : float
            Maximum duration of the simulation in seconds -> just as a safety measure against infinite loops.
        - landmark_onset_steps : int
            Temporal offset (in time steps) for the landmark amplitude peak.
        - landmark_tau_steps : int
            Temporal standard deviation (in time steps) of the landmark amplitude envelope.
        """

        net = self.network
        K = net.K
        max_steps = int(T_max / net.dt)

        # states through the time steps of the simulation
        s = np.zeros((2*K, max_steps))
        s[:, 0] = init_condition.copy()

        # initialize network state: now only phase on ring where the bump center starts
        nn_state = [initial_phase]

        # velocity input
        v_base = wlm_speed if landmarkpresent else wolm_speed
        v_noise = v_base * wm * np.random.randn()
        noisy_vel_input = v_base + v_noise
        v = (noisy_vel_input) * np.ones(max_steps)

        # landmark locations
        lm_locs = np.array(landmark_input_loc, dtype= float) # shape (L,)
        L = lm_locs.shape[0]

        # initialize lm_entry times as an array of null values
        lm_entry_times = np.full(L, np.nan) # entry times for each landmark
        


        t=0
        while nn_state[-1] < end_phase and t < max_steps -1:
            t += 1

            # Network dynamics
            prev = s[:, t-1]
            s_L = prev[:K]
            s_R = prev[K:]
            v_L, v_R = self.velocity_bias(v[t])
            g_RR, g_LL, g_RL, g_LR = self.synaptic_inp_contrib(s_L, s_R)

            # internal landmark trigger (if present)
            if landmarkpresent:
                current_phase = nn_state[-1]
                landmark, lm_entry_times = self.create_landmark_trigger(t,
                                                                        current_phase,
                                                                        lm_locs,
                                                                        lm_entry_times,
                                                                        landmark_onset_steps,
                                                                        landmark_tau_steps
                                                                        )

            else:
                landmark = np.zeros(K)
            
            # total input current
            G = self.pre_activation_current(v_L, v_R,
                                            g_RR, g_LL, g_RL, g_LR,
                                            landmark_input = landmark)

            # relu
            F = np.maximum(G, 0.0)

            # state update
            s[:, t] = prev + (F - prev) * net.dt / net.tau_s



            # tracking bump center:



    def create_landmark_trigger(
            self,
            t:int,
            phase: float,
            lm_locs:np.array,
            lm_entry_times:np.array,
            landmark_onset_steps: int,
            landmark_tau_steps: int
            )-> Tuple[np.array, np.array]:
        
        """
        If a landmark should be active at a certain time, generate its input profile.

        Logic behind:
        -------------
        - Landmarks are located at phases lm_locs[0..L-1] and have index 0,...,L-1
            lm_locs[i] = phase at which landmark i appears
        - For the current phase, find the last landmark we have passed (if any).
            - If no landmark seen yet, leave entry times null and zero landmark activity
            - If some landmark k is the last seen
                and it's the first time we enter its region (lm_entry_time was None) ==> record entry time.
        - Compute amplitude as a Gaussian in time since entry.
            - Need to compute the time distance (in step) from entry time to peak
        - Generate a spatial Gaussian bump centered at that landmark (± small shift).

        Returns
        -------
        landmark : np.ndarray of shape (K,)
            Landmark input at this time step.
        lm_entry_times : np.ndarray
            Updated array of first-entry times for each landmark.
        """
        
        L = lm_locs.shape[0]

        # 1. at first, not passed through first landmark yet
        landmark = np.zeros(self.network.K)

        # 2. Given the current phase, find the last landmark we have passed
        lm_idx = None
        for i in range(L - 1, -1, -1):  # from landmark L-1 down to first index 0
            if phase >= lm_locs[i]:
                lm_idx = i
                break

        # 3. If the first lm location was not reached yet,
        #   return zero landmark and unchanged lm_entry_times
        if lm_idx is None:
            return landmark, lm_entry_times
        

        # 4. if we are entering landmark k's region for the first time (lm_entry_time is still None)
        #   update entry time and construct landmark
        if lm_entry_times[lm_idx].isna():
            lm_entry_times[lm_idx] = t
        t_k = lm_entry_times[lm_idx]

        # time passed since entry
        time_passed = t-t_k
        




        
        
        # construct landmark input
        amp = self.construct_lm_amplitude(t, t_k, landmark_onset_steps, landmark_tau_steps)

        # Centers of the internal landmark bump (slightly shifted)
        centers = lm_locs + 3.0

        landmark = self.generate_landmark_input(
            centers=centers,
            std=5.0,
            ampl_scaling=amp,
            normalize_single=False,
        )
        return landmark
    

    def construct_lm_amplitude(self, t:int, t_k:int, landmark_onset_steps:int , landmark_tau_steps:int):
        return 50.0 * np.exp(-(t - t_k - landmark_onset_steps) ** 2/ (2.0 * landmark_tau_steps**2))

