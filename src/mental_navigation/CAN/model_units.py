import numpy as np

class PhaseRingUnit:
    """
    Base class for units that live on the 1D circular ring of K phase bins
    Provides:
    - K: number of neurons in the cells, arranged on a ring (each neuron = a phase bin)
    - base_width: width parameter for the Gaussian bump
    - create_bump(): make a normalized Gaussian bump on the ring
    """

    def __init__(self, K:int, base_width:float):
        self.K = K
        self.base_width = base_width
    
    def create_bump(self, width_scale:float = 1.0, amp:float = 1.0) -> np.ndarray:
        """
        Returns a Gaussian over indices 0, ..., K-1
        
        Parameters:
        -----------
        - width_scale: float
            multiplies base_width to set effective std of the Gaussian
        - amplitude :float
            amplitude of the bump
        
        Returns:
        --------
        bump : np.ndarray, shape:(K,)
        """
        K = self.K
        width = self.base_width * width_scale
        # indices centered around 0
        z = np.arange(-K/2, K/2)
        kernel = np.exp(-z**2 / (2 * width**2))
        kernel = amp * kernel / np.max(kernel)
        return kernel   # shape: (K,)




class ECModule(PhaseRingUnit):
    """
    One EC (grid-cell) module:
    - K neurons arranged on a ring (each neuron = a phase bin)
    - activity at time t: bump on the ring, shifted according to t / scale for every possible phase on the ring
    """

    def __init__(self, K:int, base_width:float, scale:float):
        """
        Parameters:
        -----------
        K:int
            Number of neurons / phase bins
        base_width:float
            base width parameter for the Gaussian bump
        scale:float
            temporal scaling factor for this module, characterizing frequency
            smaller scale => phase advances faster => higher effective frequency
            larger scale => phase advances slower => lower effective frequency
        """
        super().__init__(K=K, base_width=base_width)
        self.scale = scale

        # precompute base bump g_i (Gaussian over neuron indices, with static shape and dynamic position over time)
        self.bump = self.create_bump(width_scale = self.scale, amp=1.0)
    


    def phase_index(self, t: int) -> int:
        """
        Compute current phase index for the current module at global time t
        """
        t_i = int(np.round(t / self.scale))
        phi_i = (t_i -1) % self.K
        return phi_i


    def activity(self, t:int) -> np.ndarray:
        """
        Population activity vector x_i(t)
            shape: (K, )
        It is the circularly shifted version of the template module bump, according to the phase index at time t phi_i(t)
        """
        phi_i = self.phase_index(t)
        return np.roll(self.bump, phi_i)
    




class LandmarkUnit(PhaseRingUnit):
    """
    Landmark (LM) external input pattern across the ring of K phase bins

    Important: this is NOT the LM unit itself, which is a single scalar unit
    This class encodes:
    - lm_ext : np.ndarray (shape: (K,))
        external LM drive vector
        lm_ext[k] is the external LM drive vector present if the internal phase is =k
    - time dependence comes only when we sample lm_ext at phase phi_star of the LM-matched module
    """

    def __init__(self, K:int, base_width:float, lm_phases_deg:float, lm_module_index, amp:float = 1.0):
        """
        Parameters:
        -----------
        - K:
            number of bins (same as EC module)
        - base_width:
            base_width parameter for the Gaussian bump (same as EC module)
        - lm_phase_deg: float or list of float
            landmark phase in degrees 
        - amp-> amplitude of each landmark bump
        """
        super().__init__(K=K, base_width = base_width)
        self.lm_phases_deg = np.atleast_1d(lm_phases_deg).astype(float)
        self.lm_module_index = lm_module_index

        self.base_bump = self.create_bump(width_scale = 1.0, amp = amp)
        self.lm_ext = self.build_lm_ext() # shape (K,)

    
    def build_lm_ext(self) -> np.ndarray:
        """
        Sum bumps at the requested phases (if there is more than one exernal stimulus)
        Here there is no external scale:
            phi / K = lm_phase_deg / 360

        Time independent => only mapping a landmark angle to a bin
        - LM external bump itself does not evolve in time
        """
        lm_ext = np.zeros(self.K)
        for phase_deg in self.lm_phases_deg:
            idx = int(np.round(phase_deg / 360 * self.K)) % self.K
            lm_ext += np.roll(self.base_bump, idx)
        return lm_ext


    def external_input(self, phase_idx:int) -> float:
        """
        Given a current phase index (eg. LM-matched module), return the scalar external input lm_ext[phi]
        I_ext(t) = lm_ext[phi_m^* (t)] 
        """
        return float(self.lm_ext[phase_idx % self.K])

