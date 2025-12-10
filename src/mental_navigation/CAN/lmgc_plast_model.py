from .model_units import ECModule, LandmarkUnit
import numpy as np
import pandas as pd

class LMGCPlasticityModel:
    """
    LM-GC Plasticity Model with Oja's learning (Hebbian rule) rule

    Ingredients:
    -----------
    - M EC modules (each moving a Gaussian bump around a ring of K neurons)
    - 1 LM neuron (scalar activity)
    - external LM pattern sampled at LM module's phase
    - Weights W (shape (K x M)) from each module's neuron to LM 
    """

    def __init__(
        self,
        list_modules : list[ECModule],
        landmark: LandmarkUnit,
        eta: float,
        t_off: int,
        weight_init_std: float | None=None,
        seed : int | None= None
        ):
        """
        Parameters:
        -----------
        - list_modules
            the EC modules (length M)
        - landmark
            LM external pattern object
        - eta
            learning rate in Oja's rule
        - t_off
            time step at wich external input turns off
        - weight_init_std
            stdev of initial weight distribution. If none, uses 1/(K*M)
        - seed for reproducibility
        """

        if seed is not None:
            np.random.seed(seed)

        assert len(list_modules) > 0 #need at least one module in the model
        self.modules = list_modules
        self.landmark = landmark
        self.lm_module_index = landmark.lm_module_index
        self.eta = eta
        self.t_off = t_off

        self.K = self.modules[0].K
        self.M = len(list_modules)

        assert all(m.K == self.K for m in self.modules)
        assert 0 <= self.lm_module_index < self.M

        if weight_init_std is None:
            weight_init_std = 1 / (self.K * self.M)
        
        # W weight matrix (K x M)
        self.W = weight_init_std * np.random.randn(self.K, self.M)
        self.W_init = self.W.copy()

        ## logs for analyses
        self.weight_history = []
        self.learning_flags = []
        self.time_points = []
        self.activity_history = {'s':[], 's_int':[], 's_ext':[]}


    """
    Learning rule at each time step t:
    ---------------------------------
    X(t):
        matrix (K, M) of EC activities (column i x_i(t)= module i)
    s_int(t)
        sum_{i,m} W[i,m] X[i,m]
    s_ext(t)
        external input evaluated at the phase of module m^* at time t
    s => LM activity:
        s_int + s_ext during learning
        s_int only after external input stops
    W = W + lr * s * [X - s*W]
    then mean-center each column (per-module mean subtraction)
    """

    def get_initial_weights(self):
        return self.W_init

    def Oja_update(self, W, X, s, eta):
        W += eta * s * (X - s * W)
        return W

    def mean_sub(self, W):
        col_means = W.mean(axis=0, keepdims=True)
        W -= col_means
        return W

    def compute_EC_data(self, t:int):
        X = np.zeros((self.K, self.M))
        phases_track = np.zeros(self.M, dtype=int)
        for i, module in enumerate(self.modules):
            phases_track[i] = module.phase_index(t)
            X[:, i] = module.activity(t) #shape: (K,)
        return X, phases_track

    def LM_total_activity(self, s_int:float, s_ext:float|None, t:int):
        if t < self.t_off:
            s = s_int + s_ext
            return s, 1
        else:
            return s_int, 0


    def step(self, t:int) -> dict:
        """
        A single simulation step. At time t:
        - compute EC activity at time t X(t)
        - compute internal stimulus s_int, external stimulus and overall LM activity
        - update weight matrix W

        -----------------------
        phases_track = tracks the phase index (neuron bin on the ring from 0 to K-1) module i is on at time t
            shape: (M,)
        """

        # 1) compute EC modules activity
        X, phases_track = self.compute_EC_data(t)
        
        # 2) internal input s_int = sum_i {W^T_i * X_i}
        s_int = float(np.sum(self.W * X))

        # 3) external input from LM-matched module 
        phi_lm = phases_track[self.lm_module_index]
        s_ext = self.landmark.external_input(phi_lm)

        # 4) LM activity received (either s_int + s_ext or only s_int)
        s, lm_visible_flag = self.LM_total_activity(s_int, s_ext, t)
        
        # 5) Oja weight update
        self.W = self.Oja_update(self.W, X, s, self.eta)

        # 6) mean subtraction per column
        self.W = self.mean_sub(self.W)

        return {
            "X": X,
            "phases_track" : phases_track,
            "s_int": s_int,
            "s_ext": s_ext,
            "s": s,
            "learning_flag": lm_visible_flag
        }

    def run(self, T:int, snapshot_interval:int, store_hist:bool = True):
        """
        Run the simulation for T time steps
        Save a snapshot of the current state of the weights once every 'snapshot' steps
        If store_hist = True, save history of weights, phase_indices and time indices
        """
        for t in range(1, T+1):
            current_run = self.step(t)

            if store_hist and (t % snapshot_interval == 0):
                self.weight_history.append(self.W.copy())
                self.learning_flags.append(current_run["learning_flag"])
                self.time_points.append(t)
                self.activity_history['s'].append(current_run["s"])
                self.activity_history['s_int'].append(current_run["s_int"])
                self.activity_history['s_int'].append(current_run["s_ext"])



    def get_final_weights(self)-> np.ndarray:
        return self.W


    def get_history(self)-> dict:
        return {
            "time_step": np.array(self.time_points),
            "weights": np.array(self.weight_history),
            "NTS_flg": np.array(self.learning_flags),
            "total_activity": np.array(self.activity_history['s']),
            "int_input": np.array(self.activity_history['s_int']),
            "ext_input": np.array(self.activity_history['s_ext'])
            }

