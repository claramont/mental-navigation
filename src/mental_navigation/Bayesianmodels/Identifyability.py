## IDEA: Run identifyability test complete since given the fact that is only synthetic data we would be able to 
# Reproduce it completely. It is very heavy on the computation so the idea is to write a self contained script 
# that I will then run on HPC and that will save all the metrics in a CSV file. 
# I will then write another script to load that CSV file and produces the plot 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt  
from scipy.integrate import quad
from scipy.optimize import minimize

class BaseBayesianTimingModel:
    """
    Base class for Bayesian models, implements functions that are in common for the two models. 
    Implements:
    - f_bls : Bayesian least-squares estimator, t_e = E[ts | tm]
    - p_tm_given_ts : likelihood p(tm | ts, wm)
    - p_tp_given_te : likelihood p(tp | te, wp) of the path integration process
    - p_tp_given_ts : marginal p(tp | ts, wm, wp) = ∫ p(tp|te) p(tm|ts) dtm
    - neg_log_likelihood
    - fit (MLE) and simulate
    """
    def __init__(self, c= 0.5846, m= -0.1026, tm_lower= 0.0, tm_upper_factor= 2.0):
        """
        c, m: parameters of non uniform prior over ts: prior(ts) ∝ c + m*ts
        tm_lower: lower integration bound for tm
        tm_upper_factor: tm upper bound = tm_upper_factor * max(ts)
        """
        self.c = c
        self.m = m
        self.tm_lower = tm_lower
        self.tm_upper_factor = tm_upper_factor

    def meas_sd(self,ts,wm):
        """
        Measurement SD σ_m(ts) for given wm and ts.
        In different models this is wm * sqrt(ts) or wm * ts, will be overwritten
        tm given ts part, that is first internal representation tm given ts. 
        """
        raise NotImplementedError
    
    def prod_sd(self,te,wp):
        """
        Production SD σ_p(te) for given wp and te.
        In different models this is wp * sqrt(te) or wp * te.
        tp given te part, produced vector given bayesian estimate
        """
        raise NotImplementedError 
    
    def p_tm_given_ts(self,tm,ts,wm):
        """
        Mirrors Matlab Ptmts: p(tm | ts, wm) as a Gaussian centered at ts
        with SD = meas_sd(ts, wm). 
        ts,tm can be scalar or array.
        """
        ts = np.asarray(ts, dtype=float)
        tm = np.asarray(tm, dtype=float)
        sigma = self.meas_sd(ts,wm)
        var= sigma **2
        norm = 1.0 / np.sqrt(2.0*np.pi*var)
        return norm*np.exp(-0.5*(tm-ts)**2/var)
    
    def p_tp_given_te(self,tp,te,wp):
        """
        Mirrors Matlab Ptpte: p(tp | te, wp), path integration, as Gaussian centered at te
        with SD = prod_sd(te, wp).
        tp,te can be both scalars or 1D numpy arrays.
        """
        te = np.asarray(te, dtype=float)
        tp = np.asarray(tp, dtype=float)
        sigma = self.prod_sd(te,wp)
        var= sigma ** 2
        norm = 1.0 /np.sqrt(2.0*np.pi*var)
        return norm * np.exp(-0.5*(tp-te)**2/var)
    
    # Bayesian estimator 

    def f_bls(self, tm,wm,ts_min,ts_max,n_grid = 200):
        """
        Compute BLS estimate te for given tm (can be array) and wm.
        Uses discrete approximation to the integrals over ts.
        tm: scalar or 1D array
        wm: scalar
        ts_min, ts_max: support of true intervals (same as in Matlab: min(ts), max(ts))
        """
        tm = np.atleast_1d(tm).astype(float)
        ts_grid = np.linspace(ts_min, ts_max, n_grid)
        dt = ts_grid[1] - ts_grid[0]


       # Prior(ts) ∝ c + m * ts
        prior = self.c + self.m * ts_grid
       # avoid negative/ zero  prior
        prior = np.maximum(prior, 1e-9)

        te_vals = np.empty_like(tm, dtype=float)

        for i, tm_i in enumerate(tm):
            lik_ts = self.p_tm_given_ts(tm=tm_i, ts=ts_grid, wm=wm)
            weight = prior * lik_ts

            numer = np.sum(weight * ts_grid) * dt
            denom = np.sum(weight) * dt

            if denom <= 1e-15 or not np.isfinite(denom):
            # Fallback: if posterior is numerically degenerate, approximate
            # te by tm 
                te_vals[i] = tm_i 
            else:
                te_vals[i] = numer / denom
 
        # return scalar if input was scalar 
        if te_vals.size == 1:
            return te_vals[0]
        return te_vals

    #Marginal likelihood p(tp | ts, wm, wp)
    def p_tp_given_ts(self,tp,ts,wm,wp,ts_min,ts_max):
        """
        Compute p(tp | ts, wm, wp) by integrating over tm:
            ∫ p(tp | te(tm)) p(tm | ts) dtm

        tp, ts are scalars here (called once per trial in NLL).
        Computation is done from ts, computing implicitely te using the other previously defined functions. 
        """
        lower = self.tm_lower
        upper = self.tm_upper_factor * ts_max

        def integrand(tm_scalar):
            te = self.f_bls(tm=tm_scalar, wm=wm,
                            ts_min=ts_min, ts_max=ts_max)
            val = self.p_tp_given_te(tp=tp, te=te, wp=wp) * \
                    self.p_tm_given_ts(tm=tm_scalar, ts=ts, wm=wm)
            # replace NaN/inf with 0 to keep quad working 
            return np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
        
        val, _ = quad(integrand, lower, upper,
                      epsabs=1e-5, epsrel=1e-4, limit = 100)
        
        return val
    
    # Negative log likelihood (not yet finding the minimum)
    def neg_log_likelihood(self, params, ts, tp):
        """
        params: [wm, wp, offset]
        ts, tp: 1D arrays of same length
        """
        wm, wp, offset = params

        # Penalize invalid parameters
        if wm <= 0 or wp <= 0:
            return 1e6

        ts = np.asarray(ts, dtype=float)
        tp = np.asarray(tp, dtype=float)
        ts_min, ts_max = ts.min(), ts.max()

        ll = 0.0
        for ts_i, tp_i in zip(ts, tp):
            p = self.p_tp_given_ts(tp=tp_i - offset,
                                   ts=ts_i,
                                   wm=wm,
                                   wp=wp,
                                   ts_min=ts_min,
                                   ts_max=ts_max)
            # avoid log(0)
            if p <= 0.0 or not np.isfinite(p):
                p = 1e-12
            ll += np.log(p)

        return -ll
    
    # Fitting (MLE)
    def fit(self, ts, tp, w_init=(0.5, 0.5, 0.0)):
        """
        Fit parameters [wm, wp, offset] by minimizing negative log-likelihood.
        Mirrors Matlab fminsearch (Nelder–Mead).
        """
        ts = np.asarray(ts, dtype=float)
        tp = np.asarray(tp, dtype=float)

        def obj(params):
            return self.neg_log_likelihood(params, ts, tp)

        res = minimize(obj, x0=np.array(w_init, dtype=float),
                       method="Nelder-Mead")
        return res  # res.x are the MLEs
    
    # Generative part, simulate trials 
    def simulate(self, ts, wm, wp, offset, rng=None):
        """
        Simulate tm, te, tp for given ts and parameters.
        Mirrors the generative branches in the Matlab functions.
        """
        # The number of points simulated will be the same as ts the ground truth data I pass
        
        ts = np.asarray(ts, dtype=float)
        if rng is None:
            rng = np.random.default_rng()

        # Measurement stage
        sigma_m = self.meas_sd(ts, wm)
        tm = rng.normal(loc=ts, scale=sigma_m)

        # Bayesian estimator
        ts_min, ts_max = ts.min(), ts.max()
        te = self.f_bls(tm=tm, wm=wm, ts_min=ts_min, ts_max=ts_max)

        # Production stage
        sigma_p = self.prod_sd(te, wp)
        tp = rng.normal(loc=te, scale=sigma_p) + offset

        return tm, te, tp

class CounterModel(BaseBayesianTimingModel):
    """
    Reset --> mental navigation, subscalar model:
    - measurement SD: wm * sqrt(ts)
    - production SD:  wp * sqrt(te)
    """

    def meas_sd(self, ts, wm):
        ts = np.asarray(ts)
        return wm * np.sqrt(ts)

    def prod_sd(self, te, wp):
        te = np.asarray(te)
        return wp * np.sqrt(te)

class NonCounterModel(BaseBayesianTimingModel):
    """
    No-reset--> path integration, scalar model:
    - measurement SD: wm * ts
    - production SD:  wp * te
    """

    def meas_sd(self, ts, wm):
        ts = np.asarray(ts)
        return wm * ts

    def prod_sd(self, te, wp):
        te = np.asarray(te)
        return wp * te