import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize

# This file contains code to recreate bayesian models and functions to compute MSE and run simulations 
# The data produced in this file is then used to run model identifiability tests 


# Base Bayesian timing model + two subclasses

class BaseBayesianTimingModel:
    """
    Base class for Bayesian models, implements functions that are in common
    for the two models.

    Implements:
    - f_bls : Bayesian least-squares estimator, t_e = E[ts | tm]
    - p_tm_given_ts : likelihood p(tm | ts, wm)
    - p_tp_given_te : likelihood p(tp | te, wp) of the path integration process
    - p_tp_given_ts : marginal p(tp | ts, wm, wp) = ∫ p(tp|te) p(tm|ts) dtm
    - neg_log_likelihood
    - fit (MLE) and simulate
    """

    def __init__(self, c=0.5846, m=-0.1026, tm_lower=0.0, tm_upper_factor=2.0):
        self.c = c
        self.m = m
        self.tm_lower = tm_lower
        self.tm_upper_factor = tm_upper_factor

    # functions to be provided by subclasses 

    def meas_sd(self, ts, wm):
        raise NotImplementedError

    def prod_sd(self, te, wp):
        raise NotImplementedError

    #Likelihood

    def p_tm_given_ts(self, tm, ts, wm):
        """
        Likelihood p(tm | ts, wm). Gaussian centered at ts with
        variance sigma_m^2 = meas_sd(ts, wm)^2.
        """
        ts = np.asarray(ts, dtype=float)
        tm = np.asarray(tm, dtype=float)
        sigma = self.meas_sd(ts, wm)
        var = np.maximum(sigma ** 2, 1e-12)
        norm = 1.0 / np.sqrt(2.0 * np.pi * var)
        return norm * np.exp(-0.5 * (tm - ts) ** 2 / var)

    def p_tp_given_te(self, tp, te, wp):
        """
        Likelihood p(tp | te, wp). Gaussian centered at te with
        variance sigma_p^2 = prod_sd(te, wp)^2.
        """
        te = np.asarray(te, dtype=float)
        tp = np.asarray(tp, dtype=float)
        sigma = self.prod_sd(te, wp)
        var = np.maximum(sigma ** 2, 1e-12)
        norm = 1.0 / np.sqrt(2.0 * np.pi * var)
        return norm * np.exp(-0.5 * (tp - te) ** 2 / var)

    # Bayesian least squares estimator 

    def f_bls(self, tm, wm, ts_min, ts_max, n_grid=200):  
        """
        BLS estimator t_e = E[ts | tm].
        Prior(ts) ∝ c + m * ts (truncated to be non-negative).
        """
        tm = np.atleast_1d(tm).astype(float)
        ts_grid = np.linspace(ts_min, ts_max, n_grid)
        dt = ts_grid[1] - ts_grid[0]

        # Prior(ts) ∝ c + m * ts
        prior = self.c + self.m * ts_grid
        prior = np.maximum(prior, 1e-9)

        te_vals = np.empty_like(tm, dtype=float)

        for i, tm_i in enumerate(tm):
            lik_ts = self.p_tm_given_ts(tm=tm_i, ts=ts_grid, wm=wm)
            weight = prior * lik_ts

            numer = np.sum(weight * ts_grid) * dt
            denom = np.sum(weight) * dt

            if denom <= 1e-15 or not np.isfinite(denom):
                # Fallback: if posterior collapses or numeric issues,
                # just return tm_i itself.
                te_vals[i] = tm_i
            else:
                te_vals[i] = numer / denom

        if te_vals.size == 1:
            return te_vals[0]
        return te_vals

    # p(tp | ts) via numerical integration over tm 

    def p_tp_given_ts(self, tp, ts, wm, wp, ts_min, ts_max):
        """
        p(tp | ts, wm, wp) = ∫ p(tp | te(tm)) p(tm | ts) dtm
        where te(tm) = f_bls(tm).
        Integration bounds follow the MATLAB style: [tm_lower, tm_upper_factor * ts_max].
        """
        lower = self.tm_lower
        upper = self.tm_upper_factor * ts_max

        def integrand(tm_scalar):
            te = self.f_bls(tm=tm_scalar, wm=wm,
                            ts_min=ts_min, ts_max=ts_max)
            val = self.p_tp_given_te(tp=tp, te=te, wp=wp) * \
                  self.p_tm_given_ts(tm=tm_scalar, ts=ts, wm=wm)
            # Avoid NaN/inf pollution in quad
            return np.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)

        val, _ = quad(integrand, lower, upper,
                      epsabs=1e-5, epsrel=1e-4, limit=100)
        return float(val)

    # log-likelihood, fitting, and simulation 

    def neg_log_likelihood(self, params, ts, tp):
        wm, wp, offset = params

        # Keep w's positive
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
            if p <= 0.0 or not np.isfinite(p):
                p = 1e-12
            ll += np.log(p)

        return -ll

    def fit(self, ts, tp, w_init=(0.5, 0.5, 0.0)):
        ts = np.asarray(ts, dtype=float)
        tp = np.asarray(tp, dtype=float)

        def obj(params):
            return self.neg_log_likelihood(params, ts, tp)

        res = minimize(obj, x0=np.array(w_init, dtype=float),
                       method="Nelder-Mead")
        return res

    def simulate(self, ts, wm, wp, offset, rng=None):
        """
        Simulate tm, te, tp from this model.
        """
        ts = np.asarray(ts, dtype=float)
        if rng is None:
            rng = np.random.default_rng()

        # Measurement noise
        sigma_m = self.meas_sd(ts, wm)
        tm = rng.normal(loc=ts, scale=sigma_m)

        # BLS estimate
        ts_min, ts_max = ts.min(), ts.max()
        te = self.f_bls(tm=tm, wm=wm, ts_min=ts_min, ts_max=ts_max)

        # Production noise
        sigma_p = self.prod_sd(te, wp)
        tp = rng.normal(loc=te, scale=sigma_p) + offset

        return tm, te, tp


class CounterModel(BaseBayesianTimingModel):
    """Reset (mental navigation) model: subscalar variability."""
    def meas_sd(self, ts, wm):
        ts = np.asarray(ts, dtype=float)
        return wm * np.sqrt(ts)

    def prod_sd(self, te, wp):
        te = np.asarray(te, dtype=float)
        return wp * np.sqrt(te)


class NonCounterModel(BaseBayesianTimingModel):
    """No-reset (path integration) model: scalar variability."""
    def meas_sd(self, ts, wm):
        ts = np.asarray(ts, dtype=float)
        return wm * ts

    def prod_sd(self, te, wp):
        te = np.asarray(te, dtype=float)
        return wp * te


# MSE(bias+variance) analogue of MATLAB mdl.mse_bias_var

def mse_bias_var_like(ts, tp_data, tp_model):
    """
    Analogue of mdl.mse_bias_var in MATLAB:
    For each unique ts:
      - compute bias_data = mean(tp_data(ts))  - ts
      - compute bias_model = mean(tp_model(ts)) - ts
      - compute var_data, var_model (sample variance across trials)
    Then:
      score_data  = sum_t (bias_data^2  + var_data)
      score_model = sum_t (bias_model^2 + var_model)
    Return |score_data - score_model|.
    """
    ts = np.asarray(ts, dtype=float)
    tp_data = np.asarray(tp_data, dtype=float)
    tp_model = np.asarray(tp_model, dtype=float)

    biases_data, biases_model = [], []
    vars_data, vars_model = [], []

    for t in np.unique(ts):
        mask = (ts == t)
        d = tp_data[mask]
        m = tp_model[mask]

        # guard against degenerate cases
        if d.size < 2 or m.size < 2:
            continue

        biases_data.append(d.mean() - t)
        biases_model.append(m.mean() - t)
        vars_data.append(d.var(ddof=1))
        vars_model.append(m.var(ddof=1))

    biases_data = np.array(biases_data)
    biases_model = np.array(biases_model)
    vars_data = np.array(vars_data)
    vars_model = np.array(vars_model)

    score_data = np.sum(biases_data ** 2 + vars_data)
    score_model = np.sum(biases_model ** 2 + vars_model)

    return float(np.abs(score_data - score_model))


# Main simulation, with ONE generative model, TWO fitted models + timing

def run_identifiability(n_sims, seed, out_path, gen_model_name):
    """
    n_sims          : number of Monte Carlo repetitions (bb in MATLAB)
    seed            : RNG seed
    out_path        : CSV to save results
    gen_model_name  : 'counting' (CounterModel) or 'timing' (NonCounterModel)
    """

    out_path = Path(out_path)
    rng = np.random.default_rng(seed)

    # Same ts_ as MATLAB: repmat(.65:.65:3.25, [1 100])
    base_times = np.arange(0.65, 3.25 + 1e-9, 0.65)
    #ts_true = np.tile(base_times, 100), 10 trials instead of 100 to make it lighter in this test setting 
    ts_true = np.tile(base_times,100)
    n_trials = len(ts_true)

    # Same modelparams as MATLAB: [0.15 0.2 0.01]
    wm_true, wp_true, offset_true = 0.15, 0.20, 0.01
    k_params = 3  # wm, wp, offset

    # Choose generative model
    if gen_model_name.lower() == "counting":
        GenCls = CounterModel
        gen_label = "counting"
    elif gen_model_name.lower() == "timing":
        GenCls = NonCounterModel
        gen_label = "timing"
    else:
        raise ValueError("gen_model_name must be 'counting' or 'timing'")

    # Fitted models (we still fit BOTH)
    fit_models = [
        ("counting", CounterModel),
        ("timing", NonCounterModel),
    ]

    rows = []

    print(f"Ground-truth generative model: {gen_label}")
    print(f"Number of simulations: {n_sims}")
    print(f"Trials per simulation: {n_trials}")
    print("-" * 60)

    t_start = time.perf_counter()

    for bb in range(n_sims):
        # generate synthetic dataset with the chosen ground-truth model 
        gen_model = GenCls()
        tm_gen, te_gen, tp_gen = gen_model.simulate(
            ts_true, wm_true, wp_true, offset_true, rng=rng
        )

        # fit each candidate model to the SAME synthetic data 
        for fit_name, FitCls in fit_models:
            fit_model = FitCls()

            # Fit by MLE (Nelder–Mead)
            res = fit_model.fit(ts_true, tp_gen, w_init=(0.5, 0.5, 0.0))
            wm_hat, wp_hat, offset_hat = res.x

            # Negative log-likelihood at optimum
            nll = fit_model.neg_log_likelihood(res.x, ts_true, tp_gen)

            # BIC = log(N)*k + 2*negloglik    (approx to what MATLAB does)
            bic_val = np.log(n_trials) * k_params + 2.0 * nll

            # Simulate from fitted model to compute mse_bias_var-like metric
            _, _, tp_fit_gen = fit_model.simulate(
                ts_true, wm_hat, wp_hat, offset_hat, rng=rng
            )
            mse_val = mse_bias_var_like(ts_true, tp_gen, tp_fit_gen)

            rows.append(
                dict(
                    sim_id=bb,
                    generator_model=gen_label,   # fixed, single ground-truth
                    fitted_model=fit_name,       # 'counting' or 'timing'
                    wm_true=wm_true,
                    wp_true=wp_true,
                    offset_true=offset_true,
                    wm_hat=wm_hat,
                    wp_hat=wp_hat,
                    offset_hat=offset_hat,
                    negloglik=nll,
                    bic=bic_val,
                    mse_bias_var=mse_val,
                    n_trials=n_trials,
                    success=res.success,
                    fun=res.fun,
                )
            )

        # progress + timing 
        if (bb + 1) == 1 or (bb + 1) % max(1, n_sims // 10) == 0:
            elapsed = time.perf_counter() - t_start
            print(
                f"[{elapsed:7.2f} s] Finished simulation {bb + 1}/{n_sims}",
                flush=True,
            )

    total_time = time.perf_counter() - t_start
    print("-" * 60)
    print(f"Total time: {total_time:7.2f} s")
    print(f"Average per simulation: {total_time / max(1, n_sims):7.2f} s")

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved identifiability results to {out_path.resolve()}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Model identifiability simulation (BLS models)."
    )
    p.add_argument(
        "--n-sims",
        type=int,
        default=100,
        help="Number of Monte Carlo simulations (bb; default: 100)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="identifiability_results.csv",
        help="Output CSV path (default: identifiability_results.csv)",
    )
    p.add_argument(
        "--gen-model",
        type=str,
        default="counting",
        choices=["counting", "timing"],
        help="Ground-truth generative model (default: counting)",
    )
    return p.parse_args()

if __name__ == "__main__":
    n_sims= 5
    seed = 0
    out_path = "counting_sim_3.csv"
    gen_model_name = "counting"   # or "timing" for the other model
    

    run_identifiability(
        n_sims=n_sims,
        seed=seed,
        out_path=out_path,
        gen_model_name=gen_model_name,
    )
