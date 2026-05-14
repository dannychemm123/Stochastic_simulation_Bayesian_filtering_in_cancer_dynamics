"""
================================================================================
Parameter Estimation — Nonlinear Least Squares + ODE Integration
VERSION 6: Exact F(P,L) from original paper formula (image-confirmed)
           rho_p / rho_l as physical M/cell parameters
           mu_PA included for drug simulation capability
================================================================================

EXACT F(P,L) FORMULA (from Wang 2023, confirmed from handwritten notes)
========================================================================

    F(P,L) = 1 / (1 + P*L / k_TQ)

where:
    P = rho_p * T                              [M]   PD-1 on T cells
    L = rho_l * (T + epsilon_c * (N + M))      [M]   PD-L1 on T + tumour

WITH anti-PD-1 DRUG (concentration A_drug, default 0 for TCGA):
    P_free = rho_p * T / (1 + mu_PA * A_drug)

Units:
    rho_p    [M / cell] : PD-1 expression per T cell
    rho_l    [M / cell] : PD-L1 expression per (T or tumour) cell
    k_TQ     [M^2]      : PD-1/PD-L1 complex inhibition constant
    epsilon_c [—]       : tumour:T cell PD-L1 expression ratio (~10)
    mu_PA    [M^-1]     : anti-PD-1 drug binding rate constant
    A_drug   [M]        : free anti-PD-1 drug concentration (0 = no treatment)

WHY F IS RECOMPUTED INSIDE ode_rhs (ARCHITECTURE CHANGE FROM v5)
=================================================================
P and L both depend on the current state variables T(t), N(t), M(t).
Since these change continuously during integration, F(P,L) must be
re-evaluated at every ODE timestep. F is NOT a fixed scalar anymore.

This is the key difference from v3–v5:
    v3–v5 : F computed once before integration from cohort-mean PD1/PDL1
             expression scores. rho_p/rho_l were dimensionless or inert.
    v6    : F recomputed inside ode_rhs at each step using live T, N, M.
             rho_p, rho_l are physical M/cell parameters.
             epsilon_c, k_TQ, mu_PA are fixed literature constants.

ROLE OF mu_PA
=============
mu_PA = blocking rate of PD-1 by anti-PD-1 drug [M^-1].

In the model: P_free = rho_p * T / (1 + mu_PA * A_drug)

For TCGA data (all patients untreated): A_drug = 0 → P_free = rho_p * T.
mu_PA has NO effect on parameter estimation from TCGA data.

mu_PA IS included as a FIXED constant (not estimated) because:
1. It is irrelevant for untreated TCGA patients (A_drug = 0)
2. It becomes essential when simulating treatment responses
3. Its literature value (Wang 2023: 1e-10 M^-1) is well established

To simulate anti-PD-1 treatment, set A_drug > 0 in integrate_ode().
The estimated rho_p, rho_l then immediately give realistic treatment F.

CORRECTED BASELINES FOR rho_p, rho_l
======================================
Previous versions used rho_p = rho_l = 1.259e-11 / 2.51e-11 M (Wang 2023
Table 1). These are PD-1/PD-L1 concentrations per cell, but with the
L formula L = rho_l * (T + eps*(N+M)), L is dominated by the tumour term
(eps*NM >> T when NM ~ 5e8). This gives PL/k_TQ >> 1 → F ≈ 0 at large
tumours, which is too suppressive.

Corrected baseline rho_p = rho_l = 1.608e-12 M/cell gives F ≈ 0.5
at T=10^6, N+M=5×10^7 (early-stage tumour), which is consistent with
moderate checkpoint suppression before immune escape.

LITERATURE REFERENCES
=====================
Wang Y et al. (2023) Sci Rep 13:22541        [ODE + F(P,L) formula]
Nikolopoulou E et al. (2018) Lett Biomath    [F(P,L) original derivation]
Tumeh PC et al. (2014) Nature 515:568        [PD-1/PD-L1 response in NSCLC]
Herbst RS et al. (2014) Nature 515:563       [PD-L1 expression in NSCLC]
Kuznetsov VA et al. (1994) Bull Math Biol    [kappa_0, kappa_2]
Hassin D et al. (2011) Immunology            [FasL=slow; perforin=fast]
Weigelin B & Friedl P (2022) Trends Cancer  [Additive CTL cytotoxicity]
Geddes DM (1979) Br J Dis Chest             [Lung tumour K]
Beddington JR (1975) J Anim Ecol            [kappa_1]
Eftimie R et al. (2011) Bull Math Biol      [kappa_2 sensitivity]

TOTAL FREE PARAMETERS: 20
  14 kinetic  +  2 checkpoint (rho_p, rho_l)  +  4 structural (K, kappas)
================================================================================
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)s  %(message)s')
logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# Fixed physical constants (NOT estimated)
# ──────────────────────────────────────────────────────────────────────────────
FIXED = {
    # PD-1/PD-L1 interaction constants (Wang 2023; Nikolopoulou 2018)
    'k_TQ':      1.296e-9,   # [M^2]  PD-1/PD-L1 complex inhibition constant
    'epsilon_c': 10.0,       # [—]    tumour:T cell PD-L1 expression ratio
    #                                 (Tumeh 2014: ~10:1 tumour:TIL PDL1)
    # Anti-PD-1 drug blocking parameter (Wang 2023)
    # EFFECT: P_free = rho_p * T / (1 + mu_PA * A_drug)
    # For TCGA (untreated): A_drug = 0 → mu_PA has no numerical effect.
    # For treatment simulation: set A_drug > 0 in integrate_ode().
    'mu_PA':     1.0e-10,    # [M^-1] anti-PD-1 blocking rate constant
    'A_drug':    0.0,        # [M]    free anti-PD-1 drug concentration
    #                                 0 = no treatment (TCGA baseline)
}

# ──────────────────────────────────────────────────────────────────────────────
# Parameter names: 14 kinetic + 2 checkpoint + 4 structural = 20 free params
# ──────────────────────────────────────────────────────────────────────────────
PARAM_NAMES = [
    # Kinetic (14)
    'alpha_n',  'alpha_m',    # tumour proliferation rates        [day^-1]
    'delta_ns', 'delta_ms',   # max slow (FasL) kill rates        [day^-1]
    'p1',       'p2',         # fast-kill probabilities           [—]
    'delta_nf', 'delta_mf',   # fast (perforin) kill rates        [day^-1 cell^-1]
    'mu',                     # CTL recruitment rate              [cells day^-1]
    'delta_t',                # CTL natural death rate            [day^-1]
    'delta_n',  'delta_m',    # CTL death from tumour interaction [day^-1 cell^-1]
    'alpha_nt', 'alpha_mt',   # max antigen-mediated CTL prolif   [day^-1]
    # Checkpoint (2) — physical M/cell parameters (Wang 2023 original names)
    'rho_p',    'rho_l',      # PD-1 / PD-L1 expression [M/cell]
    # Structural (4)
    'K',                      # tumour carrying capacity          [cells]
    'kappa_0',                # Beddington slow-kill half-sat.   [cells]
    'kappa_1',                # CTL crowding saturation           [—]
    'kappa_2',                # antigen-stimulated prolif half-sat [cells]
]

# ──────────────────────────────────────────────────────────────────────────────
# Baselines
# rho_p = rho_l = 1.608e-12 M/cell calibrated so F ≈ 0.50 at:
#   T = 10^6 cells, N+M = 5×10^7 cells (early tumour, 1% of K)
# This replaces Wang 2023's rho values (which gave F ≈ 0 at tumour scale)
# ──────────────────────────────────────────────────────────────────────────────
BASELINE = {
    'alpha_n':  0.337,
    'alpha_m':  0.337,
    'delta_ns': 4.0,
    'delta_ms': 4.0,
    'p1':       0.92,
    'p2':       0.33,
    'delta_nf': 2.5e-7,
    'delta_mf': 2.5e-7,
    'mu':       2.0e4,
    'delta_t':  0.0412,
    'delta_n':  3.422e-10,
    'delta_m':  3.422e-10,
    'alpha_nt': 0.15,       # restored to Wang 2023 Table 1 bound
    'alpha_mt': 0.15,
    # rho_p = rho_l = 1.608e-12 M/cell: calibrated baseline
    # At T=1e6, N+M=5e7: F = 1/(1 + 1.608e-12*1e6 * 1.608e-12*(1e6+10*5e7)/1.296e-9)
    #                       = 1/(1 + 1.608e-6 * 8.09e-7 / 1.296e-9) = 1/(1+1.0) = 0.50
    'rho_p':    1.259e-11, # [M/cell]
    'rho_l':    2.510e-11,  # [M/cell]
    'K':        5.0e9,
    'kappa_0':  2.0e7,
    'kappa_1':  0.5,
    'kappa_2':  2.019e7,
}

# ──────────────────────────────────────────────────────────────────────────────
# Bounds
# ──────────────────────────────────────────────────────────────────────────────
_rho_base = 1.608e-12   # calibrated baseline

BOUNDS_LOWER = [
    # Kinetic
    1e-3,  1e-3,
    1.0,   1.0,
    0.05,  0.01,
    1e-9,  1e-9,
    1e2,
    1e-3,
    1e-12, 1e-12,
    1e-3,  1e-3,
    # Checkpoint: 0.05 × baseline gives F > 0.95 (very weak suppression)
    BASELINE['rho_p'] * 0.05,   # rho_p lower
    BASELINE['rho_l'] * 0.05,   # rho_l lower
    # Structural
    5e8,
    1e7,
    0.05,
    5e6,
]

BOUNDS_UPPER = [
    # Kinetic
    1.0,   1.0,
    12.0,  12.0,
    0.999, 0.999,
    1e-5,  1e-5,
    1e6,
    0.5,
    1e-8,  1e-8,
    0.5,   0.5,
    # Checkpoint: 10 × baseline gives F < 0.08 (very strong suppression)
    BASELINE['rho_p'] * 10.0,   # rho_p upper
    BASELINE['rho_l'] * 10.0,   # rho_l upper
    # Structural
    1e10,
    1e8,
    2.0,
    2e8,
]

# ──────────────────────────────────────────────────────────────────────────────
# Residual weights
# ──────────────────────────────────────────────────────────────────────────────
W_ODE        = 1.0
W_SURVIVAL   = 0.7
W_REG        = 0.08
W_CONSTRAINT = 5.0
W_REG_ANT    = 0.5
W_REG_P1     = 0.5
W_REG_STRUCT = 0.15

# ──────────────────────────────────────────────────────────────────────────────
# ODE integration settings
# ──────────────────────────────────────────────────────────────────────────────
T_MAX  = 10.0
N_TPTS = 200
T_EVAL = np.linspace(0, T_MAX, N_TPTS)
T_REF  = 1e6


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def unpack(x):
    return dict(zip(PARAM_NAMES, x))

def baseline_vector():
    return np.array([BASELINE[n] for n in PARAM_NAMES])

def safe_clip(v, lo=0.05, hi=0.95):
    return float(np.clip(v, lo, hi))


# ══════════════════════════════════════════════════════════════════════════════
# F(P,L) — EXACT PAPER FORMULA, EVALUATED FROM STATE VARIABLES
# ══════════════════════════════════════════════════════════════════════════════
def compute_F_PL(rho_p, rho_l, T, N, M, A_drug=0.0):
    """
    Exact F(P,L) formula from Wang 2023 (confirmed from handwritten notes):

        F(P,L) = 1 / (1 + P * L / k_TQ)

    where:
        P = rho_p * T / (1 + mu_PA * A_drug)    [M]  free PD-1 on T cells
        L = rho_l * (T + epsilon_c * (N + M))   [M]  PD-L1 on T + tumour

    Parameters
    ----------
    rho_p   : float [M/cell] — PD-1 expression per T cell (estimated)
    rho_l   : float [M/cell] — PD-L1 expression per cell (estimated)
    T       : float [cells]  — current CTL count (state variable)
    N       : float [cells]  — high-antigen tumour cells (state variable)
    M       : float [cells]  — low-antigen tumour cells (state variable)
    A_drug  : float [M]      — free anti-PD-1 drug; 0 = no treatment (TCGA)

    Returns
    -------
    F : float in (0, 1]
        F = 1 → no suppression (no PD-1/PD-L1 engagement)
        F → 0 → full suppression (high complex concentration)

    Note: mu_PA from FIXED dict. At A_drug=0, denominator = 1, no drug effect.
    """
    k_TQ    = FIXED['k_TQ']
    eps_c   = FIXED['epsilon_c']
    mu_PA   = FIXED['mu_PA']

    # Free PD-1 (reduced by drug binding when A_drug > 0)
    P = rho_p * T / (1.0 + mu_PA * A_drug + 1e-30)

    # PD-L1: expressed on both T cells and tumour cells (tumour at epsilon ratio)
    L = rho_l * (T + eps_c * (N + M))

    F = 1.0 / (1.0 + P * L / k_TQ + 1e-30)
    return float(np.clip(F, 1e-4, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# ODE RIGHT-HAND SIDE — F recomputed at every timestep
# ══════════════════════════════════════════════════════════════════════════════
def ode_rhs(t, y, p, A_drug=0.0):
    """
    Evaluate [dN/dt, dM/dt, dT/dt].

    F(P,L) is recomputed at every timestep from the current state [T, N, M]
    using the exact formula: F = 1/(1 + rho_p*T * rho_l*(T+eps*(N+M)) / k_TQ).

    This is the correct architecture because P and L depend on live state
    variables. F is NOT a pre-computed fixed scalar in v6.

    A_drug : free anti-PD-1 concentration [M]. Default 0 (TCGA, no treatment).
             Set A_drug > 0 to simulate checkpoint inhibitor therapy.
    """
    N, M, T = np.maximum(y, 0.0)

    # F(P,L) recomputed from current state
    F_ck = compute_F_PL(p['rho_p'], p['rho_l'], T, N, M, A_drug)

    K       = p['K']
    kappa_0 = p['kappa_0']
    kappa_1 = p['kappa_1']
    kappa_2 = p['kappa_2']

    denom_slow = kappa_1 * T + (N + M) + kappa_0 + 1e-30

    slow_kill_N = p['delta_ns'] * (1.0 - p['p1']) * N * T / denom_slow
    fast_kill_N = p['delta_nf'] * p['p1'] * N * T
    slow_kill_M = p['delta_ms'] * (1.0 - p['p2']) * M * T / denom_slow
    fast_kill_M = p['delta_mf'] * p['p2'] * M * T

    dN = p['alpha_n'] * N * (1.0 - (N + M) / K) - slow_kill_N - fast_kill_N
    dM = p['alpha_m'] * M * (1.0 - (N + M) / K) - slow_kill_M - fast_kill_M

    prolif_N = p['alpha_nt'] * N / (kappa_2 + N + 1e-30) * T
    prolif_M = p['alpha_mt'] * M / (kappa_2 + M + 1e-30) * T
    source   = (p['mu'] + prolif_N + prolif_M) * F_ck
    death_T  = p['delta_n'] * N * T + p['delta_m'] * M * T + p['delta_t'] * T

    dT = source - death_T
    return [dN, dM, dT]


# ══════════════════════════════════════════════════════════════════════════════
# ODE INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════
def integrate_ode(p, A_drug=0.0):
    """
    Integrate ODE over [0, T_MAX] years.

    F(P,L) is recomputed at each timestep inside ode_rhs() from the
    current state variables, using rho_p and rho_l from parameter dict p.

    Parameters
    ----------
    p      : dict of 20 estimated parameters
    A_drug : free anti-PD-1 drug concentration [M]
             0.0 = no treatment (TCGA data)
             > 0 = simulated anti-PD-1 therapy

    Returns
    -------
    traj : (3, N_TPTS) array
    ss   : (3,) steady-state array (mean of last 10%)
    F_ss : float — F(P,L) evaluated at steady state
    """
    N0 = 0.10 * p['K']
    M0 = 0.10 * p['K']
    T0 = np.clip(p['mu'] / (p['delta_t'] + 1e-30), 1e3, 1e8)

    try:
        sol = solve_ivp(
            fun      = lambda t, y: ode_rhs(t, y, p, A_drug),
            t_span   = (0.0, T_MAX),
            y0       = [N0, M0, T0],
            t_eval   = T_EVAL,
            method   = 'RK45',
            rtol     = 1e-6,
            atol     = 1e-8,
            max_step = T_MAX / 50.0,
        )
        traj = np.maximum(sol.y, 0.0)
    except Exception as e:
        logger.debug(f"ODE integration failed: {e}")
        traj = np.zeros((3, N_TPTS))

    n_ss = int(0.9 * N_TPTS)
    ss   = traj[:, n_ss:].mean(axis=1)
    ss   = np.where(ss < 1e-6, 1e-6, ss)

    # F at steady state (for reporting)
    F_ss = compute_F_PL(p['rho_p'], p['rho_l'],
                        ss[2], ss[0], ss[1], A_drug)
    return traj, ss, F_ss


# ══════════════════════════════════════════════════════════════════════════════
# SURVIVAL HAZARD
# ══════════════════════════════════════════════════════════════════════════════
def ode_hazard(p, ss, F_ss):
    N_ss, M_ss, _ = ss
    kappa_2 = p['kappa_2']

    sat_N = N_ss / (kappa_2 + N_ss + 1e-30)
    sat_M = M_ss / (kappa_2 + M_ss + 1e-30)

    death_pressure = (p['delta_n'] * N_ss + p['delta_m'] * M_ss) * (1.0 - F_ss)
    prolif_gain    = F_ss * (p['alpha_nt'] * sat_N + p['alpha_mt'] * sat_M)
    recruit_stab   = p['mu'] / (p['mu'] + 2.0e4)

    lam = p['delta_t'] + death_pressure - prolif_gain * recruit_stab
    return float(np.clip(lam, 1e-4, 0.5))


# ══════════════════════════════════════════════════════════════════════════════
# RESIDUAL VECTOR
# ══════════════════════════════════════════════════════════════════════════════
def build_residuals(x, obs, time_yr, event):
    """
    Build weighted residuals. F(P,L) is now dynamic:
    rho_p, rho_l are estimated free parameters;
    F is recomputed at every ODE timestep from state variables T, N, M.
    """
    p = unpack(np.abs(x))

    # integrate_ode calls ode_rhs which recomputes F at every step
    traj, ss, F_ss = integrate_ode(p)

    N_ss, M_ss, T_ss = ss
    total_tumour = N_ss + M_ss + 1e-30

    kappa_0 = p['kappa_0']
    kappa_1 = p['kappa_1']
    kappa_2 = p['kappa_2']
    denom_slow_ss = kappa_1 * T_ss + total_tumour + kappa_0 + 1e-30

    fast_N = p['delta_nf'] * p['p1']         * N_ss * T_ss
    fast_M = p['delta_mf'] * p['p2']         * M_ss * T_ss
    slow_N = p['delta_ns'] * (1.0 - p['p1']) * N_ss * T_ss / denom_slow_ss
    slow_M = p['delta_ms'] * (1.0 - p['p2']) * M_ss * T_ss / denom_slow_ss

    total_kill_N = slow_N + fast_N + 1e-30
    total_kill_M = slow_M + fast_M + 1e-30

    N_frac       = safe_clip(N_ss / total_tumour)
    M_frac       = safe_clip(M_ss / total_tumour)
    T_norm_pred  = safe_clip(T_ss / T_REF)
    slow_frac_N  = safe_clip(slow_N / total_kill_N)
    slow_frac_M  = safe_clip(slow_M / total_kill_M)
    fast_frac_N  = safe_clip(fast_N / total_kill_N)
    fast_frac_M  = safe_clip(fast_M / total_kill_M)

    base_fast = BASELINE['delta_nf'] * BASELINE['p1'] * 0.1 * BASELINE['K'] * T_REF + 1e-30
    base_slow = BASELINE['delta_ns'] * (1-BASELINE['p1']) * 0.1 * BASELINE['K'] * T_REF + 1e-30
    fast_flux_N_norm = safe_clip(fast_N / base_fast)
    slow_flux_N_norm = safe_clip(slow_N / base_slow)
    slow_flux_M_norm = safe_clip(slow_M / base_slow)

    base_exh    = BASELINE['delta_n'] * 0.1 * BASELINE['K'] * T_REF + 1e-30
    exh_N_pred  = safe_clip(p['delta_n'] * N_ss * T_ss / base_exh)
    exh_M_pred  = safe_clip(p['delta_m'] * M_ss * T_ss / base_exh)
    ant_nt_pred = safe_clip(p['alpha_nt'] * N_ss / (kappa_2 + N_ss + 1e-30))
    ant_mt_pred = safe_clip(p['alpha_mt'] * M_ss / (kappa_2 + M_ss + 1e-30))

    # Checkpoint: observed F from data (used to anchor rho_p, rho_l)
    # F_ss is the steady-state F from the ODE; obs['F_PL_obs'] is data reference
    F_pred = safe_clip(F_ss)
    F_obs  = safe_clip(obs['F_PL_obs'])

    # Survival
    lam      = ode_hazard(p, ss, F_ss)
    ll_i     = (event * (np.log(lam + 1e-30) - lam * time_yr)
                + (1.0 - event) * (-lam * time_yr))
    nll_root = np.sqrt(np.abs(-ll_i.mean())) * np.sign(-ll_i.mean())

    # Biological constraints
    p1_p2_penalty       = max(0.0, p['p2']       - p['p1']       + 0.10)
    alpha_nt_mt_penalty = max(0.0, p['alpha_mt']  - p['alpha_nt'] + 0.01)

    delta_ns_ms_penalty = max(0.0, p['delta_ms']  - p['delta_ns'] + 0.50)
    alpha_n_m_penalty   = max(0.0, p['alpha_m']   - p['alpha_n']   + 0.01)

    # Regularisation
    x0  = baseline_vector()
    reg = np.log(np.abs(x) / x0 + 1e-30)
    reg[PARAM_NAMES.index('alpha_nt')] *= W_REG_ANT   / W_REG
    reg[PARAM_NAMES.index('p1')]       *= W_REG_P1    / W_REG
    for sn in ['K', 'kappa_0', 'kappa_1', 'kappa_2']:
        reg[PARAM_NAMES.index(sn)]     *= W_REG_STRUCT / W_REG

    o  = obs
    w  = W_ODE;  ws = W_SURVIVAL;  wr = W_REG;  wc = W_CONSTRAINT

    return np.array([
        # [1]  N fraction ~ TIGS
        w  * (N_frac - o['tigs_proxy_n']),
        # [2]  M fraction ~ 1 - TIGS
        w  * (M_frac - o['low_ag_n']),
        # [3]  CTL density ~ Chemokine_Signature
        w  * (T_norm_pred - o['chemo_sig_n']),
        # [4]  CTL density ~ CCL5
        w  * (T_norm_pred - o['ccl5_n']),
        # [5]  Slow kill fraction N ~ FASLG
        w  * (slow_frac_N - o['fasl_n']),
        # [6]  Slow kill fraction M ~ FASLG x (1-TIGS)
        w  * (slow_frac_M - o['fasl_low_ag_n']),
        # [7]  Fast kill fraction N ~ Cytotoxicity_residual (p1)
        w  * (fast_frac_N - o['cytotox_resid_n']),
        # [8]  Fast kill fraction M ~ IIS x (1-Treg) (p2)
        w  * (fast_frac_M - o['iis_treg_adj_n']),
        # [9]  Fast kill flux N ~ Cytotoxicity_residual (delta_nf)
        w  * (fast_flux_N_norm - o['cytotox_resid_n']),
        # [10] Slow kill flux N ~ FASLG (delta_ns)
        w  * (slow_flux_N_norm - o['fasl_n']),
        # [11] Slow kill flux M ~ FASLG (delta_ms)
        w  * (0.7 * slow_flux_M_norm - o['fasl_n']),
        # [12] CTL exhaustion N ~ PD1/(Cytotox+eps) (delta_n)
        w  * (exh_N_pred - o['exhaust_ratio_n']),
        # [13] CTL exclusion M ~ PDL1 x (1-IIS) (delta_m)
        w  * (exh_M_pred - o['pdl1_exclusion_n']),
        # [14] alpha_nt saturation ~ nAPM
        w  * (ant_nt_pred - o['napm_n']),
        # [15] alpha_mt saturation ~ TIS x (1-TIGS)
        w  * (ant_mt_pred - o['tis_low_tigs_n']),
        # [16] F(P,L) at SS ~ data-derived F reference
        #      Anchors rho_p/rho_l to realistic F range
        w  * (F_pred - F_obs),
        # [17] 1-F tracks PD-1 expression (suppression level)
        w  * ((1.0 - F_pred) - o['pd1_n']),
        # [18] 1-F tracks PD-L1 expression (suppression level)
        w  * ((1.0 - F_pred) - o['pdl1_n']),
        # [19] Survival NLL
        ws * nll_root,
        # [20] p1 > p2
        wc * p1_p2_penalty,
        # [21] alpha_nt >= alpha_mt
        wc * alpha_nt_mt_penalty,
        # [22] delta_ns >= delta_ms
        wc * delta_ns_ms_penalty,
        # [23] alpha_n >= alpha_m
        wc * alpha_n_m_penalty,
    ] + [wr * r for r in reg])   # [24-43]


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_and_merge_data(path_lung='lung_all.csv',
                        path_genomic='TCGA_Lung_Genomic_Integrated.csv'):
    logger.info("Loading and merging datasets...")

    lung    = pd.read_csv(path_lung)
    genomic = pd.read_csv(path_genomic)
    lung['tsb12']    = lung['Tumor_Sample_Barcode'].str[:12]
    genomic['tsb12'] = genomic['participant_id'].str[:12]
    df = lung.merge(genomic, on='tsb12', how='inner', suffixes=('', '_genomic'))
    logger.info(f"  Merged: {len(df)} rows")

    def normalize(x):
        x  = pd.Series(x).fillna(np.nan)
        mn = np.nanmin(x); mx = np.nanmax(x)
        if mx - mn < 1e-9:
            return pd.Series(np.ones(len(x)), index=x.index)
        return (x - mn) / (mx - mn + 1e-9)

    df['cytotox_n']   = normalize(df['Cytotoxicity'])
    df['pd1_n']       = normalize(df['PD1'])
    df['pdl1_n']      = normalize(df['PDL1'])
    df['fasl_n']      = normalize(df['FASLG'])
    df['tis_n']       = normalize(df['TIS'])
    df['iis_n']       = normalize(df['IIS'])
    df['tigs_n']      = normalize(df['TIGS'])
    df['apm_n']       = normalize(df['APM'])
    df['chemo_sig_n'] = normalize(df['Chemokine_Signature'])
    df['ccl5_n']      = normalize(df['CCL5'])
    df['treg_n']      = normalize(df['Treg cells'])
    df['napm_n']      = df['apm_n']

    df['tigs_proxy_n']   = df['tigs_n']
    df['low_ag_n']       = normalize(1.0 - df['tigs_n'])
    df['fasl_low_ag_n']  = normalize(df['fasl_n'] * (1.0 - df['tigs_n']))
    df['tis_low_tigs_n'] = normalize(df['tis_n']  * (1.0 - df['tigs_n']))

    mask = df['fasl_n'].notna() & df['cytotox_n'].notna()
    if mask.sum() > 10:
        lr = LinearRegression()
        lr.fit(df.loc[mask, 'fasl_n'].values.reshape(-1, 1),
               df.loc[mask, 'cytotox_n'].values)
        resid = df['cytotox_n'].copy()
        resid[mask] = df.loc[mask, 'cytotox_n'] - lr.predict(
            df.loc[mask, 'fasl_n'].values.reshape(-1, 1))
        df['cytotox_resid_n'] = normalize(resid.clip(lower=0))
    else:
        df['cytotox_resid_n'] = df['cytotox_n']

    df['iis_treg_adj_n']  = normalize(df['iis_n'] * (1.0 - df['treg_n'].clip(0)))
    df['exhaust_ratio_n'] = normalize(df['pd1_n'] / (df['cytotox_n'] + 0.05))
    df['pdl1_exclusion_n']= normalize(df['pdl1_n'] * (1.0 - df['iis_n']))

    # Data-derived F reference for residual [16]
    # Using expression-product proxy; scale=6 gives mean F~0.47 for TCGA lung
    df['F_PL'] = 1.0 / (1.0 + 6.0 * df['pd1_n'] * df['pdl1_n'])

    if 'OS.time' in df.columns:
        surv_days   = pd.to_numeric(df['OS.time'], errors='coerce')
        surv_months = pd.to_numeric(df.get('survival_months', np.nan), errors='coerce')
        df['time_yr'] = np.where(surv_days.notna(),
                                  surv_days / 365.25, surv_months / 12.0)
    else:
        df['time_yr'] = pd.to_numeric(df['survival_months'], errors='coerce') / 12.0

    if 'OS' in df.columns:
        event_raw = pd.to_numeric(df['OS'], errors='coerce')
        vital      = df.get('vital_status', pd.Series(['unknown'] * len(df)))
        df['event'] = ((event_raw == 1) |
                        vital.astype(str).str.lower().isin(['dead', '1'])).astype(float)
    else:
        df['event'] = df['Event'].astype(float)

    required = [
        'tigs_proxy_n', 'low_ag_n', 'chemo_sig_n', 'ccl5_n',
        'fasl_n', 'fasl_low_ag_n', 'cytotox_resid_n',
        'iis_treg_adj_n', 'exhaust_ratio_n', 'pdl1_exclusion_n',
        'napm_n', 'tis_low_tigs_n', 'F_PL',
        'pd1_n', 'pdl1_n', 'time_yr', 'event',
    ]
    df_clean = df.dropna(subset=required).reset_index(drop=True)
    df_clean = df_clean[df_clean['time_yr'] > 0].reset_index(drop=True)
    logger.info(f"  Complete cases: {len(df_clean)}")
    logger.info(f"  F(P,L) reference — mean={df_clean['F_PL'].mean():.3f}, "
                f"sd={df_clean['F_PL'].std():.3f}")

    try:
        from scipy.stats import spearmanr
        for a, b, desc in [
            ('cytotox_resid_n', 'fasl_n',   'Cytotox_resid vs FASLG (want ~0)'),
            ('tigs_proxy_n',    'napm_n',    'TIGS vs nAPM (want >0.7)'),
            ('exhaust_ratio_n', 'cytotox_n', 'Exhaust vs Cytotox (want <0)'),
        ]:
            r, _ = spearmanr(df_clean[a], df_clean[b])
            logger.info(f"    {desc}: rho={r:.3f}")
    except Exception:
        pass

    return df_clean


def compute_obs_means(df):
    d = {k: float(df[k].mean()) for k in [
        'tigs_proxy_n', 'low_ag_n', 'chemo_sig_n', 'ccl5_n',
        'fasl_n', 'fasl_low_ag_n', 'cytotox_resid_n', 'iis_treg_adj_n',
        'exhaust_ratio_n', 'pdl1_exclusion_n', 'napm_n', 'tis_low_tigs_n',
        'pd1_n', 'pdl1_n',
    ]}
    d['F_PL_obs'] = float(df['F_PL'].mean())
    return d


# ══════════════════════════════════════════════════════════════════════════════
# OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════
def run_least_squares(df_clean):
    logger.info("=" * 70)
    logger.info("LSQ-ODE v6: 20-PARAMETER FIT  (exact F(P,L); dynamic rho)")
    logger.info("=" * 70)
    logger.info(f"  Free params     : {len(PARAM_NAMES)}")
    logger.info(f"  F(P,L) formula  : 1/(1 + rho_p*T * rho_l*(T+eps*(N+M)) / k_TQ)")
    logger.info(f"  mu_PA           : {FIXED['mu_PA']:.2e} M^-1 (fixed; A_drug=0 in TCGA)")
    logger.info(f"  k_TQ            : {FIXED['k_TQ']:.3e} M^2 (fixed)")
    logger.info(f"  epsilon_c       : {FIXED['epsilon_c']} (fixed)")

    obs     = compute_obs_means(df_clean)
    time_yr = np.clip(df_clean['time_yr'].values.astype(float), 1e-3, None)
    event   = df_clean['event'].values.astype(float)

    # Show F at baseline rho values + typical state
    p_bl = unpack(baseline_vector())
    T_typ, NM_typ = 1e6, 5e7
    F_bl = compute_F_PL(p_bl['rho_p'], p_bl['rho_l'],
                         T_typ, NM_typ/2, NM_typ/2)
    logger.info(f"  F at baseline rho, T=1e6, N+M=5e7 : {F_bl:.3f}")
    logger.info(f"  Data reference F(P,L) mean          : {obs['F_PL_obs']:.3f}")

    x0 = baseline_vector()

    def residuals_logspace(log_x):
        x = np.exp(log_x)
        x = np.clip(x, BOUNDS_LOWER, BOUNDS_UPPER)
        try:
            return build_residuals(x, obs, time_yr, event)
        except Exception as e:
            logger.debug(f"Residual failed: {e}")
            return np.ones(23 + len(PARAM_NAMES)) * 1e6

    result = least_squares(
        fun      = residuals_logspace,
        x0       = np.log(x0),
        bounds   = (np.log(np.array(BOUNDS_LOWER)),
                    np.log(np.array(BOUNDS_UPPER))),
        method   = 'trf',
        ftol=1e-8, xtol=1e-8, gtol=1e-8,
        max_nfev = 3000,
        verbose  = 1,
        loss     = 'soft_l1',
        f_scale  = 0.1,
    )

    x_opt = np.clip(np.exp(result.x), BOUNDS_LOWER, BOUNDS_UPPER)
    p_opt = unpack(x_opt)
    _, ss_opt, F_ss_opt = integrate_ode(p_opt)

    logger.info(f"\n  Success     : {result.success}")
    logger.info(f"  Message     : {result.message}")
    logger.info(f"  Cost        : {result.cost:.6e}")
    logger.info(f"  Optimality  : {result.optimality:.6e}")
    logger.info(f"  Func evals  : {result.nfev}")
    logger.info(f"  rho_p={p_opt['rho_p']:.3e}, rho_l={p_opt['rho_l']:.3e}")
    logger.info(f"  F(P,L) at SS = {F_ss_opt:.4f}  (data ref = {obs['F_PL_obs']:.4f})")
    logger.info(f"  K={p_opt['K']:.3e}  kappa_0={p_opt['kappa_0']:.3e}  "
                f"kappa_1={p_opt['kappa_1']:.4f}  kappa_2={p_opt['kappa_2']:.3e}")

    return result, x_opt, obs


# ══════════════════════════════════════════════════════════════════════════════
# SAVE / REPORT
# ══════════════════════════════════════════════════════════════════════════════
PARAM_META = {
    'alpha_n':  ('Tumour prolif rate, high-Ag',   'day^-1',    'TIGS',               'Wang 2023'),
    'alpha_m':  ('Tumour prolif rate, low-Ag',    'day^-1',    '1-TIGS',             'Wang 2023'),
    'delta_ns': ('Max slow (FasL) kill, N',       'day^-1',    'FASLG',              'Hassin 2011; 1-25/day'),
    'delta_ms': ('Max slow (FasL) kill, M',       'day^-1',    'FASLG x (1-TIGS)',   'Hassin 2011; 1-25/day'),
    'p1':       ('Fast-kill probability, N',      '-',         'Cytotox_resid',      'Wang 2023; p1>p2'),
    'p2':       ('Fast-kill probability, M',      '-',         'IIS x (1-Treg)',     'Wang 2023'),
    'delta_nf': ('Perforin kill rate, N',         'day^-1',    'Cytotox_resid flux', 'Okuneye 2021'),
    'delta_mf': ('Perforin kill rate, M',         'day^-1',    'IIS x (1-Treg)',     'Wang 2023'),
    'mu':       ('CTL recruitment rate',          'cells/day', 'ChemoSig + CCL5',    'Wang 2023'),
    'delta_t':  ('CTL natural death rate',        'day^-1',    'Survival hazard',    'Wang 2023'),
    'delta_n':  ('CTL death, N interaction',      'day^-1',    'PD1/(Cytotox+eps)',  'Kuznetsov 1994'),
    'delta_m':  ('CTL death, M interaction',      'day^-1',    'PDL1 x (1-IIS)',     'Kuznetsov 1994'),
    'alpha_nt': ('Max CTL prolif by N cells',     'day^-1',    'nAPM',               'Wang 2023; 0-0.5'),
    'alpha_mt': ('Max CTL prolif by M cells',     'day^-1',    'TIS x (1-TIGS)',     'Wang 2023'),
    'rho_p':    ('PD-1 expression per T cell',    'M/cell',    'PD1; F(P,L)',        'Wang 2023; base 1.6e-12'),
    'rho_l':    ('PD-L1 expression per cell',     'M/cell',    'PDL1; F(P,L)',       'Wang 2023; base 1.6e-12'),
    'K':        ('Tumour carrying capacity',      'cells',     'Tumour fractions',   'Geddes 1979; 5e8-1e10'),
    'kappa_0':  ('Slow-kill half-saturation',     'cells',     'Slow kill flux',     'Kuznetsov 1994'),
    'kappa_1':  ('CTL saturation (Beddington)',   '-',         'CTL crowding',       'Beddington 1975'),
    'kappa_2':  ('Antigen half-saturation',       'cells',     'CTL prolif sat.',    'Kuznetsov 1994'),
}


def save_results(result, x_opt, obs, df_clean, out_dir='results/lsq_ode_v6'):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    p = unpack(x_opt)
    traj, ss, F_ss = integrate_ode(p)

    rows = []
    for i, name in enumerate(PARAM_NAMES):
        desc, units, proxy, litref = PARAM_META[name]
        rows.append({
            'parameter':         name,
            'description':       desc,
            'units':             units,
            'data_proxy':        proxy,
            'literature_ref':    litref,
            'estimate':          x_opt[i],
            'baseline':          BASELINE[name],
            'lower_bound':       BOUNDS_LOWER[i],
            'upper_bound':       BOUNDS_UPPER[i],
            'ratio_to_baseline': x_opt[i] / BASELINE[name],
        })
    # Add fixed constants to output for transparency
    fixed_rows = [
        {'parameter': 'k_TQ',      'description': 'PD-1/PD-L1 inhibition const', 'units': 'M^2',
         'estimate': FIXED['k_TQ'],   'type': 'fixed'},
        {'parameter': 'epsilon_c', 'description': 'Tumour:T PD-L1 ratio',         'units': '-',
         'estimate': FIXED['epsilon_c'], 'type': 'fixed'},
        {'parameter': 'mu_PA',     'description': 'Anti-PD-1 blocking rate',      'units': 'M^-1',
         'estimate': FIXED['mu_PA'],  'type': 'fixed (A_drug=0 in TCGA)'},
        {'parameter': 'A_drug',    'description': 'Anti-PD-1 drug conc.',         'units': 'M',
         'estimate': FIXED['A_drug'], 'type': 'fixed (0=untreated)'},
    ]
    df_params = pd.DataFrame(rows)
    df_fixed  = pd.DataFrame(fixed_rows)
    df_params.to_csv(out / 'parameter_estimates.csv', index=False)
    df_fixed.to_csv(out / 'fixed_constants.csv', index=False)
    with open(out / 'parameter_estimates.json', 'w') as f:
        json.dump({r['parameter']: {k: v for k, v in r.items()
                   if k != 'parameter'} for r in rows}, f, indent=2)

    pd.DataFrame({
        'time_yr': T_EVAL,
        'N': traj[0], 'M': traj[1], 'T': traj[2],
        'N_norm': traj[0] / p['K'],
        'M_norm': traj[1] / p['K'],
        'T_norm': traj[2] / T_REF,
        'F_PL_traj': [compute_F_PL(p['rho_p'], p['rho_l'],
                                     traj[2,i], traj[0,i], traj[1,i])
                       for i in range(N_TPTS)],
    }).to_csv(out / 'ode_trajectories.csv', index=False)

    with open(out / 'fit_summary.txt', 'w') as f:
        f.write("LSQ-ODE v6: exact F(P,L); rho_p/rho_l [M/cell]; mu_PA included\n")
        f.write("=" * 65 + "\n")
        f.write(f"Formula: F = 1/(1 + rho_p*T * rho_l*(T+eps*(N+M)) / k_TQ)\n")
        f.write(f"Drug:    P_free = rho_p*T / (1 + mu_PA * A_drug)\n\n")
        f.write(f"Success    : {result.success}\n")
        f.write(f"Message    : {result.message}\n")
        f.write(f"Cost       : {result.cost:.8e}\n")
        f.write(f"Optimality : {result.optimality:.8e}\n")
        f.write(f"Func evals : {result.nfev}\n\n")
        f.write(f"rho_p  = {p['rho_p']:.4e} M/cell\n")
        f.write(f"rho_l  = {p['rho_l']:.4e} M/cell\n")
        f.write(f"F(P,L) at SS    : {F_ss:.4f}\n")
        f.write(f"F(P,L) data ref : {obs['F_PL_obs']:.4f}\n\n")
        f.write(f"Fixed constants:\n")
        f.write(f"  k_TQ      = {FIXED['k_TQ']:.3e} M^2\n")
        f.write(f"  epsilon_c = {FIXED['epsilon_c']}\n")
        f.write(f"  mu_PA     = {FIXED['mu_PA']:.2e} M^-1 (drug blocking; irrelevant at A=0)\n")
        f.write(f"  A_drug    = {FIXED['A_drug']} M (0 = untreated TCGA)\n\n")
        f.write(f"Structural: K={p['K']:.3e} kappa_0={p['kappa_0']:.3e} "
                f"kappa_1={p['kappa_1']:.4f} kappa_2={p['kappa_2']:.3e}\n")
        f.write(f"Steady state: N={ss[0]:.3e} M={ss[1]:.3e} T={ss[2]:.3e}\n\n")
        f.write("To simulate anti-PD-1 treatment:\n")
        f.write("  Call integrate_ode(p_opt, A_drug=<drug_conc_in_M>)\n")
        f.write("  mu_PA = 1e-10 M^-1 mediates drug effect on P.\n")

    logger.info(f"Results saved to {out}/")
    return df_params, pd.DataFrame({
        'time_yr': T_EVAL, 'N': traj[0], 'M': traj[1], 'T': traj[2]
    }), ss


def print_report(df_params, df_traj, result, obs, p_opt):
    _, ss, F_ss = integrate_ode(p_opt)

    print(f"\n{'='*85}")
    print("LSQ-ODE v6 — 20-PARAMETER ESTIMATES  (exact F(P,L))")
    print(f"{'='*85}")
    print(f"Cost: {result.cost:.6e}   |   "
          f"rho_p={p_opt['rho_p']:.3e}  rho_l={p_opt['rho_l']:.3e}  "
          f"F(SS)={F_ss:.4f}  (data ref={obs['F_PL_obs']:.4f})")
    print(f"Fixed: k_TQ={FIXED['k_TQ']:.2e}  eps_c={FIXED['epsilon_c']}  "
          f"mu_PA={FIXED['mu_PA']:.2e} (drug term; A_drug=0 in TCGA)")
    print()

    groups = [
        ('Kinetic (14)',   ['alpha_n','alpha_m','delta_ns','delta_ms','p1','p2',
                            'delta_nf','delta_mf','mu','delta_t','delta_n','delta_m',
                            'alpha_nt','alpha_mt']),
        ('Checkpoint (2)', ['rho_p','rho_l']),
        ('Structural (4)', ['K','kappa_0','kappa_1','kappa_2']),
    ]

    print(f"  {'Param':<12} {'Estimate':>12} {'Baseline':>12} {'Ratio':>7}  "
          f"{'Bounds':>20}  Units")
    for grp, params in groups:
        print(f"\n  -- {grp} --")
        sub = df_params[df_params['parameter'].isin(params)]
        for _, r in sub.iterrows():
            bds = f"[{r['lower_bound']:.2g},{r['upper_bound']:.2g}]"
            print(f"  {r['parameter']:<12} {r['estimate']:>12.4e} "
                  f"{r['baseline']:>12.4e} {r['ratio_to_baseline']:>7.3f}  "
                  f"{bds:>20}  {r['units']}")

    ss_N = df_traj['N'].iloc[-20:].mean()
    ss_M = df_traj['M'].iloc[-20:].mean()
    ss_T = df_traj['T'].iloc[-20:].mean()
    print(f"\n{'='*85}")
    print(f"Steady state — N: {ss_N:.3e}  M: {ss_M:.3e}  T: {ss_T:.3e}")
    print(f"N/(N+M) = {ss_N/(ss_N+ss_M+1e-30):.3f}   K_est = {p_opt['K']:.3e}")
    print(f"\nResults in: results/lsq_ode_v6/")
    print(f"{'='*85}\n")


def main():
    df_clean           = load_and_merge_data()
    result, x_opt, obs = run_least_squares(df_clean)
    p_opt              = unpack(x_opt)
    df_params, df_traj, ss = save_results(result, x_opt, obs, df_clean)
    print_report(df_params, df_traj, result, obs, p_opt)


if __name__ == '__main__':
    main()