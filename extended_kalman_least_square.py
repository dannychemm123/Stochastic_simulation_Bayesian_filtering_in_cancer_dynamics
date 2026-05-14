import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import odeint
from scipy.linalg import block_diag, expm
import os
import hashlib
import json
import warnings
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD PARAMETERS AND DATA
# ============================================================

print("="*60)
print("LOADING PARAMETERS AND DATA")
print("="*60)

# Load parameters
with open('least_square_parameters.json', 'r') as f:
    param_data = json.load(f)

params_raw = param_data['estimated_parameters']

# Map parameter names from least_square to canonical names
params = {
    'alpha_h': params_raw['alpha_n'],
    'alpha_l': params_raw['alpha_m'],
    'delta_hs': params_raw['delta_ns'],
    'delta_ls': params_raw['delta_ms'],
    'p1_baseline': params_raw['p1'],
    'p2_baseline': params_raw['p2'],
    'delta_hf': params_raw['delta_nf'],
    'delta_lf': params_raw['delta_mf'],
    'mu_baseline': params_raw['mu'],
    'delta_T': params_raw['delta_t'],
    'delta_ht': params_raw['delta_n'],
    'delta_lt': params_raw['delta_m'],
    'alpha_ht_baseline': params_raw['alpha_nt'],
    'alpha_lt_baseline': params_raw['alpha_mt'],
    'rho_p_baseline': params_raw['rho_p'],
    'rho_l_baseline': params_raw['rho_l'],
    'K': params_raw['K'],
    'kappa_0': params_raw['kappa_0'],
    'kappa_1': params_raw['kappa_1'],
    'kappa_2': params_raw['kappa_2'],
    'epsilon_c': params_raw['epsilon_c'],
    'k_TQ_baseline': params_raw['k_TQ']
}

print("Using: least_square_parameters.json")
print(f"  p1_baseline:       {params['p1_baseline']:.4f}")
print(f"  p2_baseline:       {params['p2_baseline']:.4f}")
print(f"  alpha_h:           {params['alpha_h']:.4f}")
print(f"  mu_baseline:       {params['mu_baseline']:.2f}")
print(f"  delta_hs:          {params['delta_hs']:.4f}")

# Load NSCLC clinical data
clinical = pd.read_csv('NSCLC_Clinical_Cleaned.csv')

# Map T-stage to initial tumor cell count N0
# Cell counts derived from spherical tumor volume using 1e6 cells/mm³ density,
# which is the same assumption used in cells_to_mm(). This ensures that
# cells_to_mm(H0 + L0) == tumor_size_mm at t=0 (internally consistent).
#
# Formula: cells = (4/3) * pi * (d/2)^3 * 1e6
#   T1 (d=15 mm): 1767 mm³ * 1e6 = 1.77e9 cells
#   T2 (d=40 mm): 33510 mm³ * 1e6 = 3.35e10 cells
#   T3 (d=60 mm): 113097 mm³ * 1e6 = 1.13e11 cells
#   T4 (d=80 mm): 268083 mm³ * 1e6 = 2.68e11 cells
t_stage_to_cells = {
    1.0: 1.77e9,    # T1: ~15 mm  → ~1.77 billion cells
    2.0: 3.35e10,   # T2: ~40 mm  → ~33.5 billion cells
    3.0: 1.13e11,   # T3: ~60 mm  → ~113 billion cells
    4.0: 2.68e11    # T4: ~80 mm  → ~268 billion cells
}

t_stage_to_mm = {
    1.0: 15,
    2.0: 40,
    3.0: 60,
    4.0: 80
}

clinical['H0_cells'] = clinical['t_stage'].map(t_stage_to_cells)
clinical['tumor_size_mm'] = clinical['t_stage'].map(t_stage_to_mm)
clinical = clinical.dropna(subset=['H0_cells', 'tumor_size_mm'])

print(f"\nPatients loaded: {len(clinical)}")
print(f"T-stage distribution:\n{clinical['t_stage'].value_counts().sort_index()}")

# Sanity check: verify cells_to_mm(H0_cells) matches tumor_size_mm
print("\nConsistency check (should be ~0 error):")
for stage in [1.0, 2.0, 3.0, 4.0]:
    cells = t_stage_to_cells[stage]
    vol   = cells / 1e6
    d     = 2 * (3 * vol / (4 * np.pi)) ** (1/3)
    print(f"  T{int(stage)}: {cells:.2e} cells → {d:.1f} mm  (expected {t_stage_to_mm[stage]} mm)")


# ============================================================
# STEP 2: ODE MODEL — N, M, T DYNAMICS
# ============================================================

def ode_model(y, t, p):
    """
    3-compartment tumor-immune ODE:
      dH/dt: logistic growth - slow CTL killing - fast CTL killing
      dL/dt: logistic growth - slow CTL killing - fast CTL killing
      dT/dt: (baseline + antigen-driven expansion) * F(PD-1/PDL-1) - tumor-induced death - natural death
    """
    H, L, T = y
    H = max(H, 0)
    L = max(L, 0)
    T = max(T, 0)

    # Unpack parameters
    alpha_h    = p['alpha_h']
    alpha_l    = p['alpha_l']
    K          = p['K']
    delta_hs   = p['delta_hs']
    delta_ls   = p['delta_ls']
    delta_hf   = p['delta_hf']
    delta_lf   = p['delta_lf']
    delta_ht   = p['delta_ht']
    delta_lt   = p['delta_lt']
    kappa_0    = p['kappa_0']
    kappa_1    = p['kappa_1']
    kappa_2    = p['kappa_2']
    delta_T    = p['delta_T']
    mu         = p['mu_baseline']
    alpha_ht   = p['alpha_ht_baseline']
    alpha_lt   = p['alpha_lt_baseline']
    p1         = p['p1_baseline']
    p2         = p['p2_baseline']
    epsilon_c  = p['epsilon_c']
    rho_p      = p['rho_p_baseline']
    rho_l      = p['rho_l_baseline']
    k_TQ       = p['k_TQ_baseline']

    # PD1-PDL1 immune suppression factor F(P,L)
    P = rho_p * T
    L_pd = rho_l * (T + epsilon_c * (H + L))
    F = 1.0 / (1.0 + (P * L_pd) / k_TQ)
    F = np.clip(F, 0, 1)

    # Denominator for slow killing
    denom = kappa_1 * T + (H + L) + kappa_0
    denom = max(denom, 1e-10)

    # dH/dt — High antigen tumor cells
    dH = (alpha_h * H * (1 - (H + L) / K)
          - delta_hs * (1 - p1) * (H / denom) * T
          - delta_hf * p1 * H * T)

    # dL/dt — Low antigen tumor cells
    dL = (alpha_l * L * (1 - (H + L) / K)
          - delta_ls * (1 - p2) * (L / denom) * T
          - delta_lf * p2 * L * T)

    # dT/dt — CTL T cells
    dT = ((mu
           + alpha_ht * (H / (kappa_2 + H)) * T
           + alpha_lt * (L / (kappa_2 + L)) * T) * F
          - delta_ht * H * T
          - delta_lt * L * T
          - delta_T * T)

    return [dH, dL, dT]


def cells_to_mm(H, L):
    """Convert cell count to tumor diameter in mm.
    Assumes spherical tumor with density 1e6 cells/mm³ (= 1e9 cells/cm³).
    volume_mm³ = total_cells / 1e6
    diameter   = 2 * (3V / 4π)^(1/3)
    """
    total_cells = H + L
    volume_mm3 = total_cells / 1e6
    diameter = 2 * (3 * volume_mm3 / (4 * np.pi)) ** (1/3)
    return max(diameter, 0.1)


def simulate_trajectory(H0, L0, T0, params, t_span_days):
    """Simulate deterministic ODE from initial conditions"""
    y0  = [H0, L0, T0]
    sol = odeint(ode_model, y0, t_span_days,
                 args=(params,), rtol=1e-6, atol=1e-9)
    sol = np.clip(sol, 0, None)
    return sol


def simulate_sde_trajectory(H0, L0, T0, params, t_span_days, sigma_H=1e5, sigma_L=1e5, sigma_T=1e4, seed=None):
    """Simulate stochastic SDE (Euler-Maruyama) from initial conditions"""
    rng = np.random.RandomState(seed) if seed is not None else np.random

    dt = 0.5  # internal step size for simulation stability
    X = np.zeros((len(t_span_days), 3))
    X[0] = [H0, L0, T0]

    current_x = np.array([H0, L0, T0])
    current_t = t_span_days[0]

    sigma = np.array([sigma_H, sigma_L, sigma_T])

    for i in range(1, len(t_span_days)):
        target_t = t_span_days[i]
        while current_t < target_t:
            step = min(dt, target_t - current_t)
            drift = np.array(ode_model(current_x, current_t, params))
            dW = rng.normal(0, 1, 3) * np.sqrt(step)
            current_x = current_x + drift * step + sigma * dW
            current_x = np.maximum(current_x, 1.0)
            current_t += step
        X[i] = current_x

    return X


# ============================================================
# STEP 3: EXTENDED KALMAN FILTER WITH SDE MODEL
# ============================================================

class SDE_KalmanFilter:
    """
    Extended Kalman Filter using the biological SDE as state transition.

    State vector: x = [H, L, T]  (tumor cells high-antigen, low-antigen, CTL)
    Observation:  z = tumor_size_mm = f(H, L)

    Predict step: integrate ODE forward dt days
    Update step:  incorporate new tumor size measurement
    """

    def __init__(self, params, dt_days=90):
        self.params   = params
        self.dt_days  = dt_days          # 3 months between observations
        self.t_span   = np.linspace(0, dt_days, 20)

        # Process noise Sigma — constant additive noise (LaTeX version)
        self.sigma_H = 1e5    # LaTeX: sigma_H
        self.sigma_L = 1e5    # LaTeX: sigma_L
        self.sigma_T = 1e4    # LaTeX: sigma_T

        # Observation noise R — CT measurement error (~5mm std)
        self.R = np.array([[25.0]])      # 5mm std → 25mm² variance

        # Observation matrix H — maps state to tumor size
        self.H = None                    # computed dynamically

    def predict(self, x, P):
        """
        PREDICT step: propagate state forward using Euler-Maruyama (Stable version)
        x: state [H, L, T]
        P: covariance matrix (3x3)
        """
        # 1. State Prediction (Internal EM loop for stability)
        dt_internal = 0.5
        n_steps = int(self.dt_days / dt_internal)
        x_pred = x.copy()

        for _ in range(n_steps):
            drift = np.array(ode_model(x_pred, 0, self.params))
            x_pred = x_pred + drift * dt_internal
            x_pred = np.maximum(x_pred, 1.0)

        # 2. Linearize transition (Jacobian F)
        F_jac = self._numerical_jacobian(x, interval=self.dt_days)

        # 3. Additive Process Noise Q = diag(sigma^2) * dt
        Q = np.diag([
            self.sigma_H**2 * self.dt_days,
            self.sigma_L**2 * self.dt_days,
            self.sigma_T**2 * self.dt_days
        ])

        # Predicted covariance: P_k = F P_{k-1} F^T + Q
        P_pred = F_jac @ P @ F_jac.T + Q

        return x_pred, P_pred

    def update(self, x_pred, P_pred, z_obs):
        """
        UPDATE step: incorporate tumor size measurement
        x_pred: predicted state
        P_pred: predicted covariance
        z_obs:  observed tumor size in mm
        """
        # Predicted observation (tumor size from state)
        z_pred = np.array([cells_to_mm(x_pred[0], x_pred[1])])

        # Linearized observation matrix H (∂z/∂x)
        H = self._observation_jacobian(x_pred)

        # Innovation (measurement residual)
        innovation = np.array([z_obs]) - z_pred

        # Innovation covariance
        S = H @ P_pred @ H.T + self.R

        # Kalman gain (direct scalar inversion for 1x1 matrix)
        K = P_pred @ H.T * (1.0 / S[0, 0])

        # Updated state
        x_upd = x_pred + K @ innovation
        x_upd = np.clip(x_upd, 0, None)

        # Updated covariance — Joseph stabilized form for numerical stability
        I = np.eye(len(x_pred))
        IKH  = I - K @ H
        P_upd = IKH @ P_pred @ IKH.T + K @ self.R @ K.T

        return x_upd, P_upd, float(innovation[0]), float(S[0,0])

    def _numerical_jacobian(self, x, interval, eps=1e-3):
        """Numerical Jacobian of full interval transition (F = expm(J*dt))"""
        n   = len(x)
        jac = np.zeros((n, n))

        for i in range(n):
            h = eps * max(abs(x[i]), 1.0)
            x_p    = x.copy(); x_p[i] = x[i] + h
            x_m    = x.copy(); x_m[i] = max(x[i] - h, 1e-10)
            f_p    = np.array(ode_model(x_p, 0, self.params))
            f_m    = np.array(ode_model(x_m, 0, self.params))
            jac[:, i] = (f_p - f_m) / (x_p[i] - x_m[i])

        # Transition matrix for the interval: F = exp(J * dt)
        F = expm(jac * interval)
        return F

    def _observation_jacobian(self, x, eps=1e-3):
        """Jacobian of observation function h(x) = tumor_size_mm"""
        H = np.zeros((1, 3))
        h0 = cells_to_mm(x[0], x[1])

        # dh/dH
        x_pert    = x.copy()
        x_pert[0] = max(x[0] + eps * max(abs(x[0]), 1), 1)
        H[0, 0]   = (cells_to_mm(x_pert[0], x[1]) - h0) / (x_pert[0] - x[0])

        # dh/dL
        x_pert    = x.copy()
        x_pert[1] = max(x[1] + eps * max(abs(x[1]), 1), 1)
        H[0, 1]   = (cells_to_mm(x[0], x_pert[1]) - h0) / (x_pert[1] - x[1])

        # dh/dT = 0 (T cells not directly observed)
        H[0, 2]   = 0.0

        return H


# ============================================================
# STEP 4: RUN FILTER ON EACH PATIENT
# ============================================================

print("\n" + "="*60)
print("RUNNING SDE-BASED KALMAN FILTER")
print("="*60)

# Timepoints: every 3 months (90 days)
# Timepoints: every 3 months (90 days) up to 120 months (matching chapter5.py)
# We calculate this per patient based on survival below.
# timepoints_months = [0, 3, 6, 9, 12, 18, 24]
# timepoints_days   = [t * 30.44 for t in timepoints_months]

np.random.seed(42)
kf = SDE_KalmanFilter(params, dt_days=90)

all_results = []

for _, row in clinical.iterrows():
    pid            = row['patient_id']
    H0             = row['H0_cells']
    tumor_size_mm  = row['tumor_size_mm']
    survival_months= row['survival_months']
    dead           = row['dead']
    t_stage        = row['t_stage']
    overall_stage  = row['overall_stage']
    histology      = row['histology']

    # Initial conditions
    L0 = H0 * 0.3          # 30% low-antigen tumor cells
    T0 = params['mu_baseline'] / params['delta_T']  # steady-state T cells

    # Initial state and covariance
    x = np.array([H0, L0, T0])
    P = np.diag([H0**2 * 0.1, L0**2 * 0.1, T0**2 * 0.1])

    # Define timepoints for this patient: every 3 months up to survival (max 24 months)
    t_months_list = np.arange(0, min(24, int(survival_months) + 1), 3)
    t_days_list   = t_months_list * 30.44

    # Simulate trajectories
    t_true = np.linspace(0, max(t_days_list), 200)
    
    # 1. Stochastic Truth (SDE)
    patient_seed = int(hashlib.md5(pid.encode()).hexdigest(), 16) % (2**32)
    sol_sde = simulate_sde_trajectory(H0, L0, T0, params, t_true,
                                       sigma_H=kf.sigma_H, sigma_L=kf.sigma_L, sigma_T=kf.sigma_T,
                                       seed=patient_seed)
    
    # 2. Deterministic Truth (ODE)
    sol_ode = simulate_trajectory(H0, L0, T0, params, t_true)

    # Run filter at each timepoint
    t_prev = 0
    for t_idx, (t_months, t_days) in enumerate(zip(t_months_list, t_days_list)):

        # Truths at this timepoint (interpolated)
        true_idx  = np.argmin(np.abs(t_true - t_days))
        sde_size  = cells_to_mm(sol_sde[true_idx, 0], sol_sde[true_idx, 1])
        ode_size  = cells_to_mm(sol_ode[true_idx, 0], sol_ode[true_idx, 1])

        # Noisy CT observation (based on SDE truth) with reproducible noise seed
        rng_obs   = np.random.default_rng(patient_seed + 100 + t_idx)
        obs_noise = rng_obs.normal(0, 5.0)
        obs_size  = max(1.0, sde_size + obs_noise)

        if t_idx == 0:
            # t=0: assimilate baseline CT measurement into initial state
            x, P, innovation, S = kf.update(x, P, obs_size)
            est_size  = cells_to_mm(x[0], x[1])
            H0_mat    = kf._observation_jacobian(x)
            S0_mm     = H0_mat @ P @ H0_mat.T
            unc       = float(np.sqrt(abs(S0_mm[0, 0])))

        else:
            # PREDICT
            x_pred, P_pred = kf.predict(x, P)

            # UPDATE with observation
            x, P, innovation, S = kf.update(x_pred, P_pred, obs_size)
            est_size = cells_to_mm(x[0], x[1])
            H_mat   = kf._observation_jacobian(x)
            S_mm    = H_mat @ P @ H_mat.T
            unc     = float(np.sqrt(abs(S_mm[0, 0])))

        all_results.append({
            'patient_id'        : pid,
            'timepoint_months'  : t_months,
            'sde_size_mm'       : round(sde_size, 2),
            'ode_size_mm'       : round(ode_size, 2),
            'observed_size_mm'  : round(obs_size, 2),
            'kalman_estimate_mm': round(est_size, 2),
            'kalman_uncertainty': round(unc, 2),
            'H_cells'           : round(x[0]),
            'L_cells'           : round(x[1]),
            'T_cells'           : round(x[2]),
            'innovation'        : round(innovation, 3),
            'dead'              : dead,
            'survival_months'   : survival_months,
            'overall_stage'     : overall_stage,
            'histology'         : histology,
            't_stage'           : t_stage
        })

results_df = pd.DataFrame(all_results)
print(f"✓ Filter ran on {results_df['patient_id'].nunique()} patients")
print(f"✓ Total observations: {len(results_df)}")


# ============================================================
# STEP 5: VISUALIZE (6 PATIENTS IN 2x3 GRID)
# ============================================================

print("\nGenerating plots...")

# --- FIGURE 1: Individual patient trajectories (6 patients in 2x3 grid) ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Extended Kalman Filter — Tumor Dynamics ',
             fontsize=16, fontweight='bold')

sample_pids = (results_df.groupby('patient_id')
               .size()[lambda x: x >= 3]
               .sample(min(6, results_df['patient_id'].nunique()),
                       random_state=42).index)

for ax, pid in zip(axes.flatten(), sample_pids):
    pdata = results_df[results_df['patient_id'] == pid].sort_values('timepoint_months')

    t_months = pdata['timepoint_months'].values
    sde_s    = pdata['sde_size_mm'].values
    ode_s    = pdata['ode_size_mm'].values
    obs_s    = pdata['observed_size_mm'].values
    est_s    = pdata['kalman_estimate_mm'].values
    unc      = pdata['kalman_uncertainty'].values

    ax.plot(t_months, sde_s, 'g--', linewidth=2, label='SDE Truth', alpha=0.7)
    ax.plot(t_months, ode_s, 'k:',  linewidth=2, label='ODE Baseline', alpha=0.6)
    ax.scatter(t_months, obs_s, color='red', s=80, zorder=5,
               label='CT observation', marker='x', linewidth=2)
    ax.plot(t_months, est_s, 'b-o', linewidth=2.5, markersize=6,
            label='Kalman estimate')
    ax.fill_between(t_months,
                    np.maximum(0, est_s - 1.96 * unc),
                    est_s + 1.96 * unc,
                    alpha=0.25, color='blue', label='95% CI')

    stage  = pdata['overall_stage'].iloc[0]
    dead   = pdata['dead'].iloc[0]
    surv   = pdata['survival_months'].iloc[0]
    status = 'Died' if dead == 1 else 'Alive'

    ax.set_title(f'Patient {pid} | {stage}\n{status} ({surv:.0f} mo)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (months)', fontsize=10)
    ax.set_ylabel('Tumor size (mm)', fontsize=10)
    ax.set_xlim(0, 24)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=9)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4,
           fontsize=11, bbox_to_anchor=(0.5, -0.02))

plt.subplots_adjust(bottom=0.12, hspace=0.35, wspace=0.3)
plt.savefig('Fig1_SDE_Kalman_Trajectories_least_square.png', dpi=150, bbox_inches='tight')
print("✓ Fig1 saved")


# --- FIGURE 2: N, M, T cell dynamics for one patient ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('ODE State Variables — Cell Population Dynamics',
             fontsize=14, fontweight='bold')

sample_pid = results_df['patient_id'].iloc[0]
pdata = results_df[results_df['patient_id'] == sample_pid]
t = pdata['timepoint_months'].values

ax = axes[0]
ax.plot(t, pdata['H_cells'].values / 1e9, 'b-o', linewidth=2.5, markersize=6, label='High Antigen (H)')
ax.set_title('H — High Antigen Tumor Cells', fontweight='bold', fontsize=12)
ax.set_xlabel('Time (months)', fontsize=10)
ax.set_ylabel('Cell count (billions)', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')
ax.fill_between(t, 0, pdata['H_cells'].values / 1e9, alpha=0.2, color='blue')
ax.legend(fontsize=10)

ax = axes[1]
ax.plot(t, pdata['L_cells'].values / 1e9, 'r-o', linewidth=2.5, markersize=6, label='Low Antigen (L)')
ax.set_title('L — Low Antigen Tumor Cells', fontweight='bold', fontsize=12)
ax.set_xlabel('Time (months)', fontsize=10)
ax.set_ylabel('Cell count (billions)', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')
ax.fill_between(t, 0, pdata['L_cells'].values / 1e9, alpha=0.2, color='red')
ax.legend(fontsize=10)

ax = axes[2]
ax.plot(t, pdata['T_cells'].values / 1e3, 'g-o', linewidth=2.5, markersize=6, label='CTL (T)')
ax.set_title('T — CTL T Cells', fontweight='bold', fontsize=12)
ax.set_xlabel('Time (months)', fontsize=10)
ax.set_ylabel('Cell count (thousands)', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')
ax.fill_between(t, 0, pdata['T_cells'].values / 1e3, alpha=0.2, color='green')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('Fig2_Cell_Dynamics_least_square.png', dpi=150, bbox_inches='tight')
print("✓ Fig2 saved")


# --- FIGURE 3: Filter performance ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Extended Kalman Filter Performance', fontsize=14, fontweight='bold')

ax = axes[0]
err_kalman = (results_df['kalman_estimate_mm'] - results_df['sde_size_mm']).abs()
err_obs    = (results_df['observed_size_mm']   - results_df['sde_size_mm']).abs()

ax.hist(err_kalman, bins=30, alpha=0.6, color='blue',
        label=f'Kalman (mean={err_kalman.mean():.1f}mm)', edgecolor='black', linewidth=1.2)
ax.hist(err_obs,    bins=30, alpha=0.6, color='red',
        label=f'Raw obs (mean={err_obs.mean():.1f}mm)', edgecolor='black', linewidth=1.2)
ax.set_xlabel('Absolute error (mm)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('A. Error Distribution', fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

ax = axes[1]
innov_by_time = results_df.groupby('timepoint_months')['innovation'].agg(['mean', 'std']).reset_index()
ax.plot(innov_by_time['timepoint_months'], innov_by_time['mean'], 'b-o', linewidth=2.5, markersize=7)
ax.fill_between(innov_by_time['timepoint_months'],
                innov_by_time['mean'] - innov_by_time['std'],
                innov_by_time['mean'] + innov_by_time['std'],
                alpha=0.25, color='blue')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Timepoint (months)', fontsize=11)
ax.set_ylabel('Innovation (obs - predicted) mm', fontsize=11)
ax.set_title('B. Filter Innovation Over Time\n(should converge to ~0)',
             fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')

ax = axes[2]
stage_order  = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
stage_colors = ['green', 'blue', 'orange', 'red']
unc_by_stage = results_df.groupby('overall_stage')['kalman_uncertainty'].mean()
stages  = [s for s in stage_order if s in unc_by_stage.index]
vals    = [unc_by_stage[s] for s in stages]
colors  = [stage_colors[stage_order.index(s)] for s in stages]
bars = ax.bar(stages, vals, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Mean uncertainty (mm)', fontsize=11)
ax.set_title('C. Filter Uncertainty by Stage', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('Fig3_Filter_Performance_least_square.png', dpi=150, bbox_inches='tight')
print("✓ Fig3 saved")

# Save results
results_df.to_csv('SDE_Kalman_Results_least_square.csv', index=False)
print("\n✓ SDE_Kalman_Results_least_square.csv saved")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Parameter file:        least_square_parameters.json")
print(f"Patients filtered:     {results_df['patient_id'].nunique()}")
print(f"Mean Kalman error:     {err_kalman.mean():.2f} mm")
print(f"Mean raw obs error:    {err_obs.mean():.2f} mm")
print(f"Improvement:           {((err_obs.mean()-err_kalman.mean())/err_obs.mean()*100):.1f}%")