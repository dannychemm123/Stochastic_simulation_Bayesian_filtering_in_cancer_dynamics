import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import cholesky, block_diag
import json
import os
import hashlib
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('Agg')
# ============================================================
# STEP 1: LOAD PARAMETERS AND DATA
# ============================================================

print("="*60)
print("LOADING PARAMETERS AND DATA")
print("="*60)

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
print(f"  p1_baseline:  {params['p1_baseline']:.4f}")
print(f"  p2_baseline:  {params['p2_baseline']:.4f}")
print(f"  alpha_h:     {params['alpha_h']:.4f}")
print(f"  mu_baseline:  {params['mu_baseline']:.2f}")
print(f"  delta_hs:     {params['delta_hs']:.4f}")

clinical = pd.read_csv('NSCLC_Clinical_Cleaned.csv')

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

clinical['H0_cells']      = clinical['t_stage'].map(t_stage_to_cells)
clinical['tumor_size_mm'] = clinical['t_stage'].map(t_stage_to_mm)
clinical = clinical.dropna(subset=['H0_cells', 'tumor_size_mm'])

print(f"\nPatients loaded: {len(clinical)}")
print(f"T-stage distribution:\n{clinical['t_stage'].value_counts().sort_index()}")


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
    H = max(H, 0); L = max(L, 0); T = max(T, 0)

    alpha_h  = p['alpha_h'];   alpha_l  = p['alpha_l']
    K        = p['K']
    delta_hs = p['delta_hs'];  delta_ls = p['delta_ls']
    delta_hf = p['delta_hf'];  delta_lf = p['delta_lf']
    delta_ht = p['delta_ht'];  delta_lt = p['delta_lt']
    kappa_0  = p['kappa_0'];   kappa_1  = p['kappa_1'];   kappa_2 = p['kappa_2']
    delta_T  = p['delta_T']
    mu       = p['mu_baseline']
    alpha_ht = p['alpha_ht_baseline']
    alpha_lt = p['alpha_lt_baseline']
    p1       = p['p1_baseline'];  p2 = p['p2_baseline']
    epsilon_c= p['epsilon_c']
    rho_p    = p['rho_p_baseline']
    rho_l    = p['rho_l_baseline']
    k_TQ     = p['k_TQ_baseline']

    P = rho_p * T
    L_pd = rho_l * (T + epsilon_c * (H + L))
    F = np.clip(1.0 / (1.0 + (P * L_pd) / k_TQ), 0, 1)

    denom = max(kappa_1 * T + (H + L) + kappa_0, 1e-10)

    dH = (alpha_h * H * (1 - (H + L) / K)
          - delta_hs * (1 - p1) * (H / denom) * T
          - delta_hf * p1 * H * T)

    dL = (alpha_l * L * (1 - (H + L) / K)
          - delta_ls * (1 - p2) * (L / denom) * T
          - delta_lf * p2 * L * T)

    dT = ((mu
           + alpha_ht * (H / (kappa_2 + H)) * T
           + alpha_lt * (L / (kappa_2 + L)) * T) * F
          - delta_ht * H * T
          - delta_lt * L * T
          - delta_T * T)

    return [dH, dL, dT]


def cells_to_mm(H, L):
    """Convert total cell count to tumor diameter in mm (assuming sphere)."""
    volume_mm3 = max(H + L, 0) / 1e6
    return max(2 * (3 * volume_mm3 / (4 * np.pi)) ** (1/3), 0.1)


# Sanity check
print("\nConsistency check (should be ~0 error):")
for stage in [1.0, 2.0, 3.0, 4.0]:
    cells = t_stage_to_cells[stage]
    mm = cells_to_mm(cells, 0)
    expected = t_stage_to_mm[stage]
    print(f"  T{int(stage)}: {cells:.2e} cells → {mm:.1f} mm  (expected {expected} mm)")


def cells_to_mm_vec(H_arr, L_arr):
    """Vectorized version for arrays of particles."""
    volume = np.maximum(H_arr + L_arr, 0) / 1e6
    return np.maximum(2.0 * (3.0 * volume / (4.0 * np.pi)) ** (1.0/3.0), 0.1)


def simulate_trajectory(N0, M0, T0, params, t_span):
    """Simulate deterministic ODE from initial conditions"""
    sol = odeint(ode_model, [N0, M0, T0], t_span,
                 args=(params,), rtol=1e-6, atol=1e-9)
    return np.clip(sol, 0, None)


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
# STEP 3: UNSCENTED KALMAN FILTER WITH SDE MODEL
# ============================================================

class SDE_UnscentedKalmanFilter:
    """
    Unscented Kalman Filter (UKF) using the van der Merwe scaled unscented transform.

    State:       x = [H, L, T]
    Observation: z = tumor_size_mm = cells_to_mm(H, L)

    Unlike the EKF, no Jacobians are needed. The state distribution is
    approximated by 2n+1 deterministically chosen sigma points propagated
    through the nonlinear ODE and observation function exactly.

    Parameters (van der Merwe):
      alpha  — sigma-point spread around mean (1e-3 to 1)
      beta   — prior knowledge of distribution (2 optimal for Gaussian)
      kappa  — secondary scaling (0 is standard)
      lambda_ = alpha^2 * (n + kappa) - n
    """

    def __init__(self, params, dt_days=90, alpha=1e-3, beta=2.0, kappa=0.0):
        self.params   = params
        self.dt_days  = dt_days
        self.t_span   = np.linspace(0, dt_days, 20)
        self.n        = 3                        # state dimension

        # UKF scaling parameters
        self.alpha  = alpha
        self.beta   = beta
        self.kappa  = kappa
        lam         = alpha**2 * (self.n + kappa) - self.n
        self.lambda_ = lam

        # Sigma-point weights for mean and covariance
        n_sp = 2 * self.n + 1
        self.Wm = np.full(n_sp, 1.0 / (2 * (self.n + lam)))
        self.Wc = np.full(n_sp, 1.0 / (2 * (self.n + lam)))
        self.Wm[0] = lam / (self.n + lam)
        self.Wc[0] = lam / (self.n + lam) + (1 - alpha**2 + beta)

        # Observation noise R — CT measurement error (~5mm std → 25mm² variance)
        self.R = np.array([[25.0]])

        # Process noise Sigma — constant additive noise (LaTeX version)
        self.sigma_H = 1e5    # LaTeX: sigma_H
        self.sigma_L = 1e5    # LaTeX: sigma_L
        self.sigma_T = 1e4    # LaTeX: sigma_T

    # ----------------------------------------------------------
    # Sigma-point generation
    # ----------------------------------------------------------
    def _sigma_points(self, x, P):
        """
        Compute 2n+1 sigma points from mean x and covariance P.
        Uses Cholesky decomposition: sqrt_matrix = chol((n+λ)P)
        """
        n   = self.n
        lam = self.lambda_
        try:
            sqrt_P = cholesky((n + lam) * P, lower=True)
        except Exception:
            # Fallback: add small jitter for numerical safety
            sqrt_P = cholesky((n + lam) * (P + 1e-8 * np.eye(n)), lower=True)

        sigmas      = np.zeros((2 * n + 1, n))
        sigmas[0]   = x
        for i in range(n):
            sigmas[i + 1]     = x + sqrt_P[:, i]
            sigmas[i + 1 + n] = x - sqrt_P[:, i]

        # Clip negative cell counts
        sigmas = np.clip(sigmas, 0, None)
        return sigmas

    # ----------------------------------------------------------
    # Unscented transform helper
    # ----------------------------------------------------------
    def _unscented_transform(self, sigmas_f, noise_cov):
        """
        Compute mean and covariance from propagated sigma points + noise.
        sigmas_f: (2n+1, dim) — propagated sigma points
        noise_cov: additive noise covariance
        Returns: mean, covariance
        """
        x_mean = np.sum(self.Wm[:, None] * sigmas_f, axis=0)
        diff   = sigmas_f - x_mean
        cov    = (self.Wc[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0)
        cov   += noise_cov
        return x_mean, cov

    # ----------------------------------------------------------
    # PREDICT step
    # ----------------------------------------------------------
    def predict(self, x, P):
        """
        Propagate each sigma point through the Euler-Maruyama step (Stable version).
        Returns predicted mean x_pred and covariance P_pred.
        """
        sigmas = self._sigma_points(x, P)
        
        # Propagate sigma points through EM loop for stability
        sigmas_pred = np.zeros_like(sigmas)
        dt_internal = 0.5
        n_steps = int(self.dt_days / dt_internal)
        
        for i, sp in enumerate(sigmas):
            sp_curr = sp.copy()
            for _ in range(n_steps):
                drift = np.array(ode_model(sp_curr, 0, self.params))
                sp_curr = sp_curr + drift * dt_internal
                sp_curr = np.maximum(sp_curr, 1.0)
            sigmas_pred[i] = sp_curr
            
        # State-proportional Process Noise (5% CV)
        x_pred_approx = np.sum(self.Wm[:, None] * sigmas_pred, axis=0)
        q_std = np.array([
            0.05 * np.maximum(x_pred_approx[0], 1e6),
            0.05 * np.maximum(x_pred_approx[1], 1e6),
            0.05 * np.maximum(x_pred_approx[2], 1e3)
        ])
        Q = np.diag(q_std**2)
        
        x_pred, P_pred = self._unscented_transform(sigmas_pred, Q)
        return x_pred, P_pred, sigmas_pred      # return sigmas for UPDATE reuse

    # ----------------------------------------------------------
    # UPDATE step
    # ----------------------------------------------------------
    def update(self, x_pred, P_pred, sigmas_pred, z_obs):
        """
        Update state with scalar tumor-size observation z_obs (mm).
        Uses same propagated sigma points from predict step.
        """
        # Map sigma points through observation function
        z_sigmas = np.array([[cells_to_mm(sp[0], sp[1])] for sp in sigmas_pred])  # (2n+1, 1)

        z_pred = np.sum(self.Wm[:, None] * z_sigmas, axis=0)   # (1,)

        # Innovation covariance S
        dz     = z_sigmas - z_pred
        S      = (self.Wc[:, None, None] * dz[:, :, None] * dz[:, None, :]).sum(axis=0) + self.R

        # Cross-covariance P_xz
        dx     = sigmas_pred - x_pred
        P_xz   = (self.Wc[:, None, None] * dx[:, :, None] * dz[:, None, :]).sum(axis=0)  # (3,1)

        # Kalman gain
        K = P_xz @ np.linalg.inv(S)                            # (3,1)

        # Innovation
        innovation = np.array([z_obs]) - z_pred                 # (1,)

        # Updated state
        x_upd = x_pred + K @ innovation
        x_upd = np.clip(x_upd, 0, None)

        # Updated covariance
        P_upd = P_pred - K @ S @ K.T
        
        # Stability check: Ensure symmetry and positive-definiteness
        P_upd = 0.5 * (P_upd + P_upd.T)
        min_eig = np.min(np.linalg.eigvals(P_upd))
        if min_eig < 0:
            P_upd += (1e-6 - min_eig) * np.eye(3)
            
        return x_upd, P_upd, float(innovation[0]), float(S[0, 0])

    # ----------------------------------------------------------
    # Observation uncertainty in mm (for plotting)
    # ----------------------------------------------------------
    def obs_uncertainty_mm(self, x, P):
        """
        Propagate state covariance through h(x) via unscented transform
        to get uncertainty in mm (more accurate than delta method).
        """
        sigmas = self._sigma_points(x, P)
        z_sigmas = np.array([[cells_to_mm(sp[0], sp[1])] for sp in sigmas])
        z_mean = np.sum(self.Wm[:, None] * z_sigmas, axis=0)
        dz = z_sigmas - z_mean
        S_mm = (self.Wc[:, None, None] * dz[:, :, None] * dz[:, None, :]).sum(axis=0)
        return float(np.sqrt(max(S_mm[0, 0], 0)))


# ============================================================
# STEP 4: RUN UKF ON EACH PATIENT
# ============================================================

print("\n" + "="*60)
print("RUNNING SDE-BASED UNSCENTED KALMAN FILTER")
print("="*60)

# Timepoints: every 3 months (90 days) up to 120 months (matching chapter5.py)
# We calculate this per patient based on survival below.
# timepoints_months = [0, 3, 6, 9, 12, 18, 24]
# timepoints_days   = [t * 30.44 for t in timepoints_months]

np.random.seed(42)
ukf = SDE_UnscentedKalmanFilter(params, dt_days=90, alpha=0.1, beta=2.0, kappa=0.0)

all_results = []

for _, row in clinical.iterrows():
    pid             = row['patient_id']
    H0              = row['H0_cells']
    tumor_size_mm   = row['tumor_size_mm']
    survival_months = row['survival_months']
    dead            = row['dead']
    t_stage         = row['t_stage']
    overall_stage   = row['overall_stage']
    histology       = row['histology']

    # Initial conditions — 70/30 H/L split so cells_to_mm(H0,L0) == tumor_size_mm
    H0_total = row['H0_cells']
    H0 = 0.7 * H0_total
    L0 = 0.3 * H0_total
    T0 = params['mu_baseline'] / params['delta_T']

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
                                       sigma_H=ukf.sigma_H, sigma_L=ukf.sigma_L, sigma_T=ukf.sigma_T,
                                       seed=patient_seed)
                                       
    # 2. Deterministic Truth (ODE)
    sol_ode = simulate_trajectory(H0, L0, T0, params, t_true)

    # Run filter at each timepoint
    # Run filter at each timepoint
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
            x, P, innovation, S = ukf.update(x, P, ukf._sigma_points(x, P), obs_size)
            est_size  = cells_to_mm(x[0], x[1])
            pred_size = est_size   # no prior prediction at t=0
            unc       = ukf.obs_uncertainty_mm(x, P)

        else:
            x_pred, P_pred, sigmas_pred = ukf.predict(x, P)
            pred_size = cells_to_mm(x_pred[0], x_pred[1])
            x, P, innovation, S = ukf.update(x_pred, P_pred, sigmas_pred, obs_size)
            est_size = cells_to_mm(x[0], x[1])
            unc      = ukf.obs_uncertainty_mm(x, P)

        all_results.append({
            'patient_id'        : pid,
            'timepoint_months'  : t_months,
            'sde_size_mm'       : round(sde_size,  2),
            'ode_size_mm'       : round(ode_size,  2),
            'observed_size_mm'  : round(obs_size,  2),
            'ukf_predicted_mm'  : round(pred_size, 2),
            'ukf_estimate_mm'   : round(est_size,  2),
            'ukf_uncertainty_mm': round(unc,        2),
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
print(f"✓ UKF ran on {results_df['patient_id'].nunique()} patients")
print(f"✓ Total observations: {len(results_df)}")


# ============================================================
# STEP 5: VISUALIZE (6 PATIENTS IN 2x3 GRID)
# ============================================================

print("\nGenerating plots...")

# --- FIGURE 1: Individual patient trajectories (6 patients in 2x3 grid) ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Unscented Kalman Filter — Tumor Dynamics',
             fontsize=16, fontweight='bold')

sample_pids = (results_df.groupby('patient_id')
               .size()[lambda x: x >= 3]
               .sample(min(6, results_df['patient_id'].nunique()),
                       random_state=42).index)

for ax, pid in zip(axes.flatten(), sample_pids):
    pdata  = results_df[results_df['patient_id'] == pid].sort_values('timepoint_months')
    t_months = pdata['timepoint_months'].values
    sde_s    = pdata['sde_size_mm'].values
    ode_s    = pdata['ode_size_mm'].values
    obs_s    = pdata['observed_size_mm'].values
    est_s    = pdata['ukf_estimate_mm'].values
    unc      = pdata['ukf_uncertainty_mm'].values

    # True trajectories
    ax.plot(t_months, sde_s, 'g--', linewidth=2, label='SDE Truth', alpha=0.7)
    ax.plot(t_months, ode_s, 'k:',  linewidth=2, label='ODE Baseline', alpha=0.6)
    ax.scatter(t_months, obs_s, color='red', s=80, zorder=5, label='CT observation', marker='x', linewidth=2)
    ax.plot(t_months, est_s, 'b-o', linewidth=2.5, markersize=6, label='UKF estimate')
    ax.fill_between(t_months,
                    np.maximum(0, est_s - 1.96 * unc),
                    est_s + 1.96 * unc,
                    alpha=0.25, color='blue', label='95% CI')

    stage  = pdata['overall_stage'].iloc[0]
    dead   = pdata['dead'].iloc[0]
    surv   = pdata['survival_months'].iloc[0]
    status = 'Died' if dead == 1 else 'Alive'

    ax.set_title(f'Patient {pid} | {stage}\n{status} ({surv:.0f} mo)', fontsize=11, fontweight='bold')
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
plt.savefig('Fig1_UKF_Trajectories_least_square.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Fig1 saved")


# --- FIGURE 2: N, M, T cell dynamics ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('UKF State Variables — Cell Population Dynamics',
             fontsize=14, fontweight='bold')

sample_pid = results_df['patient_id'].iloc[0]
pdata      = results_df[results_df['patient_id'] == sample_pid]
t          = pdata['timepoint_months'].values

ax = axes[0]
ax.plot(t, pdata['H_cells'].values / 1e9, 'b-o', linewidth=2.5, markersize=6, label='High Antigen (H)')
ax.fill_between(t, 0, pdata['H_cells'].values / 1e9, alpha=0.2, color='blue')
ax.set_title('H — High Antigen Tumor Cells', fontweight='bold', fontsize=12)
ax.set_xlabel('Time (months)', fontsize=10); ax.set_ylabel('Cell count (billions)', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--'); ax.legend(fontsize=10)

ax = axes[1]
ax.plot(t, pdata['L_cells'].values / 1e9, 'r-o', linewidth=2.5, markersize=6, label='Low Antigen (L)')
ax.fill_between(t, 0, pdata['L_cells'].values / 1e9, alpha=0.2, color='red')
ax.set_title('L — Low Antigen Tumor Cells', fontweight='bold', fontsize=12)
ax.set_xlabel('Time (months)', fontsize=10); ax.set_ylabel('Cell count (billions)', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--'); ax.legend(fontsize=10)

ax = axes[2]
ax.plot(t, pdata['T_cells'].values / 1e3, 'g-o', linewidth=2.5, markersize=6, label='CTL (T)')
ax.fill_between(t, 0, pdata['T_cells'].values / 1e3, alpha=0.2, color='green')
ax.set_title('T — CTL T Cells', fontweight='bold', fontsize=12)
ax.set_xlabel('Time (months)', fontsize=10); ax.set_ylabel('Cell count (thousands)', fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--'); ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('Fig2_UKF_Cell_Dynamics_least_square.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Fig2 saved")


# --- FIGURE 3: Filter performance ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('UKF Performance', fontsize=14, fontweight='bold')

err_ukf = (results_df['ukf_estimate_mm'] - results_df['sde_size_mm']).abs()
err_obs = (results_df['observed_size_mm']   - results_df['sde_size_mm']).abs()

ax = axes[0]
ax.hist(err_ukf, bins=30, alpha=0.6, color='blue',
        label=f'UKF (mean={err_ukf.mean():.1f}mm)', edgecolor='black', linewidth=1.2)
ax.hist(err_obs, bins=30, alpha=0.6, color='red',
        label=f'Raw obs (mean={err_obs.mean():.1f}mm)', edgecolor='black', linewidth=1.2)
ax.set_xlabel('Absolute error (mm)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('A. Error Distribution', fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

ax = axes[1]
innov_by_time = results_df.groupby('timepoint_months')['innovation'].agg(
    ['mean', 'std']).reset_index()
ax.plot(innov_by_time['timepoint_months'], innov_by_time['mean'], 'b-o', linewidth=2.5, markersize=7)
ax.fill_between(innov_by_time['timepoint_months'],
                innov_by_time['mean'] - innov_by_time['std'],
                innov_by_time['mean'] + innov_by_time['std'],
                alpha=0.25, color='blue')
ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Timepoint (months)', fontsize=11)
ax.set_ylabel('Innovation (obs - predicted) mm', fontsize=11)
ax.set_title('B. Filter Innovation Over Time\n(should converge to ~0)', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')

ax = axes[2]
stage_order  = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
stage_colors = ['green', 'blue', 'orange', 'red']
unc_by_stage = results_df.groupby('overall_stage')['ukf_uncertainty_mm'].mean()
stages  = [s for s in stage_order if s in unc_by_stage.index]
vals    = [unc_by_stage[s] for s in stages]
colors  = [stage_colors[stage_order.index(s)] for s in stages]
bars = ax.bar(stages, vals, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Mean uncertainty in mm (UKF)', fontsize=11)
ax.set_title('C. Filter Uncertainty by Stage', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('Fig3_UKF_Performance_least_square.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Fig3 saved")

# Save results
results_df.to_csv('UKF_Export.csv', index=False)
print("\n✓ UKF_Export.csv saved (includes ukf_predicted_mm for chapter5_from_csv.py)")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Parameter file:        least_square_parameters.json")
print(f"Patients filtered:     {results_df['patient_id'].nunique()}")
print(f"Mean UKF error:        {err_ukf.mean():.2f} mm")
print(f"Mean raw obs error:    {err_obs.mean():.2f} mm")
print(f"Improvement:           {((err_obs.mean()-err_ukf.mean())/err_obs.mean()*100):.1f}%")
