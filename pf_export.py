import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import json
import multiprocessing as mp
from functools import partial
import hashlib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: LOAD PARAMETERS AND DATA
# ============================================================

print("="*60)
print("LOADING PARAMETERS AND DATA")
print("="*60)

with open('least_square_parameters.json', 'r') as f:
    param_data = json.load(f)

params_raw = param_data['estimated_parameters']

params = {
    'alpha_h':           params_raw['alpha_n'],
    'alpha_l':           params_raw['alpha_m'],
    'delta_hs':          params_raw['delta_ns'],
    'delta_ls':          params_raw['delta_ms'],
    'p1_baseline':       params_raw['p1'],
    'p2_baseline':       params_raw['p2'],
    'delta_hf':          params_raw['delta_nf'],
    'delta_lf':          params_raw['delta_mf'],
    'mu_baseline':       params_raw['mu'],
    'delta_T':           params_raw['delta_t'],
    'delta_ht':          params_raw['delta_n'],
    'delta_lt':          params_raw['delta_m'],
    'alpha_ht_baseline': params_raw['alpha_nt'],
    'alpha_lt_baseline': params_raw['alpha_mt'],
    'rho_p_baseline':    params_raw['rho_p'],
    'rho_l_baseline':    params_raw['rho_l'],
    'K':                 params_raw['K'],
    'kappa_0':           params_raw['kappa_0'],
    'kappa_1':           params_raw['kappa_1'],
    'kappa_2':           params_raw['kappa_2'],
    'epsilon_c':         params_raw['epsilon_c'],
    'k_TQ_baseline':     params_raw['k_TQ']
}

print("Using: least_square_parameters.json")
print(f"  p1_baseline:  {params['p1_baseline']:.4f}")
print(f"  p2_baseline:  {params['p2_baseline']:.4f}")
print(f"  alpha_h:      {params['alpha_h']:.4f}")
print(f"  mu_baseline:  {params['mu_baseline']:.2f}")
print(f"  delta_hs:     {params['delta_hs']:.4f}")

clinical = pd.read_csv('NSCLC_Clinical_Cleaned.csv')

# ── Cell counts consistent with cells_to_mm (rho = 1e6 cells/mm³) ──────────
# total = (4/3)*pi*(d/2)^3 * 1e6;  H0 = 0.7*total;  L0 = 0.3*total
# so cells_to_mm(H0, L0) == d  exactly at t=0
t_stage_to_cells = {
    1.0: 1.77e9,    # T1: ~15 mm
    2.0: 3.35e10,   # T2: ~40 mm
    3.0: 1.13e11,   # T3: ~60 mm
    4.0: 2.68e11    # T4: ~80 mm
}

t_stage_to_mm = {1.0: 15, 2.0: 40, 3.0: 60, 4.0: 80}

clinical['H0_cells']      = clinical['t_stage'].map(t_stage_to_cells)
clinical['tumor_size_mm'] = clinical['t_stage'].map(t_stage_to_mm)
clinical = clinical.dropna(subset=['H0_cells', 'tumor_size_mm'])

print(f"\nPatients loaded: {len(clinical)}")
print(f"T-stage distribution:\n{clinical['t_stage'].value_counts().sort_index()}")


# ============================================================
# STEP 2: ODE MODEL — scalar (for SDE truth) and vectorised (for PF)
# ============================================================

def ode_model(y, t, p):
    """Scalar ODE — used only for SDE ground-truth simulation."""
    H, L, T = y
    H = max(H, 0); L = max(L, 0); T = max(T, 0)

    P    = p['rho_p_baseline'] * T
    L_pd = p['rho_l_baseline'] * (T + p['epsilon_c'] * (H + L))
    F    = np.clip(1.0 / (1.0 + (P * L_pd) / p['k_TQ_baseline']), 0, 1)
    denom = max(p['kappa_1']*T + H + L + p['kappa_0'], 1e-10)

    dH = (p['alpha_h'] * H * (1 - (H+L)/p['K'])
          - p['delta_hs'] * (1-p['p1_baseline']) * (H/denom) * T
          - p['delta_hf'] * p['p1_baseline'] * H * T)

    dL = (p['alpha_l'] * L * (1 - (H+L)/p['K'])
          - p['delta_ls'] * (1-p['p2_baseline']) * (L/denom) * T
          - p['delta_lf'] * p['p2_baseline'] * L * T)

    dT = ((p['mu_baseline']
           + p['alpha_ht_baseline'] * (H/(p['kappa_2']+H)) * T
           + p['alpha_lt_baseline'] * (L/(p['kappa_2']+L)) * T) * F
          - p['delta_ht'] * H * T
          - p['delta_lt'] * L * T
          - p['delta_T'] * T)

    return [dH, dL, dT]


def ode_model_vec(particles, p):
    """
    Vectorised ODE — operates on ALL particles simultaneously.
    particles: (N, 3) array  [H, L, T] per row
    Returns:   (N, 3) drift array
    FIX: replaces the inner Python loop in predict(), giving ~50-100x speedup.
    """
    H = np.maximum(particles[:, 0], 0.0)
    L = np.maximum(particles[:, 1], 0.0)
    T = np.maximum(particles[:, 2], 0.0)

    P    = p['rho_p_baseline'] * T
    L_pd = p['rho_l_baseline'] * (T + p['epsilon_c'] * (H + L))
    F    = np.clip(1.0 / (1.0 + (P * L_pd) / p['k_TQ_baseline']), 0, 1)
    denom = np.maximum(p['kappa_1']*T + H + L + p['kappa_0'], 1e-10)

    dH = (p['alpha_h'] * H * (1 - (H+L)/p['K'])
          - p['delta_hs'] * (1-p['p1_baseline']) * (H/denom) * T
          - p['delta_hf'] * p['p1_baseline'] * H * T)

    dL = (p['alpha_l'] * L * (1 - (H+L)/p['K'])
          - p['delta_ls'] * (1-p['p2_baseline']) * (L/denom) * T
          - p['delta_lf'] * p['p2_baseline'] * L * T)

    dT = ((p['mu_baseline']
           + p['alpha_ht_baseline'] * (H/(p['kappa_2']+H)) * T
           + p['alpha_lt_baseline'] * (L/(p['kappa_2']+L)) * T) * F
          - p['delta_ht'] * H * T
          - p['delta_lt'] * L * T
          - p['delta_T'] * T)

    return np.stack([dH, dL, dT], axis=1)


def cells_to_mm(H, L):
    """Scalar: cell count → tumour diameter (mm)."""
    volume_mm3 = max(H + L, 0) / 1e6
    return max(2 * (3 * volume_mm3 / (4 * np.pi)) ** (1/3), 0.1)


def cells_to_mm_vec(H_arr, L_arr):
    """Vectorised: cell count arrays → tumour diameter array (mm)."""
    volume = np.maximum(H_arr + L_arr, 0) / 1e6
    return np.maximum(2.0 * (3.0 * volume / (4.0 * np.pi)) ** (1.0/3.0), 0.1)


# Sanity check
print("\nConsistency check  (cells_to_mm(0.7*total, 0.3*total) should == d):")
for stage in [1.0, 2.0, 3.0, 4.0]:
    total    = t_stage_to_cells[stage]
    H0, L0   = 0.7 * total, 0.3 * total          # correct 70/30 split
    mm       = cells_to_mm(H0, L0)
    expected = t_stage_to_mm[stage]
    err      = abs(mm - expected)
    print(f"  T{int(stage)}: {total:.2e} cells → {mm:.1f} mm  "
          f"(expected {expected} mm, error {err:.2f} mm)")


def simulate_sde_trajectory(H0, L0, T0, params, t_span_days,
                             sigma_H=1e5, sigma_L=1e5, sigma_T=1e4, seed=None):
    """Euler-Maruyama SDE simulation (scalar, for ground truth)."""
    rng = np.random.RandomState(seed) if seed is not None else np.random
    dt  = 0.5
    X   = np.zeros((len(t_span_days), 3))
    X[0]      = [H0, L0, T0]
    current_x = np.array([H0, L0, T0], dtype=float)
    current_t = t_span_days[0]
    sigma     = np.array([sigma_H, sigma_L, sigma_T])
    for i in range(1, len(t_span_days)):
        target_t = t_span_days[i]
        while current_t < target_t:
            step      = min(dt, target_t - current_t)
            drift     = np.array(ode_model(current_x, current_t, params))
            dW        = rng.normal(0, 1, 3) * np.sqrt(step)
            current_x = np.maximum(current_x + drift*step + sigma*dW, 1.0)
            current_t += step
        X[i] = current_x
    return X


def simulate_trajectory(H0, L0, T0, params, t_span):
    """Deterministic ODE simulation (for ODE baseline)."""
    sol = odeint(ode_model, [H0, L0, T0], t_span,
                 args=(params,), rtol=1e-6, atol=1e-9)
    return np.clip(sol, 0, None)


# ============================================================
# STEP 3: PARTICLE FILTER
# ============================================================

class SDE_ParticleFilter:
    """
    Sequential Importance Resampling Particle Filter.

    Two fixes vs original:
    1. predict() uses vectorised ODE (ode_model_vec) — no inner Python loop.
    2. obs_uncertainty_mm() reports the posterior std of the particle cloud
       only, WITHOUT adding R.  Adding R was the bug that caused sigma_bar
       to equal CT noise (~5 mm) regardless of filter quality.
    """

    def __init__(self, params, N_particles=500, dt_days=90):
        self.params      = params
        self.N_particles = N_particles
        self.dt_days     = dt_days
        self.R           = 5.0          # CT obs noise std (mm)
        self.sigma_H     = 1e5
        self.sigma_L     = 1e5
        self.sigma_T     = 1e4
        self.resample_threshold = 0.5

    def _initialize_particles(self, x0):
        particles = np.zeros((self.N_particles, 3))
        for i in range(3):
            particles[:, i] = x0[i] * (
                1 + np.random.normal(0, 0.05, self.N_particles))
        particles = np.clip(particles, 0, None)
        weights   = np.ones(self.N_particles) / self.N_particles
        return particles, weights

    def predict(self, particles, weights):
        """
        Vectorised Euler-Maruyama propagation.
        FIX: replaces double Python loop with single NumPy call per EM step.
        """
        dt_internal = 0.5
        n_steps     = int(self.dt_days / dt_internal)
        pts         = particles.copy()

        for _ in range(n_steps):
            drift = ode_model_vec(pts, self.params)   # (N,3) — NO inner loop
            pts   = np.maximum(pts + drift * dt_internal, 1.0)

        # Additive diffusion for the full interval
        sigma = np.array([self.sigma_H, self.sigma_L, self.sigma_T])
        dW    = np.random.normal(0, 1, pts.shape) * np.sqrt(self.dt_days)
        pts   = np.maximum(pts + sigma * dW, 1.0)

        return pts, weights

    def update(self, particles_pred, weights, z_obs):
        z_pred     = cells_to_mm_vec(particles_pred[:, 0], particles_pred[:, 1])
        likelihood = np.exp(-0.5 * ((z_pred - z_obs) / self.R) ** 2)
        weights_new = weights * likelihood
        weights_new = weights_new / (np.sum(weights_new) + 1e-20)
        N_eff = 1.0 / np.sum(weights_new ** 2)
        if N_eff < self.resample_threshold * self.N_particles:
            particles_pred, weights_new = self._systematic_resample(
                particles_pred, weights_new)
        return particles_pred, weights_new, z_pred, likelihood

    def _systematic_resample(self, particles, weights):
        positions = (np.arange(self.N_particles) +
                     np.random.uniform()) / self.N_particles
        cumsum  = np.cumsum(weights)
        indices = np.clip(np.searchsorted(cumsum, positions),
                          0, self.N_particles - 1)
        return particles[indices].copy(), \
               np.ones(self.N_particles) / self.N_particles

    def get_state_estimate(self, particles, weights):
        return np.sum(weights[:, None] * particles, axis=0)

    def get_state_covariance(self, particles, weights, x_hat):
        diff = particles - x_hat
        return np.sum(
            weights[:, None, None] * diff[:, :, None] * diff[:, None, :],
            axis=0)

    def obs_uncertainty_mm(self, particles, weights):
        """
        FIX: returns the POSTERIOR std of the particle cloud in observation
        space only — does NOT add R.

        Original code added self.R (=5 mm) unconditionally, which forced
        sigma_bar ≈ 5 mm for every patient and stage, making the reported
        uncertainty indistinguishable from raw CT noise even when the
        particle cloud was tightly concentrated.

        The posterior std correctly reflects how much the filter has reduced
        uncertainty relative to the prior; R is the measurement noise and
        belongs in the likelihood, not in the reported posterior width.
        """
        z_pred = cells_to_mm_vec(particles[:, 0], particles[:, 1])
        z_mean = np.sum(weights * z_pred)
        z_std  = np.sqrt(np.sum(weights * (z_pred - z_mean) ** 2))
        return z_std          # posterior std only — R removed


# ============================================================
# STEP 4: PER-PATIENT FILTER FUNCTION
# ============================================================

def run_patient(idx_row_tuple, params):
    pid             = idx_row_tuple[1]['patient_id']
    H0_total        = idx_row_tuple[1]['H0_cells']
    tumor_size_mm   = idx_row_tuple[1]['tumor_size_mm']
    survival_months = idx_row_tuple[1]['survival_months']
    dead            = idx_row_tuple[1]['dead']
    t_stage         = idx_row_tuple[1]['t_stage']
    overall_stage   = idx_row_tuple[1]['overall_stage']
    idx             = idx_row_tuple[0]

    # FIX: correct 70/30 split so H0+L0 = total and cells_to_mm == tumor_size_mm
    H0 = 0.7 * H0_total
    L0 = 0.3 * H0_total
    T0 = params['mu_baseline'] / params['delta_T']

    pf = SDE_ParticleFilter(params, N_particles=500, dt_days=90)
    particles, weights = pf._initialize_particles(np.array([H0, L0, T0]))

    t_true = np.linspace(0, 24 * 30.44, 200)

    patient_seed = int(hashlib.md5(pid.encode()).hexdigest(), 16) % (2**32)
    sol_sde = simulate_sde_trajectory(H0, L0, T0, params, t_true,
                                      sigma_H=pf.sigma_H,
                                      sigma_L=pf.sigma_L,
                                      sigma_T=pf.sigma_T,
                                      seed=patient_seed)
    sol_ode = simulate_trajectory(H0, L0, T0, params, t_true)

    # Define timepoints for this patient: every 3 months up to survival (max 24 months)
    t_months_list = np.arange(0, min(24, int(survival_months) + 1), 3)
    t_days_list   = t_months_list * 30.44

    patient_results = []
    for t_idx, (t_months, t_days) in enumerate(zip(t_months_list, t_days_list)):

        if t_months > survival_months:
            break

        true_idx = np.argmin(np.abs(t_true - t_days))
        sde_size = cells_to_mm(sol_sde[true_idx, 0], sol_sde[true_idx, 1])
        ode_size = cells_to_mm(sol_ode[true_idx, 0], sol_ode[true_idx, 1])

        np.random.seed(42 + idx * 1000 + t_idx)
        obs_size = max(1.0, sde_size + np.random.normal(0, 5.0))

        if t_idx == 0:
            particles, weights, z_pred, likelihood = pf.update(
                particles, weights, obs_size)
            x_hat     = pf.get_state_estimate(particles, weights)
            pred_size = cells_to_mm(x_hat[0], x_hat[1])  # no prior pred at t=0
        else:
            particles, weights = pf.predict(particles, weights)
            x_pred_hat = pf.get_state_estimate(particles, weights)
            pred_size  = cells_to_mm(x_pred_hat[0], x_pred_hat[1])
            particles, weights, z_pred, likelihood = pf.update(
                particles, weights, obs_size)
            x_hat = pf.get_state_estimate(particles, weights)

        cov       = pf.get_state_covariance(particles, weights, x_hat)
        unc       = pf.obs_uncertainty_mm(particles, weights)
        est_size  = cells_to_mm(x_hat[0], x_hat[1])
        innovation = obs_size - float(np.sum(weights * z_pred))

        patient_results.append({
            'patient_id'       : pid,
            'timepoint_months' : t_months,
            'sde_size_mm'      : round(sde_size,   2),
            'ode_size_mm'      : round(ode_size,   2),
            'observed_size_mm' : round(obs_size,   2),
            'pf_predicted_mm'  : round(pred_size,  2),
            'pf_estimate_mm'   : round(est_size,   2),
            'pf_uncertainty_mm': round(unc,         4),
            'H_cells'          : round(x_hat[0]),
            'L_cells'          : round(x_hat[1]),
            'T_cells'          : round(x_hat[2]),
            'innovation'       : round(innovation,  3),
            'dead'             : dead,
            'survival_months'  : survival_months,
            'overall_stage'    : overall_stage,
            't_stage'          : t_stage,
            'N_particles'      : 500
        })

    return patient_results


# ============================================================
# STEP 5: RUN IN PARALLEL
# ============================================================

print("\n" + "="*60)
print("RUNNING PARTICLE FILTER (500 PARTICLES/PATIENT, VECTORISED)")
print("="*60)

row_dicts = [(idx, row.to_dict()) for idx, row in clinical.iterrows()]

n_workers = max(1, mp.cpu_count() - 1)
print(f"Using {n_workers} workers...")

task_fn = partial(run_patient, params=params)

with mp.Pool(processes=n_workers) as pool:
    results_list = pool.map(task_fn, row_dicts)

all_results = [item for sublist in results_list for item in sublist]
results_df  = pd.DataFrame(all_results)

print(f"✓ Particle filter ran on {results_df['patient_id'].nunique()} patients")
print(f"✓ Total observations: {len(results_df)}")


# ============================================================
# STEP 6: VISUALISE
# ============================================================

print("\nGenerating plots...")

# --- Figure 1: 6 patient trajectories ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Particle Filter — Tumor Dynamics',
             fontsize=16, fontweight='bold')

sample_pids = (results_df.groupby('patient_id')
               .size()[lambda x: x >= 3]
               .sample(min(6, results_df['patient_id'].nunique()),
                       random_state=42).index)

for ax, pid in zip(axes.flatten(), sample_pids):
    pdata    = results_df[results_df['patient_id'] == pid].sort_values(
        'timepoint_months')
    t        = pdata['timepoint_months'].values
    sde_s    = pdata['sde_size_mm'].values
    ode_s    = pdata['ode_size_mm'].values
    obs_s    = pdata['observed_size_mm'].values
    est_s    = pdata['pf_estimate_mm'].values
    unc      = pdata['pf_uncertainty_mm'].values

    ax.plot(t, sde_s, 'g--', lw=2,   label='SDE truth',      alpha=0.7)
    ax.plot(t, ode_s, 'k:',  lw=2,   label='ODE baseline',   alpha=0.6)
    ax.scatter(t, obs_s, color='red', s=80, zorder=5,
               label='CT observation', marker='x', linewidths=2)
    ax.plot(t, est_s, 'b-o', lw=2.5, markersize=6,
               label='PF estimate')
    ax.fill_between(t,
                    np.maximum(0, est_s - 1.96*unc),
                    est_s + 1.96*unc,
                    alpha=0.25, color='blue', label='95% CI')

    stage  = pdata['overall_stage'].iloc[0]
    dead   = pdata['dead'].iloc[0]
    surv   = pdata['survival_months'].iloc[0]
    status = 'Died' if dead == 1 else 'Alive'
    ax.set_title(f'Patient {pid} | {stage}\n{status} ({surv:.0f} mo)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Time (months)', fontsize=10)
    ax.set_ylabel('Tumor size (mm)', fontsize=10)
    ax.set_xlim(0, 24); ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=9)

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5,
           fontsize=10, bbox_to_anchor=(0.5, -0.02))
plt.subplots_adjust(bottom=0.12, hspace=0.35, wspace=0.3)
plt.savefig('Fig1_PF_Trajectories_least_square.png', dpi=150,
            bbox_inches='tight')
print("✓ Fig1 saved")

# --- Figure 2: Cell dynamics ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Particle Filter State Variables — Cell Population Dynamics',
             fontsize=14, fontweight='bold')

sample_pid = results_df['patient_id'].iloc[0]
pdata      = results_df[results_df['patient_id'] == sample_pid]
t          = pdata['timepoint_months'].values

for ax, col, label, color, unit, scale in [
    (axes[0], 'H_cells', 'High Antigen (H)', 'blue',  'billions', 1e9),
    (axes[1], 'L_cells', 'Low Antigen (L)',  'red',   'billions', 1e9),
    (axes[2], 'T_cells', 'CTL (T)',          'green', 'thousands',1e3),
]:
    vals = pdata[col].values / scale
    ax.plot(t, vals, f'{color[0]}-o', lw=2.5, markersize=6, label=label)
    ax.fill_between(t, 0, vals, alpha=0.2, color=color)
    ax.set_title(f'{col.split("_")[0]} — {label}',
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('Time (months)', fontsize=10)
    ax.set_ylabel(f'Cell count ({unit})', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('Fig2_PF_Cell_Dynamics_least_square.png', dpi=150,
            bbox_inches='tight')
print("✓ Fig2 saved")

# --- Figure 3: Performance ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Particle Filter Performance', fontsize=14, fontweight='bold')

err_pf  = (results_df['pf_estimate_mm'] - results_df['sde_size_mm']).abs()
err_obs = (results_df['observed_size_mm'] - results_df['sde_size_mm']).abs()

ax = axes[0]
ax.hist(err_pf,  bins=30, alpha=0.6, color='blue',
        label=f'PF (mean={err_pf.mean():.1f} mm)',
        edgecolor='black', linewidth=1.2)
ax.hist(err_obs, bins=30, alpha=0.6, color='red',
        label=f'Raw obs (mean={err_obs.mean():.1f} mm)',
        edgecolor='black', linewidth=1.2)
ax.set_xlabel('Absolute error (mm)', fontsize=11)
ax.set_ylabel('Count', fontsize=11)
ax.set_title('A. Error Distribution', fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

ax = axes[1]
innov = results_df.groupby('timepoint_months')['innovation'].agg(
    ['mean','std']).reset_index()
ax.plot(innov['timepoint_months'], innov['mean'], 'b-o', lw=2.5, markersize=7)
ax.fill_between(innov['timepoint_months'],
                innov['mean'] - innov['std'],
                innov['mean'] + innov['std'],
                alpha=0.25, color='blue')
ax.axhline(0, color='red', linestyle='--', lw=2)
ax.set_xlabel('Timepoint (months)', fontsize=11)
ax.set_ylabel('Innovation (obs − predicted) mm', fontsize=11)
ax.set_title('B. Filter Innovation Over Time\n(should converge to ~0)',
             fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')

ax = axes[2]
stage_order  = ['Stage I', 'Stage II', 'Stage III', 'Stage IV']
stage_colors = ['green', 'blue', 'orange', 'red']
unc_by_stage = results_df.groupby('overall_stage')['pf_uncertainty_mm'].mean()
stages  = [s for s in stage_order if s in unc_by_stage.index]
vals    = [unc_by_stage[s] for s in stages]
colors  = [stage_colors[stage_order.index(s)] for s in stages]
bars = ax.bar(stages, vals, color=colors, alpha=0.7,
              edgecolor='black', linewidth=1.5)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{val:.2f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('Mean posterior uncertainty (mm)', fontsize=11)
ax.set_title('C. Filter Uncertainty by Stage', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

plt.tight_layout()
plt.savefig('Fig3_PF_Performance_least_square.png', dpi=150,
            bbox_inches='tight')
print("✓ Fig3 saved")

results_df.to_csv('PF_Export.csv', index=False)
print("\n✓ PF_Export.csv saved (includes pf_predicted_mm for chapter5_from_csv.py)")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Patients filtered:     {results_df['patient_id'].nunique()}")
print(f"Particles per patient: 500")
print(f"Mean PF error:         {err_pf.mean():.2f} mm")
print(f"Mean raw obs error:    {err_obs.mean():.2f} mm")
print(f"Improvement:           "
      f"{(err_obs.mean()-err_pf.mean())/err_obs.mean()*100:.1f}%")
print(f"Mean PF uncertainty:   "
      f"{results_df['pf_uncertainty_mm'].mean():.2f} mm  "
      f"(should now be << 5 mm)")