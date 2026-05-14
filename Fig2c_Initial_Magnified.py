import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import json

# 1. Load parameters
with open('least_square_parameters.json', 'r') as f:
    param_data = json.load(f)
params_raw = param_data['estimated_parameters']

params = {
    'alpha_h': params_raw['alpha_n'], 'alpha_l': params_raw['alpha_m'],
    'K': params_raw['K'], 'delta_hs': params_raw['delta_ns'],
    'delta_ls': params_raw['delta_ms'], 'delta_hf': params_raw['delta_nf'],
    'delta_lf': params_raw['delta_mf'], 'delta_ht': params_raw['delta_n'],
    'delta_lt': params_raw['delta_m'], 'kappa_0': params_raw['kappa_0'],
    'kappa_1': params_raw['kappa_1'], 'kappa_2': params_raw['kappa_2'],
    'delta_T': params_raw['delta_t'], 'mu_baseline': params_raw['mu'],
    'alpha_ht_baseline': params_raw['alpha_nt'], 'alpha_lt_baseline': params_raw['alpha_mt'],
    'p1_baseline': params_raw['p1'], 'p2_baseline': params_raw['p2'],
}
rho_p, rho_l, k_TQ, epsilon_c, mu_PA = params_raw['rho_p'], params_raw['rho_l'], params_raw['k_TQ'], params_raw['epsilon_c'], params_raw['mu_PA']

def compute_F(T, N, M, A_drug):
    P = rho_p * T / (1.0 + mu_PA * A_drug + 1e-30)
    L = rho_l * (T + epsilon_c * (N + M))
    F = 1.0 / (1.0 + P * L / k_TQ + 1e-30)
    return float(np.clip(F, 1e-4, 1.0))

def ode_dynamic_F(y, t, p, A_drug):
    H, L, T = max(y[0], 0), max(y[1], 0), max(y[2], 0)
    A_curr = A_drug * (1.0 - np.exp(-t / 120.0))
    F = compute_F(T, H, L, A_curr)
    denom = max(p['kappa_1'] * T + (H + L) + p['kappa_0'], 1e-10)
    dH = (p['alpha_h'] * H * (1 - (H + L) / p['K']) - p['delta_hs'] * (1 - p['p1_baseline']) * (H / denom) * T - p['delta_hf'] * p['p1_baseline'] * H * T)
    dL = (p['alpha_l'] * L * (1 - (H + L) / p['K']) - p['delta_ls'] * (1 - p['p2_baseline']) * (L / denom) * T - p['delta_lf'] * p['p2_baseline'] * L * T)
    dT = ((p['mu_baseline'] + p['alpha_ht_baseline'] * (H / (p['kappa_2'] + H)) * T + p['alpha_lt_baseline'] * (L / (p['kappa_2'] + L)) * T) * F - p['delta_ht'] * H * T - p['delta_lt'] * L * T - p['delta_T'] * T)
    return [dH, dL, dT]

def initial_conditions_for_F(F_target, T0):
    if F_target >= 0.999: NM = 1e3
    elif F_target <= 0.011: NM = 1.572e10
    else:
        numerator = (1.0 / F_target - 1.0) * k_TQ / (rho_p * rho_l * T0)
        NM = max((numerator - T0) / epsilon_c, 1e3)
    return NM * 0.75, NM * 0.25

# 2. Setup Simulation
T0 = params['mu_baseline'] / params['delta_T']
t_months = np.linspace(0, 3, 500) # ZOOMED: 0 to 3 months
t_days = t_months * 30.44
F_targets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
A_drug_treated = 4.4e11

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Magnified View: Initial 3 Months of Treatment Dynamics', fontsize=18, fontweight='bold', y=0.98)

for idx, (ax, Ft) in enumerate(zip(axes.flatten(), F_targets)):
    H0, L0 = initial_conditions_for_F(Ft, T0)
    y0 = [H0, L0, T0]
    
    # Baseline
    sol_b = odeint(ode_dynamic_F, y0, t_days, args=(params, 0.0))
    # Treated
    sol_t = odeint(ode_dynamic_F, y0, t_days, args=(params, A_drug_treated))

    ax2 = ax.twinx()
    
    # Treated Tumor
    lH, = ax.plot(t_months, sol_t[:, 0]/1e6, color='#2166ac', lw=2, label='H (Treated)')
    lL, = ax.plot(t_months, sol_t[:, 1]/1e6, color='#d6604d', lw=2, label='L (Treated)')
    # Treated CTL
    lT, = ax2.plot(t_months, sol_t[:, 2]/1e3, color='#1a9850', lw=2, ls='--', label='CTL (Treated)')
    
    # Baseline comparison (faint)
    ax.plot(t_months, (sol_b[:, 0]+sol_b[:, 1])/1e6, color='black', alpha=0.1, lw=1)

    ax.set_title(f'Initial F ≈ {Ft:.1f}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Time (months)')
    ax.set_ylabel('Tumor (Millions)')
    ax2.set_ylabel('CTL (Thousands)', color='#1a9850')
    ax2.tick_params(axis='y', labelcolor='#1a9850')
    ax.set_xlim(0, 3)
    ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('Fig2c_Initial_Magnified.png', dpi=150)
print("✓ Saved: Fig2c_Initial_Magnified.png")
