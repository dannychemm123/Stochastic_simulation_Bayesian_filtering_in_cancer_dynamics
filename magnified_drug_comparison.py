import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import json
import warnings
warnings.filterwarnings('ignore')

# 1. Load parameters
with open('least_square_parameters.json', 'r') as f:
    param_data = json.load(f)
params_raw = param_data['estimated_parameters']

p = {
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

T0 = p['mu_baseline'] / p['delta_T']
A_drug = 4.4e11
TAU = 120.0
F_targets = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
COLORS = {'H': '#2166ac', 'L': '#d6604d', 'T': '#1a9850'}

def compute_F(T, N, M, A):
    P = rho_p * T / (1.0 + mu_PA * A + 1e-30)
    L = rho_l * (T + epsilon_c * (N + M))
    F = 1.0 / (1.0 + P * L / k_TQ + 1e-30)
    return float(np.clip(F, 1e-4, 1.0))

def ode_func(y, t, A_max):
    H, L, T = max(y[0], 0), max(y[1], 0), max(y[2], 0)
    A_curr = A_max * (1.0 - np.exp(-t / TAU))
    F = compute_F(T, H, L, A_curr)
    denom = max(p['kappa_1'] * T + (H + L) + p['kappa_0'], 1e-10)
    dH = (p['alpha_h'] * H * (1 - (H + L) / p['K']) - p['delta_hs'] * (1 - p['p1_baseline']) * (H / denom) * T - p['delta_hf'] * p['p1_baseline'] * H * T)
    dL = (p['alpha_l'] * L * (1 - (H + L) / p['K']) - p['delta_ls'] * (1 - p['p2_baseline']) * (L / denom) * T - p['delta_lf'] * p['p2_baseline'] * L * T)
    dT = ((p['mu_baseline'] + p['alpha_ht_baseline'] * (H / (p['kappa_2'] + H)) * T + p['alpha_lt_baseline'] * (L / (p['kappa_2'] + L)) * T) * F - p['delta_ht'] * H * T - p['delta_lt'] * L * T - p['delta_T'] * T)
    return [dH, dL, dT]

def get_initial(Ft):
    if Ft >= 0.999: NM = 1e3
    elif Ft <= 0.011: NM = 1.572e10
    else: NM = max(((1.0/Ft - 1.0)*k_TQ/(rho_p*rho_l*T0) - T0)/epsilon_c, 1e3)
    return [NM*0.75, NM*0.25, T0]

# 2. Main Combined Plot
fig, axes = plt.subplots(4, 3, figsize=(18, 20))
t_months = np.linspace(0, 3, 200)
t_days = t_months * 30.44

print("Simulating ODE and SDE magnified trajectories...")
for i, Ft in enumerate(F_targets):
    y0 = get_initial(Ft)
    
    # ODE ROW (Determinstic)
    ax_ode = axes[i // 3 * 2, i % 3]
    ax2_ode = ax_ode.twinx()
    sol_ode = odeint(ode_func, y0, t_days, args=(A_drug,))
    ax_ode.plot(t_months, sol_ode[:,0]/1e6, color=COLORS['H'], lw=2, label='H = High Antigen')
    ax_ode.plot(t_months, sol_ode[:,1]/1e6, color=COLORS['L'], lw=2, label='L = Low Antigen')
    ax2_ode.plot(t_months, sol_ode[:,2]/1e3, color=COLORS['T'], lw=2, ls='-', label='T = T cells')
    ax_ode.set_title(f'ODE: Initial F ≈ {Ft:.1f}', fontweight='bold')
    if i == 0: 
        lns = ax_ode.get_lines() + ax2_ode.get_lines()
        lbls = [l.get_label() for l in lns]
        ax_ode.legend(lns, lbls, loc='upper right', fontsize=10)
    ax_ode.set_xlim(0, 3); ax_ode.grid(True, alpha=0.1)

    # SDE ROW (Stochastic)
    ax_sde = axes[i // 3 * 2 + 1, i % 3]
    ax2_sde = ax_sde.twinx()
    s_runs = []
    for r in range(10):
        # Mini inline SDE solver for simplicity
        curr_x, curr_t, dt = np.array(y0), 0.0, 0.1
        res = []
        for td in t_days:
            while curr_t < td:
                d = np.array(ode_func(curr_x, curr_t, A_drug))
                curr_x = np.maximum(curr_x + d*0.1 + np.array([5e5,5e5,1e4])*np.random.normal(0,1,3)*np.sqrt(0.1), 1.0)
                curr_t += 0.1
            res.append(curr_x.copy())
        s_runs.append(np.array(res))
    
    hm, hs = np.mean([r[:,0]/1e6 for r in s_runs], axis=0), np.std([r[:,0]/1e6 for r in s_runs], axis=0)
    lm, ls = np.mean([r[:,1]/1e6 for r in s_runs], axis=0), np.std([r[:,1]/1e6 for r in s_runs], axis=0)
    tm, ts = np.mean([r[:,2]/1e3 for r in s_runs], axis=0), np.std([r[:,2]/1e3 for r in s_runs], axis=0)
    
    ax_sde.fill_between(t_months, hm-hs, hm+hs, color=COLORS['H'], alpha=0.1)
    ax_sde.fill_between(t_months, lm-ls, lm+ls, color=COLORS['L'], alpha=0.1)
    ax2_sde.fill_between(t_months, tm-ts, tm+ts, color=COLORS['T'], alpha=0.1)
    ax_sde.plot(t_months, hm, color=COLORS['H'], lw=2, label='H = High Antigen')
    ax_sde.plot(t_months, lm, color=COLORS['L'], lw=2, label='L = Low Antigen')
    ax2_sde.plot(t_months, tm, color=COLORS['T'], lw=2, ls='-', label='T = T cells')
    ax_sde.set_title(f'SDE: Initial F ≈ {Ft:.1f}', fontweight='bold')
    if i == 0: 
        lns = ax_sde.get_lines() + ax2_sde.get_lines()
        lbls = [l.get_label() for l in lns]
        ax_sde.legend(lns, lbls, loc='upper right', fontsize=10)
    ax_sde.set_xlim(0, 3); ax_sde.grid(True, alpha=0.1)

plt.tight_layout()
plt.savefig('Fig_Magnified_Combined.png', dpi=150)
print("✓ Saved: Fig_Magnified_Combined.png")
