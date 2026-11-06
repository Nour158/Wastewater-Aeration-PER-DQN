"""Compare five constant blower settings in the simplified wastewater environment.
Run from project root: python .\02_Python_Code\validate_environment.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wastewater_env import WastewaterAerationEnv

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / '01_Dataset' / 'processed' / 'dry_influent.csv'
RESULTS = ROOT / '03_Results' / 'validation'
RESULTS.mkdir(parents=True, exist_ok=True)

using_data = DATA_PATH.exists()
print(f"Input source: {'BSM1-derived processed dry influent' if using_data else 'synthetic constant influent (fallback)'}")
if not using_data:
    print('NOTE: No processed CSV found; this tests the synthetic fallback, not BSM1 data.')

records = []
summary = []
for action, pct in enumerate((0, 25, 50, 75, 100)):
    env = WastewaterAerationEnv(csv_path=DATA_PATH if using_data else None)
    obs, _ = env.reset(seed=42)  # identical starting point for every blower setting
    initial_do, initial_bod = float(obs[0]), float(obs[1])
    total_reward = 0.0
    total_energy = 0.0
    do_values = []
    bod_values = []
    rewards = []
    for step in range(env.cfg.episode_steps):
        obs, reward, terminated, truncated, info = env.step(action)
        do = float(info['do_mg_l'])
        bod = float(info['bod_proxy_mg_l'])
        energy = float(info['energy_proxy'])
        if not np.all(np.isfinite(obs)) or not np.isfinite(reward):
            raise RuntimeError(f'Non-finite value at blower={pct}%, step={step+1}')
        if not (0 <= do <= env.cfg.do_saturation_mg_l and 0 <= bod <= 1000):
            raise RuntimeError(f'State out of bounds at blower={pct}%, step={step+1}')
        total_reward += float(reward)
        total_energy += energy
        do_values.append(do)
        bod_values.append(bod)
        rewards.append(float(reward))
        records.append({'blower_pct': pct, 'step': step+1,
                        'hours': (step+1)*env.cfg.dt_hours,
                        'DO_mg_L': do, 'BOD_proxy_mg_L': bod,
                        'energy_proxy': energy, 'reward': float(reward)})
        if terminated or truncated:
            break
    summary.append({
        'blower_pct': pct,
        'initial_DO': initial_do, 'final_DO': do_values[-1],
        'mean_DO': np.mean(do_values),
        'pct_steps_DO_below_2': 100*np.mean(np.array(do_values) < env.cfg.min_do_mg_l),
        'pct_steps_DO_2_to_3': 100*np.mean((np.array(do_values) >= 2) & (np.array(do_values) <= 3)),
        'initial_BOD_proxy': initial_bod, 'final_BOD_proxy': bod_values[-1],
        'mean_BOD_proxy': np.mean(bod_values),
        'energy_proxy_total': total_energy,
        'total_reward': total_reward,
        'steps': len(do_values),
    })
    env.close()

history = pd.DataFrame(records)
comparison = pd.DataFrame(summary)
history.to_csv(RESULTS / 'blower_timeseries.csv', index=False)
comparison.to_csv(RESULTS / 'blower_comparison.csv', index=False)

print('\nComparison (all blower settings start from the same state and influent index):')
print(comparison[['blower_pct','final_DO','final_BOD_proxy','energy_proxy_total','total_reward','pct_steps_DO_below_2']].round(2).to_string(index=False))

fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
for pct, group in history.groupby('blower_pct'):
    axes[0].plot(group['hours'], group['DO_mg_L'], label=f'{pct}%')
    axes[1].plot(group['hours'], group['BOD_proxy_mg_L'], label=f'{pct}%')
    axes[2].plot(group['hours'], group['energy_proxy'].cumsum(), label=f'{pct}%')
axes[0].axhspan(2, 3, alpha=0.12, label='Illustrative DO target 2–3')
axes[0].set_ylabel('DO (mg/L)')
axes[1].set_ylabel('BOD proxy (mg/L)')
axes[2].set_ylabel('Cumulative energy proxy (unitless)')
axes[2].set_xlabel('Simulation time (hours)')
for ax in axes:
    ax.grid(alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
fig.suptitle('Fixed blower settings — simplified tank model')
fig.tight_layout()
fig.savefig(RESULTS / 'blower_validation.png', dpi=160)
plt.close(fig)

print(f'\nSaved: {RESULTS / "blower_comparison.csv"}')
print(f'Saved: {RESULTS / "blower_timeseries.csv"}')
print(f'Saved: {RESULTS / "blower_validation.png"}')
print('\nChecks: finite values and configured state bounds passed.')
print('WARNING: Passing checks does not prove physical realism; clipping can conceal instability.')
print('Energy is a normalized u^2 proxy, not measured electricity (kWh).')
