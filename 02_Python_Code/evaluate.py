"""Evaluate wastewater aeration policies on held-out data or weather-event windows.

Event mode is a diagnostic, NOT a held-out robustness benchmark: event windows
may overlap the training and validation periods of the dry-trained agents.
"""
import argparse
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from controllers import Fixed, RuleBased
from rl_core import Agent
from wastewater_env import WastewaterAerationEnv

ROOT = Path(__file__).resolve().parents[1]
EVENT_WINDOWS = {'rain': (708, 1096), 'storm': (750, 1203)}  # inclusive indices


def evaluate(controller, data, seed, episodes=20):
    env = WastewaterAerationEnv(csv_path=data)
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        metrics, actions = [], []
        for _ in range(env.cfg.episode_steps):
            action = int(controller.act(obs))
            actions.append(action)
            obs, reward, terminated, truncated, info = env.step(action)
            metrics.append((reward, info['do_mg_l'], info['bod_proxy_mg_l'],
                            info['energy_proxy'], info['blower_fraction']))
            if terminated or truncated:
                break
        if not metrics:
            raise RuntimeError('Environment returned no steps')
        arr = np.asarray(metrics, dtype=float)
        counts = np.bincount(actions, minlength=5)
        result = {
            'reward': arr[:, 0].sum(),
            'do_below_2_pct': 100 * np.mean(arr[:, 1] < 2),
            'do_above_3_pct': 100 * np.mean(arr[:, 1] > 3),
            'mean_do': arr[:, 1].mean(),
            'mean_bod_proxy': arr[:, 2].mean(),
            'energy_proxy_sum': arr[:, 3].sum(),
            'blower_switches': int(np.count_nonzero(np.diff(arr[:, 4]))),
            'final_bod_proxy': arr[-1, 2],
            'mean_blower_fraction': arr[:, 4].mean(),
        }
        for i, count in enumerate(counts):
            result[f'action_{i}_pct'] = 100 * count / len(actions)
        rows.append(result)
    df = pd.DataFrame(rows)
    summary = df.mean(numeric_only=True).to_dict()
    summary['reward_std'] = float(df['reward'].std(ddof=0))
    return summary, df

def evaluate_full_window(controller, data):
    """Evaluate every influent row exactly once, in order."""
    from dataclasses import replace
    from config import TankConfig

    n = len(pd.read_csv(data))

    cfg = replace(TankConfig(), episode_steps=n)
    env = WastewaterAerationEnv(csv_path=data, config=cfg)

    obs, _ = env.reset(seed=900, options={"start_idx": 0})

    rows = []
    actions = []

    for step in range(n):
        action = int(controller.act(obs))
        actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)

        rows.append({
            "step": step,
            "reward": reward,
            "do_mg_l": info["do_mg_l"],
            "bod_proxy_mg_l": info["bod_proxy_mg_l"],
            "energy_proxy": info["energy_proxy"],
            "blower_fraction": info["blower_fraction"],
        })

        if terminated or truncated:
            break

    df = pd.DataFrame(rows)

    if len(df) != n:
        raise RuntimeError("Full-window evaluation ended early")

    summary = {
        "reward": df["reward"].sum(),
        "do_below_2_pct": 100 * (df["do_mg_l"] < 2).mean(),
        "do_above_3_pct": 100 * (df["do_mg_l"] > 3).mean(),
        "mean_do": df["do_mg_l"].mean(),
        "mean_bod_proxy": df["bod_proxy_mg_l"].mean(),
        "energy_proxy_sum": df["energy_proxy"].sum(),
        "blower_switches": int(
            np.count_nonzero(np.diff(df["blower_fraction"]))
        ),
        "final_bod_proxy": df["bod_proxy_mg_l"].iloc[-1],
        "mean_blower_fraction": df["blower_fraction"].mean(),
    }

    for i in range(5):
        summary[f"action_{i}_pct"] = (
            100 * actions.count(i) / len(actions)
        )

    return summary, df

def run_full_window_comparison(
    controllers, csv_path, metrics_dir, name
):
    summaries = []

    for controller_name, controller in controllers.items():
        summary, steps = evaluate_full_window(
            controller, csv_path
        )

        summaries.append({
            "controller": controller_name,
            **summary
        })

        steps.to_csv(
            metrics_dir / f"full_steps_{controller_name}_{name}.csv",
            index=False
        )

    results = pd.DataFrame(summaries)

    output = metrics_dir / f"full_evaluation_{name}.csv"
    results.to_csv(output, index=False)

    print(f"\nFULL-WINDOW EVALUATION: {name}")
    print(results.to_string(index=False))
    print("Saved:", output)

    return results

def make_controllers(train_dataset, model_seed):
    controllers = {'fixed_50': Fixed(2), 'fixed_75': Fixed(3), 'rule_based': RuleBased()}
    for algorithm in ('dqn', 'per'):
        path = ROOT / '03_Results' / 'models' / f'{algorithm}_{train_dataset}_seed{model_seed}.pt'
        if not path.exists():
            print('Model unavailable:', path)
            continue
        agent = Agent(prioritized=(algorithm == 'per'), seed=model_seed)
        agent.load(path)
        controllers[algorithm] = agent
    return controllers


def run_comparison(controllers, csv_path, seed, episodes, metrics_dir, name):
    summaries = []
    for controller_name, controller in controllers.items():
        metrics, per_episode = evaluate(controller, csv_path, seed, episodes)
        summaries.append({'controller': controller_name, **metrics})
        per_episode.to_csv(metrics_dir / f'episodes_{controller_name}_{name}.csv', index=False)
    summary = pd.DataFrame(summaries)
    output = metrics_dir / f'evaluation_{name}.csv'
    summary.to_csv(output, index=False)
    print(f'\n{name}\n{summary.to_string(index=False, float_format=lambda x: f"{x:.3f}")}')
    print('Saved', output)
    return summary


def save_figures(df, figure_dir, name):
    for metric, ylabel in [('energy_proxy_sum', 'Normalized energy proxy sum'),
                           ('do_below_2_pct', 'DO below 2 mg/L (%)'),
                           ('reward', 'Total reward')]:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(df['controller'], df[metric])
        ax.set_ylabel(ylabel)
        ax.set_title(name.replace('_', ' '))
        fig.tight_layout()
        fig.savefig(figure_dir / f'{metric}_{name}.png', dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['dry', 'rain', 'storm'], default='dry')
    parser.add_argument('--train-dataset', choices=['dry', 'rain', 'storm'], default='dry')
    parser.add_argument('--model-seed', type=int, default=42)
    parser.add_argument('--seed', type=int, default=900)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--mode', choices=['test', 'event'], default='test',
                        help='test: chronological held-out split; event: event-window diagnostic')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be at least 1')
    if args.mode == 'event' and args.dataset == 'dry':
        parser.error('--mode event requires --dataset rain or storm')

    metrics_dir = ROOT / '03_Results' / 'metrics'
    figure_dir = ROOT / '03_Results' / 'figures'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    controllers = make_controllers(args.train_dataset, args.model_seed)

    if args.mode == 'test':
        data = ROOT / '01_Dataset' / 'splits' / f'{args.dataset}_test.csv'
        if not data.exists():
            raise FileNotFoundError(f'Missing {data}; run data_split.py')
        name = f'trained_{args.train_dataset}_tested_{args.dataset}'
        df = run_comparison(controllers, data, args.seed, args.episodes, metrics_dir, name)
        save_figures(df, figure_dir, name)
    else:
        start, end = EVENT_WINDOWS[args.dataset]
        processed = ROOT / '01_Dataset' / 'processed'
        dry_path = processed / 'dry_influent.csv'
        event_path = processed / f'{args.dataset}_influent.csv'
        dry = pd.read_csv(dry_path)
        event = pd.read_csv(event_path)
        if len(dry) != len(event) or end >= len(dry):
            raise ValueError('Dataset lengths/window do not match expected processed data')
        dry_window = dry.iloc[start:end + 1].reset_index(drop=True)
        event_window = event.iloc[start:end + 1].reset_index(drop=True)
        if dry_window.equals(event_window):
            raise ValueError('Event and dry windows are identical; no weather contrast possible')
        print(f'\nEVENT DIAGNOSTIC: {args.dataset}, rows {start}–{end} inclusive')
        print('WARNING: these windows overlap training/validation; NOT held-out test results.')
        print('Both scenarios use identical evaluation seeds and row boundaries.')
        with tempfile.TemporaryDirectory(prefix='wastewater_event_') as tmp:
            dry_csv = Path(tmp) / 'dry_window.csv'
            event_csv = Path(tmp) / 'event_window.csv'
            dry_window.to_csv(dry_csv, index=False)
            event_window.to_csv(event_csv, index=False)
            prefix = f'event_{args.dataset}_rows_{start}_{end}_trained_{args.train_dataset}'
            dry_name = f'{prefix}_reference_dry'
            event_name = f'{prefix}_weather'
            dry_results = run_full_window_comparison(
                controllers, dry_csv, metrics_dir, dry_name
            )

            event_results = run_full_window_comparison(
                controllers, event_csv, metrics_dir, event_name
            )
            save_figures(event_results, figure_dir, event_name)
            comparison = event_results.set_index('controller').select_dtypes(include='number') - \
                         dry_results.set_index('controller').select_dtypes(include='number')
            comparison.insert(0, 'controller', comparison.index)
            comparison.to_csv(metrics_dir / f'evaluation_{prefix}_delta_weather_minus_dry.csv', index=False)
            print('\nWEATHER MINUS SAME-WINDOW DRY (positive DO violation/energy = worse):')
            cols = ['controller', 'reward', 'do_below_2_pct', 'do_above_3_pct', 'energy_proxy_sum']
            print(comparison[cols].to_string(index=False, float_format=lambda x: f'{x:+.3f}'))
    print('Figures saved to', figure_dir)


if __name__ == '__main__':
    main()
