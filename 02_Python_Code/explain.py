"""Post-hoc explanations of trained DQN/PER-DQN policies (surrogate + SHAP)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from rl_core import Agent, normalize
from wastewater_env import WastewaterAerationEnv

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['DO_mg_L', 'BOD_proxy_mg_L', 'previous_blower', 'influent_BOD_proxy', 'flow_m3_h']
ACTIONS = ['0%', '25%', '50%', '75%', '100%']


def rollout(agent, csv_path, seed, episodes):
    env = WastewaterAerationEnv(csv_path=csv_path)
    states, actions = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        for _ in range(env.cfg.episode_steps):
            a = int(agent.act(obs, epsilon=0.0))
            states.append(np.array(obs, dtype=np.float32).copy())
            actions.append(a)
            obs, _, terminated, truncated, _ = env.step(a)
            if terminated or truncated:
                break
    return np.asarray(states), np.asarray(actions)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--algorithm', choices=['dqn', 'per'], default='per')
    p.add_argument('--train-dataset', choices=['dry', 'rain', 'storm'], default='dry')
    p.add_argument('--dataset', choices=['dry', 'rain', 'storm'], default='dry', help='Explanation evaluation dataset')
    p.add_argument('--model-seed', type=int, default=42)
    p.add_argument('--seed', type=int, default=314)
    p.add_argument('--episodes', type=int, default=12)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--skip-shap', action='store_true')
    p.add_argument('--shap-samples', type=int, default=30)
    args = p.parse_args()

    model_path = ROOT / '03_Results' / 'models' / f'{args.algorithm}_{args.train_dataset}_seed{args.model_seed}.pt'
    if not model_path.exists():
        raise FileNotFoundError(f'Model missing: {model_path}')
    train_csv = ROOT / '01_Dataset' / 'splits' / f'{args.dataset}_train.csv'
    test_csv = ROOT / '01_Dataset' / 'splits' / f'{args.dataset}_test.csv'
    for path in (train_csv, test_csv):
        if not path.exists():
            raise FileNotFoundError(path)
    agent = Agent(prioritized=(args.algorithm == 'per'), seed=args.model_seed)
    agent.load(model_path)
    agent.online.eval()

    X_train, y_train = rollout(agent, train_csv, args.seed, args.episodes)
    X_test, y_test = rollout(agent, test_csv, args.seed + 1000, args.episodes)
    tree = DecisionTreeClassifier(max_depth=args.depth, min_samples_leaf=20, random_state=args.seed)
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
    fidelity = accuracy_score(y_test, pred)
    majority = np.bincount(y_test, minlength=5).max() / len(y_test)
    balanced = balanced_accuracy_score(y_test, pred)
    prefix = f'{args.algorithm}_trained_{args.train_dataset}_explained_{args.dataset}_seed{args.model_seed}'
    metrics_dir = ROOT / '03_Results' / 'metrics'
    figures_dir = ROOT / '03_Results' / 'figures'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rules = export_text(tree, feature_names=FEATURES)
    (metrics_dir / f'rules_{prefix}.txt').write_text(
        f'Model: {model_path.name}\n'
        f'Surrogate fidelity on test: {fidelity:.4f}\n'
        f'Majority-action baseline: {majority:.4f}\n'
        f'Balanced fidelity: {balanced:.4f}\n'
        'IMPORTANT: rules approximate the neural policy; not physical safety rules.\n'
        'Test rows are held out from tree fitting, but may overlap model training if train-dataset differs.\n\n'
        + rules, encoding='utf-8')
    cm = confusion_matrix(y_test, pred, labels=range(5))
    pd.DataFrame(cm, index=ACTIONS, columns=ACTIONS).to_csv(metrics_dir / f'confusion_{prefix}.csv')
    pd.DataFrame({'feature': FEATURES, 'tree_importance': tree.feature_importances_}).to_csv(
        metrics_dir / f'tree_importance_{prefix}.csv', index=False)
    fig, ax = plt.subplots(figsize=(18, 9))
    plot_tree(tree, feature_names=FEATURES, class_names=ACTIONS, filled=True, rounded=True,
              fontsize=8, ax=ax)
    fig.tight_layout()
    fig.savefig(figures_dir / f'tree_{prefix}.png', dpi=150)
    plt.close(fig)
    print(f'{args.algorithm.upper()} surrogate fidelity: {fidelity:.3f}; majority baseline: {majority:.3f}; balanced fidelity: {balanced:.3f}')
    print('Rules saved:', metrics_dir / f'rules_{prefix}.txt')
    print('Tree image saved:', figures_dir / f'tree_{prefix}.png')

    if args.skip_shap:
        return
    try:
        import shap
        import torch
    except ImportError:
        print('SHAP or PyTorch unavailable. Install: python -m pip install shap')
        return
    rng = np.random.default_rng(args.seed)
    bg_idx = rng.choice(len(X_train), size=min(40, len(X_train)), replace=False)
    test_idx = rng.choice(len(X_test), size=min(args.shap_samples, len(X_test)), replace=False)
    background = np.stack([normalize(x) for x in X_train[bg_idx]])
    explained = np.stack([normalize(x) for x in X_test[test_idx]])
    def q_values(normalized_states):
        x = torch.as_tensor(np.asarray(normalized_states, dtype=np.float32))
        with torch.no_grad():
            return agent.online(x).cpu().numpy()
    print('Computing SHAP Q-value explanations (this can take a few minutes)...')
    explainer = shap.KernelExplainer(q_values, background)
    values = explainer.shap_values(explained, nsamples=100, silent=True)
    # SHAP versions return either [samples, features, actions] or list[action][samples, features].
    if isinstance(values, list):
        vals = np.stack([np.asarray(v) for v in values], axis=-1)
    else:
        vals = np.asarray(values)
    if vals.shape != (len(explained), len(FEATURES), 5):
        raise ValueError(f'Unexpected SHAP dimensions: {vals.shape}')
    selected_actions = np.argmax(q_values(explained), axis=1)
    selected_shap = vals[np.arange(len(explained)), :, selected_actions]
    importance = np.mean(np.abs(selected_shap), axis=0)
    pd.DataFrame({'feature': FEATURES, 'mean_abs_shap_selected_action_q': importance}).sort_values(
        'mean_abs_shap_selected_action_q', ascending=False).to_csv(
        metrics_dir / f'shap_importance_{prefix}.csv', index=False)
    np.savez_compressed(metrics_dir / f'shap_values_{prefix}.npz',
                        shap_values=vals, normalized_states=explained,
                        selected_actions=selected_actions, feature_names=np.asarray(FEATURES))
    fig, ax = plt.subplots(figsize=(8, 4))
    order = np.argsort(importance)
    ax.barh(np.asarray(FEATURES)[order], importance[order])
    ax.set_xlabel('Mean absolute SHAP contribution to chosen action Q-value')
    ax.set_title(f'SHAP: {args.algorithm.upper()} on {args.dataset} test')
    fig.tight_layout()
    fig.savefig(figures_dir / f'shap_importance_{prefix}.png', dpi=160)
    plt.close(fig)
    print('SHAP importance saved:', figures_dir / f'shap_importance_{prefix}.png')
    print('NOTE: SHAP explains predicted Q-values, not causal effects or regulatory compliance.')

if __name__ == '__main__':
    main()
