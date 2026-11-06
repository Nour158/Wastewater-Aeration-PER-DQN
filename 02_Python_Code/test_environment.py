from pathlib import Path
from gymnasium.utils.env_checker import check_env
from wastewater_env import WastewaterAerationEnv

ROOT = Path(__file__).resolve().parents[1]
csv = ROOT / '01_Dataset' / 'processed' / 'dry_influent.csv'
env = WastewaterAerationEnv(csv_path=csv if csv.exists() else None)
check_env(env, skip_render_check=True)
obs, info = env.reset(seed=42)
print('Initial state [DO, BOD, previous blower, influent BOD proxy, flow]:', obs)
for step in range(10):
    action = 2  # 50% blower
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step+1:02d} | DO={info['do_mg_l']:.2f} | BOD proxy={info['bod_proxy_mg_l']:.2f} | reward={reward:.3f}")
    if terminated or truncated:
        break
print('Environment smoke test passed!')
