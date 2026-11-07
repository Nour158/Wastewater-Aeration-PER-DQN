from pathlib import Path
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from config import TankConfig

class WastewaterAerationEnv(gym.Env):
    """Illustrative, single well-mixed tank. Not the full BSM1/ASM1 model."""
    metadata = {'render_modes': []}
    ACTIONS = np.array([0., .25, .5, .75, 1.], dtype=np.float32)

    def __init__(self, csv_path=None, config=None):
        super().__init__()
        self.cfg = config or TankConfig()
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        # [DO, BOD, previous blower, incoming BOD proxy, flow m3/h]
        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=np.array([20, 1000, 1, 1000, 10000], dtype=np.float32),
            dtype=np.float32,
        )
        if csv_path is not None:
            self.data = pd.read_csv(csv_path)
            required = {'bod_in_proxy_mg_l', 'flow_m3_h'}
            if not required.issubset(self.data.columns):
                raise ValueError(f'CSV needs columns: {sorted(required)}')
            self.data = self.data.dropna(subset=list(required))
            self.data = self.data[(self.data['flow_m3_h'] > 0) & (self.data['bod_in_proxy_mg_l'] >= 0)]
            if self.data.empty:
                raise ValueError('No usable influent rows')
            self.bod_series = self.data['bod_in_proxy_mg_l'].to_numpy(float)
            self.flow_series = self.data['flow_m3_h'].to_numpy(float)
        else:
            # Synthetic conditions for smoke tests before BSM1 files are available.
            self.bod_series = np.array([180.0])
            self.flow_series = np.array([50.0])
        self.reset()

    def _influent(self):
        i = self.idx % len(self.bod_series)
        return float(self.bod_series[i]), float(self.flow_series[i])

    def _obs(self):
        bod_in, flow = self._influent()
        return np.array([self.do, self.bod, self.prev_u, bod_in, flow], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0 if len(self.bod_series) == 1 else int(self.np_random.integers(len(self.bod_series)))
        self.steps = 0
        self.do = 2.5
        self.bod = 40.0
        self.prev_u = 0.5
        return self._obs(), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action: {action}')
        u = float(self.ACTIONS[int(action)])
        bod_in, flow = self._influent()
        cfg = self.cfg
        dilution = flow / cfg.volume_m3
        oxygen_factor = self.do / (cfg.k_o_mg_l + self.do)
        biodegradation = cfg.k_b_per_hour * self.bod * oxygen_factor
        d_bod = dilution * (bod_in - self.bod) - biodegradation
        d_do = (cfg.kla_max_per_hour * u * (cfg.do_saturation_mg_l - self.do)
                - cfg.oxygen_per_bod * biodegradation
                + dilution * (cfg.influent_do_mg_l - self.do))
        next_bod = float(np.clip(self.bod + cfg.dt_hours * d_bod, 0, 1000))
        next_do = float(np.clip(self.do + cfg.dt_hours * d_do, 0, cfg.do_saturation_mg_l))
        low_do_penalty = max(0.0, cfg.min_do_mg_l - next_do) ** 2
        high_do_penalty = max(0.0, next_do - cfg.max_do_mg_l) ** 2
        # BOD limit is illustrative and generally unreachable in this simple tank.
        # Cap its influence so the agent cannot ignore dissolved oxygen control.
        bod_penalty = min(4.0, (max(0.0, next_bod - cfg.bod_limit_mg_l) / 30.0) ** 2)
        energy_penalty = u ** 2
        switching_penalty = abs(u - self.prev_u)
        reward = -(cfg.low_do_weight * low_do_penalty
                   + cfg.high_do_weight * high_do_penalty
                   + cfg.bod_weight * bod_penalty
                   + cfg.energy_weight * energy_penalty
                   + cfg.switching_weight * switching_penalty)
        self.do, self.bod, self.prev_u = next_do, next_bod, u
        self.idx += 1
        self.steps += 1
        truncated = self.steps >= cfg.episode_steps
        info = {
            'do_mg_l': next_do, 'bod_proxy_mg_l': next_bod,
            'blower_fraction': u, 'energy_proxy': energy_penalty,
            'd_do_mg_l_per_h': d_do, 'd_bod_mg_l_per_h': d_bod,
            'influent_bod_proxy_mg_l': bod_in, 'flow_m3_h': flow,
            'reward_low_do': -cfg.low_do_weight * low_do_penalty,
            'reward_high_do': -cfg.high_do_weight * high_do_penalty,
            'reward_bod': -cfg.bod_weight * bod_penalty,
            'reward_energy': -cfg.energy_weight * energy_penalty,
            'reward_switching': -cfg.switching_weight * switching_penalty,
        }
        return self._obs(), float(reward), False, truncated, info
