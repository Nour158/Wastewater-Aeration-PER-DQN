"""Smoke/consistency tests; not a physical validation study."""
import numpy as np
from wastewater_env import WastewaterAerationEnv
from config import TankConfig

def main():
    env=WastewaterAerationEnv(config=TankConfig())
    obs,_=env.reset(seed=10)
    assert obs.shape==(5,)
    for action in range(5):
        obs,_=env.reset(seed=10)
        obs,reward,term,trunc,info=env.step(action)
        assert np.all(np.isfinite(obs)) and np.isfinite(reward)
        assert 0<=info['do_mg_l']<=8 and 0<=info['bod_proxy_mg_l']<=1000
        breakdown=sum(info[k] for k in ('reward_low_do','reward_high_do','reward_bod','reward_energy','reward_switching'))
        assert np.isclose(reward,breakdown)
    print('PASS: finite observations, bounds and reward decomposition')
    print('NOTE: numerical tests do not establish physical calibration or BSM1 equivalence')
if __name__=='__main__':main()
