import numpy as np
class RuleBased:
    def __init__(self,target=2.5):self.target=target
    def act(self,obs):
        do=float(obs[0]);previous=float(obs[2]);error=self.target-do
        desired=np.clip(previous+.30*error,0,1)
        return int(np.argmin(abs(np.array([0,.25,.5,.75,1])-desired)))
class Fixed:
    def __init__(self,action=3):self.action=action
    def act(self,obs):return self.action
