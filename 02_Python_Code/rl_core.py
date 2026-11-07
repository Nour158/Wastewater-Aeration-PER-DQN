"""Shared DQN and PER-DQN implementations; PyTorch."""
import random
from collections import deque
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Scales are fixed from physically interpretable observation bounds, never fitted on test data.
SCALE = np.array([8., 200., 1., 200., 1000.], dtype=np.float32)
def normalize(obs):
    return np.clip(np.asarray(obs, dtype=np.float32) / SCALE, 0., 5.)

class QNetwork(nn.Module):
    def __init__(self, n_state=5, n_actions=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_state,128), nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,n_actions))
    def forward(self,x): return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=50000, prioritized=False, alpha=.6):
        self.capacity=capacity; self.prioritized=prioritized; self.alpha=alpha
        self.data=[]; self.priorities=[]; self.pos=0
    def __len__(self): return len(self.data)
    def add(self,s,a,r,ns,done):
        sample=(normalize(s),int(a),float(r),normalize(ns),float(done))
        p=max(self.priorities,default=1.)
        if len(self.data)<self.capacity:
            self.data.append(sample); self.priorities.append(p)
        else:
            self.data[self.pos]=sample; self.priorities[self.pos]=p
        self.pos=(self.pos+1)%self.capacity
    def sample(self,n,beta=.4):
        n=min(n,len(self.data))
        if self.prioritized:
            probs=np.asarray(self.priorities,dtype=np.float64)**self.alpha
            probs=probs/probs.sum()
            ids=np.random.choice(len(self.data),size=n,replace=False,p=probs)
            weights=(len(self.data)*probs[ids])**(-beta)
            weights=weights/weights.max()
        else:
            ids=np.random.choice(len(self.data),size=n,replace=False)
            weights=np.ones(n)
        batch=list(zip(*(self.data[i] for i in ids)))
        return [np.asarray(v) for v in batch],ids,np.asarray(weights,dtype=np.float32)
    def update(self,ids,td):
        if self.prioritized:
            for i,e in zip(ids,td): self.priorities[int(i)]=float(abs(e)+1e-5)

class Agent:
    def __init__(self, prioritized=False,lr=5e-4,gamma=.99,seed=42):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        self.online=QNetwork();self.target=QNetwork()
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=lr)
        self.memory=ReplayBuffer(prioritized=prioritized)
        self.gamma=gamma;self.updates=0
    def act(self,obs,epsilon=0.):
        if random.random()<epsilon: return random.randrange(5)
        with torch.no_grad():
            x=torch.tensor(normalize(obs)).unsqueeze(0)
            return int(self.online(x).argmax(1).item())
    def learn(self,batch_size=64,beta=.4):
        if len(self.memory)<batch_size:return None
        (s,a,r,ns,d),ids,w=self.memory.sample(batch_size,beta)
        s=torch.tensor(s,dtype=torch.float32); ns=torch.tensor(ns,dtype=torch.float32)
        a=torch.tensor(a,dtype=torch.int64);r=torch.tensor(r,dtype=torch.float32)
        d=torch.tensor(d,dtype=torch.float32);w=torch.tensor(w,dtype=torch.float32)
        q=self.online(s).gather(1,a[:,None]).squeeze(1)
        with torch.no_grad():
            # Double DQN target reduces overestimation.
            next_actions=self.online(ns).argmax(1,keepdim=True)
            next_q=self.target(ns).gather(1,next_actions).squeeze(1)
            target=r+self.gamma*(1-d)*next_q
        td=target-q
        loss=(w*F.smooth_l1_loss(q,target,reduction='none')).mean()
        self.optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(),10.)
        self.optimizer.step();self.memory.update(ids,td.detach().numpy())
        self.updates+=1
        if self.updates%250==0:self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())
    def save(self,path):torch.save({'state_dict':self.online.state_dict()},path)
    def load(self,path):
        ck=torch.load(path,map_location='cpu',weights_only=True)
        self.online.load_state_dict(ck['state_dict']);self.target.load_state_dict(ck['state_dict'])
