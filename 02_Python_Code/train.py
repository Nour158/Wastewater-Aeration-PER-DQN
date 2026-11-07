import argparse,csv,random
from pathlib import Path
import numpy as np
from rl_core import Agent
from wastewater_env import WastewaterAerationEnv
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--algorithm',choices=['dqn','per'],required=True)
    p.add_argument('--episodes',type=int,default=600);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--dataset',choices=['dry','rain','storm'],default='dry')
    args=p.parse_args();random.seed(args.seed);np.random.seed(args.seed)
    data=ROOT/f'01_Dataset/splits/{args.dataset}_train.csv'
    if not data.exists():raise FileNotFoundError(f'{data} missing; run prepare_dataset.py and data_split.py')
    env=WastewaterAerationEnv(csv_path=data);agent=Agent(prioritized=args.algorithm=='per',seed=args.seed)
    results=[]; models=ROOT/'03_Results/models';models.mkdir(parents=True,exist_ok=True)
    for ep in range(args.episodes):
        obs,_=env.reset(seed=args.seed+ep);reward_sum=0.;losses=[];actions=[];low_do_steps=0
        epsilon=max(.05,1.-.95*ep/max(1,args.episodes*.8))
        beta=min(1.,.4+.6*ep/max(1,args.episodes))
        for step in range(env.cfg.episode_steps):
            action=agent.act(obs,epsilon)
            new_obs,reward,terminated,truncated,info=env.step(action)
            actions.append(action);low_do_steps+=int(info['do_mg_l']<env.cfg.min_do_mg_l)
            # Truncation is a time limit; bootstrap remains valid for value learning.
            agent.memory.add(obs,action,reward,new_obs,terminated)
            loss=agent.learn(beta=beta)
            if loss is not None:losses.append(loss)
            obs=new_obs;reward_sum+=reward
            if terminated or truncated:break
        results.append({'episode':ep+1,'reward':reward_sum,'epsilon':epsilon,'loss':float(np.mean(losses)) if losses else '', 'do_below_2_pct':100*low_do_steps/max(1,len(actions)), 'mean_blower_fraction':float(np.mean(np.asarray(actions)*.25)), 'action_counts':','.join(str(actions.count(i)) for i in range(5))})
        if (ep+1)%25==0:print(args.algorithm,'episode',ep+1,'reward',round(reward_sum,1))
    model=models/f'{args.algorithm}_{args.dataset}_seed{args.seed}.pt';agent.save(model)
    out=ROOT/'03_Results/metrics'/f'train_{args.algorithm}_{args.dataset}_seed{args.seed}.csv';out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=results[0]);writer.writeheader();writer.writerows(results)
    print('Saved',model,'and',out)
if __name__=='__main__':main()
