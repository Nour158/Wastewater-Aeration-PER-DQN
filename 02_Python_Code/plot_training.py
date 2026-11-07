from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
files=list((ROOT/'03_Results/metrics').glob('train_*.csv'))
if not files:raise SystemExit('No training logs found')
fig,ax=plt.subplots(figsize=(9,5))
for f in files:
    df=pd.read_csv(f);ax.plot(df.episode,df.reward.rolling(15,min_periods=1).mean(),label=f.stem)
ax.set(xlabel='Episode',ylabel='15-episode rolling reward',title='Training curves');ax.legend();fig.tight_layout()
out=ROOT/'03_Results/figures/training_curves.png';out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out,dpi=160);print(out)
