"""Chronological nonoverlapping dataset split; no test leakage."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    inp=ROOT/'01_Dataset/processed';out=ROOT/'01_Dataset/splits';out.mkdir(parents=True,exist_ok=True)
    for weather in ('dry','rain','storm'):
        f=inp/f'{weather}_influent.csv'
        if not f.exists(): print('Missing',f);continue
        df=pd.read_csv(f); n=len(df)
        if n<300: raise ValueError(f'{weather}: at least 300 samples required')
        a=int(.7*n);b=int(.85*n)
        for name,part in [('train',df.iloc[:a]),('val',df.iloc[a:b]),('test',df.iloc[b:])]:
            path=out/f'{weather}_{name}.csv';part.to_csv(path,index=False);print(path,len(part))
if __name__=='__main__':main()
