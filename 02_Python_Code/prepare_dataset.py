from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / '01_Dataset' / 'raw'
PROCESSED = ROOT / '01_Dataset' / 'processed'
COLUMNS = ['t','Si','Ss','Xi','Xs','Xbh','Xba','Xp','So','Sno','Snh','Snd','Xnd','Salk','Q']


def prepare_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r'\s+', comment='#')
    if not set(COLUMNS).issubset(df.columns):
        # Some distributions omit a header; detect and handle this case.
        df = pd.read_csv(path, sep=r'\s+', comment='#', header=None, names=COLUMNS)
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['t','Ss','Xs','Q']).sort_values('t').drop_duplicates('t')
    df = df[(df['Q'] > 0) & (df['Ss'] >= 0) & (df['Xs'] >= 0)].copy()
    if df.empty:
        raise ValueError(f'No valid rows in {path}')
    # Simplified, explicitly assumed BOD-equivalent proxy; NOT measured BOD.
    df['bod_in_proxy_mg_l'] = 0.6 * (df['Ss'] + df['Xs'])
    df['flow_m3_h'] = df['Q'] / 24.0  # BSM1 Q in m3/day
    df['time_days'] = df['t']
    return df[['time_days','bod_in_proxy_mg_l','flow_m3_h']]


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    found = 0
    for weather, name in [('dry','Inf_dry_2006.txt'),('rain','Inf_rain_2006.txt'),('storm','Inf_strm_2006.txt')]:
        path = RAW / name
        if not path.exists():
            print(f'SKIP missing {path}')
            continue
        out = prepare_one(path)
        dest = PROCESSED / f'{weather}_influent.csv'
        out.to_csv(dest, index=False)
        print(f'{weather}: {len(out)} rows -> {dest}')
        found += 1
    if not found:
        print('No BSM1 text files found. Place them in 01_Dataset/raw and rerun.')

if __name__ == '__main__':
    main()
