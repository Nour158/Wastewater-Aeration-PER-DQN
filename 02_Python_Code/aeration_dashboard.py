"""Interactive educational aeration tank simulation.
Run: streamlit run 02_Python_Code/aeration_dashboard.py
Uses the same simplified equations as WastewaterAerationEnv; not a full BSM1 model.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Aeration Tank | Live Simulator', page_icon='💧', layout='wide')
st.title('💧 Wastewater Aeration Tank — Interactive Simulation')
st.caption('Educational single-tank BOD-proxy/DO model. BSM1 influent is used as input, not as measured BOD or calibrated effluent performance.')

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / '01_Dataset' / 'processed'
files = sorted(PROCESSED.glob('*.csv')) if PROCESSED.exists() else []

with st.sidebar:
    st.header('Controls')
    options = ['Synthetic constant influent'] + [str(p.relative_to(ROOT)) for p in files]
    source = st.selectbox('Influent data source', options)
    mode = st.radio('Blower control', ['Fixed blower', 'DO feedback controller'])
    blower_pct = st.slider('Fixed blower (%)', 0, 100, 50, 25, disabled=mode != 'Fixed blower')
    st.caption('Feedback controller: increases blower when DO < 2, decreases when DO > 3 mg/L.')
    hours = st.slider('Simulation duration (hours)', 1, 48, 24)
    start_idx = st.number_input('Dataset starting row', min_value=0, value=0, step=1, disabled=source == options[0])
    st.subheader('Initial conditions')
    initial_do = st.slider('Initial DO (mg/L)', 0.0, 8.0, 2.5, 0.1)
    initial_bod = st.slider('Initial BOD proxy (mg/L)', 0.0, 250.0, 40.0, 1.0)
    st.subheader('Illustrative tank parameters')
    volume = st.number_input('Tank volume (m³)', min_value=100.0, value=1000.0, step=100.0)
    kb = st.slider('BOD degradation constant (1/hour)', 0.01, 0.30, 0.08, 0.01)
    kla_max = st.slider('Maximum oxygen transfer KLa (1/hour)', 0.1, 5.0, 2.0, 0.1)
    st.caption('Time step = 15 min. Energy is a normalized u² proxy, not measured kWh.')

DT = 0.25
N = int(hours / DT)

if source == options[0]:
    incoming = np.full(N, 180.0)
    flows = np.full(N, 50.0)
    source_label = 'Synthetic illustrative influent'
else:
    path = ROOT / source
    try:
        data = pd.read_csv(path)
        cols = {'bod_in_proxy_mg_l', 'flow_m3_h'}
        if not cols.issubset(data.columns):
            st.error(f'{path.name} must contain columns: {sorted(cols)}')
            st.stop()
        data = data.dropna(subset=list(cols))
        data = data[(data['bod_in_proxy_mg_l'] >= 0) & (data['flow_m3_h'] > 0)].reset_index(drop=True)
        if data.empty:
            st.error('No valid rows found in this dataset.')
            st.stop()
        idx = (int(start_idx) + np.arange(N)) % len(data)
        incoming = data['bod_in_proxy_mg_l'].to_numpy(dtype=float)[idx]
        flows = data['flow_m3_h'].to_numpy(dtype=float)[idx]
        source_label = path.name
    except Exception as exc:
        st.error(f'Could not read CSV: {exc}')
        st.stop()

if 'frame' not in st.session_state:
    st.session_state.frame = 0

# The same simplified Euler updates and reward as the current project code.
def simulate():
    do, bod, previous_u = float(initial_do), float(initial_bod), 0.5
    rows = [{'hour': 0.0, 'do': do, 'bod': bod, 'blower': previous_u,
             'energy_total': 0.0, 'reward_total': 0.0, 'bod_in': float(incoming[0]),
             'flow': float(flows[0]), 'oxygen_added': 0.0, 'oxygen_consumed': 0.0,
             'flow_oxygen': 0.0, 'bod_removed_rate': 0.0, 'raw_do': do, 'raw_bod': bod}]
    energy_total, reward_total = 0.0, 0.0
    for i in range(N):
        if mode == 'Fixed blower':
            u = blower_pct / 100.0
        else:
            # Deliberately simple rule-based baseline, NOT trained DQN/PER-DQN.
            u = float(np.clip(previous_u + (0.25 if do < 2 else (-0.25 if do > 3 else 0)), 0, 1))
        q, bod_in = float(flows[i]), float(incoming[i])
        dilution = q / volume
        oxygen_factor = do / (0.5 + do)
        removal = kb * bod * oxygen_factor
        oxygen_added = kla_max * u * (8.0 - do)
        oxygen_consumed = 0.5 * removal
        flow_oxygen = dilution * (0.5 - do)
        d_bod = dilution * (bod_in - bod) - removal
        d_do = oxygen_added - oxygen_consumed + flow_oxygen
        raw_do = do + DT * d_do
        raw_bod = bod + DT * d_bod
        do = float(np.clip(raw_do, 0, 8))
        bod = float(np.clip(raw_bod, 0, 1000))
        low = max(0, 2.0 - do) ** 2
        high = max(0, do - 3.0) ** 2
        bod_penalty = (max(0, bod - 30) / 30) ** 2
        reward = -(5 * low + high + 2 * bod_penalty + u**2 + 0.1 * abs(u - previous_u))
        energy_total += u**2 * DT  # time-integrated normalized proxy
        reward_total += reward
        rows.append({'hour': (i + 1) * DT, 'do': do, 'bod': bod, 'blower': u,
                     'energy_total': energy_total, 'reward_total': reward_total,
                     'bod_in': bod_in, 'flow': q, 'oxygen_added': oxygen_added,
                     'oxygen_consumed': oxygen_consumed, 'flow_oxygen': flow_oxygen,
                     'bod_removed_rate': removal, 'raw_do': raw_do, 'raw_bod': raw_bod})
        previous_u = u
    return pd.DataFrame(rows)

df = simulate()
st.caption(f'Input source: {source_label} • {N} steps • Fixed 15-minute timestep')

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    if st.button('◀ Restart', use_container_width=True):
        st.session_state.frame = 0
with c2:
    if st.button('Next 15 minutes ▶', use_container_width=True):
        st.session_state.frame = min(N, st.session_state.frame + 1)
with c3:
    frame = st.slider('Simulation time (hours)', 0.0, float(hours), min(float(hours), st.session_state.frame * DT), DT)
    st.session_state.frame = int(round(frame / DT))

current = df.iloc[st.session_state.frame]
DO = current['do']
BOD = current['bod']
blower = current['blower']

m1, m2, m3, m4 = st.columns(4)
m1.metric('Dissolved oxygen', f'{DO:.2f} mg/L', 'Target: 2–3 mg/L', delta_color='off')
m2.metric('BOD proxy in tank', f'{BOD:.1f} mg/L')
m3.metric('Blower power', f'{blower * 100:.0f}%')
m4.metric('Cumulative energy proxy', f'{current["energy_total"]:.2f}', help='Integral of blower fraction squared over hours; NOT kWh')

st.subheader('Tank process at selected time')
left, right = st.columns([1.3, 1])
with left:
    # Native HTML/CSS tank illustration; no external image dependencies.
    oxygen_color = '#0b8f8c' if 2 <= DO <= 3 else ('#e69f00' if DO > 3 else '#cb4848')
    fill_pct = 66
    bubbles = ' '.join(['○'] * max(1, int(blower * 14)))
    st.markdown(f'''<div style="border:2px solid #64748b;border-radius:18px;padding:14px;background:#f8fafc;color:#0f172a;">
      <div style="display:flex;justify-content:space-between;gap:8px;font-weight:600"><span>⬇ Incoming wastewater<br><small>{current['flow']:.1f} m³/h · organic proxy {current['bod_in']:.1f} mg/L</small></span><span>➡ Mixed tank outlet<br><small>Current tank proxy: {BOD:.1f} mg/L</small></span></div>
      <div style="position:relative;height:205px;margin:16px 5%;border:4px solid #475569;border-top:0;border-radius:0 0 24px 24px;overflow:hidden;background:#e2e8f0">
        <div style="position:absolute;bottom:0;width:100%;height:{fill_pct}%;background:linear-gradient(180deg,#a7d8f3,#4b9cc7);padding:15px;text-align:center;color:#083344;font-weight:bold;">
          <div style="font-size:23px;color:{oxygen_color};background:#ffffffbb;border-radius:12px">DO: {DO:.2f} mg/L</div>
          <div style="margin-top:8px;font-size:14px">Microorganisms consume oxygen<br>while degrading organic matter</div>
          <div style="font-size:20px;letter-spacing:4px;white-space:normal">{bubbles}</div>
        </div>
      </div>
      <div style="text-align:center;font-weight:600">⬆ Air blower: {blower*100:.0f}%</div>
    </div>''', unsafe_allow_html=True)
    st.caption('Schematic visualization only. Bubbles illustrate blower intensity, not simulated fluid dynamics.')
with right:
    st.write('**Instantaneous process rates** (mg/L/hour)')
    st.write(f'🟢 Oxygen transferred from air: **+{current["oxygen_added"]:.2f}**')
    st.write(f'🔴 Oxygen used by microbes: **−{current["oxygen_consumed"]:.2f}**')
    st.write(f'🔄 Flow effect on oxygen: **{current["flow_oxygen"]:+.2f}**')
    st.write(f'🦠 Biological BOD-proxy removal: **{current["bod_removed_rate"]:.2f}**')
    st.write(f'🎯 DO target: **2–3 mg/L**')
    st.write(f'🏆 Cumulative reward: **{current["reward_total"]:.1f}**')
    if DO < 2:
        st.error('Low oxygen: DO is below the illustrative target range.')
    elif DO > 3:
        st.warning('Excess oxygen: DO is above the illustrative target range.')
    else:
        st.success('DO is within the illustrative target range.')

st.subheader('What happens over time?')
chart_df = df.iloc[:st.session_state.frame + 1].set_index('hour')
a, b = st.columns(2)
with a:
    st.markdown('**Dissolved oxygen (mg/L)**')
    st.line_chart(chart_df[['do']], height=250)
with b:
    st.markdown('**Tank BOD proxy vs incoming BOD proxy (mg/L)**')
    st.line_chart(chart_df[['bod', 'bod_in']], height=250)
c, d = st.columns(2)
with c:
    st.markdown('**Blower fraction (0 to 1)**')
    st.line_chart(chart_df[['blower']], height=230)
with d:
    st.markdown('**Cumulative normalized energy proxy**')
    st.line_chart(chart_df[['energy_total']], height=230)

st.subheader('Full-run summary')
summary = pd.DataFrame([{
    'Final DO (mg/L)': round(float(df['do'].iloc[-1]), 2),
    'Final BOD proxy (mg/L)': round(float(df['bod'].iloc[-1]), 2),
    'Time below DO 2 (%)': round(float((df['do'].iloc[1:] < 2).mean() * 100), 1),
    'Time above DO 3 (%)': round(float((df['do'].iloc[1:] > 3).mean() * 100), 1),
    'Energy proxy (u²·hours)': round(float(df['energy_total'].iloc[-1]), 2),
    'Total reward': round(float(df['reward_total'].iloc[-1]), 1),
    'Unclipped DO outside [0,8]': int(((df['raw_do'].iloc[1:] < 0) | (df['raw_do'].iloc[1:] > 8)).sum()),
    'Unclipped BOD outside [0,1000]': int(((df['raw_bod'].iloc[1:] < 0) | (df['raw_bod'].iloc[1:] > 1000)).sum()),
}])
st.dataframe(summary, hide_index=True, use_container_width=True)
st.download_button('Download full simulation CSV', df.to_csv(index=False).encode('utf-8'), file_name='aeration_simulation.csv', mime='text/csv')
st.info('This dashboard is a simplified educational model. The BSM1-derived organic proxy is not measured BOD; tank concentration is not a verified effluent compliance result. Feedback control is a simple rule-based baseline, not trained DQN or PER-DQN.')
