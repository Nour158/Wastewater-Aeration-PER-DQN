# Wastewater Aeration Control — DQN / PER-DQN / XAI

**Status: runnable implementation package, NOT an empirically validated or fully trained final submission.** The BSM1 raw data is not bundled, so models, trained weights, real plots and empirical comparison numbers are intentionally not fabricated.

## Important scientific limitations
- Single-tank illustrative BOD/DO mass balance, **not full BSM1/ASM1**.
- `bod_in_proxy_mg_l = 0.6 * (Ss + Xs)` is an assumed COD-derived proxy, **not measured BOD**.
- Blower energy `u²` is a normalized energy proxy, **not kWh**.
- 30 mg/L threshold is illustrative, **not a regulatory claim**.
- Train/test windows are chronological; no overlapping source rows. Episodes start at random indices and may wrap within their own split (no crossing train/test), which should be considered when interpreting temporal realism.
- Environment state is clipped; inspect for numerical instability before drawing engineering conclusions.
- Reward weights, kinetic coefficients, HRT and DO dynamics require expert calibration before real-world use.
- The Pygame interface offers manual and rule-based DO control; its PER-DQN button remains disabled pending separate integration.

## Setup (PowerShell, from project root)
```powershell
python -m pip install -r requirements.txt
python .\02_Python_Code\prepare_dataset.py
python .\02_Python_Code\data_split.py
python .\02_Python_Code\validate_model.py
python .\02_Python_Code\test_environment.py
python .\02_Python_Code\pygame_aeration_lab.py
```

Place original `Inf_dry_2006.txt`, `Inf_rain_2006.txt`, `Inf_strm_2006.txt` in `01_Dataset/raw/` before preprocessing. Source: IWA BSM1 benchmark (iwa-mia.org/benchmarking/). Verify license and cite the dataset's original authors in your submission.

## Train and evaluate
```powershell
python .\02_Python_Code\train.py --algorithm dqn --episodes 300
python .\02_Python_Code\train.py --algorithm per --episodes 300
python .\02_Python_Code\plot_training.py
python .\02_Python_Code\evaluate.py --dataset dry --episodes 10
python .\02_Python_Code\explain.py --dataset dry
```
Train for additional random seeds and all three datasets for a robust report. Model filenames assume seed 42 for evaluation; adjust code if using a different seed.

## Research workflow
1. Validate data conversion, timestep, kinetic assumptions and reward.
2. Compare fixed-blower and rule-based baselines with DQN and PER-DQN on held-out data.
3. Run multiple seeds; report mean and dispersion, DO violations, BOD proxy, normalized energy, switching and reward.
4. Interpret SHAP Q-value attributions; extracted tree rules approximate **agent actions**, not guaranteed safe control.
5. Document limitations and only then prepare claims for the presentation.
