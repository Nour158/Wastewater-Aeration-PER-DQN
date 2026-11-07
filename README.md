# Wastewater Treatment Aeration Tank Control Using PER-DQN and Rule Extraction

> **Project status:** Dataset preparation, a simplified wastewater environment, reward design, and a Pygame visualization have been implemented. The Pygame lab has been shown running. **Model validation, controller benchmarking, DQN/PER-DQN training, and explainability are still pending.**

## 1. Project overview

This project explores how reinforcement learning can control an aeration blower in a simulated wastewater treatment tank. The aim is to maintain sufficient **dissolved oxygen (DO)**, reduce an **organic-load/BOD proxy**, and limit **aeration energy use**.

The planned controller is **Deep Q-Network with Prioritized Experience Replay (PER-DQN)**. Its performance will be compared with a standard DQN and simpler controllers. Explainable AI (SHAP and decision-tree rule extraction) will be used to interpret the learned control policy.

**Important scope limitation:** This is an illustrative single, well-mixed tank model, **not** a calibrated implementation of the full IWA BSM1/ASM1 wastewater process. The BOD quantity is derived from influent data and is **not measured effluent BOD**. Energy is represented by a normalized `blower_fraction ** 2` proxy, **not electricity in kWh**. Any threshold values are experimental settings, not verified discharge standards.

## 2. Current implementation

### Implemented / demonstrated

- Obtained BSM1 dry, rain, and storm influent time-series files.
- Prepared CSV influent data with columns including `bod_in_proxy_mg_l` and `flow_m3_h`.
- Implemented a Gymnasium environment with discrete blower actions.
- Implemented simplified BOD/DO dynamics and a multi-objective reward.
- Ran an environment smoke test successfully.
- Ran a fixed-blower validation comparison (0%, 25%, 50%, 75%, 100%).
- Built a Pygame interactive aeration lab; a running screenshot was provided.

### Not yet complete or verified

- Independent checking of dataset units, physical assumptions, and numerical stability.
- Comprehensive tests on all dry/rain/storm conditions.
- Formal rule-based baseline evaluation.
- Standard DQN training and evaluation.
- PER-DQN training and evaluation.
- SHAP explanations, surrogate rules, and final comparative results.
- Loading trained AI controllers into Pygame.

## 3. Project layout

The expected layout is:

```text
Wastewater_Aeration_PER_DQN/
├── 01_Dataset/
│   ├── raw/                     # Original BSM1 influent text files
│   ├── processed/               # Prepared CSV files
│   └── DATASET_README.md
├── 02_Python_Code/
│   ├── config.py
│   ├── prepare_dataset.py
│   ├── wastewater_env.py
│   ├── test_environment.py
│   ├── validate_environment.py
│   ├── pygame_aeration_lab.py
│   ├── replay_buffer.py         # Planned / verify contents
│   ├── dqn_agent.py             # Planned / verify contents
│   ├── per_dqn_agent.py         # Planned / verify contents
│   ├── rule_based_controller.py # Planned / verify contents
│   ├── train.py                 # Planned / verify contents
│   ├── evaluate.py              # Planned / verify contents
│   ├── explain_shap.py          # Planned / verify contents
│   └── extract_rules.py         # Planned / verify contents
├── 03_Results/
│   ├── validation/
│   └── interactive_experiments/ # Pygame CSV exports, when saved
├── 04_Presentation/
├── 05_References/
├── requirements.txt             # Location may differ
└── README.md
```

> **Note:** The structure above is the intended project organization. Some listed AI scripts may be placeholders or not yet implemented. Check the actual repository before running them.

## 4. Setup (Windows / VS Code / PowerShell)

Open the project folder in VS Code and run these commands from the **project root**.

### Create and activate a virtual environment (first time only)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If `.venv` already exists, activate it instead of recreating it.

### Install dependencies

```powershell
python -m pip install numpy pandas matplotlib gymnasium torch scikit-learn shap seaborn tqdm pygame
```

If a verified `requirements.txt` is available, you can instead use:

```powershell
python -m pip install -r requirements.txt
```

Tested in the team's environment with Python 3.13 and Pygame 2.6.1; package compatibility on other machines should be checked.

## 5. Running the current project

From the project root:

**Environment smoke test**

```powershell
python .\02_Python_Code\test_environment.py
```

Expected success message:

```text
Environment smoke test passed!
```

**Fixed-blower validation**

```powershell
python .\02_Python_Code\validate_environment.py
```

Expected outputs include:

```text
03_Results/validation/blower_comparison.csv
03_Results/validation/blower_timeseries.csv
03_Results/validation/blower_validation.png
```

**Interactive Pygame lab**

```powershell
python .\02_Python_Code\pygame_aeration_lab.py
```

The Pygame lab displays an animated aeration basin, live DO/BOD proxy values, influent information, history plots, and a tuning panel. It supports manual blower adjustment, a basic automatic DO mode, dataset selection, and experiment CSV export. **PER-DQN control is not available until a model is trained and integrated.**

If an older copy of `pygame_aeration_lab.py` fails with `TypeError: unsupported operand type(s) for +: 'int' and 'list'`, inspect the label-position calculation around line 316: remove the square brackets around the `.index(label)` expression.

## 6. Environment model

### State (observation)

```text
[DO, BOD_proxy, previous_blower_fraction, influent_BOD_proxy, flow_m3_h]
```

### Available actions

| Action index | Blower setting |
|---|---:|
| 0 | 0% |
| 1 | 25% |
| 2 | 50% |
| 3 | 75% |
| 4 | 100% |

### Simplified process equations

```text
oxygen_factor = DO / (K_O + DO)
biodegradation = k_b * BOD_proxy * oxygen_factor

dBOD/dt = (Q / V) * (BOD_in_proxy - BOD_proxy) - biodegradation

dDO/dt = KLa_max * u * (DO_saturation - DO)
         - oxygen_per_bod * biodegradation
         + (Q / V) * (DO_in - DO)
```

The simulation advances in 15-minute steps (`dt_hours = 0.25`) in the current configuration. These equations are simplified and must not be presented as the full BSM1 model.

### Reward

The revised reward penalizes:

1. DO below the minimum limit.
2. DO above the desired upper range.
3. BOD proxy above the illustrative threshold.
4. Aeration energy proxy (`u²`).
5. Frequent or large blower-setting changes.

```text
reward = -(
    low_do_weight   * low_do_penalty
  + high_do_weight  * high_do_penalty
  + bod_weight      * bod_penalty
  + energy_weight   * energy_penalty
  + switching_weight * switching_penalty
)
```

The reward weights are experimental design choices, not physically calibrated costs.

## 7. Latest fixed-blower comparison

The following results came from the **BSM1-derived processed dry influent** under one shared initial condition and influent start index:

| Fixed blower | Final DO (mg/L) | Final BOD proxy (mg/L) | Energy proxy total | Total reward | Steps with DO < 2 mg/L |
|---|---:|---:|---:|---:|---:|
| 0% | 0.00 | 124.07 | 0.0 | -5708.41 | 100.00% |
| 25% | 1.00 | 113.55 | 6.0 | -4383.06 | 86.46% |
| 50% | 2.75 | 109.68 | 24.0 | -3510.05 | 68.75% |
| 75% | 3.97 | 108.72 | 54.0 | -3479.96 | 0.00% |
| 100% | 4.75 | 108.36 | 96.0 | -3591.69 | 0.00% |

**Interpretation:** 75% received the highest (least negative) reward among these *fixed settings*, but its final DO exceeded the desired 2–3 mg/L range. The results do **not** prove it is the optimal policy. A dynamic controller may perform better by changing blower power over time.

## 8. Next developer: ordered handover checklist

- [ ] **1. Validate dataset preprocessing:** Confirm BSM1 flow `Q` is converted from m³/day to m³/h exactly once (`Q / 24`), and confirm 15-minute sampling.
- [ ] **2. Validate process model:** Check equations, units, numerical integration, state clipping, tank volume, hydraulic residence time, and plausible parameter ranges.
- [ ] **3. Test all conditions:** Run dry, rain, and storm datasets; save reproducible results and note limitations.
- [ ] **4. Implement rule-based baseline:** Create/test automatic DO control and save comparison metrics.
- [ ] **5. Prepare learning data:** Make chronological train/validation/test splits without crossing boundaries; normalize observations using training data only.
- [ ] **6. Implement standard DQN:** Q-network, replay buffer, epsilon-greedy exploration, target network, and checkpoint saving.
- [ ] **7. Train/evaluate DQN:** Record reward curves, DO violations, BOD proxy, energy proxy, switching, and stability.
- [ ] **8. Implement PER-DQN:** Prioritized replay, TD-error priorities, and importance-sampling correction.
- [ ] **9. Train/evaluate PER-DQN:** Use comparable seeds, training budgets, and held-out conditions.
- [ ] **10. Compare controllers fairly:** Fixed blower, rule-based, DQN, and PER-DQN on identical test scenarios.
- [ ] **11. Explain the trained agent:** Generate SHAP explanations and extract IF–THEN rules from a surrogate decision tree; measure surrogate fidelity.
- [ ] **12. Connect models to Pygame:** Add trained DQN/PER-DQN controller modes and display actions and metrics live.
- [ ] **13. Finalize deliverables:** Organize reproducible scripts, datasets, plots, methodology, assumptions, references, presentation, and live demo.

**Do not train the final AI model before verifying the environment and data units.** Changing physics or reward weights after training generally requires retraining.

## 9. Suggested evaluation metrics

Report all controllers on the same scenarios:

- Percentage of steps with DO below 2 mg/L.
- Percentage of steps with DO above 3 mg/L.
- Mean/minimum/maximum DO (mg/L).
- Mean/final BOD proxy (mg/L).
- Sum of normalized energy proxy (`u²` per step).
- Cumulative reward.
- Number/magnitude of blower changes.
- Training convergence and variation across random seeds.
- Runtime and surrogate-rule fidelity, where relevant.

## 10. Data source and reproducibility notes

- **Influent data:** International Water Association (IWA), Benchmark Simulation Model No. 1 (BSM1), dry/rain/storm influent time series: <https://iwa-mia.org/benchmarking/>.
- Record the exact downloaded dataset versions and preprocessing formulas in `01_Dataset/DATASET_README.md`.
- Record random seeds, environment configuration, reward weights, training hyperparameters, and data splits for every experiment.
- Do not interpret an RL reward improvement alone as proof of real-world energy savings or effluent compliance.

## 11. Recommended first action for the next team member

Open `02_Python_Code/config.py`, `prepare_dataset.py`, and `wastewater_env.py`; verify the input units and one-step calculations; then rerun `test_environment.py` and `validate_environment.py`. Once the model is checked, implement the rule-based controller and establish a reproducible baseline before training DQN.
