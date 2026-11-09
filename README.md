# Wastewater Aeration Tank Control Using DQN, PER-DQN and Explainable AI

## Project Overview

This project investigates reinforcement learning for energy-aware aeration control in a simulated wastewater treatment tank.

It compares conventional Deep Q-Network (DQN) and Prioritized Experience Replay DQN (PER-DQN) against fixed-blower and rule-based controllers.

The project also applies explainable AI techniques to investigate learned decisions and extract interpretable surrogate rules.

**Project status:** Implemented and experimentally evaluated in a simplified simulation environment. Not validated for deployment in a real wastewater treatment plant.

## Objectives

- Simulate dissolved oxygen (DO) and organic-load dynamics.
- Train DQN and PER-DQN aeration controllers.
- Compare learned policies with fixed and rule-based baselines.
- Evaluate reward, DO violations, normalized aeration energy and blower switching.
- Investigate policy decisions using SHAP and surrogate decision trees.
- Assess learning improvement and consistency across random seeds.

## Dataset

The project uses influent data derived from the International Water Association Benchmark Simulation Model No. 1 (IWA BSM1).

Three influent scenarios are included:

- Dry weather
- Rain weather
- Storm weather

The preprocessing pipeline extracts influent characteristics and calculates:

`bod_in_proxy_mg_l = 0.6 * (Ss + Xs)`

`flow_m3_h = Q / 24`

The BOD quantity is an illustrative COD-derived proxy, not directly measured biochemical oxygen demand.

### Dataset Verification

Local raw influent data were compared numerically against an independently downloaded official BSM1 release.

The relevant `Ss`, `Xs` and `Q` columns matched within the comparison tolerance for the reference rows examined.

The chronological dataset split is:

- Training: 70%
- Validation: 15%
- Testing: 15%

**Important limitation:** The processed dry, rain and storm test partitions are identical. Therefore, evaluation on these three test files does not establish independent weather-scenario generalization.

Separate rain and storm event-window diagnostics were conducted using matched dry-weather reference windows. These event windows overlap the training/validation period and are not independent held-out tests.

See `01_Dataset/DATASET_DOCUMENTATION.md` for dataset provenance, preprocessing and verification details.

## Simulation Environment

The project uses a simplified single-tank model with:

- Dissolved oxygen dynamics
- Organic-load proxy dynamics
- Aeration-dependent oxygen transfer
- Influent dilution
- Simplified biological degradation
- Discrete blower control actions

The available blower fractions are:

`0%, 25%, 50%, 75%, 100%`

The reward penalizes low DO, excessive DO, excessive organic-load proxy, normalized aeration energy and blower switching.

This model is **not a complete BSM1/ASM1 wastewater treatment simulation**.

## Controllers

Five controllers are evaluated:

1. Fixed blower at 50%
2. Fixed blower at 75%
3. Rule-based controller
4. DQN
5. PER-DQN

DQN uses conventional experience replay, while PER-DQN prioritizes replay samples according to their learning relevance.

## Training

Both learning algorithms were trained on the dry-weather training partition.

Training configuration used in the completed experiments:

- 600 episodes per run
- Three random seeds: 42, 43 and 44
- Two learning algorithms: DQN and PER-DQN
- Six trained model checkpoints

The saved training histories include episode reward, exploration epsilon, training loss, low-DO violation percentage, blower fraction and action distribution.

### Training Findings

Average episode rewards improved substantially from the first 100 to the final 100 episodes.

| Metric | DQN | PER-DQN |
|---|---:|---:|
| Early mean reward | −656.22 | −657.81 |
| Late mean reward | −100.06 | −99.98 |
| Reward improvement | +556.16 | +557.82 |
| Across-seed late reward SD | 1.95 | 2.45 |

These results support substantial learning improvement and relatively consistent late-training rewards across three seeds.

However, reward was still improving toward episode 600. A definitive convergence plateau has not been demonstrated.

## Held-Out Test Evaluation

The trained models were evaluated on the chronological dry-weather test partition.

Each model seed was evaluated using the same evaluation random seed (900) and 20 episodes.

### Verified Multi-Seed Results

| Metric | DQN | PER-DQN | Rule-Based |
|---|---:|---:|---:|
| Mean reward | −61.461 | −62.348 | −65.202 |
| DO below 2 mg/L (%) | 14.583 | 14.132 | 3.802 |
| DO above 3 mg/L (%) | 2.396 | 4.931 | 4.896 |
| Energy proxy sum | 22.891 | 23.694 | 26.750 |
| Blower switches | 25.70 | 25.40 | 18.95 |

Values for DQN and PER-DQN are averaged across three trained model seeds.

DQN achieved slightly higher average reward and lower normalized energy than PER-DQN in this evaluation.

PER-DQN had a slightly lower average percentage of low-DO violations.

The rule-based controller had substantially fewer low-DO violations than either learned controller.

These findings do not establish that PER-DQN universally outperforms DQN or that the learned policies meet real-world operational safety requirements.

Detailed results are saved in `03_Results/metrics/verified_multiseed_summary.csv`.

## Rain and Storm Event Diagnostics

Additional experiments compare weather-event windows against matching dry-weather reference windows.

- Rain: rows 708–1096
- Storm: rows 750–1203

These experiments use the same window boundaries for the weather event and dry reference.

The final 96 processed observations in each event window are identical to the corresponding dry-weather inputs.

These experiments are useful for analyzing simulated disturbance responses but are **not independent held-out weather-generalization tests**.

## Explainable AI

The project includes explainability and surrogate modeling components:

- SHAP-based feature importance and Q-value attribution
- Surrogate decision-tree models
- Extracted interpretable decision rules
- Surrogate agreement evaluation

Surrogate rules approximate the actions selected by trained agents. They do not automatically provide safe or physically validated control rules.

Explainability outputs are stored in `03_Results/figures/` and `03_Results/metrics/`.

## Installation

From the project root, create and activate a Python virtual environment if needed, then install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Data Preparation

Place the following source files in `01_Dataset/raw/`:

- `Inf_dry_2006.txt`
- `Inf_rain_2006.txt`
- `Inf_strm_2006.txt`

Then run:

```powershell
python .\02_Python_Code\prepare_dataset.py
python .\02_Python_Code\data_split.py
python .\02_Python_Code\validate_model.py
python .\02_Python_Code\test_environment.py
```

## Training and Evaluation

Train a model using the project's training script and its documented command-line options.

Example:

```powershell
python .\02_Python_Code\train.py --algorithm dqn --episodes 600
python .\02_Python_Code\train.py --algorithm per --episodes 600
```

Evaluate an existing trained model:

```powershell
python .\02_Python_Code\evaluate.py --dataset dry --train-dataset dry --model-seed 42 --seed 900 --episodes 20 --mode test
```

Repeat with model seeds 43 and 44 for multi-seed comparison.

**Note:** The standard evaluation script uses a common output filename across model seeds. Preserve or rename each result before running the next seed to avoid overwriting earlier results.

## Project Results

The `03_Results` directory contains:

- `models/` — trained DQN and PER-DQN checkpoints
- `metrics/` — training histories, evaluation results and explainability metrics
- `figures/` — evaluation charts and explainability visualizations

## Scientific Limitations

1. The tank model is simplified and does not implement the full BSM1/ASM1 process.
2. The organic-load quantity is a proxy rather than measured BOD.
3. The normalized blower energy proxy is not electricity consumption in kWh.
4. The 30 mg/L organic-load threshold is illustrative, not a regulatory limit.
5. The three processed weather test partitions are identical.
6. Weather-event diagnostics overlap training/validation periods.
7. Random episode starts and within-partition wrapping limit temporal realism.
8. State clipping may conceal numerical or physical modeling problems.
9. Convergence to a stable training plateau has not been established.
10. Reward weights and kinetic parameters require calibration before real-world use.
11. Surrogate decision rules are explanatory approximations, not validated safety controllers.
12. The Pygame interface supports manual and rule-based operation; learned-controller integration into the interface is not established.

## Conclusion

DQN and PER-DQN both learned substantially improved aeration policies in the simplified simulated environment.

Across the verified held-out dry-weather evaluations, DQN achieved slightly better reward and energy results, while PER-DQN had slightly fewer low-DO violations.

The rule-based controller maintained substantially better protection against low DO, highlighting the need for stronger safety constraints before practical deployment.

The project demonstrates a reproducible workflow for reinforcement learning, controller comparison and explainability in a benchmark-derived wastewater aeration simulation.