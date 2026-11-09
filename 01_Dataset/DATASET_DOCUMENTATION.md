# Dataset Documentation and Provenance

## 1. Dataset Identity

**Project:** Wastewater Treatment Aeration Tank Control Using PER-DQN and Rule Extraction

**Dataset:** Benchmark Simulation Model No. 1 (BSM1) Influent Scenarios

**Source Organization:** International Water Association (IWA) Task Group on Benchmarking of Control Strategies for Wastewater Treatment Plants

**Official Technical Documentation:**

[BSM1 General Description — IWA Technical Report](https://iwa-mia.org/wp-content/uploads/2019/04/BSM_TG_Tech_Report_no_1_BSM1_General_Description.pdf)

**Reference Implementation:**

[Benchmark Simulation Models — GitHub](https://github.com/wwtmodels/Benchmark-Simulation-Models)

---

## 2. Dataset Purpose

The BSM1 benchmark provides standardized wastewater influent scenarios for studying and comparing treatment-process control strategies.

Our project uses these influent scenarios as external disturbances for a simplified aeration tank simulator.

The reinforcement learning agent learns by interacting with this simulator rather than from labeled examples of optimal blower actions.

---

## 3. Dataset Provenance and Reliability

BSM1 is an established, scientifically documented wastewater treatment simulation benchmark.

The benchmark provides realistic influent patterns, but its files must not be described as verified raw measurements collected directly from a particular operating wastewater treatment plant.

The data are suitable for reproducible simulation experiments. However, they do not independently establish:

- The accuracy of our simplified aeration tank model.
- The safety of the trained controller in a real wastewater treatment plant.
- The controller's generalization to real operational conditions.

**Verification Status:** The official benchmark documentation and original filenames have been identified. Local file checksums have been recorded. Byte-for-byte verification against independently obtained official reference files is still pending.

---

## 4. Original Dataset Files

| Filename | Scenario | Description |
|---|---|---|
| `Inf_dry_2006.txt` | Dry | Two weeks of dynamic dry-weather influent |
| `Inf_rain_2006.txt` | Rain | Dry-weather baseline with a rainfall event during the second week |
| `Inf_strm_2006.txt` | Storm | Dry-weather baseline with storm events during the second week |

The BSM1 documentation specifies the following units:

| Variable | Unit |
|---|---|
| Time | Days |
| Influent flow | m³/day |
| Wastewater component concentrations | g/m³ |

The original files include soluble and particulate wastewater components and influent flow.

---

## 5. Data Preprocessing

Our project prepares a simplified dataset using the following variables:

- **Q:** Influent flow.
- **Ss:** Readily biodegradable substrate.
- **Xs:** Slowly biodegradable substrate.
- **BOD proxy:** Project-defined approximation of organic load.

### 5.1 BOD Proxy Calculation

The project-defined BOD proxy is calculated as:

```python
bod_in_proxy = 0.6 * (Ss + Xs)
```

This formula is a modeling approximation. It is not a directly measured laboratory BOD value or an official BSM1 BOD conversion.

### 5.2 Flow Conversion

Influent flow is converted from cubic meters per day to cubic meters per hour:

```python
flow_m3h = flow_m3day / 24
```

### 5.3 Concentration Units

Concentrations expressed in g/m³ are numerically equivalent to mg/L:

```text
1 g/m³ = 1 mg/L
```

**Implementation Note:** These transformations should be checked against the current `prepare_dataset.py` implementation.

---

## 6. Dataset Size and Splitting

The project documentation reports **1,345 observations per weather scenario**, representing approximately 14 days of data sampled every 15 minutes, including the endpoint.

### Chronological Dataset Split

| Partition | Percentage | Rows per scenario |
|---|---:|---:|
| Training | 70% | 941 |
| Validation | 15% | 202 |
| Testing | 15% | 202 |
| **Total** | **100%** | **1,345** |

The observations are divided chronologically rather than randomly shuffled to reduce leakage between neighboring time steps.

The dataset is used to provide environmental inputs to a reinforcement learning simulator. Therefore, there is no supervised-learning target label for the optimal blower action.

---

## 7. Weather-Testing Limitation — Confirmed

SHA-256 checks performed on **9 November 2026** confirmed that the following three processed test datasets are byte-for-byte identical:

- `dry_test.csv`
- `rain_test.csv`
- `storm_test.csv`

All three files have the same SHA-256 checksum:

```text
3C4876586F275CCB4924BA164E06E2CAE24FE86A6A0DBE18BE98D5FF819144B4
```

### Interpretation

The existing test partitions do not represent three independent weather conditions.

Although the original dry, rain, and storm influent files contain different data, their processed test partitions are identical.

Therefore, evaluations using these test partitions must not be described as independent rain- and storm-weather robustness tests.

### Required Future Validation

Additional evaluation using genuinely distinct, non-overlapping held-out weather-event periods is required to assess generalization to rain and storm conditions.

---

## 8. Real Versus Simulated Information

| Component | Classification |
|---|---|
| BSM1 influent scenarios | Scientifically documented benchmark data |
| Influent flow | Derived from benchmark inputs |
| Influent BOD proxy | Calculated approximation |
| Tank dissolved oxygen (DO) | Simulated |
| Tank BOD proxy | Simulated |
| Blower response | Simulated |
| Energy metric | Normalized mathematical proxy |
| DQN/PER-DQN rewards | Simulation-based outcomes |

**Important:** The project does not claim validation using measured effluent quality or actual plant electricity consumption.

---

## 9. Dataset Integrity Verification

On **9 November 2026**, SHA-256 checksums were calculated for the local raw BSM1 influent files.

### 9.1 Original Raw File Checksums

| Original File | SHA-256 |
|---|---|
| `Inf_dry_2006.txt` | `5C6767A1D7E8051D3FF3C2FB4EC959D97911C5CAF692253179D2CB1CC4BF4F8F` |
| `Inf_rain_2006.txt` | `B2FFC6A2581AEDD3C56FB98592C974B3D6678917A4B791B5E2426AF54914DC6D` |
| `Inf_strm_2006.txt` | `B7E0322E487D04160B24D51CA3703B9DD5A5D3CBA38B2229A1EB5B9B0F71C6FA` |

The three raw files have different SHA-256 hashes, confirming that their contents differ.

### 9.2 Processed Test File Checksums

| Processed File | SHA-256 |
|---|---|
| `dry_test.csv` | `3C4876586F275CCB4924BA164E06E2CAE24FE86A6A0DBE18BE98D5FF819144B4` |
| `rain_test.csv` | `3C4876586F275CCB4924BA164E06E2CAE24FE86A6A0DBE18BE98D5FF819144B4` |
| `storm_test.csv` | `3C4876586F275CCB4924BA164E06E2CAE24FE86A6A0DBE18BE98D5FF819144B4` |

The matching hashes confirm that the processed test files are byte-for-byte identical.

### 9.3 Verification Status

| Verification Task | Status |
|---|---|
| Identify official BSM1 documentation | Completed |
| Identify original dataset filenames | Completed |
| Calculate local raw-file SHA-256 hashes | Completed |
| Check processed test-file duplication | Completed |
| Compare raw files with independently downloaded official copies | Pending |
| Verify current preprocessing implementation against documentation | Pending |

### 9.4 Verification Interpretation

These checks establish local file integrity and identify duplicate datasets.

However, checksums alone do not establish scientific validity or prove that the files match the official BSM1 distribution.

Official-source comparison remains necessary before claiming byte-for-byte provenance verification.

### 9.5 Numerical Comparison with the Official BSM1 Release

An independent reference copy was downloaded from the IWA Benchmark Simulation Models GitHub repository, release `BSM1_v1`.

**Official reference:** https://github.com/wwtmodels/Benchmark-Simulation-Models/releases/download/BSM1_v1/BSM1.zip

The local files were compared against the corresponding official ASCII influent files:

| Local File | Official Reference |
|---|---|
| `Inf_dry_2006.txt` | `dryinfluent.ascii` |
| `Inf_rain_2006.txt` | `raininfluent.ascii` |
| `Inf_strm_2006.txt` | `storminfluent.ascii` |

#### Dataset Structure

The official reference files contain **1,344 rows and 10 columns**, while the local files contain **1,345 rows and 15 columns**.

The comparison aligned the 10 corresponding variables and compared the first 1,344 observations, excluding the additional final local row.

Numerical comparison used `numpy.allclose()` with relative and absolute tolerances of `1e-6`.

#### Numerical Comparison Results

| Scenario | Matching Columns | Differences |
|---|---|---|
| Dry | 10 of 10 | None within tolerance |
| Rain | 9 of 10 | `Si` differs in 197 rows |
| Storm | 9 of 10 | `Si` differs in 71 rows |

The maximum absolute differences in `Si` were:

- **Rain:** 20.0 g/m³
- **Storm:** approximately 22.29158 g/m³

In the differing observations, the local `Si` values were 30 g/m³, whereas the official reference contained varying values.

The project's primary influent variables, `Ss`, `Xs`, and `Q`, matched the official reference within tolerance across all three scenarios.

#### Additional Endpoint Row Verification

The local files contain an additional observation at `t = 14.0` days, whereas the official reference ends at approximately `t = 13.98958333` days.

The final local observation was compared with the first observation, excluding the time column, using `numpy.allclose()` with `rtol=0` and `atol=1e-8`.

| Scenario | Final Row Repeats First Row |
|---|---|
| Dry | No |
| Rain | No |
| Storm | Yes |

Therefore, the additional observation cannot be universally described as a repeated endpoint. The origin of the additional dry and rain endpoint values remains unverified.

#### Verification Conclusion

The numerical comparison establishes agreement within the specified tolerance between the project's primary influent variables (`Ss`, `Xs`, and `Q`) and the official BSM1 reference for the first 1,344 observations across all three scenarios.

However, the following limitations remain:

1. The local files contain an additional observation at day 14.
2. The rain dataset differs from the official reference in `Si` for 197 observations.
3. The storm dataset differs from the official reference in `Si` for 71 observations.
4. The three processed test CSV files have identical SHA-256 hashes, indicating byte-for-byte identical contents. They therefore do not provide independent weather-scenario test partitions.

**Overall conclusion:** The dataset comparison provides strong evidence that the project's primary influent inputs are consistent with the official BSM1 benchmark. However, complete numerical identity between the local files and the official reference has not been established. The documented differences and evaluation limitations must be considered when interpreting the project's results.

### 9.6 Weather-Scenario Independence and Test Partition Analysis

To investigate the identical SHA-256 hashes of the three processed test CSV files, the processed dry, rain, and storm influent datasets were compared across their chronological training, validation, and testing partitions.

The comparison used the two model input features:

- `bod_in_proxy_mg_l`
- `flow_m3_h`

Rows were considered different when at least one feature differed beyond an absolute tolerance of `1e-8`.

#### Results

| Partition | Number of Rows | Dry vs Rain: Different Rows | Dry vs Storm: Different Rows |
|---|---:|---:|---:|
| Training | 941 | 137 | 11 |
| Validation | 202 | 60 | 60 |
| Testing | 202 | 0 | 0 |

The training and validation partitions contain differences between weather scenarios.

However, the final 202 observations of all three processed datasets are identical.

This explains why `dry_test.csv`, `rain_test.csv`, and `storm_test.csv` have identical SHA-256 hashes.

#### Interpretation

The chronological 70/15/15 splitting procedure is correctly implemented in `data_split.py`, assuming the processed inputs remain sorted by time.

However, separate filenames do not guarantee independent test conditions.

The current held-out test partitions represent the same processed influent sequence across dry, rain, and storm scenarios. Consequently, they cannot establish independent performance under distinct rain or storm conditions.

Weather-event evaluation must therefore be reported separately from chronological held-out testing.

Any weather-event windows overlapping model training or validation periods must be identified as diagnostic evaluations rather than independent held-out test results.

#### Conclusion

The dataset preparation and chronological splitting procedures are consistent with the documented implementation.

The weather-scenario comparison confirms that the current test partitions

---

## 10. Dataset and Model Limitations

1. **Benchmark versus real measurements:** BSM1 influent scenarios are benchmark inputs, not independently verified measurements from a specific operating plant.

2. **Simplified environment:** The project uses a simplified single-tank model rather than the complete BSM1 wastewater treatment process.

3. **Approximate BOD:** The BOD proxy is calculated from substrate concentrations and is not a laboratory-measured BOD value.

4. **Energy approximation:** Energy consumption is represented by a normalized mathematical proxy rather than calibrated electricity consumption in kWh.

5. **Identical test partitions:** The existing dry, rain, and storm test CSVs are confirmed to be identical and cannot demonstrate independent weather-event generalization.

6. **Real-world validation:** Generalization to real wastewater treatment plants has not been demonstrated.

---

## 11. References

1. International Water Association Task Group on Benchmarking of Control Strategies for Wastewater Treatment Plants. *Benchmark Simulation Model No. 1 (BSM1): General Description*. IWA Technical Report. [Official documentation](https://iwa-mia.org/wp-content/uploads/2019/04/BSM_TG_Tech_Report_no_1_BSM1_General_Description.pdf).

2. Gernaey, K. V., Jeppsson, U., Vanrolleghem, P. A., & Copp, J. B. (Eds.). (2014). *Benchmarking of Control Strategies for Wastewater Treatment Plants*. IWA Publishing, Scientific and Technical Report No. 23.

3. IWA Benchmarking Task Group. *Benchmark Simulation Models — Reference Implementation*. [GitHub repository](https://github.com/wwtmodels/Benchmark-Simulation-Models).

---

## 12. Conclusion

The project uses a recognized wastewater simulation benchmark with documented scientific origins.

Local integrity checks confirm that the three original influent files contain different data, while the processed dry, rain, and storm test partitions are identical.

The dataset is suitable for controlled simulation experiments, provided that its benchmark nature and modeling limitations are clearly acknowledged.

**Final verification status:** Dataset provenance is documented, local integrity checks are completed, and independent official-file comparison remains pending.