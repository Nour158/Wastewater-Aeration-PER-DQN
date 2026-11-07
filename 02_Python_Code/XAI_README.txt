WASTEWATER AERATION XAI PATCH

Copy explain.py into your project's 02_Python_Code folder (replace old explain.py).
Keep your models, datasets, train.py and evaluate.py unchanged.

From the project root, with (.venv) active:
python -m pip install shap scikit-learn matplotlib
python .\02_Python_Code\explain.py --algorithm dqn --dataset dry --train-dataset dry --model-seed 42
python .\02_Python_Code\explain.py --algorithm per --dataset dry --train-dataset dry --model-seed 42

For a quick tree-only test, add --skip-shap.

Outputs:
03_Results/metrics/rules_*.txt
03_Results/metrics/confusion_*.csv
03_Results/metrics/tree_importance_*.csv
03_Results/metrics/shap_importance_*.csv
03_Results/metrics/shap_values_*.npz
03_Results/figures/tree_*.png
03_Results/figures/shap_importance_*.png

The surrogate tree approximates neural-network actions; fidelity measures agreement,
not correctness. SHAP measures contributions to estimated Q-values, not causal effects.

MULTI-SEED TRAINING (OPTIONAL, LATER):
python .\02_Python_Code\train.py --algorithm dqn --dataset dry --episodes 600 --seed 43
python .\02_Python_Code\train.py --algorithm per --dataset dry --episodes 600 --seed 43
python .\02_Python_Code\train.py --algorithm dqn --dataset dry --episodes 600 --seed 44
python .\02_Python_Code\train.py --algorithm per --dataset dry --episodes 600 --seed 44
Then evaluate each with --model-seed 43 or 44. Do not average results from
weather windows as if they were independent held-out tests.
