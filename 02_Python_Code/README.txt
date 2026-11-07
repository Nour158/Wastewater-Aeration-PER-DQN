Wastewater Aeration Pygame AI Integration

1. Copy pygame_aeration_lab.py into your project's 02_Python_Code folder (back up original first).
2. Keep existing rl_core.py, config.py, wastewater_env.py, and trained models.
3. Required models:
   03_Results/models/dqn_dry_seed42.pt
   03_Results/models/per_dry_seed42.pt
4. Run from the project root, with your .venv activated:
   python .\02_Python_Code\pygame_aeration_lab.py
5. Choose Manual, Auto DO, DQN, or PER-DQN. DQN/PER-DQN buttons are disabled if model loading fails.
6. Use Reset to restart the 24-hour episode. Use Save CSV for a trace.

Important:
- Models trained on dry data; rain/storm mode is a robustness demonstration, not held-out event validation.
- AI modes force training-time default physical/reward settings and lock sliders to prevent misleading out-of-distribution comparisons.
- This is a simplified illustrative tank and normalized energy proxy, not calibrated BSM1 or physical kWh.
- AI control is greedy (epsilon=0); switching mode mid-episode is a demo, not a controlled scientific comparison. Reset between comparisons.
