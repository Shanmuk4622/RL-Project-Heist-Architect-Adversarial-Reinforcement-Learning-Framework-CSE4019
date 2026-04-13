# 📊 Heist Architect V2: Training Results & Analytics

This document tracks the evaluation metrics, performance expectations, and analytics derived from deploying the Heist Architect framework on dual NVIDIA Tesla T4 instances.

## 📈 Performance Dashboards

Rather than logging thousands of raw numbers to the terminal, the `.ipynb` notebook generates a unified `training_dashboard.png` visualizing the following dimensions:

1. **Reward Trajectories:** The foundational PPO reward curve measuring the Architect's defensive success against the Robber's grid-solving success.
2. **Win Rate Rolling Average:** A strict 100-episode window indicating true traversal dominance. Curriculum boundaries are shattered when this line crosses 0.55.
3. **Triggered Alarms & Vaults:** Micro-metrics capturing whether the Robber adopts stealth behavior (fewer sub-zero alarm penalties) or aggressive hoarding (maximizing vaults).
4. **Episode Step-Lengths:** Identifies when the Robber realizes that artificially extending the clock is suboptimal against increasing guard proximities.

## ⚡ ELO Arms Race Dynamics

The implementation of `EloRater` acts as the definitive indicator of model convergence:
*   **The Rookie Stage:** Robber ELO initially spikes because a single randomly walking guard provides little map coverage.
*   **The Overfit Rebound:** The Architect network learns to cluster cameras near vaults, causing an aggressive negative slope in Robber ELO.
*   **Equilibrium:** Under PPO constraints, neither side can easily exploit the other. LSTM hidden states are fully utilized, pushing the agents into high-level spatial deduction rather than memorized routes.

## 💾 Model Checkpoints & Cloud Strategy

Thanks to automated Hugging Face integration, results are fully decoupled from local Kaggle runtimes.
- You can access the definitive model repository and all historical iterations at `Shanmuk4622/heist-architect-v2`.
- `architect.pt` and `robber.pt` weights are automatically synced every 500 episodes alongside live `metrics.json` outputs. 
- These `.pt` files are ready to be plugged into any local PyGame rendering loop to visually observe the strategies our agents have developed.
