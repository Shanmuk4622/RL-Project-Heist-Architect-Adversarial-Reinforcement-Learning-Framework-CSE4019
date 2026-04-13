# 🏛️ Heist Architect V2: Adversarial PPO Framework

## 🎯 The Concept
The **Heist Architect** is an asymmetric, adversarial reinforcement learning framework where two intelligent agents are pitted against each other in a dynamic grid-world environment. 

1. **The Architect (The Creator):** 
   - Responsible for designing an impenetrable fortress. 
   - Actions include setting up impenetrable walls, determining optimal security camera placements, and establishing dynamic patrol routes for guards.
   - Constrained by a budget (e.g., specific number of cameras and guards based on difficulty).

2. **The Robber (The Solver):**
   - Responsible for breaching the fortress, navigating the terrain, dodging guards, and successfully robbing all vaults without being caught.
   - Evaluated by traversal efficiency and stealth.

This is a **zero-sum game**: the Architect's reward is directly tied to the Robber's failure, forcing an arms race where both agents must continuously evolve their strategies to survive.

---

## ⚙️ Key Technical Innovations (V2)

The Kaggle transition brings massive architectural improvements over our previous iteration:

*   **⚡ True Dual-GPU Asymmetric Training:** Instead of forcing both agents into a mirrored `DataParallel` block that bottlenecks VRAM, the Architect is assigned exclusively to `cuda:0` and the Robber to `cuda:1`. This allows us to scale parameter sizes significantly without OOM (Out of Memory) crashes.
*   **🧠 Genuine PPO Loops:** Transitioned from basic softmax outputs to a highly optimized Proximal Policy Optimization (PPO) loop utilizing Generalized Advantage Estimation (GAE), Critic Value updates, and exact LSTM-hidden-state batch management.
*   **⚔️ ELO-Based Self-Play:** Rather than always fighting the latest opponent, agents sample from a `SelfPlayPool`. This forces the Architect/Robber to fight random *past versions* of their opponent, completely eliminating "Strategy Collapse" (where agents overfit to one specific sequence and forget how to be robust).
*   **🎓 Dynamic Curriculum Learning:** The environment's initial state is dead simple (1 guard, 2 cameras). The `CurriculumManager` monitors the Robber's win rate. Only when the Robber hits a >55% win rate over 100 episodes does the environment physically upgrade to the next difficulty tier.
*   **☁️ Resumable Hugging Face Sync:** Because Kaggle limits continuous compute sessions, the trainer securely pushes model weights and metrics to Hugging Face every 500 episodes. If a kernel crashes, the runtime will download the latest snapshot and seamlessly resume.
