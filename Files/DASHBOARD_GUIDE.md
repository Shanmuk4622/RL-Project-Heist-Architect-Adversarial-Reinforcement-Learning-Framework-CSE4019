# 🖥️ Heist Architect — Web Dashboard Guide

Welcome to the **Heist Architect Web Dashboard**! This interactive control center serves as the primary visual and managerial interface for the adversarial reinforcement learning framework. 

This guide will explain how to spin up the web space, what each panel does, and how you can manage your AI training directly from your browser.

---

## 🚀 1. How to Launch the Web Space

The entire web space is built using Flask and WebSocket (Socket.io) to provide real-time updates without refreshing the page.

To launch the dashboard, open your command prompt/terminal and run:

```bash
# 1. Activate your environment
conda activate cv_conda

# 2. Start the visualization server
python main.py visualize --port 5000
```

Once the terminal outputs that the server is running, open your web browser and go to:
👉 **[http://localhost:5000](http://localhost:5000)**

*(Note: Keep the terminal window open, as it acts as the backend server for the web app!)*

---

## 🗺️ 2. Grid Visualization (Left Panel)

The main focus of the application is the **Environment Grid**. It provides a live, 20x20 visual representation of the Heist map.

### What you are watching:
- **Tiles:** Includes the Start (Neon Green), Vault (Pink), Walls (Slate 3D Blocks), Cameras (Purple borders), and Guards (Orange borders).
- **The Vision Cone (Purple Hint):** Represents the real-time "Danger Zones". If the Solver steps into these tiles, it gets caught!
- **The Solver (Gold Circle):** Watch the AI actively move and leave a path trail across the grid.
- **Tick Counter:** Shows the exact time-step of the current simulation out of `max_steps`.

**👁️ Toggle Visibility Button:** At the top right of the grid, the Eye icon toggles the display of the purple surveillance vision cones.

---

## 📋 3. Game Log (Bottom Left Panel)

Below the grid is the **Game Log**. This is a persistent tabular record of every episode run during your session. 
- It tracks the budget, generated layout (Cameras, Walls, Guards), solve rates, and rewards.
- The **Mode** column tells you whether an episode was run automatically during training (`Auto`) or if you triggered it manually (`🎮 Interactive`).
- If you freeze an agent during interactive mode, you'll see a snowflake icon (`❄️A` for Architect or `❄️S` for Solver) indicating that agent wasn't learning during that run.

---

## ⚙️ 4. Training & Control Panels (Right Side)

The right side of the dashboard is your control center. It contains several cards to direct the reinforcement learning execution.

### A. Training Controls
This is how you initiate deep-learning loops.
- **Episodes:** Total number of map iterations to train.
- **Solver Attempts / Layout:** How many times the Solver tries to beat a single map before the Architect updates its strategy.
- **🚀 Start Training:** Begins automated adversarial training. A progress bar will appear.
- **🎬 Run Demo:** Runs a single test episode using the latest model checkpoint weights.

### B. Interactive Episode (Manual Control)
Want to step in and test specific parameters without running a full multi-hour training block? Use this panel.
- **Budget / Episodes:** Force the Architect to use exactly `N` budget and run `X` episodes.
- **Temperature:** Controls the Architect's randomness. Higher values (`>1.5`) create chaotic, unpredictable mazes. Lower values (`<0.5`) force it to use its absolute most confident security setups.
- **Freeze Toggles:** 
  - *Freeze Architect* prevents the Architect's neural network from updating (useful to let the Solver "catch up" on a particularly hard map).
  - *Freeze Solver* prevents the Solver from updating.
- **Asset Restrictions:** Uncheck Cameras or Guards to force the Architect to build purely wall mazes or camera-only intersections.

### C. Path Simulation
Allows you to load historic checkpoints and retroactively simulate environments from past training epochs. 
- Select a checkpoint from the dropdown to load a specific historical state of the AI, then click **Simulate Demo**.

### D. Live Metrics & Training Progress (Charts)
- **Live Metrics Board:** Constantly updates with percentages detailing who is winning.
  - *Solve Rate (Green)* vs *Detection Rate (Red)*.
- **Charts:** Two dynamic spline charts that map the `Rewards` (Architect vs Solver) and the `Rates` over time. The charts use smoothed curves to help you identify when the agents hit an equilibrium zone.

---

## 💡 Best Practices for Managing the Workspace

1. **Start Simple:** Before doing a 1000 episode training run, use the **Interactive block** with a low budget (e.g. `Budget = 5`, unchecked Cameras/Guards) to watch the Solver learn basic wall-hugging mechanics.
2. **Watch the Equilibrium:** Check the **Rates** chart. If the Solve Rate drops to `0%` for too long, the Architect made the level impossible or too hard. You might want to pause or run some Interactive episodes with the Architect Frozen (`❄️A`) so the Solver can figure it out.
3. **Save Your Results:** Every 50 episodes during training, the backend automatically saves `.pt` network checkpoints to your directory, so you'll never lose progress if you accidentally close the web tab!
