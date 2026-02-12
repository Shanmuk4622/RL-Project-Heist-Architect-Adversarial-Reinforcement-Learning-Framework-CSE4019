# 🏦 Heist Architect: Adversarial Reinforcement Learning Framework

> *Two AI agents play an infinite game of cops-and-robbers. One designs the security system. The other breaks in. Over thousands of rounds, they both get dangerously good at their jobs.*

---

## 📖 Table of Contents

1. [What Is This Project?](#-what-is-this-project)
2. [The Game — How It Works](#-the-game--how-it-works)
3. [The Two Agents](#-the-two-agents)
4. [The Environment — The 2D Grid World](#-the-environment--the-2d-grid-world)
5. [Security Components](#-security-components)
6. [How the Agents Learn (Reinforcement Learning)](#-how-the-agents-learn-reinforcement-learning)
7. [The Reward System — The Rules of the Game](#-the-reward-system--the-rules-of-the-game)
8. [The Training Loop — How They Get Smarter](#-the-training-loop--how-they-get-smarter)
9. [The Curriculum — Gradually Increasing Difficulty](#-the-curriculum--gradually-increasing-difficulty)
10. [Neural Network Architecture](#-neural-network-architecture)
11. [Project Structure](#-project-structure)
12. [How to Run](#-how-to-run)
13. [What to Expect During Training](#-what-to-expect-during-training)
14. [The Science Behind It](#-the-science-behind-it)

---

## 🎯 What Is This Project?

Imagine you're building an AI that designs the **perfect bank vault security system** — cameras placed at the best angles, guards walking the smartest patrol routes, walls funneling intruders into traps. Now imagine a *second* AI whose only job is to **break in** without getting caught.

**That's this project.**

It's a **dual-agent adversarial reinforcement learning** framework. Two neural networks compete against each other:

- **The Architect** 🔒 — designs security layouts (where to put walls, cameras, and guards)
- **The Solver** 🕵️ — tries to sneak past the security and reach the vault

Neither agent is programmed with strategies. They **learn from scratch** through trial and error, developing increasingly clever tactics just by competing with each other.

The beauty of the system is **emergent complexity**: nobody tells the Architect to create chokepoints or synchronized patrol patterns. Nobody tells the Solver to time camera rotations or hide behind walls. These strategies **emerge naturally** from the competition.

---

## 🎮 The Game — How It Works

Each round (called an "episode") follows this cycle:

```
┌──────────────────────────────────────────────────────────┐
│                   ONE EPISODE                            │
│                                                          │
│  1. ARCHITECT designs a security layout                  │
│     → Places walls, cameras, and guards on the grid      │
│     → Must stay within a budget (can't spam cameras)     │
│     → Must leave a valid path from START to VAULT        │
│                                                          │
│  2. SOLVER attempts to reach the vault                   │
│     → Starts at the START tile                           │
│     → Can move UP, DOWN, LEFT, RIGHT, or WAIT            │
│     → Must avoid camera vision cones and guard patrols   │
│     → Cameras ROTATE and guards MOVE over time           │
│                                                          │
│  3. EPISODE ENDS when one of three things happen:        │
│     ✅ Solver reaches the VAULT → Solver wins            │
│     🚨 Solver is DETECTED → Architect wins               │
│     ⏰ Solver runs out of TIME → Draw (both penalized)   │
│                                                          │
│  4. BOTH agents learn from the outcome                   │
│     → Architect adjusts where it places security         │
│     → Solver adjusts how it navigates                    │
│                                                          │
│  5. Repeat thousands of times...                         │
└──────────────────────────────────────────────────────────┘
```

---

## 🤖 The Two Agents

### The Architect 🔒 (The Security Designer)

The Architect's job is to create a security layout that **catches the Solver**. Think of it as a level designer for the hardest stealth game ever made.

**What it controls:**
- **Where to place walls** — blocks the Solver's path, forces detours
- **Where to place cameras** — rotating surveillance with triangular vision cones
- **Where to place guards** — moving patrols along defined routes
- **Camera settings** — angle, rotation speed, field of view
- **Guard patrol paths** — which tiles the guards walk between

**Constraints:**
- Has a **budget** — can't place unlimited security (wall=1 point, camera=3, guard=5)
- Must leave a **valid path** from Start to Vault (can't just wall off the vault)
- Budget increases over training time (curriculum learning)

**What it sees:**
- An empty grid with Start and Vault positions
- It outputs a complete layout in one shot — all walls, cameras, and guards at once

### The Solver 🕵️ (The Infiltrator)

The Solver's job is to navigate from the Start tile to the Vault tile **without being spotted** by any camera or guard.

**What it controls:**
- Movement: UP, DOWN, LEFT, RIGHT
- Can also **WAIT** — stay still for one time step (crucial for timing camera rotations!)

**What it sees:**
- The **occupancy grid** — where walls, cameras, guards, start, and vault are
- The **visibility map** — a real-time overlay showing which tiles are currently under surveillance (the "danger zones")
- Its **own position** relative to the vault
- A **time feature** — helps it reason about camera rotation cycles

**The challenge:**
- Cameras **rotate over time** — a safe tile now might be visible in 3 steps
- Guards **move along patrol routes** — the Solver must time its movements
- The grid can have **chokepoints** — narrow corridors forced by wall placement
- Every step costs a tiny penalty — can't just wait forever

---

## 🗺️ The Environment — The 2D Grid World

The game takes place on a 2D grid (default: 20×20 tiles, but configurable).

```
####################
#S.................#      S = Start (Solver spawns here)
#..................#      V = Vault (Solver's goal)
#..####............#      # = Wall (blocks movement & vision)
#..................#      C = Camera (has rotating vision cone)
#......C...........#      G = Guard (moves along patrol path)
#..................#      . = Empty (walkable tile)
#..................#
#..................#
#..........G.......#
#..................#
#..................#
#..................#
#..................#
#..................#
#..................#
#..................#
#..................#
#.................V#
####################
```

**Key rules:**
- Border tiles are always walls (the grid has a permanent perimeter)
- Start is always at **(1, 1)** (top-left interior)
- Vault is always at **(grid-2, grid-2)** (bottom-right interior)
- Only the interior tiles can have security assets placed on them

---

## 🔐 Security Components

### Walls 🧱
- **Cost:** 1 budget point
- **Behavior:** Static obstacle, doesn't move
- **Effect:** Blocks the Solver's movement AND blocks camera/guard line-of-sight
- **Strategic use:** The Architect learns to create corridors and chokepoints that funnel the Solver into surveilled areas

### Cameras 📷
- **Cost:** 3 budget points
- **Behavior:** Fixed position, but the **heading rotates** over time
- **Settings:** Field of View (FOV angle), rotation speed, vision range
- **Vision cone:** A triangular area extending from the camera in the direction it's facing
- **Line-of-sight:** Walls block the camera's vision (no seeing through walls!)
- **Detection:** If the Solver steps into a camera's vision cone → **CAUGHT!**
- **Strategic use:** The Architect learns to place cameras at intersections and chokepoints where rotation sweeps cover maximum area

### Guards 💂
- **Cost:** 5 budget points (most expensive)
- **Behavior:** Moves along a **patrol path** — a loop of waypoints
- **Settings:** Speed, vision range, FOV angle
- **Vision cone:** Similar to cameras, but moves with the guard and points in the direction of movement
- **Detection:** Same as cameras — Solver in vision cone = caught
- **Strategic use:** Guards are mobile, making them harder to predict. The Architect learns patrol routes that complement camera blind spots

### The Visibility Map 🔴
At every time step, the system computes a **Dynamic Visibility Map** — a grid overlaying the environment that shows which tiles are currently "under surveillance."

```
Time Step 5:                    Time Step 10:
####################            ####################
#S.................#            #S........XXX......#
#..................#            #.........XX........#
#..####............#            #..####...X.........#
#............XXX...#            #..................#
#......C...XXXXX...#  ────►     #......C...........#
#..........XXXX....#            #....XXXXX..........#
#...........XXX....#            #....XXXX...........#
#..................#            #.....XXX...........#
####################            ####################

X = Currently visible to security (DANGER ZONE)
```

The visibility map changes every tick as cameras rotate and guards move. The Solver sees this map and must plan its route through the gaps.

---

## 🧠 How the Agents Learn (Reinforcement Learning)

### What is Reinforcement Learning?

Both agents learn through **trial and error**, the same way you'd learn a video game:

1. **Try something** (take an action)
2. **See what happens** (get a reward or penalty)
3. **Adjust your strategy** (update your neural network)
4. Repeat thousands of times

The agents don't have any pre-programmed strategies. They start with **completely random behavior** and gradually learn what works.

### The Algorithm: PPO (Proximal Policy Optimization)

Both agents use **PPO**, one of the most stable RL algorithms. Here's the intuition:

- The agent's neural network outputs a **probability distribution** over all possible actions
- During training, the agent takes actions, collects rewards, and asks: *"Was this action better or worse than expected?"*
- If an action gave more reward than expected → make it **more likely**
- If an action gave less reward → make it **less likely**
- PPO adds a "clip" that prevents the agent from changing its strategy too drastically in one update (prevents catastrophic forgetting)

### Reward Shaping — Teaching the Solver to Navigate

A critical design decision. If you only reward the Solver for reaching the vault (+10) and penalize detection (-1), it takes forever to learn because reaching the vault by random chance on a 20×20 grid is extremely unlikely.

**Solution: Distance-based reward shaping**

```
Every step, the Solver gets:
  +0.1  for each tile it moves CLOSER to the vault
  -0.1  for each tile it moves FARTHER from the vault
  -0.01 per step (encourages efficiency)
  +0.05 proximity bonus when very close to vault (within 3 tiles)

At the end:
  +10.0 for reaching the vault
  -1.0  for getting detected
  +2.0  partial credit at timeout (based on how close it got)
```

This gives the Solver **continuous feedback** — even early in training when it has no idea what it's doing, moving toward the vault feels good and moving away feels bad. This is what makes learning possible.

### Generalized Advantage Estimation (GAE)

The Solver also uses **GAE** — a technique that balances between two questions:

1. "How much reward did I *actually* get from this point?" (looking backward — more accurate but noisy)
2. "How much reward does my value function *predict* I'll get?" (looking forward — less noisy but might be wrong)

GAE smoothly blends these two signals, giving the Solver stable and efficient learning.

---

## 💰 The Reward System — The Rules of the Game

Rewards are designed as a **zero-sum game** — what helps one agent hurts the other.

### Solver Rewards (Per Step)

| Event | Reward | Why |
|---|---|---|
| Move closer to vault | +0.1 | Guides learning toward the goal |
| Move away from vault | -0.1 | Discourages wandering |
| Each time step | -0.01 | Encourages efficiency |
| Near vault (≤3 tiles) | +0.05 × proximity | Bonus for getting close |
| **Reach the vault** | **+10.0** | **Main objective — big reward** |
| **Get detected** | **-1.0** | **Penalty for being spotted** |
| Timeout (partial credit) | 0 to +2.0 | Based on final distance to vault |

### Architect Rewards (Per Layout)

The Architect gets one reward per layout, based on how the Solver performed across multiple attempts:

| Event | Reward | Why |
|---|---|---|
| Detection rate × 1.0 | 0 to +1.0 | More detections = better security |
| Solver succeeds >80% | -0.5 | Security is too easy |
| Solver succeeds 20-60% | +0.2 | Bonus for "challenging but fair" layouts |
| Invalid level (no path) | -1.0 | Can't just wall off the vault |

The ideal for the Architect: make levels where the Solver *can* succeed but *usually doesn't* (the sweet spot is ~30% solve rate).

---

## 🔄 The Training Loop — How They Get Smarter

Training follows a **GAN-style adversarial loop** — similar to how GANs (Generative Adversarial Networks) work in image generation:

```
┌─────────────────────────────────────────────────────────┐
│              ADVERSARIAL TRAINING LOOP                  │
│                                                         │
│  For each episode:                                      │
│                                                         │
│  ┌───────────────┐                                      │
│  │   ARCHITECT   │──generates layout──►┌──────────┐     │
│  │  Neural Net   │                     │   GRID   │     │
│  └───────────────┘                     │ + walls  │     │
│         ▲                              │ + cams   │     │
│         │                              │ + guards │     │
│    gets reward                         └────┬─────┘     │
│    based on                                 │           │
│    solve_rate                               ▼           │
│         │                          ┌────────────────┐   │
│         │                          │  SOLVER tries  │   │
│         │                          │  20 attempts   │   │
│         │                          │  (per layout)  │   │
│         │                          └───────┬────────┘   │
│         │                                  │            │
│         └──────────────────────────────────┘            │
│                                                         │
│  Both neural networks update (PPO)                      │
│  Next episode: Architect designs a NEW layout           │
└─────────────────────────────────────────────────────────┘
```

**Per episode:**
1. **Architect generates** a complete security layout (walls + cameras + guards)
2. **Layout validated** — must have a path from Start to Vault
3. **Solver attempts** the level 20 times (same layout, different attempts)
4. **Results tracked** — what % of attempts succeeded vs. detected vs. timed out
5. **Both agents update** their neural networks based on rewards
6. **Repeat** with a new layout

The key insight: the Architect sees the **aggregate results** (solve rate), not individual Solver moves. It learns patterns like "putting a camera at this intersection catches the Solver more often."

---

## 📈 The Curriculum — Gradually Increasing Difficulty

You can't throw a beginner into the hardest level. Training uses **curriculum learning** — a gradual ramp-up:

```
    Difficulty
    ▲
    │                                          ┌──────────
    │                              ┌───────────┘ Phase 4
    │                              │ Expert
    │               ┌──────────────┘ budget=22
    │               │ Phase 3        cameras+guards
    │    ┌──────────┘ Full Security
    │    │ Phase 2    budget=15
    │    │ +Cameras   cameras+guards
    ├────┘ budget=8
    │ Phase 1
    │ Walls Only
    │ budget=5
    │
    │ WARMUP (no security at all)
    └──────────────────────────────────────────────────── Episodes
    0   30    80        200              400        500
```

| Phase | Episodes | Budget | Assets | Purpose |
|---|---|---|---|---|
| **Warmup** | Pre-training (30) | 0 | Empty grid | Solver learns basic navigation toward vault |
| **Phase 1** | 1 – 80 | 5 | Walls only | Solver learns to navigate around obstacles |
| **Phase 2** | 81 – 200 | 8 | Walls + Cameras | Solver learns to avoid rotating vision cones |
| **Phase 3** | 201 – 400 | 15 | Full security | Both agents enter adversarial competition |
| **Phase 4** | 400+ | 22 | Expert | High-budget, complex layouts |

**Why warmup matters:** Without the warmup phase, the Solver gets detected instantly and never learns anything. The warmup gives it time to learn that "walk toward the vault" is a good strategy, before introducing any threats.

---

## 🏗️ Neural Network Architecture

### Solver Network (CNN + LSTM — 550K parameters)

```
Input: 3-channel grid image (10×10 or 20×20)
  Channel 1: Occupancy grid (walls, cameras, guards normalized)
  Channel 2: Visibility map (binary — which tiles are dangerous)
  Channel 3: Position gradient (solver position + distance gradient to vault)

       ┌──────────────────────────────────┐
Input ─►│  Conv2D(3→32, 3×3) + ReLU       │
       │  Conv2D(32→64, 3×3) + ReLU       │  ◄── CNN: "What does the grid
       │  Conv2D(64→64, 3×3) + ReLU       │       look like?"
       └──────────────┬───────────────────┘
                      │ flatten
                      ▼
              ┌───────────────┐
              │  LSTM(256)    │  ◄── LSTM: "What's changing over time?"
              │  (hidden=128) │       (remembers camera rotation pattern)
              └──────┬────────┘
                     │
              ┌──────┴──────┐
              │             │
              ▼             ▼
        ┌──────────┐  ┌──────────┐
        │  Policy  │  │  Value   │  ◄── Two-headed output
        │  Head    │  │  Head    │
        │  (5 acts)│  │  (1 val) │
        └──────────┘  └──────────┘
              │             │
              ▼             ▼
        action probs    state value
        [WAIT, U, D,    "how good is
         L, R]           this state?"
```

**Why LSTM?** Cameras rotate over time. The Solver needs to **remember** what it saw a few steps ago to predict when a camera will rotate away, creating a safe window. A simple feedforward network can't do this — it has no memory.

### Architect Network (CNN Encoder-Decoder — 407K parameters)

```
Input: 1-channel grid (just walls/start/vault structure)

       ┌──────────────────────────────────┐
Input ─►│  ENCODER                         │
       │  Conv2D(1→32) + ReLU             │
       │  Conv2D(32→64) + ReLU            │  ◄── Compress grid into
       │  Conv2D(64→128) + ReLU           │      a compact representation
       └──────────────┬───────────────────┘
                      │
       ┌──────────────┴───────────────────┐
       │  DECODER                          │
       │  ConvTranspose2D(128→64) + ReLU  │
       │  ConvTranspose2D(64→32) + ReLU   │  ◄── Expand back to grid
       │  ConvTranspose2D(32→3) + Sigmoid │      with per-tile probabilities
       └──────────────┬───────────────────┘
                      │
                      ▼
            Placement Probabilities
            [ wall_prob, camera_prob, guard_prob ]
            at every (row, col)

       + Separate heads for camera FOV, rotation speed, heading
       + Value head for critic
```

**How layout generation works:**
1. Network outputs a probability for each asset type at each tile
2. We sample from these probabilities (with temperature for exploration)
3. We place assets in order of probability until budget runs out
4. If the layout has no valid path → layout rejected, Architect penalized

---

## 📁 Project Structure

```
RL/
├── main.py                         # CLI entry point (train / demo / visualize)
├── requirements.txt                # Python dependencies
├── configs/
│   └── default.yaml                # All hyperparameters in one place
│
├── heist_architect/                # Core RL framework
│   ├── __init__.py
│   ├── environment.py              # The 2D grid simulation engine
│   │                               # - Step logic, collision detection
│   │                               # - Observation generation for neural nets
│   │                               # - Distance-based reward shaping
│   │
│   ├── rewards.py                  # Zero-sum reward calculator
│   │                               # - Architect: detection rate based
│   │                               # - Solver: vault reach, detection penalty
│   │
│   ├── networks.py                 # Neural network definitions (PyTorch)
│   │                               # - SolverNetwork: CNN + LSTM
│   │                               # - ArchitectNetwork: Encoder-Decoder CNN
│   │
│   ├── training.py                 # The adversarial training loop
│   │                               # - Warmup → Curriculum → Full adversarial
│   │                               # - Metrics tracking, checkpointing
│   │
│   ├── utils.py                    # Helpers: GPU detection, BFS pathfinding
│   │
│   ├── agents/
│   │   ├── architect.py            # Architect agent (PPO, layout generation)
│   │   └── solver.py               # Solver agent (PPO + GAE, navigation)
│   │
│   └── components/
│       ├── security.py             # Wall, Camera, Guard classes
│       │                           # - Camera raycasting for vision cones
│       │                           # - Guard patrol path movement
│       ├── visibility.py           # Dynamic visibility map computation
│       └── budget.py               # Asset costs and budget management
│
├── visualization/                  # Web-based monitoring dashboard
│   ├── server.py                   # Flask + WebSocket server
│   ├── index.html                  # Dashboard UI
│   ├── style.css                   # Dark theme with glassmorphism
│   └── app.js                      # Canvas renderer + live charts
│
├── checkpoints/                    # Saved model weights (created during training)
└── logs/                           # Training metrics JSON (created during training)
```

---

## 🚀 How to Run

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA (optional but ~10x faster)
- Conda (recommended for environment management)

### Setup
```bash
conda activate cv_conda       # or your preferred environment
pip install -r requirements.txt
```

### Train the Agents
```bash
# Full training (500 episodes, ~30-40 min with GPU)
python main.py train

# Quick test (smaller grid, fewer episodes)
python main.py train --episodes 200 --grid-size 10

# Custom settings
python main.py train --episodes 1000 --grid-size 15 --solver-attempts 30
```

### Run a Demo
```bash
# Watch a single episode play out (uses trained models if available)
python main.py demo
python main.py demo --grid-size 10
```

### Visualization Dashboard
```bash
python main.py visualize
# Open http://localhost:5000 in your browser
```

The dashboard shows:
- **Live grid** — watch the Solver navigate in real-time
- **Camera cones** — rotating purple triangles
- **Guard paths** — dashed orange patrol routes
- **Training charts** — solve rate, rewards, and losses over time
- **Controls** — start training, run demos, toggle visibility

---

## 📊 What to Expect During Training

### Early Training (Episodes 1-80, Walls Only)
```
Solve rate:  80-100%    ← Solver learned basic navigation in warmup
Detection:   0%         ← No cameras or guards yet
Architect:   Learning wall placement, creating basic mazes
```

### Mid Training (Episodes 81-200, + Cameras)
```
Solve rate:  Drops to 30-60%  ← Cameras now catch the Solver
Detection:   20-40%           ← Solver is adapting to vision cones
Architect:   Placing cameras at intersections and chokepoints
Solver:      Learning to WAIT for cameras to rotate past
```

### Late Training (Episodes 200-400, Full Security)
```
Solve rate:  20-50%     ← Competitive equilibrium forming
Detection:   30-50%     ← Both agents are skilled
Architect:   Complex layouts with coordinated cameras + guards
Solver:      Timing movements, using walls as cover
```

### Expert (Episodes 400+)
```
Solve rate:  ~30-40%    ← Nash equilibrium zone
Detection:   ~40-50%    ← Intense competition
Architect:   Discovers advanced patterns (patrol synchronization)
Solver:      Exploits blind spots and timing windows
```

---

## 🔬 The Science Behind It

### What Makes This Different?

In traditional RL, an agent learns to solve a **fixed environment** — the same maze every time. This project is different because the environment itself is **an agent that's trying to beat you**.

This creates **Non-Stationary Adversarial Learning**:
- The Solver can't just memorize one solution — the layout changes every episode
- The Architect can't just place cameras randomly — the Solver adapts
- Both agents must **generalize** — learn principles, not patterns

### Emergent Strategies

These aren't programmed — they emerge naturally from the competition:

**Architect discovers:**
- **Chokepoints** — Forcing the Solver through narrow corridors covered by cameras
- **Dead ends** — Walls that look like paths but lead into surveilled areas
- **Patrol synchronization** — Guards timed so there's no safe window
- **Camera crossfire** — Multiple cameras covering each other's blind spots

**Solver discovers:**
- **Wall hugging** — Using walls as cover from camera vision
- **Timing** — Waiting for cameras to rotate away before sprinting past
- **Indirect paths** — Taking longer routes that avoid detection
- **Patience** — Using the WAIT action at critical moments

### Connection to Game Theory

The training converges toward a **Nash Equilibrium** — a state where neither agent can unilaterally improve by changing its strategy. At equilibrium:
- The Architect can't catch the Solver more often without making invalid levels
- The Solver can't reach the vault more often without getting detected
- The solve rate stabilizes around 30-40% — the natural balance point

This is the same mathematical framework behind poker AI, military strategy, and economic modeling.

---

## ⚙️ Configuration

All hyperparameters are in `configs/default.yaml`:

```yaml
environment:
  grid_rows: 20          # Grid dimensions
  grid_cols: 20
  max_steps: 200         # Steps before timeout

budget:
  wall: 1                # Cost per wall tile
  camera: 3              # Cost per camera
  guard: 5               # Cost per guard

training:
  learning_rate: 0.001   # How fast agents learn
  gamma: 0.99            # How much agents value future rewards
  entropy_coeff: 0.05    # Exploration encouragement
  ppo_epochs: 3          # Update passes per batch
```

---

*Built with PyTorch, NumPy, Flask, and SocketIO. GPU-accelerated via CUDA.*
