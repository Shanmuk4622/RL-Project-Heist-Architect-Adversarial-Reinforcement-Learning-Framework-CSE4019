# 🎬 Heist Architect — Full Presentation Guide
**File:** `Heist_Architect_Presentation_CSE4019.pptx`
**Slides:** 19 | **Duration:** ~20–25 minutes | **Theme:** Dark glassmorphism

---

## 📐 Design System (for manual edits in PowerPoint)

### Colour Palette
| Name | Hex | Used For |
|---|---|---|
| Dark Background | `#0D1117` | All slide backgrounds |
| Mid Background | `#141E2E` | Cards & panels |
| Accent Cyan | `#00B4D8` | Primary headings, Architect elements |
| Accent Amber | `#FFB700` | Stats, Solver elements, highlights |
| Accent Coral | `#FF476F` | Danger, detection, collapse phases |
| Accent Mint | `#06D6A0` | Success, solve rate, recovery phases |
| White | `#FFFFFF` | Primary body text |
| Light Gray | `#C8D8E8` | Secondary body text |
| Dim Gray | `#556677` | Captions, metadata, footnotes |

### Fonts
> All text in this presentation uses **Calibri** (PowerPoint default). If you want to upgrade:
> - Titles → **Inter Bold** or **Outfit Bold**
> - Body → **Inter Regular** or **DM Sans**
> - Code-style text → **Fira Code** or **Consolas**

### Slide Dimensions
- **16:9 Widescreen** — 13.33 × 7.5 inches (33.87 × 19.05 cm)
- Do NOT change slide dimensions after saving — content will reflow.

---

## 📑 Slide-by-Slide Breakdown

---

### Slide 1 — Title / Cover
**Purpose:** Set the tone. Grab attention immediately.

**What's on it:**
- Large title: **HEIST ARCHITECT**
- Subtitle: *Adversarial Reinforcement Learning Framework*
- Tagline: *"Two AI agents play an infinite game of cops-and-robbers..."*
- Phase I & Phase II badges (Cyan & Amber)
- Timeline strip: Local Training → Kaggle Cloud → 17,000 Episodes
- Footer: Course code, tech stack, HuggingFace repo

**Speaker Script (30 sec):**
> *"Our project is called Heist Architect. Two neural networks compete in an infinite adversarial game — one builds a bank vault security system, and the other tries to break in. We did this in two phases: a local training run to prove the concept, then a full cloud scale-up on Kaggle GPUs."*

**Edits you might want to make:**
- Add your **team member names** in the footer strip (bottom gray bar)
- Add your **register numbers** next to the course code
- Change the tagline if you have a punchier one-liner

---

### Slide 2 — Agenda / Table of Contents
**Purpose:** Orient the audience. Show that you have depth.

**What's on it:**
- Left column (Cyan): Phase I topics (8 items)
- Right column (Amber): Phase II topics (8 items)

**Speaker Script (20 sec):**
> *"Here's our agenda. We have two major phases — Phase I covers the framework design and local training. Phase II covers what happened when we scaled to 17,000 episodes on cloud GPUs."*

**Edits you might want to make:**
- Reorder items if your personal presentation order differs
- Add slide numbers next to each item if your panel prefers it

---

### Slide 3 — Problem Statement
**Purpose:** Establish "why this matters" before showing the solution.

**What's on it:**
- A centred quote card explaining the core problem
- Three challenge cards: Static Environments / Single-Agent Limitation / No Adversarial Co-Learning

**Speaker Script (1.5 min):**
> *"Standard RL trains one agent on one fixed environment. The agent memorises solutions. But in the real world, an attacker and defender both adapt. Existing RL benchmarks like MiniGrid have no intelligent opponent — the walls don't fight back. We built a framework where the environment itself is an adversary."*

**Edits you might want to make:**
- Replace the three challenge cards with your own framing if needed
- The quote card text can be shortened — it's long

---

### Slide 4 — Adversarial RL Concept
**Purpose:** Explain the theoretical backbone without being too academic.

**What's on it:**
- MDP (standard RL) vs Markov Game (this project) side-by-side comparison
- VS badge in the centre
- Four key concept cards: Non-Stationarity / Co-Adaptation / Nash Equilibrium / ELO Rating

**Speaker Script (2 min):**
> *"In standard RL, an agent learns to solve a fixed environment. Our project is a Markov Game — both agents act, both influence state transitions, and both adapt. This creates non-stationarity: the 'correct answer' changes every time your opponent learns something new. The target of training isn't convergence to a fixed policy — it's convergence to a Nash Equilibrium."*

**Key term to emphasise:** **Non-Stationarity** — this is what makes the project scientifically interesting. Hammer it.

---

### Slide 5 — The Two Agents
**Purpose:** Clearly separate the two neural networks and their roles.

**What's on it:**
- Architect card (left, Cyan): constraints, controls, network info
- Solver card (right, Amber): observations, actions, network info

**Speaker Script (2 min):**
> *"The Architect is the security designer. It outputs a probability map over the entire 20×20 grid, placing walls, cameras, and guards within a budget. It has 407,000 parameters in an encoder-decoder CNN.*
>
> *The Solver is the infiltrator. It sees three input channels — the occupancy grid, the danger zone map, and its own position. Critically, it has an LSTM layer, which gives it memory. It needs to remember where a camera was pointing 5 steps ago to know when it will rotate away."*

**Edits you might want to make:**
- If you want to highlight specific parameter counts, those are already correct
- Budget numbers are correct: Wall=1, Camera=3, Guard=5

---

### Slide 6 — Grid World Environment
**Purpose:** Make the environment concrete and visual.

**What's on it:**
- ASCII art grid with a live snapshot of the environment
- Legend for every symbol (S, V, C, G, #, X, .)
- Key rules callout at bottom right

**Speaker Script (1 min):**
> *"The game happens on a 20×20 tile grid. S is where the Solver starts, top-left. V is the Vault, bottom-right. The X tiles are the danger zones — areas currently under surveillance. The Solver sees this danger map updating in real time as cameras rotate and guards move."*

**Edits you might want to make:**
- If you have a screenshot of the actual dashboard grid, **replace the ASCII art** with a real screenshot (Insert → Picture)
- Screenshot the dashboard at `localhost:5000` with the eye icon ON to show all vision cones

---

### Slide 7 — Security Components
**Purpose:** Deep-dive into what the Architect can place.

**What's on it:**
- Three vertical cards: Wall (Gray), Camera (Cyan), Guard (Amber)
- Each card shows cost, behavior, and strategic use

**Speaker Script (1.5 min):**
> *"Three asset types. Walls cost 1 point — static obstacles that also block camera vision. Cameras cost 3 points — they don't move, but their vision cone rotates. This is what forces the Solver to time its movements. Guards cost 5 points — they're mobile, and their vision cone follows their direction of movement. The Architect learns that combining a narrow wall corridor with a rotating camera and a guard is a near-perfect trap."*

---

### Slide 8 — Reward System
**Purpose:** Show the mathematical rules of the game.

**What's on it:**
- Left: Solver rewards (7 entries with colour-coded value tags)
- Right: Architect rewards (4 entries)
- Bottom right: "Architect's Sweet Spot" callout box

**Speaker Script (1.5 min):**
> *"This is a zero-sum game. What helps one agent hurts the other. The Solver gets +10 for reaching the Vault and -1 for being detected. But the key design decision is distance shaping — every step, the Solver gets +0.1 for moving closer. Without this, reaching the vault by random chance on a 20×20 grid takes forever to learn.*
>
> *The Architect's sweet spot is producing layouts where the Solver succeeds about 30% of the time — challenging but not impossible. That bonus is built into the reward function."*

---

### Slide 9 — Curriculum Learning
**Purpose:** Show the staged training approach.

**What's on it:**
- 5 vertical phase cards: Warmup / Phase I / Phase II / Phase III / Phase IV
- Each card shows episode range, budget, description, solve rate, detection rate

**Speaker Script (1.5 min):**
> *"You can't throw a beginner into the hardest level. We use curriculum learning — starting with an empty grid, then adding walls, then cameras, then guards, then full budget. This is critical because without warmup, the Solver gets detected immediately and never learns that 'walk toward the vault' is a good strategy. Curriculum isn't just convenience — it's a stability tool."*

**Key moment:** Point to the Phase III card with "0→50%" solve rate — this is the dramatic collapse and recovery narrative.

---

### Slide 10 — Neural Network Architecture
**Purpose:** Show the technical depth of your implementation.

**What's on it:**
- Solver network: 5-layer diagram (Input→CNN×3→LSTM→Policy/Value heads)
- Architect network: 5-layer diagram (Input→Encoder×3→Decoder×3→Placement Map)
- Footnotes explaining WHY LSTM / WHY temperature sampling

**Speaker Script (2 min):**
> *"Both networks use PPO — Proximal Policy Optimization — for stable learning. The Solver has a CNN backbone for spatial feature extraction, then an LSTM for temporal memory. Camera cycles are periodic — the LSTM remembers the period and predicts when the coast is clear.*
>
> *The Architect uses an encoder-decoder — like a U-Net. It compresses the grid into a latent representation, then upsamples back to a probability map of where to place each asset type. We sample from this map with temperature — high temperature gives chaotic, exploratory layouts; low temperature gives confident, optimised placements."*

---

### Slide 11 — PHASE I Results (Local Run)
**Purpose:** Show concrete numbers from Phase I training.

**What's on it:**
- 4 stat boxes: 59.5% Solve Rate / 40.5% Detection Rate / 0.335 Arch Reward / 7.813 Solver Reward
- 4 phase narrative cards (one per curriculum phase)

**Speaker Script (2 min):**
> *"Phase I ran 500 episodes locally in about 2 hours. Final solve rate was 59.5% against 40.5% detection. The most interesting moment was Phase III — when guards were introduced at Episode 200. The Solver's win rate instantly hit 0%. One hundred episodes of dying. Then the LSTM figured out guard timing and clawed back to 50-50 balance — exactly the Nash Equilibrium predicted by theory.*
>
> *Phase IV showed the framework is robust — even at maximum budget, the Solver maintained a 60% average, proving the neural architecture can handle expert-level challenge."*

---

### Slide 12 — PHASE II Introduction
**Purpose:** Transition from "proof of concept" to "full-scale science."

**What's on it:**
- Context paragraph explaining why Phase II
- 5 upgrade cards: Scale / Hardware / Curriculum / ELO / Cloud Storage
- Large insight callout: collapse-recovery cycling

**Speaker Script (1 min):**
> *"Phase I proved the concept. Phase II asks: what happens at 34× scale? We moved to Kaggle's dual Tesla T4 GPUs and trained for 17,000 episodes. The key insight from Phase II was that the training curve is not smooth — it's evidence of adversarial cycling — phases of collapse and recovery that are theoretically predicted but rarely observed empirically."*

---

### Slide 13 — Training Dynamics
**Purpose:** The single most impactful data slide — the 11-milestone table.

**What's on it:**
- 11-row table: Episode / Stage / Win Rate / ELO Diff / Interpretation
- Colour-coded by outcome (Mint=strong, Coral=collapse)

**Speaker Script (2.5 min):**
> *"This table is the story of the training run. The Solver starts strong — 0.96 win rate at episode 50. It reaches Master level by episode 300 with ELO +810. Then it collapses — at episode 2950, win rate is 0.00 and ELO is -1060. The Architect found vault fortification.*
>
> *But then the Solver recovers. By episode 3500 it's back to 1.00. It holds that for 6,000 episodes — all the way to episode 10,000. Then a second collapse, even deeper: ELO -1317 at episode 15,500. The deepest valley in the entire run. Then — full recovery to ELO +1093 at episode 17,000. This is adversarial cycling. This is the science."*

**This is your 🌟 star slide — spend the most time here.**

---

### Slide 14 — ELO Arms Race
**Purpose:** Explain what ELO tells us and what the cycles mean.

**What's on it:**
- ELO explanation paragraph
- Four stage narratives: Rookie / Overfit Rebound / Recovery Window / Final Arms Race
- ep17000 best checkpoint highlight bar

**Speaker Script (1.5 min):**
> *"ELO is borrowed from chess. It answers: relative to each other, who's winning? Positive ELO means the Solver dominates; negative means the Architect dominates. The Rookie stage shows the Solver winning easily. The Overfit Rebound shows the Architect discovering vault clustering — a devastating strategy the Solver hadn't seen. Recovery is the Solver finding a superior counter-strategy. ep17000 is the best observable checkpoint — win rate 1.00, ELO +1093."*

---

### Slide 15 — Web Dashboard
**Purpose:** Show the engineering work — the visualization layer.

**What's on it:**
- 4 panel cards: Grid Visualization / Training Controls / Live Metrics / Path Simulation

**Speaker Script (1.5 min):**
> *"We also built a real-time web dashboard using Flask and WebSocket. You can watch the Solver navigate the grid live, toggle the purple vision cones on and off, run interactive episodes with custom budgets and temperatures, and compare checkpoint behaviors side by side."*

**🔑 Pro tip:** If you can do a live demo, do it HERE. Open `localhost:5000`, run an interactive episode with Budget=15, and show the Solver waiting for guards to pass.

---

### Slide 16 — PHASE II Results
**Purpose:** The numbers from the cloud run.

**What's on it:**
- 4 stat boxes: Win-Rate 1.00 / ELO +1093 / Reward +2.3 / 17,000 Episodes
- Full 5-phase timeline summary (Phase A through E)

**Speaker Script (1.5 min):**
> *"At the end of 17,000 episodes, the best checkpoint — ep17000 — achieves a 1.00 win rate and ELO of +1093 against a Master-level Architect. Phase A dominated quickly. Phases B and D were catastrophic collapses. Phases C and E were recoveries that each exceeded the previous peak. The system never settled permanently — it cycled, which is exactly what adversarial theory predicts."*

---

### Slide 17 — Emergent Strategies & Nash Equilibrium
**Purpose:** The most intellectually impressive slide — show that nobody programmed these tactics.

**What's on it:**
- Left card: 6 Architect-discovered tactics
- Right card: 6 Solver-discovered tactics
- Nash Equilibrium explanation card at bottom

**Speaker Script (2 min):**
> *"Nobody told the Architect to create patrol synchronisation — timing guards so there's no safe window. Nobody told the Solver to wall-hug or use the WAIT action for camera timing windows. These emerged from thousands of iterations of competitive pressure. The Solver's LSTM remembered guard positions from 5 steps ago and used that to predict safe movement windows.*
>
> *Phase I converged at 30% solve / 40% detect — the Nash Equilibrium. Phase II showed deeper cycles but always returned to competitive balance. This is the same mathematical framework behind poker AI and military strategy models."*

---

### Slide 18 — References
**Purpose:** Academic credibility. 12 papers, all from 2015–2023.

**What's on it:**
- 12 reference cards, two columns of 6, each with number tag / authors / full citation

**Highlighted references to mention:**
- **[1] Schulman et al. 2017** — PPO, the algorithm both agents use
- **[2] Baker et al. 2020** — OpenAI hide-and-seek, the closest related work
- **[6] OpenAI 2021** — Asymmetric self-play, direct inspiration
- **[9] Littman 1994** — Markov Games, the theoretical backbone

**Speaker Script (30 sec):**
> *"Our framework builds on 12 key papers. PPO from Schulman 2017 is our core learning algorithm. Baker et al.'s hide-and-seek paper from OpenAI is the closest related work. Littman's Markov Games paper from 1994 is the theoretical foundation."*

---

### Slide 19 — Conclusion
**Purpose:** Land the key takeaways and close strong.

**What's on it:**
- Large "CONCLUSIONS" header
- 5 checkmark cards: each with a bold claim + supporting evidence

**Speaker Script (1 min):**
> *"To conclude: Phase I proved the concept — Nash Equilibrium in 500 episodes. Phase II deepened the science — adversarial cycling over 17,000 episodes. Emergent strategies appeared that we never programmed. The architecture scales — 550K + 407K parameters remain competitive at Master curriculum level. And game theory is empirically validated — Nash Equilibrium, Markov Games, ELO dynamics — all reproducible in our custom environment."*

---

## 🎮 Live Demo Checklist (If Presenting with Dashboard)

Run this before your presentation starts:

```bash
# Terminal 1 — Launch dashboard
conda activate cv_conda
python main.py visualize --port 5000

# OR for Phase II workspace
python main.py dashboard --host 127.0.0.1 --port 5000
```

**Demo sequence (3 minutes):**
1. Open `http://localhost:5000`
2. Set **Budget = 15** in Interactive Episode panel
3. Click **🎬 Run Demo** — let one episode play
4. Toggle the 👁️ **Eye Icon** — show the purple vision cones sweeping
5. Point at the **Gold Circle (Solver)** — notice it stopping and waiting for guards to pass
6. Say: *"It's only doing that because of its LSTM memory module"*
7. Open the **Live Metrics card** — show Solve Rate vs Detection Rate
8. Load **ep17000** from the checkpoint dropdown — click **Simulate Demo**

---

## ✏️ Common Manual Edits in PowerPoint

### Adding Team Member Names
1. Go to **Slide 1 (Title)** → Footer / bottom gray strip
2. Add: `Team: [Name 1] (RegNo)  |  [Name 2] (RegNo)  |  [Name 3] (RegNo)`
3. Font: Calibri 9pt, color `#556677`

### Adding Slide Numbers
1. Insert → Header & Footer → Slide Number → Apply to All

### Replacing ASCII Grid with Real Screenshot
1. Take a screenshot of the dashboard grid at `localhost:5000`
2. Slide 6: Delete the ASCII textbox
3. Insert → Picture → select your screenshot
4. Resize to fit the left card area (~5.5 × 5.5 inches)

### Changing a Stat Box Value
- Slide 11 and 16 have stat boxes
- Click the large number text → edit directly
- The coloured value text is a textbox, not a shape — just double-click

### Adding Images to Slides
- Best spots: Slide 6 (grid screenshot), Slide 15 (dashboard screenshot)
- Slides 11/16 (could add training curve charts from `reports/assets/`)
- Charts are available at: `reports/assets/training_winrate_milestones.png` and `training_elo_milestones.png`

---

## 📊 How to Insert the Training Charts (Slide 13 or 16)

The ELO and Win-Rate charts are already generated at:
```
reports/assets/training_elo_milestones.png
reports/assets/training_winrate_milestones.png
```

To add them to Slide 16:
1. Delete one of the timeline text rows to make space
2. Insert → Picture → select `training_winrate_milestones.png`
3. Resize to ~6 inches wide, place on the right half of the slide

---

## 🕐 Timing Guide

| Slide | Time | Notes |
|---|---|---|
| 1 — Title | 0:30 | Fast opener |
| 2 — Agenda | 0:20 | Just namecheck sections |
| 3 — Problem | 1:30 | Explain "why adversarial" |
| 4 — RL Concept | 2:00 | Hammer non-stationarity |
| 5 — Agents | 2:00 | LSTM memory is key point |
| 6 — Environment | 1:00 | If demo available, switch here |
| 7 — Security | 1:30 | Emphasise cost/strategy tradeoff |
| 8 — Rewards | 1:30 | Zero-sum + distance shaping |
| 9 — Curriculum | 1:30 | "Stability tool not just convenience" |
| 10 — Networks | 2:00 | LSTM why / temperature why |
| 11 — Phase I Results | 2:00 | ⭐ Guard collapse moment |
| 12 — Phase II Intro | 1:00 | Bridge between phases |
| 13 — Training Dynamics | 2:30 | ⭐⭐ THE KEY SLIDE |
| 14 — ELO | 1:30 | Explain cycles |
| 15 — Dashboard | 1:30 | Or live demo |
| 16 — Phase II Results | 1:30 | Numbers confirmation |
| 17 — Emergent | 2:00 | ⭐ Nobody programmed these |
| 18 — References | 0:30 | Quick namecheck |
| 19 — Conclusion | 1:00 | Land the takeaways |
| **Total** | **~27 min** | Adjust if time-limited |

---

## ❓ Anticipated Panel Questions & Answers

**Q: Why PPO and not DQN or A3C?**
> PPO is the most stable policy gradient algorithm for continuous and discrete action spaces. DQN is off-policy and struggles with multi-agent settings. A3C is less sample-efficient. PPO's clipping prevents catastrophic policy updates, which is critical in non-stationary adversarial environments.

**Q: Is your system truly at Nash Equilibrium?**
> In theory, exact Nash Equilibrium is difficult to reach in deep learning. What we observe are partial equilibria and competitive cycles — which is exactly what adversarial RL theory predicts. The ~30% solve rate in Phase I and the ELO cycling in Phase II are empirical evidence of this near-equilibrium behavior.

**Q: Why did you use an LSTM specifically?**
> Camera rotation is periodic — the Solver needs temporal memory to predict when a camera will rotate away. A feedforward CNN has no memory; it sees only the current frame. The LSTM maintains a hidden state that acts as a memory buffer across time-steps.

**Q: What is the baseline comparison?**
> Our baseline is a random Solver agent vs a random Architect. The trained system significantly outperforms this — the Solver achieves 60%+ solve rates against expert layouts that a random agent cannot penetrate. A future extension could compare against A* pathfinding as a handcrafted baseline.

**Q: How does the budget system prevent trivial solutions?**
> If the Architect walls off the Vault entirely, our BFS validator rejects the layout and penalises the Architect with -1.0 reward. This forces it to create solvable-but-hard layouts. The "challenging but fair" bonus (+0.2) for 20–60% solve rates further incentivises difficulty calibration.

**Q: Could this transfer to real security system design?**
> Conceptually yes — this is a simplified model of the adversarial security design problem. Real extensions would include 3D environments, multi-Solver agents, and interpretability tools to explain why the Architect places cameras in specific positions.

---

## 📁 File Reference

```
RL-Project-Heist-Architect-Adversarial-RL/
├── Heist_Architect_Presentation_CSE4019.pptx   ← YOUR PRESENTATION
├── generate_ppt.py                              ← Script to regenerate
├── reports/assets/
│   ├── training_winrate_milestones.png          ← Insert into Slide 13/16
│   └── training_elo_milestones.png              ← Insert into Slide 14
├── image.png                                    ← Dashboard preview (Slide 6)
├── Files/
│   ├── README.md                                ← Phase I architecture source
│   ├── RESULTS.md                               ← Phase I result numbers
│   └── DASHBOARD_GUIDE.md                       ← Dashboard panel descriptions
├── README_TRAINING.md                           ← Phase II milestone table source
├── README_THEORY.md                             ← Theory & terminology source
└── kaggle_training/README_RESULTS.md           ← ELO & Kaggle run details
```

---

*Generated for CSE4019 — Adversarial Reinforcement Learning Framework — 2025-2026*
