# Heist Architect v2 Theory Primer

## 1. The Big Question

Most reinforcement learning projects answer a single-agent question:

> How does one policy learn to behave well in one environment?

Heist Architect v2 asks a harder question:

> How do two intelligent policies learn against each other when one is trying to make the task harder and the other is trying to solve it?

That makes the project a study of adversarial learning, strategic adaptation, and instability under changing opponents.

## 2. The Core Worldview

This project is not mainly about pathfinding.

It is about strategic interaction in a security-like environment.

There are two roles:

- Architect: creates the challenge.
- Solver / Robber: tries to overcome the challenge.

The environment is therefore not static. It is produced by another learner.

That single fact changes almost everything about the theory.

## 3. What the Main Terms Mean

### 3.1 Agent

An agent is a decision-making system that observes a state and chooses an action.

In this project, both Architect and Solver are agents.

### 3.2 Policy

A policy is the rule or neural network that maps observation to action.

If the policy changes, the agent’s behavior changes.

# Heist Architect v2 Theory Primer

## 1. The Big Question

Most reinforcement learning projects answer a single-agent question:

> How does one policy learn to behave well in one environment?

Heist Architect v2 asks a harder question:

> How do two intelligent policies learn against each other when one is trying to make the task harder and the other is trying to solve it?

That makes the project a study of adversarial learning, strategic adaptation, and instability under changing opponents.

## 2. The Core Worldview

This project is not mainly about pathfinding.

It is about strategic interaction in a security-like environment.

There are two roles:

- Architect: creates the challenge.
- Solver / Robber: tries to overcome the challenge.

The environment is therefore not static. It is produced by another learner.

That single fact changes almost everything about the theory.

## 3. What the Main Terms Mean

### 3.1 Agent

An agent is a decision-making system that observes a state and chooses an action.

In this project, both Architect and Solver are agents.

### 3.2 Policy

A policy is the rule or neural network that maps observation to action.

If the policy changes, the agent’s behavior changes.

### 3.3 Environment

The environment is the system that responds to actions.

Here the environment includes layout structure, movement rules, visibility, and time-based interactions.

### 3.4 Observation

An observation is what the agent can see at a given moment.

It is not necessarily the full truth of the world. Many RL systems are partially observable, meaning the agent sees only part of the state.

### 3.5 State

The state is the underlying configuration of the world.

In a grid-world security setting, this may include walls, cameras, guards, positions, and time-step information.

### 3.6 Action

An action is a choice made by the policy.

Examples:
- Solver moves
- Solver waits
- Architect places security elements

### 3.7 Reward

Reward is the numerical feedback signal used for learning.

Positive reward encourages a behavior. Negative reward discourages it.

### 3.8 Return

Return is the accumulated reward over a trajectory.

In RL, the policy is trained to maximize expected return, not just immediate reward.

### 3.9 Episode

An episode is one complete run from start to finish.

In this project, an episode corresponds to one adversarial game instance and its outcome.

### 3.10 Checkpoint

A checkpoint is a saved model snapshot.

It freezes the policy at a moment in training so it can be reused, compared, or restored later.

## 4. Why This is Different from Ordinary RL

In ordinary single-agent RL:

- the environment is mostly fixed
- the policy learns a stable task
- performance usually improves against the same rules

In Heist Architect v2:

- the opponent also learns
- the task distribution moves
- improvements for one side change the learning problem for the other side

This means the problem is strategically non-stationary.

## 5. Why the Environment is a Moving Target

The Architect is not a passive map generator.

It actively changes the layout to make the Solver’s job harder.

The Solver is not just memorizing paths.

It must adapt to changing wall patterns, vision cones, patrol timing, and reachable routes.

Because both sides change, the effective training data distribution changes over time.

That is why standard “train once, converge once” intuition does not fit very well here.

## 6. The Markov Game View

A normal Markov Decision Process (MDP) assumes one decision-maker acting in an environment.

This project is better understood as a Markov game.

In simple terms:

- both sides act
- both sides influence the state transitions
- both sides receive feedback
- both sides adapt to the other side

This is the theoretical foundation behind adversarial reinforcement learning.

## 7. Non-Stationarity Explained Simply

Non-stationarity means the rules of the learning problem effectively change over time.

Example:

- If the Solver learns a good route, the Architect can create a layout that blocks it.
- If the Architect creates a strong trap, the Solver must find a new strategy.

So the “correct answer” is not stable.

This causes:

- sudden drops in performance
- oscillating strength between the two policies
- phase-like transitions in training curves

## 8. Why Curriculum Matters

Curriculum means the task difficulty is introduced gradually.

This is important because if both policies start too hard:

- the Solver may fail before learning basics
- the Architect may learn noisy or uninformative layouts
- the training signal can become too unstable to use

Curriculum lets the system learn in stages:

1. basic movement and layout reasoning
2. stronger obstacle interactions
3. full strategic competition

So curriculum is not just convenience. It is a stability tool.

## 9. What ELO Means Here

ELO is a relative skill signal.

It is useful because absolute reward alone can be misleading in a changing opponent environment.

Interpretation:

- higher Solver ELO means the Solver is currently doing better relative to the Architect
- lower Solver ELO means the Architect is currently winning the strategic exchange

Important limitation:

ELO here is not a universal truth about intelligence. It is a comparative score inside one competitive system.

## 10. Why Collapse Happens

Collapse is the name for a long period where performance sharply worsens.

In adversarial RL, collapse can happen for several theoretical reasons:

- overfitting to a narrow opponent behavior
- reward landscape shifts after the opponent improves
- policy becomes too specialized
- one side finds a temporary exploit and the other side has not adapted yet

Collapse does not automatically mean the system is broken.

Sometimes it is a normal consequence of the game changing underneath the learner.

## 11. Why Recovery Happens

Recovery happens when the learner rediscover a policy that works under the new opponent distribution.

This can happen because:

- the saved checkpoint is stronger than the current drifted policy
- the opponent distribution changes again
- the policy learns a more transferable strategy
- a previously forgotten behavior becomes useful again

Your logs show recovery after deep collapse, which is a classic sign of adversarial cycling rather than simple monotonic training.

## 12. Why Checkpoints Matter Theoretically

Checkpoints are not just backups.

They are evidence of different strategic regimes.

That is why one should keep checkpoints from:

- early dominance
- collapse
- recovery
- final strong performance

If you only keep the best checkpoint, you lose the ability to study how the system behaved across regimes.

## 13. Why Visual Replay Matters

The dashboard is a conceptual instrument, not only a UI.

It helps answer questions like:

- Does the Solver hesitate at the right moments?
- Does the Architect create genuine structural traps?
- Are cameras and guards producing meaningful pressure?
- Is the layout complexity strategic or just decorative?

This matters because scalar metrics can hide the real reason a policy is succeeding or failing.

## 14. Strategy vs Memorization

One of the most important theoretical distinctions in this project is:

- memorization: the policy performs well because it knows a specific pattern
- strategy: the policy performs well because it understands a general principle

Adversarial RL pushes toward strategy because the opponent changes patterns over time.

This is why generalization is central to the project.

## 15. Exploration and Exploitation

Exploration means trying new actions or structures.
Exploitation means using what already works.

In this project, both sides need both:

- the Solver must sometimes try uncertain routes or wait timings
- the Architect must sometimes try less obvious layouts

Too little exploration leads to brittle behavior.
Too much exploration prevents stable learning.

The aim is controlled exploration.

## 16. Equilibrium: Important but Not Perfectly Realized

In theory, adversarial systems often aim toward an equilibrium-like state where neither side can improve much by changing alone.

In deep learning practice, exact equilibrium is hard to reach.

What usually appears instead:

- cycles
- temporary dominance
- partial equilibria
- regime shifts

So training should be interpreted as a process of moving through competitive regimes, not as a single clean convergence event.

## 17. How to Read the Training History

The training history should be interpreted as evidence of the system’s strategic phases.

Visible patterns from your log:

- early fast rise in solver strength
- a first large collapse
- a recovery into strong positive performance
- a second deep collapse
- a final recovery to a strong endpoint

This kind of curve is exactly what adversarial co-adaptation often looks like.

## 18. What ep17000 Means in Theory Terms

ep17000 is not magic.

It is the strongest visible checkpoint in the final recovery regime.

Theoretically, that means:

- the Solver is performing strongly against the current Architect distribution
- the system has recovered from previous instability
- the checkpoint is useful as a default evaluation model

But it should still be tested against collapse-era checkpoints if you want a complete scientific picture.

## 19. Common Terms in Plain Language

### Policy
The decision-making brain.

### Reward
The score signal that tells the policy what is good or bad.

### State
The current condition of the world.

### Observation
What the agent can perceive about that condition.

### Episode
One full game or run.

### Checkpoint
A saved snapshot of the policy.

### Non-stationarity
The rules are effectively changing because the opponent is changing.

### Curriculum
A planned difficulty progression.

### ELO
A relative strength estimate.

### Collapse
A major drop in performance.

### Recovery
A return to strong performance after collapse.

## 20. Final Theoretical Takeaway

The project is best understood as a competitive learning system where intelligence emerges from pressure, adaptation, and instability.

The most important idea is not the grid itself.

The most important idea is that the environment is adversarially shaped by another learner.

That is what makes the project theoretically interesting.

And that is why the training history contains alternating phases of rise, collapse, and recovery.

In this project, both sides need both:

- the Solver must sometimes try uncertain routes or wait timings
- the Architect must sometimes try less obvious layouts

Too little exploration leads to brittle behavior.
Too much exploration prevents stable learning.

The aim is controlled exploration.

## 16. Equilibrium: Important but Not Perfectly Realized

In theory, adversarial systems often aim toward an equilibrium-like state where neither side can improve much by changing alone.

In deep learning practice, exact equilibrium is hard to reach.

What usually appears instead:

- cycles
- temporary dominance
- partial equilibria
- regime shifts

So training should be interpreted as a process of moving through competitive regimes, not as a single clean convergence event.

## 17. How to Read the Training History

The training history should be interpreted as evidence of the system’s strategic phases.

Visible patterns from your log:

- early fast rise in solver strength
- a first large collapse
- a recovery into strong positive performance
- a second deep collapse
- a final recovery to a strong endpoint

This kind of curve is exactly what adversarial co-adaptation often looks like.

## 18. What ep17000 Means in Theory Terms

ep17000 is not magic.

It is the strongest visible checkpoint in the final recovery regime.

Theoretically, that means:

- the Solver is performing strongly against the current Architect distribution
- the system has recovered from previous instability
- the checkpoint is useful as a default evaluation model

But it should still be tested against collapse-era checkpoints if you want a complete scientific picture.

## 19. Common Terms in Plain Language

### Policy
The decision-making brain.

### Reward
The score signal that tells the policy what is good or bad.

### State
The current condition of the world.

### Observation
What the agent can perceive about that condition.

### Episode
One full game or run.

### Checkpoint
A saved snapshot of the policy.

### Non-stationarity
The rules are effectively changing because the opponent is changing.

### Curriculum
A planned difficulty progression.

### ELO
A relative strength estimate.

### Collapse
A major drop in performance.

### Recovery
A return to strong performance after collapse.

## 20. Final Theoretical Takeaway

The project is best understood as a competitive learning system where intelligence emerges from pressure, adaptation, and instability.

The most important idea is not the grid itself.

The most important idea is that the environment is adversarially shaped by another learner.

That is what makes the project theoretically interesting.

And that is why the training history contains alternating phases of rise, collapse, and recovery.
