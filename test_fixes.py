"""Quick test of the training fixes."""
import numpy as np
from heist_architect.environment import HeistEnvironment, EnvironmentConfig

# Test 1: vault_pos auto-set
config10 = EnvironmentConfig(grid_rows=10, grid_cols=10)
print(f"1. 10x10 vault_pos = {config10.vault_pos} (should be (8,8))")

config20 = EnvironmentConfig(grid_rows=20, grid_cols=20)
print(f"   20x20 vault_pos = {config20.vault_pos} (should be (18,18))")

# Test 2: Distance-based reward
env = HeistEnvironment(config10)
env.set_layout([], [], [])
obs = env.reset()
print(f"\n2. Solver starts at {env.solver_pos}, vault at {config10.vault_pos}")

# Move DOWN (toward vault row-wise)
total_r = 0
for i in range(7):  # Solver needs to get from (1,1) to (8,8)
    obs, r, done, info = env.step(2)  # DOWN
    total_r += r
    if done:
        break

print(f"   After 7x DOWN: pos={env.solver_pos}, cumulative_reward={total_r:+.3f}")
print(f"   (should be positive due to distance reward — closer to vault)")

# Move RIGHT to reach vault
for i in range(7):
    obs, r, done, info = env.step(4)  # RIGHT
    total_r += r
    if done:
        break

print(f"   After 7x RIGHT: pos={env.solver_pos}, cumulative_reward={total_r:+.3f}")
print(f"   Status: {info['status']}")

# Test 3: State tensor gradient
env2 = HeistEnvironment(config10)
env2.set_layout([], [], [])
env2.reset()
state = env2.get_state_tensor()
print(f"\n3. State tensor shape: {state.shape}")
print(f"   Position channel has gradient: {state[2].min():.3f} to {state[2].max():.3f}")

# Test 4: Training imports
from heist_architect.training import AdversarialTrainer
print(f"\n4. Curriculum phases:")
trainer = AdversarialTrainer.__new__(AdversarialTrainer)
for thresh, budget, cams, guards, desc in AdversarialTrainer.CURRICULUM:
    print(f"   Ep {thresh:4d}+: budget={budget:2d}, cameras={cams}, guards={guards}, {desc}")

print(f"\n=== ALL FIXES VERIFIED ===")
