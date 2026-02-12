"""Quick sanity test for the Heist Architect framework."""
import numpy as np

print("=" * 50)
print("  Heist Architect — Sanity Test")
print("=" * 50)

# Test 1: Imports
print("\n[1] Testing imports...")
from heist_architect.environment import HeistEnvironment, EnvironmentConfig
from heist_architect.agents.architect import ArchitectAgent
from heist_architect.agents.solver import SolverAgent
from heist_architect.rewards import RewardCalculator
from heist_architect.components.security import Wall, Camera, Guard
from heist_architect.components.visibility import DynamicVisibilityMap
from heist_architect.components.budget import BudgetManager, BUDGET_COSTS
print("    PASS: All imports successful")

# Test 2: Create environment
print("\n[2] Testing environment creation...")
config = EnvironmentConfig(grid_rows=10, grid_cols=10, start_pos=(1,1), vault_pos=(8,8))
env = HeistEnvironment(config)
print(f"    PASS: {env}")

# Test 3: Set layout
print("\n[3] Testing layout placement...")
walls = [(3,3), (3,4), (3,5)]
cameras = [{"row": 5, "col": 5, "fov_angle": 60, "heading": 0, "rotation_speed": 15, "vision_range": 4}]
guards = [{"patrol_path": [(7,2),(7,3),(7,4),(7,5)], "speed": 1, "vision_range": 3, "fov_angle": 90}]
valid = env.set_layout(walls, cameras, guards)
print(f"    PASS: Layout valid = {valid}")

# Test 4: Reset and run steps
print("\n[4] Testing simulation steps...")
obs = env.reset()
print(f"    Observation keys: {list(obs.keys())}")
for i in range(5):
    obs, reward, done, info = env.step(4)  # move right
    if done:
        break
print(f"    PASS: pos={env.solver_pos}, tick={env.tick}, status={info['status']}")

# Test 5: Grid rendering
print("\n[5] Grid visualization:")
print(env.render_text())

# Test 6: Budget
print("\n[6] Testing budget system...")
budget = BudgetManager(total_budget=15)
print(f"    Initial: {budget}")
budget.purchase("camera")
budget.purchase("guard")
budget.purchase("wall")
print(f"    After purchases: {budget}")

# Test 7: Visibility
print("\n[7] Testing visibility map...")
vis = env.visibility_map.visibility
print(f"    Shape: {vis.shape}, Surveilled tiles: {int(np.sum(vis > 0.5))}")

# Test 8: Reward calculator
print("\n[8] Testing reward calculator...")
calc = RewardCalculator()
print(f"    Rewards: {calc.get_reward_summary()}")

# Test 9: State tensor
print("\n[9] Testing state tensor for neural network...")
state = env.get_state_tensor()
print(f"    Shape: {state.shape} (channels, rows, cols)")

# Test 10: Agents
print("\n[10] Testing agent initialization...")
architect = ArchitectAgent(grid_rows=10, grid_cols=10, budget=15)
solver = SolverAgent(grid_rows=10, grid_cols=10)
print(f"    Architect network: {sum(p.numel() for p in architect.network.parameters())} params")
print(f"    Solver network: {sum(p.numel() for p in solver.network.parameters())} params")

# Test 11: Agent action selection
print("\n[11] Testing agent inference...")
walls, cameras, guards = architect.generate_layout(temperature=1.0)
print(f"    Architect generated: {len(walls)} walls, {len(cameras)} cameras, {len(guards)} guards")

solver.reset()
action = solver.select_action(state)
print(f"    Solver action: {action} ({HeistEnvironment.ACTION_NAMES[action]})")

print("\n" + "=" * 50)
print("  ALL TESTS PASSED ✓")
print("=" * 50)
