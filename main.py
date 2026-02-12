"""
Heist Architect — Main Entry Point
====================================
CLI interface for training, demo, and visualization.

Usage:
    python main.py train [--episodes N] [--grid-size N]
    python main.py demo
    python main.py visualize
"""

import argparse
import yaml
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def cmd_train(args):
    """Run adversarial training."""
    from heist_architect.environment import EnvironmentConfig
    from heist_architect.training import AdversarialTrainer
    
    config = EnvironmentConfig(
        grid_rows=args.grid_size,
        grid_cols=args.grid_size,
        max_steps=args.max_steps,
        start_pos=(1, 1),
        vault_pos=(args.grid_size - 2, args.grid_size - 2),
        architect_budget=8,  # starts at curriculum minimum
    )
    
    trainer = AdversarialTrainer(
        config=config,
        total_episodes=args.episodes,
        solver_episodes_per_layout=args.solver_attempts,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
    )
    
    trainer.train()


def cmd_demo(args):
    """Run a demo episode and print results."""
    from heist_architect.environment import HeistEnvironment, EnvironmentConfig
    from heist_architect.agents.architect import ArchitectAgent
    from heist_architect.agents.solver import SolverAgent
    
    config = EnvironmentConfig(
        grid_rows=args.grid_size,
        grid_cols=args.grid_size,
        start_pos=(1, 1),
        vault_pos=(args.grid_size - 2, args.grid_size - 2),
    )
    
    env = HeistEnvironment(config)
    architect = ArchitectAgent(
        grid_rows=config.grid_rows,
        grid_cols=config.grid_cols,
        budget=15
    )
    solver = SolverAgent(
        grid_rows=config.grid_rows,
        grid_cols=config.grid_cols,
    )
    
    # Load models if available
    arch_path = os.path.join(args.save_dir, f"architect_ep{args.checkpoint}.pt")
    solver_path = os.path.join(args.save_dir, f"solver_ep{args.checkpoint}.pt")
    
    if os.path.exists(arch_path):
        architect.load(arch_path)
        print(f"Loaded Architect from {arch_path}")
    if os.path.exists(solver_path):
        solver.load(solver_path)
        print(f"Loaded Solver from {solver_path}")
    
    # Generate layout
    walls, cameras, guards = architect.generate_layout(temperature=0.5)
    is_valid = env.set_layout(walls, cameras, guards)
    
    print(f"\n{'='*40}")
    print(f"  Demo Episode")
    print(f"  Level valid: {is_valid}")
    print(f"  Walls: {len(walls)}, Cameras: {len(cameras)}, Guards: {len(guards)}")
    print(f"{'='*40}\n")
    
    # Run solver
    obs = env.reset()
    solver.reset()
    state = env.get_state_tensor()
    
    print("Initial grid:")
    print(env.render_text())
    print()
    
    for step in range(config.max_steps):
        action = solver.select_action(state)
        obs, reward, done, info = env.step(action)
        state = env.get_state_tensor()
        
        if step % 20 == 0 or done:
            action_name = env.ACTION_NAMES.get(action, "?")
            print(f"Step {step:3d}: {action_name:5s} -> pos={env.solver_pos} "
                  f"reward={reward:+.2f} status={info['status']}")
        
        if done:
            break
    
    print(f"\nFinal grid:")
    print(env.render_text())
    print(f"\nOutcome: {info['status']}")
    print(f"Total steps: {env.tick}")


def cmd_visualize(args):
    """Start the visualization server."""
    from visualization.server import create_app
    
    app, socketio = create_app(
        save_dir=args.save_dir,
        grid_size=args.grid_size,
    )
    
    print(f"\n{'='*40}")
    print(f"  Heist Architect Visualization")
    print(f"  Open: http://localhost:{args.port}")
    print(f"{'='*40}\n")
    
    socketio.run(app, host='0.0.0.0', port=args.port, debug=False)


def main():
    parser = argparse.ArgumentParser(
        description="Heist Architect: Adversarial RL Framework"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Run adversarial training")
    train_parser.add_argument("--episodes", type=int, default=500,
                              help="Total training episodes")
    train_parser.add_argument("--grid-size", type=int, default=20,
                              help="Grid dimensions (NxN)")
    train_parser.add_argument("--max-steps", type=int, default=200,
                              help="Max steps per episode")
    train_parser.add_argument("--solver-attempts", type=int, default=20,
                              help="Solver attempts per layout")
    train_parser.add_argument("--save-dir", default="checkpoints",
                              help="Model save directory")
    train_parser.add_argument("--log-dir", default="logs",
                              help="Log directory")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run a demo episode")
    demo_parser.add_argument("--grid-size", type=int, default=20)
    demo_parser.add_argument("--save-dir", default="checkpoints")
    demo_parser.add_argument("--checkpoint", type=int, default=500,
                             help="Checkpoint episode to load")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Start visualization server")
    viz_parser.add_argument("--port", type=int, default=5000)
    viz_parser.add_argument("--grid-size", type=int, default=20)
    viz_parser.add_argument("--save-dir", default="checkpoints")
    
    args = parser.parse_args()
    
    if args.command == "train":
        cmd_train(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "visualize":
        cmd_visualize(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
