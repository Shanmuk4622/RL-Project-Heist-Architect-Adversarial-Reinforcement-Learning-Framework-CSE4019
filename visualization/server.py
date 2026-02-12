"""
Visualization Server — Flask + WebSocket for live training visualization.
Streams environment state to the browser dashboard.
"""

import os
import sys
import json
import threading
from flask import Flask, send_from_directory, jsonify
from flask_socketio import SocketIO

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_app(save_dir="checkpoints", grid_size=20):
    """Create the Flask app with SocketIO."""
    
    viz_dir = os.path.dirname(os.path.abspath(__file__))
    
    app = Flask(__name__, static_folder=viz_dir)
    app.config['SECRET_KEY'] = 'heist-architect-viz'
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    # Shared state
    state = {
        "trainer": None,
        "training_thread": None,
        "latest_env_state": None,
        "latest_metrics": None,
        "is_training": False,
    }
    
    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------
    
    @app.route('/')
    def index():
        return send_from_directory(viz_dir, 'index.html')
    
    @app.route('/style.css')
    def css():
        return send_from_directory(viz_dir, 'style.css')
    
    @app.route('/app.js')
    def js():
        return send_from_directory(viz_dir, 'app.js')
    
    @app.route('/api/status')
    def status():
        return jsonify({
            "is_training": state["is_training"],
            "has_state": state["latest_env_state"] is not None,
        })
    
    @app.route('/api/metrics')
    def get_metrics():
        log_path = os.path.join(os.path.dirname(viz_dir), "logs", "training_metrics.json")
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                return jsonify(json.load(f))
        return jsonify({})
    
    # -----------------------------------------------------------------------
    # WebSocket Events  
    # -----------------------------------------------------------------------
    
    @socketio.on('connect')
    def handle_connect():
        print('[Viz] Client connected')
        if state["latest_env_state"]:
            socketio.emit('env_state', state["latest_env_state"])
    
    @socketio.on('start_training')
    def handle_start_training(data):
        if state["is_training"]:
            socketio.emit('error', {'message': 'Training already in progress'})
            return
        
        episodes = data.get('episodes', 500)
        solver_attempts = data.get('solver_attempts', 20)
        
        from heist_architect.environment import EnvironmentConfig
        from heist_architect.training import AdversarialTrainer
        
        config = EnvironmentConfig(
            grid_rows=grid_size,
            grid_cols=grid_size
        )
        
        trainer = AdversarialTrainer(
            config=config,
            total_episodes=episodes,
            solver_episodes_per_layout=solver_attempts,
            save_dir=save_dir,
        )
        state["trainer"] = trainer
        state["is_training"] = True
        
        def training_callback(episode, metrics, env_state):
            state["latest_env_state"] = env_state
            state["latest_metrics"] = metrics
            socketio.emit('env_state', env_state)
            socketio.emit('training_update', {
                'episode': episode,
                'metrics': metrics,
            })
        
        def run_training():
            try:
                trainer.train(callback=training_callback)
            finally:
                state["is_training"] = False
                socketio.emit('training_complete', {})
        
        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()
        state["training_thread"] = thread
        
        socketio.emit('training_started', {'episodes': episodes})
    
    @socketio.on('run_demo')
    def handle_demo(data=None):
        from heist_architect.environment import HeistEnvironment, EnvironmentConfig
        from heist_architect.agents.architect import ArchitectAgent
        from heist_architect.agents.solver import SolverAgent
        
        config = EnvironmentConfig(grid_rows=grid_size, grid_cols=grid_size)
        env = HeistEnvironment(config)
        architect = ArchitectAgent(grid_rows=grid_size, grid_cols=grid_size, budget=15)
        solver = SolverAgent(grid_rows=grid_size, grid_cols=grid_size)
        
        # Try to load trained models
        arch_path = os.path.join(save_dir, "architect_ep500.pt")
        solver_path = os.path.join(save_dir, "solver_ep500.pt")
        if os.path.exists(arch_path):
            architect.load(arch_path)
        if os.path.exists(solver_path):
            solver.load(solver_path)
        
        # Generate and run
        walls, cameras, guards = architect.generate_layout(temperature=0.5)
        env.set_layout(walls, cameras, guards)
        obs = env.reset()
        solver.reset()
        state_tensor = env.get_state_tensor()
        
        frames = []
        for step in range(config.max_steps):
            frames.append(env.get_environment_state())
            action = solver.select_action(state_tensor)
            obs, reward, done, info = env.step(action)
            state_tensor = env.get_state_tensor()
            if done:
                frames.append(env.get_environment_state())
                break
        
        # Stream frames
        for i, frame in enumerate(frames):
            socketio.emit('demo_frame', {
                'frame': frame,
                'step': i,
                'total': len(frames),
            })
            socketio.sleep(0.1)
        
        socketio.emit('demo_complete', {
            'outcome': info.get('status', 'unknown'),
            'steps': len(frames),
        })
    
    return app, socketio


if __name__ == '__main__':
    app, socketio = create_app()
    print("Starting visualization server on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
