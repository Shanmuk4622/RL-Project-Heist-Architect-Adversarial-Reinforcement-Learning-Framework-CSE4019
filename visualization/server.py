"""
Visualization Server — Flask + WebSocket for live training visualization.
Streams environment state to the browser dashboard.

Features:
- Live training updates via WebSocket
- Interactive episode mode (custom budget, freeze agents, etc.)
- Game log API (persistent episode history)
- Demo playback
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
    project_dir = os.path.dirname(viz_dir)
    
    app = Flask(__name__, static_folder=viz_dir)
    app.config['SECRET_KEY'] = 'heist-architect-viz'
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    # Shared state — trainer persists across training + interactive sessions
    state = {
        "trainer": None,
        "training_thread": None,
        "latest_env_state": None,
        "latest_metrics": None,
        "is_training": False,
    }
    
    def _get_or_create_trainer(episodes=500, solver_attempts=20):
        """Get existing trainer or create a new one."""
        if state["trainer"] is not None:
            return state["trainer"]
        
        from heist_architect.environment import EnvironmentConfig
        from heist_architect.training import AdversarialTrainer
        
        config = EnvironmentConfig(
            grid_rows=grid_size,
            grid_cols=grid_size,
            start_pos=(1, 1),
            vault_pos=(grid_size - 2, grid_size - 2),
        )
        
        trainer = AdversarialTrainer(
            config=config,
            total_episodes=episodes,
            solver_episodes_per_layout=solver_attempts,
            save_dir=save_dir,
            log_dir=os.path.join(project_dir, "logs"),
        )
        
        # Auto-resume if checkpoints exist
        latest = trainer.find_latest_checkpoint()
        if latest:
            trainer.resume_from_checkpoint()
        
        state["trainer"] = trainer
        return trainer
    
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
    def api_status():
        return jsonify({
            "is_training": state["is_training"],
            "has_state": state["latest_env_state"] is not None,
            "has_trainer": state["trainer"] is not None,
            "global_episode": state["trainer"].global_episode if state["trainer"] else 0,
        })
    
    @app.route('/api/metrics')
    def api_metrics():
        log_path = os.path.join(project_dir, "logs", "training_metrics.json")
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                return jsonify(json.load(f))
        return jsonify({})
    
    @app.route('/api/game_log')
    def api_game_log():
        """Return the full game log."""
        log_path = os.path.join(project_dir, "logs", "game_log.json")
        if os.path.exists(log_path):
            with open(log_path, 'r') as f:
                return jsonify(json.load(f))
        if state["trainer"]:
            return jsonify(state["trainer"].get_game_log())
        return jsonify([])
    
    # -----------------------------------------------------------------------
    # WebSocket Events  
    # -----------------------------------------------------------------------
    
    @socketio.on('connect')
    def handle_connect():
        print('[Viz] Client connected')
        if state["latest_env_state"]:
            socketio.emit('env_state', state["latest_env_state"])
        
        # Send existing game log
        if state["trainer"]:
            socketio.emit('game_log_update', {
                'log': state["trainer"].get_game_log(),
                'global_episode': state["trainer"].global_episode,
            })
    
    @socketio.on('start_training')
    def handle_start_training(data):
        if state["is_training"]:
            socketio.emit('error', {'message': 'Training already in progress'})
            return
        
        episodes = data.get('episodes', 500)
        solver_attempts = data.get('solver_attempts', 20)
        resume = data.get('resume', True)  # Auto-resume by default
        
        trainer = _get_or_create_trainer(episodes, solver_attempts)
        trainer.total_episodes = episodes
        trainer.solver_episodes = solver_attempts
        state["is_training"] = True
        
        def training_callback(episode, metrics, env_state):
            state["latest_env_state"] = env_state
            state["latest_metrics"] = metrics
            socketio.emit('env_state', env_state)
            socketio.emit('training_update', {
                'episode': episode,
                'metrics': metrics,
            })
            # Send latest game log entry
            if trainer.game_log:
                socketio.emit('game_log_entry', trainer.game_log[-1].to_dict())
        
        def run_training():
            try:
                trainer.train(callback=training_callback, resume=resume)
            finally:
                state["is_training"] = False
                socketio.emit('training_complete', {
                    'global_episode': trainer.global_episode,
                    'game_log': trainer.get_game_log()[-10:],  # Last 10 entries
                })
        
        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()
        state["training_thread"] = thread
        
        socketio.emit('training_started', {
            'episodes': episodes,
            'resume_from': trainer.global_episode if resume else 0,
        })
    
    @socketio.on('run_interactive')
    def handle_interactive(data):
        """Run interactive episodes with custom parameters."""
        if state["is_training"]:
            socketio.emit('error', {'message': 'Training in progress — wait until it finishes'})
            return
        
        # Get or create trainer (will auto-resume)
        trainer = _get_or_create_trainer()
        
        num_episodes = data.get('num_episodes', 1)
        budget = data.get('budget', 15)
        freeze_architect = data.get('freeze_architect', False)
        freeze_solver = data.get('freeze_solver', False)
        temperature = data.get('temperature', 1.0)
        solver_attempts = data.get('solver_attempts', 20)
        allow_cameras = data.get('allow_cameras', True)
        allow_guards = data.get('allow_guards', True)
        
        state["is_training"] = True
        
        def interactive_callback(episode, metrics, env_state):
            state["latest_env_state"] = env_state
            socketio.emit('env_state', env_state)
            socketio.emit('training_update', {
                'episode': episode,
                'metrics': metrics,
            })
            if trainer.game_log:
                socketio.emit('game_log_entry', trainer.game_log[-1].to_dict())
        
        def run_interactive():
            try:
                socketio.emit('interactive_started', {
                    'num_episodes': num_episodes,
                    'budget': budget,
                    'freeze_architect': freeze_architect,
                    'freeze_solver': freeze_solver,
                })
                
                results = trainer.run_interactive_episodes(
                    num_episodes=num_episodes,
                    budget=budget,
                    freeze_architect=freeze_architect,
                    freeze_solver=freeze_solver,
                    temperature=temperature,
                    solver_attempts=solver_attempts,
                    allow_cameras=allow_cameras,
                    allow_guards=allow_guards,
                    callback=interactive_callback,
                )
                
                socketio.emit('interactive_complete', {
                    'num_episodes': num_episodes,
                    'results': results,
                    'global_episode': trainer.global_episode,
                })
            finally:
                state["is_training"] = False
        
        thread = threading.Thread(target=run_interactive, daemon=True)
        thread.start()
    
    @socketio.on('get_game_log')
    def handle_get_game_log(data=None):
        """Send the full game log to the client."""
        if state["trainer"]:
            socketio.emit('game_log_update', {
                'log': state["trainer"].get_game_log(),
                'global_episode': state["trainer"].global_episode,
            })
        else:
            # Try loading from file
            log_path = os.path.join(project_dir, "logs", "game_log.json")
            if os.path.exists(log_path):
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
                socketio.emit('game_log_update', {
                    'log': log_data,
                    'global_episode': len(log_data),
                })
            else:
                socketio.emit('game_log_update', {'log': [], 'global_episode': 0})
    
    @socketio.on('run_demo')
    def handle_demo(data=None):
        if state["is_training"]:
            socketio.emit('error', {'message': 'Training in progress'})
            return
            
        trainer = _get_or_create_trainer()
        
        socketio.emit('demo_started', {})
        
        # Generate and run (using default demo parameters)
        # We can now use simulate_episode for better demos too, but keep legacy demo_episode for now
        # OR better: use simulate_episode for demo too to get the "clean" behavior
        # Let's switch standard demo to use simulate_episode with default params
        demo = trainer.simulate_episode(budget=15, solver_attempts=1)
        
        # Stream frames
        for i, frame in enumerate(demo["frames"]):
            socketio.emit('demo_frame', {
                'frame': frame,
                'step': i,
                'total': len(demo["frames"]),
            })
            socketio.sleep(0.1)
        
        socketio.emit('demo_complete', {
            'outcome': demo["outcome"],
            'steps': demo["total_steps"],
        })

    @socketio.on('get_checkpoints')
    def handle_get_checkpoints(data=None):
        """Return list of available checkpoints."""
        trainer = _get_or_create_trainer()
        checkpoints = trainer.list_checkpoints()
        socketio.emit('checkpoints_list', {'checkpoints': checkpoints})

    @socketio.on('run_simulation')
    def handle_simulation(data):
        """Run a simulation with specific checkpoint and parameters."""
        if state["is_training"]:
            socketio.emit('error', {'message': 'Training in progress'})
            return
        
        episode = int(data.get('episode', 0))
        budget = int(data.get('budget', 15))
        solver_attempts = int(data.get('solver_attempts', 1))
        
        trainer = _get_or_create_trainer()
        state["is_training"] = True
        
        def run_sim():
            try:
                # Load checkpoint
                if not trainer.load_checkpoint(episode):
                    socketio.emit('error', {'message': f'Could not load checkpoint for episode {episode}'})
                    return
                
                socketio.emit('simulation_started', {
                    'episode': episode,
                    'budget': budget
                })
                
                # Run simulation
                result = trainer.simulate_episode(budget=budget, solver_attempts=solver_attempts)
                
                # Stream frames
                for i, frame in enumerate(result["frames"]):
                    socketio.emit('simulation_frame', {
                        'frame': frame,
                        'step': i,
                        'total': len(result["frames"]),
                    })
                    socketio.sleep(0.1) # Smooth playback
                
                socketio.emit('simulation_complete', {
                    'outcome': result["outcome"],
                    'steps': result["total_steps"],
                    'reward': result["reward"]
                })
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                socketio.emit('error', {'message': str(e)})
            finally:
                state["is_training"] = False
        
        thread = threading.Thread(target=run_sim, daemon=True)
        thread.start()
    
    return app, socketio


if __name__ == '__main__':
    app, socketio = create_app()
    print("Starting visualization server on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
