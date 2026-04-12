conda activate cv_conda

# Train (~500 episodes)
python main.py train

# Quick test
python main.py train --episodes 50 --grid-size 10

# Demo episode
python main.py demo

# Visualization dashboard (http://localhost:5000)
python main.py visualize