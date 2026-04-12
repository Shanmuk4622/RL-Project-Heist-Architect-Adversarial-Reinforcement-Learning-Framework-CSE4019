# 🎬 Presentation Guide: Demonstrating the Framework

If you need to present or shoot a video showcasing the Heist Architect framework, follow this structural script to capture the audience's attention and easily explain the complex reinforcement learning happening under the hood.

## The Core Concept (1 Minute)
Start by setting the stage.
* **The Pitch:** *"Welcome to Heist Architect. This is a custom zero-sum game between two AI brains that I built. One builds a bank vault security system (The Architect), and the other tries to break into it (The Solver)."*
* **The Rules:** If the Architect builds an impossible map with no way in, it is penalized. If it makes an easy map, the Solver wins and the Architect is penalized. Because of this dynamic, it is forced to create fair but extremely difficult mazes.

## Displaying the Environment (2 Minutes)
Open the Dashboard at `localhost:5000` and break down the visual grid.
* **Colors:** Explain standard assets. The Pink tile is the vault. The Slate tiles are walls.
* **The Threats:** Point out the Purple Cameras and Orange Guards. 
* **The Eye Icon (👁️):** Click the visibility toggle in the dashboard. Show the audience the massive sweeping purple vision cones that the Solver has to dodge.

## Running the Live Demonstration (3 Minutes)
This is where you showcase the `Live Broadcast Mode` I implemented.
1. Run an **Interactive Episode** with `Budget = 15`. 
2. Point out how the **Solver (Gold Circle)** is physically stopping and waiting behind walls so that the patrol guards pass by before it makes a run for a chokepoint. 
3. Mention that it only knows how to time this because of its **LSTM Memory module**—it can remember where the guards were a second ago!

## Reviewing the Charts and Metrics (1-2 Minutes)
Open the `RESULTS.md` logic or point to the **Live Metrics Chart** on the dashboard UI.
* **The A-HA Moment:** Show the audience how the Solver's win rate tanked the minute Guards were introduced at Episode 200, but because of Reinforcement Learning, the AI slowly evolved and fought its way back up to a 60% win-rate against impossible odds.
* **Nash Equilibrium:** State that the AIs reached a "Nash Equilibrium"—an optimal baseline where neither AI completely dominates the other, proving the curriculum was balanced.
