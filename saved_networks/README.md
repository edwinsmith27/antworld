# Saved Networks

This directory contains saved neural networks from successful ants during evolution.

## How it works

- After each generation, the neural networks of surviving ants are automatically saved to JSON files
- Files are named with the pattern: `survivors_YYYYMMDD_HHMMSS.json`
- When starting a new simulation, the most recent saved networks can be loaded to continue evolution
- This allows for persistent learning across simulation runs

## File Format

Each JSON file contains:
- `timestamp`: When the networks were saved
- `generation`: The generation number
- `step`: The total step count
- `best_fitness`: The highest fitness score in this generation
- `count`: Number of saved networks
- `networks`: Array of network data including weights, biases, and fitness scores

## Usage

By default, the simulation will automatically:
1. Save successful networks after each generation
2. Load the most recent saved networks when starting a new simulation

To disable loading saved networks:
```python
from main import run_simulation
run_simulation(load_saved=False)
```
