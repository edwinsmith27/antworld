# antworld

A neural network simulation for ant world where ants with neural networks learn to navigate their environment through evolution.

## Overview

Ant World is an evolutionary simulation environment where:
- **World**: A 2D space (default 500x500) where ants exist and navigate
- **Ants**: Each ant has its own neural network "brain" that controls its behavior
- **Food System**: Ants must find food to survive; health depletes over time
- **Evolution**: Ants compete for survival based on fitness (distance traveled and food collected)
- **Natural Selection**: Unsuccessful ants die out, successful ants reproduce with mutations
- **Continuous Learning**: The simulation runs indefinitely until manually stopped
- **Neural Network Persistence**: Successful ant brains are saved and loaded across simulation runs

## Features

- **Large 2D world environment** (default 500x500)
- **Ants with neural network brains** for autonomous decision making
- **Food system**:
  - Food items spawn randomly and at simulation start
  - Ants must collect food to maintain health
  - Health depletes over time, requiring constant foraging
- **Advanced sensor inputs**:
  - Normalized position (x, y)
  - Distance to world edges
  - **Distance to nearest food** (enables food-seeking behavior)
- **Movement system** with boundary collision detection
- **Evolutionary algorithm** with fitness-based selection
- **Generational learning** through neural network mutation
- **Neural network persistence**:
  - Successful networks automatically saved after each generation
  - Networks loaded on next simulation start for continuous evolution
- **Continuous simulation** that runs until user interrupts
- **Death and respawn mechanism** for low-performing ants
- **Reproduction system** where successful ants pass on their neural networks
- **Memory leak protection** with proper cleanup of dead ant components
- **Detailed statistics tracking** per generation (best/average/worst fitness, survivors)
- **Visualization options**:
  - Real-time visualization with food display
  - Configurable animation speed (0ms = maximum speed)
  - Displays ant health, food collected, and positions

## Installation

Requirements:
- numpy>=1.21.0
- matplotlib>=3.3.0
- flask>=2.0.0
- flask-cors>=3.0.0

```bash
pip install -r requirements.txt
```

## Usage

**Run the web-based simulation (recommended):**

```bash
python web_app.py
```

Then open your browser to http://localhost:5000

The web UI provides:
- Configuration page to set simulation parameters
- Real-time visualization with animated ants and food
- Live statistics and ant tracking
- Interactive controls (start, stop, reset)

**Run the simulation with matplotlib visualization:**

```bash
python main_visual.py
```

This provides:
- Animated visualization (default: 500x500 world, 0ms delay for maximum speed)
- Food displayed as green circles
- Ant trails showing movement history
- Live statistics panel

**Run the simulation (console only, no visualization):**

```bash
python main.py
```

This runs the simulation without visualization for maximum performance.

**Run tests:**

```bash
python -m unittest test_antworld.py
```

## Neural Network Architecture

The ant's brain is a simple feedforward neural network:
- **Input layer**: 5 neurons
  1. Normalized X position
  2. Normalized Y position
  3. Distance to right edge
  4. Distance to bottom edge
  5. **Distance to nearest food** (new!)
- **Hidden layer**: 8 neurons (tanh activation)
- **Output layer**: 4 neurons (softmax) representing actions: up, right, down, left

## Saved Networks

Successful ant neural networks are automatically saved after each generation to the `saved_networks/` directory. These networks are loaded on the next simulation start, allowing continuous evolution across runs.

To disable loading saved networks:
```python
from main import run_simulation
run_simulation(load_saved=False)
```

See `saved_networks/README.md` for more details.

## Project Structure

- **ant.py**: Ant class with neural network and network persistence functions
- **world.py**: World class with food system and collision detection
- **main.py**: Evolutionary simulation runner (console)
- **main_visual.py**: Visual simulation runner with matplotlib
- **visualizer.py**: Visualization module with food display
- **web_app.py**: Flask web application with REST API
- **templates/**: HTML templates for web UI
- **test_antworld.py**: Comprehensive unit tests
- **saved_networks/**: Directory for persistent neural networks

## How It Works

1. **World Creation**: A large 2D grid world (default 500x500) is created with initial food placement (20 items)
2. **Initial Population**: 20 ants are spawned at random positions. If saved networks exist, they are loaded; otherwise, random neural networks are created
3. **Survival Mechanics**: Each ant starts with full health that depletes over time. Ants must find and consume food to restore health
4. **Generation Loop**: Each generation runs for 200 steps where ants:
   - Sense their environment (position, distances to edges, distance to nearest food)
   - Process inputs through their neural network
   - Select and perform an action (move up/down/left/right)
   - Lose health each step and collect food to survive
   - Track fitness based on distance traveled and food collected
5. **Natural Selection**: At the end of each generation:
   - Ants are evaluated based on their fitness (distance traveled)
   - Low-performing ants die out (below fitness threshold or bottom quartile)
   - Successful ants' neural networks are saved to disk
   - Survivors proceed to the next generation
6. **Reproduction**: The population is replenished by:
   - Keeping survivors with their learned behaviors
   - Creating offspring from survivors with mutated neural networks
7. **Evolution**: Over many generations, the population evolves better food-seeking and navigation strategies

The simulation runs continuously, allowing you to observe the evolutionary process in real-time until you stop it with Ctrl+C. Neural networks are persistently saved, so evolution continues across simulation runs.
