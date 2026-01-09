# antworld

A neural network simulation for ant world where ants with neural networks learn to navigate their environment through evolution.

## Overview

Ant World is an evolutionary simulation environment where:
- **World**: A 2D space (up to 500x500) where ants exist and navigate
- **Ants**: Each ant has its own neural network "brain" that controls its behavior
- **Evolution**: Ants compete for survival based on fitness (distance traveled)
- **Natural Selection**: Unsuccessful ants die out, successful ants reproduce with mutations
- **Continuous Learning**: The simulation runs indefinitely until manually stopped

## Features

- Large 2D world environment (up to 500x500)
- Ants with neural network brains for decision making
- Basic sensor inputs (position, distance to edges)
- Movement system with boundary collision
- **Evolutionary algorithm** with fitness-based selection
- **Generational learning** through neural network mutation
- **Continuous simulation** that runs until user interrupts
- **Death mechanism** for low-performing ants
- **Reproduction system** where successful ants pass on their neural networks
- Detailed statistics tracking per generation (best/average/worst fitness, survivors)

## Installation

```

Requirements:
- numpy>=1.21.0
- matplotlib>=3.3.0
- flask>=2.0.0
- flask-cors>=3.0.0bash
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
- Real-time visualization with animated ants
- Live statistics and ant tracking
- Interactive controls (start, stop, reset)

Run the simulation with matplotlib visualization:

```bash
python main_visual.py
```

Run the simulation (console only):

```bash
python main.py
```

Run tests:

```bweb_app.py**: Flask web application with REST API
- **templates/**: HTML templates for web UI
  - **config.html**: Configuration page
  - **simulation.html**: Real-time visualization page
- **main.py**: Main simulation runner (console output)
- **main_visual.py**: Visual simulation runner with matplotlib
- **visualizer.py**: Visualization module for matplotlib

## Architecture

- **world.py**: World class representing the environment
- **ant.py**: Ant class with neural network for navigation
- **main.py**: Main simulation runner (console output)
- **main_visual.py**: Visual simulation runner with matplotlib
- **visualizer.py**: Visualization module for real-time display
- **test_antworld.py**: Unit tests for all components

## How It Works

1. **World Creation**: A large 2D grid world (default 500x500) is created
2. **Initial Population**: 20 ants are spawned at random positions, each with a random neural network
3. **Generation Loop**: Each generation runs for 200 steps where ants:
   - Sense their environment (position, distances to edges)
   - Process inputs through their neural network
   - Select and perform an action (move up/down/left/right)
   - Track fitness based on distance traveled
4. **Natural Selection**: At the end of each generation:
   - Ants are evaluated based on their fitness (distance traveled)
   - Low-performing ants die out (below fitness threshold or bottom quartile)
   - Successful ants survive to the next generation
5. **Reproduction**: The population is replenished by:
   - Keeping survivors with their learned behaviors
   - Creating offspring from survivors with mutated neural networks
6. **Evolution**: Over many generations, the population evolves more successful navigation strategies

The simulation runs continuously, allowing you to observe the evolutionary process in real-time until you stop it with Ctrl+C.
