# antworld

A neural network simulation for ant world where ants with neural networks learn to navigate their environment.

## Overview

Ant World is a simulation environment where:
- **World**: A 2D space where ants exist and navigate
- **Ants**: Each ant has its own neural network "brain" that controls its behavior
- **Learning**: Ants can learn and improve through neural network mutation

## Features

- 2D world environment with configurable dimensions
- Ants with neural network brains for decision making
- Basic sensor inputs (position, distance to edges)
- Movement system with boundary collision
- Learning mechanism through neural network mutation
- Statistics tracking (distance traveled, steps taken)
- **Web-based UI** with configuration page and real-time visualization
- **Interactive controls** to start, stop, and reset simulation
- Real-time visualization showing ants navigating the world with colored trails

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

1. **World Creation**: A 2D grid world is created with specified dimensions
2. **Ant Creation**: Multiple ants are spawned at random positions, each with a random neural network
3. **Simulation Loop**: Each step, ants:
   - Sense their environment (position, distances to edges)
   - Process inputs through their neural network
   - Select and perform an action (move up/down/left/right)
4. **Learning**: Neural networks can be mutated to explore different behaviors
