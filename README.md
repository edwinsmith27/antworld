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

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the simulation:

```bash
python main.py
```

Run tests:

```bash
python -m unittest test_antworld.py
```

## Architecture

- **world.py**: World class representing the environment
- **ant.py**: Ant class with neural network for navigation
- **main.py**: Main simulation runner
- **test_antworld.py**: Unit tests for all components

## How It Works

1. **World Creation**: A 2D grid world is created with specified dimensions
2. **Ant Creation**: Multiple ants are spawned at random positions, each with a random neural network
3. **Simulation Loop**: Each step, ants:
   - Sense their environment (position, distances to edges)
   - Process inputs through their neural network
   - Select and perform an action (move up/down/left/right)
4. **Learning**: Neural networks can be mutated to explore different behaviors
