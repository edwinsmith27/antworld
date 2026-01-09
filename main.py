"""Main simulation runner for ant world."""
from world import World
from ant import Ant
import numpy as np


def run_simulation(num_ants=5, num_steps=100, world_width=50, world_height=50):
    """
    Run the ant world simulation.
    
    Args:
        num_ants: Number of ants to create
        num_steps: Number of simulation steps
        world_width: Width of the world
        world_height: Height of the world
    """
    print("=== Ant World Simulation ===")
    print(f"World size: {world_width}x{world_height}")
    print(f"Number of ants: {num_ants}")
    print(f"Simulation steps: {num_steps}")
    print()
    
    # Create world
    world = World(width=world_width, height=world_height)
    
    # Create and add ants at random positions
    for i in range(num_ants):
        x = np.random.randint(0, world_width)
        y = np.random.randint(0, world_height)
        ant = Ant(x=x, y=y)
        world.add_ant(ant)
        print(f"Ant {i+1} created at position ({x}, {y})")
    
    print("\nStarting simulation...\n")
    
    # Run simulation
    for step in range(num_steps):
        world.step()
        
        # Print progress every 20 steps
        if (step + 1) % 20 == 0:
            print(f"Step {step + 1}/{num_steps}")
            for i, ant in enumerate(world.ants):
                print(f"  Ant {i+1}: position=({ant.x}, {ant.y}), distance_traveled={ant.distance_traveled:.2f}")
    
    print("\n=== Simulation Complete ===")
    print("\nFinal ant statistics:")
    for i, ant in enumerate(world.ants):
        print(f"Ant {i+1}:")
        print(f"  Final position: ({ant.x}, {ant.y})")
        print(f"  Total distance traveled: {ant.distance_traveled:.2f}")
        print(f"  Steps taken: {ant.steps_taken}")
    
    # Demonstrate learning by mutating the best ant
    print("\n=== Learning Phase ===")
    best_ant = max(world.ants, key=lambda a: a.distance_traveled)
    best_ant_idx = world.ants.index(best_ant)
    print(f"Best performing ant: Ant {best_ant_idx + 1} (distance: {best_ant.distance_traveled:.2f})")
    print("Applying learning mutation to best ant's neural network...")
    best_ant.learn()
    print("Neural network mutated successfully!")


if __name__ == "__main__":
    # Run simulation with default parameters
    run_simulation()
