"""Visual simulation runner for ant world."""
from visualizer import visualize_simulation


if __name__ == "__main__":
    # Run visualized simulation
    # Parameters:
    # - num_ants: Number of ants in the simulation
    # - num_steps: How many steps to simulate
    # - world_width: Width of the world
    # - world_height: Height of the world  
    # - interval: Animation speed (lower = faster, in milliseconds)
    
    visualize_simulation(
        num_ants=5,
        num_steps=200,
        world_width=50,
        world_height=50,
        interval=50  # 50ms between frames
    )
