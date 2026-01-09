"""Visual simulation runner for ant world."""
from visualizer import visualize_simulation


if __name__ == "__main__":
    # Run visualized simulation
    # Parameters:
    # - num_ants: Number of ants in the simulation
    # - num_steps: How many steps to simulate
    # - world_width: Width of the world
    # - world_height: Height of the world  
    # - interval: Animation speed (0 = as fast as possible, in milliseconds)
    # - enable_visualization: Whether to show visualization (False = faster)
    
    visualize_simulation(
        num_ants=20,
        num_steps=200,
        world_width=500,
        world_height=500,
        interval=0,  # 0ms = no delay, run as fast as possible
        enable_visualization=True  # Set to False for faster simulation without graphics
    )
