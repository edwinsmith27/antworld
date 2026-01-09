"""Visualization module for ant world simulation."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import numpy as np


class AntWorldVisualizer:
    """Visualizes the ant world simulation using matplotlib."""
    
    def __init__(self, world, figsize=(10, 10)):
        """
        Initialize the visualizer.
        
        Args:
            world: The World object to visualize
            figsize: Figure size tuple (width, height)
        """
        self.world = world
        self.figsize = figsize
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.setup_plot()
        
        # Ant visualization elements
        self.ant_artists = []
        self.trail_lines = []
        
        # Statistics text
        self.stats_text = None
        
        # History for trails
        self.ant_histories = [[] for _ in self.world.ants]
        
    def setup_plot(self):
        """Set up the plot with world boundaries."""
        self.ax.set_xlim(-1, self.world.width + 1)
        self.ax.set_ylim(-1, self.world.height + 1)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_title('Ant World Simulation')
        self.ax.grid(True, alpha=0.3)
        
        # Draw world boundary
        boundary = patches.Rectangle(
            (0, 0), self.world.width, self.world.height,
            linewidth=2, edgecolor='black', facecolor='lightblue', alpha=0.2
        )
        self.ax.add_patch(boundary)
        
    def initialize_ants(self):
        """Initialize ant visual elements."""
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.world.ants)))
        
        for i, ant in enumerate(self.world.ants):
            # Create ant marker (circle)
            ant_circle = plt.Circle(
                (ant.x, ant.y), 0.5,
                color=colors[i], ec='black', linewidth=1.5, zorder=10
            )
            self.ax.add_patch(ant_circle)
            self.ant_artists.append(ant_circle)
            
            # Create trail line
            trail_line, = self.ax.plot([], [], '-', color=colors[i], alpha=0.3, linewidth=1)
            self.trail_lines.append(trail_line)
            
            # Initialize history with starting position
            self.ant_histories[i].append((ant.x, ant.y))
    
    def draw_food(self):
        """Draw food items on the plot."""
        # Remove old food artists
        if hasattr(self, 'food_artists'):
            for artist in self.food_artists:
                artist.remove()
        
        self.food_artists = []
        
        # Draw current food
        for food in self.world.food:
            food_circle = plt.Circle(
                (food.x, food.y), 0.8,
                color='green', ec='darkgreen', linewidth=1.5, zorder=5, alpha=0.7
            )
            self.ax.add_patch(food_circle)
            self.food_artists.append(food_circle)
            
    def update(self, frame):
        """
        Update function for animation.
        
        Args:
            frame: Frame number (simulation step)
            
        Returns:
            List of artists to update
        """
        # Step the simulation
        self.world.step()
        
        # Update ant positions and trails
        for i, ant in enumerate(self.world.ants):
            # Update ant position
            self.ant_artists[i].center = (ant.x, ant.y)
            
            # Update history
            self.ant_histories[i].append((ant.x, ant.y))
            
            # Update trail (keep last 50 positions)
            if len(self.ant_histories[i]) > 50:
                self.ant_histories[i] = self.ant_histories[i][-50:]
            
            xs, ys = zip(*self.ant_histories[i])
            self.trail_lines[i].set_data(xs, ys)
        
        # Draw food
        self.draw_food()
        
        # Update statistics
        if self.stats_text:
            self.stats_text.remove()
        
        stats_str = f"Step: {frame + 1}\n"
        stats_str += f"Food: {len(self.world.food)}\n"
        for i, ant in enumerate(self.world.ants):
            stats_str += f"Ant {i+1}: pos=({ant.x},{ant.y}) hp={ant.health:.0f} food={ant.food_collected}\n"
        
        self.stats_text = self.ax.text(
            0.02, 0.98, stats_str,
            transform=self.ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        artists = self.ant_artists + self.trail_lines + [self.stats_text]
        if hasattr(self, 'food_artists'):
            artists.extend(self.food_artists)
        return artists
    
    def animate(self, num_steps=100, interval=50):
        """
        Run the animated simulation.
        
        Args:
            num_steps: Number of simulation steps
            interval: Milliseconds between frames
        """
        self.initialize_ants()
        
        anim = FuncAnimation(
            self.fig, self.update,
            frames=num_steps,
            interval=interval,
            blit=True,
            repeat=False
        )
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def snapshot(self):
        """Display a single snapshot of the current world state."""
        self.initialize_ants()
        
        # Update once to show current state
        self.update(0)
        
        plt.tight_layout()
        plt.show()


def visualize_simulation(num_ants=5, num_steps=200, world_width=50, world_height=50, interval=0):
    """
    Run a visualized ant world simulation.
    
    Args:
        num_ants: Number of ants to create
        num_steps: Number of simulation steps
        world_width: Width of the world
        world_height: Height of the world
        interval: Milliseconds between animation frames (0 = as fast as possible)
    """
    from world import World
    from ant import Ant
    
    print("=== Ant World Visual Simulation ===")
    print(f"World size: {world_width}x{world_height}")
    print(f"Number of ants: {num_ants}")
    print(f"Simulation steps: {num_steps}")
    print(f"Animation interval: {interval}ms")
    print("\nCreating ants...")
    
    # Create world
    world = World(width=world_width, height=world_height)
    
    # Create and add ants at random positions
    for i in range(num_ants):
        x = np.random.randint(5, world_width - 5)
        y = np.random.randint(5, world_height - 5)
        ant = Ant(x=x, y=y)
        world.add_ant(ant)
        print(f"  Ant {i+1} created at position ({x}, {y})")
    
    print(f"\nInitial food placed: {len(world.food)}")
    print("\nStarting visualization...")
    print("Close the window to end the simulation.\n")
    
    # Create visualizer and run animation
    viz = AntWorldVisualizer(world)
    viz.animate(num_steps=num_steps, interval=interval)
