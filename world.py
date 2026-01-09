"""World class representing the environment where ants live."""
import numpy as np


class World:
    """A 2D world space where ants can navigate."""
    
    def __init__(self, width=100, height=100):
        """
        Initialize the world.
        
        Args:
            width: Width of the world
            height: Height of the world
        """
        self.width = width
        self.height = height
        self.ants = []
        
    def add_ant(self, ant):
        """Add an ant to the world."""
        self.ants.append(ant)
        ant.world = self
        
    def step(self):
        """Advance the simulation by one time step."""
        for ant in self.ants:
            ant.step()
            
    def is_valid_position(self, x, y):
        """Check if a position is within world bounds."""
        return 0 <= x < self.width and 0 <= y < self.height
