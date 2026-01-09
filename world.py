"""World class representing the environment where ants live."""
import numpy as np


class Food:
    """A food item in the world."""
    
    def __init__(self, x, y, energy=50):
        """
        Initialize a food item.
        
        Args:
            x: X position
            y: Y position
            energy: Energy value of the food
        """
        self.x = x
        self.y = y
        self.energy = energy


class World:
    """A 2D world space where ants can navigate."""
    
    def __init__(self, width=100, height=100, food_spawn_rate=0.1):
        """
        Initialize the world.
        
        Args:
            width: Width of the world
            height: Height of the world
            food_spawn_rate: Probability of spawning food each step (0-1)
        """
        self.width = width
        self.height = height
        self.ants = []
        self.food = []
        self.food_spawn_rate = food_spawn_rate
        self.step_count = 0
        
    def add_ant(self, ant):
        """Add an ant to the world."""
        self.ants.append(ant)
        ant.world = self
        
    def spawn_food(self):
        """Randomly spawn food in the world."""
        if np.random.random() < self.food_spawn_rate:
            x = np.random.randint(0, self.width)
            y = np.random.randint(0, self.height)
            food = Food(x, y, energy=np.random.randint(30, 70))
            self.food.append(food)
    
    def check_food_collision(self, ant):
        """Check if an ant is at a food location and consume it."""
        for food in self.food[:]:
            # Check if ant is close enough to food (within 1 unit)
            distance = np.sqrt((ant.x - food.x)**2 + (ant.y - food.y)**2)
            if distance < 1.5:
                ant.health = min(ant.max_health, ant.health + food.energy)
                ant.food_collected += 1
                self.food.remove(food)
                return True
        return False
    
    def respawn_ant(self, dead_ant):
        """Respawn an ant with a new neural network at a random position."""
        from ant import Ant
        x = np.random.randint(5, self.width - 5)
        y = np.random.randint(5, self.height - 5)
        new_ant = Ant(x=x, y=y)
        return new_ant
        
    def step(self):
        """Advance the simulation by one time step."""
        self.step_count += 1
        
        # Spawn food randomly
        self.spawn_food()
        
        # Update ants
        dead_ants = []
        for ant in self.ants:
            ant.step()
            
            # Check for food collision
            self.check_food_collision(ant)
            
            # Check if ant died
            if ant.health <= 0:
                dead_ants.append(ant)
        
        # Respawn dead ants
        for dead_ant in dead_ants:
            self.ants.remove(dead_ant)
            new_ant = self.respawn_ant(dead_ant)
            self.add_ant(new_ant)
            
    def is_valid_position(self, x, y):
        """Check if a position is within world bounds."""
        return 0 <= x < self.width and 0 <= y < self.height
