"""Ant class with neural network for navigation."""
import numpy as np
import json
import os
from datetime import datetime


class NeuralNetwork:
    """Simple neural network for ant decision making."""
    
    def __init__(self, input_size=4, hidden_size=8, output_size=4):
        """
        Initialize neural network with random weights.
        
        Args:
            input_size: Number of input neurons (sensor inputs)
            hidden_size: Number of hidden layer neurons
            output_size: Number of output neurons (movement directions)
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights randomly
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.5
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.5
        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)
        
    def forward(self, inputs):
        """
        Forward pass through the network.
        
        Args:
            inputs: Input array
            
        Returns:
            Output array with action probabilities
        """
        # Hidden layer
        hidden = np.tanh(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        # Output layer
        output = np.dot(hidden, self.weights_hidden_output) + self.bias_output
        # Apply softmax for action selection
        exp_output = np.exp(output - np.max(output))
        return exp_output / exp_output.sum()
    
    def copy(self):
        """
        Create a deep copy of this neural network.
        
        Returns:
            A new NeuralNetwork instance with copied weights
        """
        new_nn = NeuralNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        )
        new_nn.weights_input_hidden = self.weights_input_hidden.copy()
        new_nn.weights_hidden_output = self.weights_hidden_output.copy()
        new_nn.bias_hidden = self.bias_hidden.copy()
        new_nn.bias_output = self.bias_output.copy()
        return new_nn
    
    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        """
        Mutate network weights for learning/evolution.
        
        Args:
            mutation_rate: Probability of mutating each weight
            mutation_scale: Scale of mutations
        """
        for weights in [self.weights_input_hidden, self.weights_hidden_output]:
            mask = np.random.random(weights.shape) < mutation_rate
            weights[mask] += np.random.randn(np.sum(mask)) * mutation_scale
    
    def to_dict(self):
        """
        Convert neural network to dictionary for serialization.
        
        Returns:
            Dictionary representation of the network
        """
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'weights_input_hidden': self.weights_input_hidden.tolist(),
            'weights_hidden_output': self.weights_hidden_output.tolist(),
            'bias_hidden': self.bias_hidden.tolist(),
            'bias_output': self.bias_output.tolist()
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Create neural network from dictionary.
        
        Args:
            data: Dictionary representation of the network
            
        Returns:
            A new NeuralNetwork instance
        """
        nn = cls(
            input_size=data['input_size'],
            hidden_size=data['hidden_size'],
            output_size=data['output_size']
        )
        nn.weights_input_hidden = np.array(data['weights_input_hidden'])
        nn.weights_hidden_output = np.array(data['weights_hidden_output'])
        nn.bias_hidden = np.array(data['bias_hidden'])
        nn.bias_output = np.array(data['bias_output'])
        return nn


class Ant:
    """An ant agent that uses a neural network to navigate the world."""
    
    def __init__(self, x=0, y=0, max_health=100):
        """
        Initialize an ant.
        
        Args:
            x: Initial x position
            y: Initial y position
            max_health: Maximum health value
        """
        self.x = x
        self.y = y
        self.world = None
        self.brain = NeuralNetwork(input_size=5, hidden_size=8, output_size=4)
        self.steps_taken = 0
        self.distance_traveled = 0.0
        self.max_health = max_health
        self.health = max_health
        self.food_collected = 0
        self.health_decay_rate = 0.5  # Health lost per step
        
    def sense(self):
        """
        Get sensor inputs from the environment.
        
        Returns:
            Array of sensor values
        """
        if self.world is None:
            return np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        
        # Find nearest food
        distance_to_food = 1.0  # Default: far away (normalized)
        if self.world.food:
            min_distance = float('inf')
            for food in self.world.food:
                distance = np.sqrt((self.x - food.x)**2 + (self.y - food.y)**2)
                if distance < min_distance:
                    min_distance = distance
            # Normalize distance by diagonal of world
            max_distance = np.sqrt(self.world.width**2 + self.world.height**2)
            distance_to_food = min(1.0, min_distance / max_distance)
        
        # Simple sensors: normalized position, distances to edges, and distance to food
        sensor_inputs = np.array([
            self.x / self.world.width,  # Normalized x position
            self.y / self.world.height,  # Normalized y position
            (self.world.width - self.x) / self.world.width,  # Distance to right edge
            (self.world.height - self.y) / self.world.height,  # Distance to bottom edge
            distance_to_food,  # Distance to nearest food (normalized)
        ])
        return sensor_inputs
    
    def act(self, action):
        """
        Perform an action (move in a direction).
        
        Args:
            action: Action index (0=up, 1=right, 2=down, 3=left)
        """
        old_x, old_y = self.x, self.y
        
        # Movement directions: up, right, down, left
        movements = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        dx, dy = movements[action]
        
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Only move if within bounds
        if self.world and self.world.is_valid_position(new_x, new_y):
            self.x = new_x
            self.y = new_y
            # Track distance
            self.distance_traveled += np.sqrt((self.x - old_x)**2 + (self.y - old_y)**2)
    
    def step(self):
        """Execute one step: sense, decide, act."""
        # Get sensor inputs
        inputs = self.sense()
        
        # Get action from neural network
        action_probs = self.brain.forward(inputs)
        action = np.argmax(action_probs)
        
        # Perform action
        self.act(action)
        self.steps_taken += 1
        
        # Decrease health over time
        self.health -= self.health_decay_rate
        self.health = max(0, self.health)  # Don't go below 0
    
    def learn(self):
        """Apply learning by mutating the neural network."""
        self.brain.mutate(mutation_rate=0.1, mutation_scale=0.1)


def save_successful_networks(ants, generation=0, step=0, save_dir='saved_networks'):
    """
    Save neural networks of successful ants to disk.
    
    Args:
        ants: List of successful ant instances
        generation: Current generation number
        step: Current step number
        save_dir: Directory to save networks
    """
    if not ants:
        return
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"survivors_{timestamp}.json")
    
    # Calculate best fitness
    best_fitness = max(ant.distance_traveled for ant in ants)
    
    # Prepare data
    data = {
        'timestamp': timestamp,
        'generation': generation,
        'step': step,
        'best_fitness': float(best_fitness),
        'count': len(ants),
        'networks': []
    }
    
    # Save each ant's network
    for ant in ants:
        network_data = {
            'fitness': float(ant.distance_traveled),
            'food_collected': ant.food_collected,
            'distance_traveled': float(ant.distance_traveled),
            'brain': ant.brain.to_dict()
        }
        data['networks'].append(network_data)
    
    # Write to file
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved {len(ants)} networks to {filename}")
    return filename


def load_networks(filename):
    """
    Load neural networks from a saved file.
    
    Args:
        filename: Path to the saved networks file
        
    Returns:
        List of NeuralNetwork instances
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    
    networks = []
    for network_data in data['networks']:
        brain = NeuralNetwork.from_dict(network_data['brain'])
        networks.append(brain)
    
    print(f"Loaded {len(networks)} networks from {filename}")
    return networks


def get_latest_saved_networks(save_dir='saved_networks'):
    """
    Get the most recently saved networks file.
    
    Args:
        save_dir: Directory containing saved networks
        
    Returns:
        Path to the latest networks file, or None if no files exist
    """
    if not os.path.exists(save_dir):
        return None
    
    files = [f for f in os.listdir(save_dir) if f.startswith('survivors_') and f.endswith('.json')]
    if not files:
        return None
    
    # Sort by filename (timestamp is in the filename)
    files.sort(reverse=True)
    return os.path.join(save_dir, files[0])
