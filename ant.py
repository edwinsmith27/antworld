"""Ant class with neural network for navigation."""
import numpy as np


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
        self.brain = NeuralNetwork(input_size=4, hidden_size=8, output_size=4)
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
            return np.array([0.5, 0.5, 0.5, 0.5])
        
        # Simple sensors: normalized position and distances to edges
        sensor_inputs = np.array([
            self.x / self.world.width,  # Normalized x position
            self.y / self.world.height,  # Normalized y position
            (self.world.width - self.x) / self.world.width,  # Distance to right edge
            (self.world.height - self.y) / self.world.height,  # Distance to bottom edge
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
