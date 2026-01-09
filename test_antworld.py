"""Unit tests for ant world simulation."""
import unittest
import numpy as np
import os
import json
import tempfile
from world import World, Food
from ant import Ant, NeuralNetwork, save_successful_networks, load_networks


class TestWorld(unittest.TestCase):
    """Test cases for World class."""
    
    def test_world_creation(self):
        """Test world is created with correct dimensions."""
        world = World(width=50, height=50)
        self.assertEqual(world.width, 50)
        self.assertEqual(world.height, 50)
        self.assertEqual(len(world.ants), 0)
    
    def test_initial_food_placement(self):
        """Test initial food is placed in the world."""
        world = World(width=50, height=50, initial_food_count=10)
        self.assertEqual(len(world.food), 10)
    
    def test_add_ant(self):
        """Test adding ant to world."""
        world = World()
        ant = Ant(x=10, y=10)
        world.add_ant(ant)
        self.assertEqual(len(world.ants), 1)
        self.assertEqual(ant.world, world)
    
    def test_is_valid_position(self):
        """Test position validation."""
        world = World(width=10, height=10)
        self.assertTrue(world.is_valid_position(5, 5))
        self.assertTrue(world.is_valid_position(0, 0))
        self.assertTrue(world.is_valid_position(9, 9))
        self.assertFalse(world.is_valid_position(-1, 5))
        self.assertFalse(world.is_valid_position(5, -1))
        self.assertFalse(world.is_valid_position(10, 5))
        self.assertFalse(world.is_valid_position(5, 10))
    
    def test_food_collision(self):
        """Test ant collecting food."""
        world = World(width=50, height=50, initial_food_count=0)
        ant = Ant(x=25, y=25)
        world.add_ant(ant)
        
        # Reduce ant's health so it can benefit from food
        ant.health = 50
        initial_health = ant.health
        
        # Place food near ant
        food = Food(x=25, y=25, energy=50)
        world.food.append(food)
        
        # Check collision
        world.check_food_collision(ant)
        
        # Verify food was collected
        self.assertEqual(len(world.food), 0)
        self.assertEqual(ant.food_collected, 1)
        self.assertGreater(ant.health, initial_health)


class TestNeuralNetwork(unittest.TestCase):
    """Test cases for NeuralNetwork class."""
    
    def test_neural_network_creation(self):
        """Test neural network is created with correct dimensions."""
        nn = NeuralNetwork(input_size=5, hidden_size=8, output_size=4)
        self.assertEqual(nn.input_size, 5)
        self.assertEqual(nn.hidden_size, 8)
        self.assertEqual(nn.output_size, 4)
        self.assertEqual(nn.weights_input_hidden.shape, (5, 8))
        self.assertEqual(nn.weights_hidden_output.shape, (8, 4))
    
    def test_forward_pass(self):
        """Test neural network forward pass."""
        nn = NeuralNetwork(input_size=5, hidden_size=8, output_size=4)
        inputs = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        output = nn.forward(inputs)
        self.assertEqual(output.shape, (4,))
        # Check output sums to 1 (softmax property)
        self.assertAlmostEqual(np.sum(output), 1.0, places=5)
    
    def test_mutate(self):
        """Test neural network mutation."""
        nn = NeuralNetwork()
        original_weights = nn.weights_input_hidden.copy()
        nn.mutate(mutation_rate=1.0, mutation_scale=0.1)
        # Check that weights have changed
        self.assertFalse(np.array_equal(original_weights, nn.weights_input_hidden))
    
    def test_copy(self):
        """Test neural network deep copy."""
        nn = NeuralNetwork()
        nn_copy = nn.copy()
        
        # Check that copies have same values
        self.assertTrue(np.array_equal(nn.weights_input_hidden, nn_copy.weights_input_hidden))
        self.assertTrue(np.array_equal(nn.weights_hidden_output, nn_copy.weights_hidden_output))
        
        # Check that they are independent (modifying one doesn't affect the other)
        nn.weights_input_hidden[0, 0] = 999.0
        self.assertNotEqual(nn.weights_input_hidden[0, 0], nn_copy.weights_input_hidden[0, 0])
    
    def test_to_dict_and_from_dict(self):
        """Test neural network serialization and deserialization."""
        nn = NeuralNetwork(input_size=5, hidden_size=8, output_size=4)
        
        # Convert to dict
        nn_dict = nn.to_dict()
        
        # Check dict structure
        self.assertIn('input_size', nn_dict)
        self.assertIn('weights_input_hidden', nn_dict)
        
        # Convert back from dict
        nn_restored = NeuralNetwork.from_dict(nn_dict)
        
        # Check that restored network is identical
        self.assertEqual(nn.input_size, nn_restored.input_size)
        self.assertTrue(np.array_equal(nn.weights_input_hidden, nn_restored.weights_input_hidden))
        self.assertTrue(np.array_equal(nn.weights_hidden_output, nn_restored.weights_hidden_output))


class TestAnt(unittest.TestCase):
    """Test cases for Ant class."""
    
    def test_ant_creation(self):
        """Test ant is created at correct position."""
        ant = Ant(x=10, y=20)
        self.assertEqual(ant.x, 10)
        self.assertEqual(ant.y, 20)
        self.assertIsNotNone(ant.brain)
        self.assertEqual(ant.steps_taken, 0)
    
    def test_ant_sense(self):
        """Test ant sensor inputs."""
        world = World(width=100, height=100)
        ant = Ant(x=50, y=50)
        world.add_ant(ant)
        sensors = ant.sense()
        self.assertEqual(sensors.shape, (5,))
        self.assertEqual(sensors[0], 0.5)  # x position normalized
        self.assertEqual(sensors[1], 0.5)  # y position normalized
    
    def test_ant_movement(self):
        """Test ant can move."""
        world = World(width=100, height=100)
        ant = Ant(x=50, y=50)
        world.add_ant(ant)
        
        # Test moving right (action 1)
        ant.act(1)
        self.assertEqual(ant.x, 51)
        self.assertEqual(ant.y, 50)
        
        # Test moving down (action 2)
        ant.act(2)
        self.assertEqual(ant.x, 51)
        self.assertEqual(ant.y, 51)
    
    def test_ant_boundary_collision(self):
        """Test ant respects world boundaries."""
        world = World(width=10, height=10)
        ant = Ant(x=0, y=0)
        world.add_ant(ant)
        
        # Try to move up (out of bounds)
        ant.act(0)
        self.assertEqual(ant.x, 0)
        self.assertEqual(ant.y, 0)
        
        # Try to move left (out of bounds)
        ant.act(3)
        self.assertEqual(ant.x, 0)
        self.assertEqual(ant.y, 0)
    
    def test_ant_step(self):
        """Test ant step execution."""
        world = World(width=100, height=100)
        ant = Ant(x=50, y=50)
        world.add_ant(ant)
        
        initial_steps = ant.steps_taken
        ant.step()
        self.assertEqual(ant.steps_taken, initial_steps + 1)
    
    def test_distance_to_food_sensor(self):
        """Test ant can sense distance to food."""
        world = World(width=100, height=100, initial_food_count=0)
        ant = Ant(x=50, y=50)
        world.add_ant(ant)
        
        # Place food at a known location
        food = Food(x=60, y=60, energy=50)
        world.food.append(food)
        
        # Get sensor inputs
        sensors = ant.sense()
        
        # Check that distance to food sensor is populated (index 4)
        self.assertGreater(sensors[4], 0.0)
        self.assertLess(sensors[4], 1.0)


class TestNetworkSaving(unittest.TestCase):
    """Test cases for saving and loading neural networks."""
    
    def test_save_and_load_networks(self):
        """Test saving and loading networks to/from disk."""
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some ants with different fitness
            ants = []
            for i in range(3):
                ant = Ant(x=10*i, y=10*i)
                ant.distance_traveled = 100.0 * (i + 1)
                ant.food_collected = i
                ants.append(ant)
            
            # Save networks
            filename = save_successful_networks(ants, generation=1, step=100, save_dir=tmpdir)
            self.assertIsNotNone(filename)
            self.assertTrue(os.path.exists(filename))
            
            # Load networks
            loaded_networks = load_networks(filename)
            self.assertEqual(len(loaded_networks), 3)
            
            # Verify networks are functional
            for network in loaded_networks:
                inputs = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
                output = network.forward(inputs)
                self.assertEqual(output.shape, (4,))


if __name__ == "__main__":
    unittest.main()
