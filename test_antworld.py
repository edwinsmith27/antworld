"""Unit tests for ant world simulation."""
import unittest
import numpy as np
from world import World
from ant import Ant, NeuralNetwork


class TestWorld(unittest.TestCase):
    """Test cases for World class."""
    
    def test_world_creation(self):
        """Test world is created with correct dimensions."""
        world = World(width=50, height=50)
        self.assertEqual(world.width, 50)
        self.assertEqual(world.height, 50)
        self.assertEqual(len(world.ants), 0)
    
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


class TestNeuralNetwork(unittest.TestCase):
    """Test cases for NeuralNetwork class."""
    
    def test_neural_network_creation(self):
        """Test neural network is created with correct dimensions."""
        nn = NeuralNetwork(input_size=4, hidden_size=8, output_size=4)
        self.assertEqual(nn.input_size, 4)
        self.assertEqual(nn.hidden_size, 8)
        self.assertEqual(nn.output_size, 4)
        self.assertEqual(nn.weights_input_hidden.shape, (4, 8))
        self.assertEqual(nn.weights_hidden_output.shape, (8, 4))
    
    def test_forward_pass(self):
        """Test neural network forward pass."""
        nn = NeuralNetwork(input_size=4, hidden_size=8, output_size=4)
        inputs = np.array([0.5, 0.5, 0.5, 0.5])
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
        self.assertEqual(sensors.shape, (4,))
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


if __name__ == "__main__":
    unittest.main()
