"""Main simulation runner for ant world."""
from world import World
from ant import Ant
import numpy as np


class EvolutionarySimulation:
    """Manages the evolutionary ant world simulation."""
    
    # Simulation constants
    GENERATION_LENGTH = 200  # Steps per generation
    FITNESS_THRESHOLD_RATIO = 0.1  # Fitness threshold as ratio of world size
    MIN_FITNESS_THRESHOLD = 10.0  # Absolute minimum fitness threshold
    MIN_SURVIVAL_RATE = 0.25  # Minimum 25% of population survives each generation
    PROGRESS_REPORT_INTERVAL = 50  # Report progress every N steps
    
    def __init__(self, initial_ants=20, world_width=500, world_height=500):
        """
        Initialize the evolutionary simulation.
        
        Args:
            initial_ants: Number of ants to start with
            world_width: Width of the world (max 500)
            world_height: Height of the world (max 500)
        """
        self.world = World(width=world_width, height=world_height)
        self.generation = 0
        self.step_count = 0
        self.generation_length = self.GENERATION_LENGTH
        # Calculate minimum fitness based on world size
        max_distance = min(world_width, world_height) * self.FITNESS_THRESHOLD_RATIO
        self.min_fitness_threshold = max(self.MIN_FITNESS_THRESHOLD, max_distance)
        self.initial_ant_count = initial_ants
        self.running = True
        
        # Create initial population
        self._create_initial_population()
        
    def _create_initial_population(self):
        """Create the initial population of ants."""
        for i in range(self.initial_ant_count):
            x = np.random.randint(0, self.world.width)
            y = np.random.randint(0, self.world.height)
            ant = Ant(x=x, y=y)
            self.world.add_ant(ant)
    
    def _calculate_fitness(self, ant):
        """
        Calculate fitness score for an ant.
        
        Args:
            ant: The ant to evaluate
            
        Returns:
            Fitness score (higher is better)
        """
        # Fitness based on distance traveled
        return ant.distance_traveled
    
    def _selection_and_reproduction(self):
        """
        Select successful ants and create next generation.
        Unsuccessful ants die out.
        """
        if not self.world.ants:
            # If all ants died, create new random population
            print("All ants died! Creating new random population...")
            self._create_initial_population()
            return
        
        # Calculate fitness for all ants
        fitness_scores = [(ant, self._calculate_fitness(ant)) for ant in self.world.ants]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Print statistics
        print(f"\nGeneration {self.generation} Statistics:")
        print(f"  Total ants: {len(self.world.ants)}")
        print(f"  Best fitness: {fitness_scores[0][1]:.2f}")
        print(f"  Average fitness: {np.mean([f for _, f in fitness_scores]):.2f}")
        print(f"  Worst fitness: {fitness_scores[-1][1]:.2f}")
        
        # Select survivors (ensure minimum survivor count, then apply fitness threshold)
        survivors = []
        min_survivors = max(2, int(len(fitness_scores) * self.MIN_SURVIVAL_RATE))
        
        # First, guarantee minimum survivors from top performers
        for i in range(min(min_survivors, len(fitness_scores))):
            survivors.append(fitness_scores[i][0])
        
        # Then add any additional ants that meet the fitness threshold
        for ant, fitness in fitness_scores[min_survivors:]:
            if fitness >= self.min_fitness_threshold:
                survivors.append(ant)
        
        print(f"  Survivors: {len(survivors)}/{len(self.world.ants)}")
        
        # Create next generation
        new_ants = []
        
        # Keep the survivors in new positions
        for ant in survivors:
            # Reset ant position and stats for next generation
            new_ant = Ant(
                x=np.random.randint(0, self.world.width),
                y=np.random.randint(0, self.world.height)
            )
            # Deep copy the brain from survivor
            new_ant.brain = ant.brain.copy()
            new_ants.append(new_ant)
        
        # Create offspring from survivors through mutation
        while len(new_ants) < self.initial_ant_count:
            # Select random parent
            parent = survivors[np.random.randint(0, len(survivors))]
            
            # Create child ant
            child = Ant(
                x=np.random.randint(0, self.world.width),
                y=np.random.randint(0, self.world.height)
            )
            
            # Copy parent's brain and mutate
            child.brain = parent.brain.copy()
            child.learn()  # Apply mutation
            
            new_ants.append(child)
        
        # Replace old population with new generation
        self.world.ants = new_ants
        for ant in self.world.ants:
            ant.world = self.world
        
        self.generation += 1
    
    def run(self):
        """Run the simulation continuously until interrupted."""
        print("=== Evolutionary Ant World Simulation ===")
        print(f"World size: {self.world.width}x{self.world.height}")
        print(f"Initial population: {self.initial_ant_count} ants")
        print(f"Generation length: {self.generation_length} steps")
        print(f"Minimum fitness threshold: {self.min_fitness_threshold}")
        print("\nPress Ctrl+C to stop the simulation\n")
        print("Starting simulation...\n")
        
        try:
            while self.running:
                # Run one generation
                for step in range(self.generation_length):
                    self.world.step()
                    self.step_count += 1
                    
                    # Print progress periodically
                    if (step + 1) % self.PROGRESS_REPORT_INTERVAL == 0:
                        alive_count = len(self.world.ants)
                        if alive_count > 0:
                            avg_distance = np.mean([ant.distance_traveled for ant in self.world.ants])
                            print(f"Generation {self.generation}, Step {step + 1}/{self.generation_length} - Avg distance: {avg_distance:.2f}")
                
                # Evolution: selection and reproduction
                self._selection_and_reproduction()
                
        except KeyboardInterrupt:
            print("\n\nSimulation interrupted by user")
            self._print_final_statistics()
    
    def _print_final_statistics(self):
        """Print final simulation statistics."""
        print("\n=== Final Statistics ===")
        print(f"Total generations completed: {self.generation}")
        print(f"Total steps simulated: {self.step_count}")
        
        if self.world.ants:
            print(f"\nCurrent population: {len(self.world.ants)} ants")
            fitness_scores = [self._calculate_fitness(ant) for ant in self.world.ants]
            print(f"Best current fitness: {max(fitness_scores):.2f}")
            print(f"Average current fitness: {np.mean(fitness_scores):.2f}")
        else:
            print("\nNo ants survived")


def run_simulation(num_ants=20, world_width=500, world_height=500):
    """
    Run the evolutionary ant world simulation.
    
    Args:
        num_ants: Number of ants to start with
        world_width: Width of the world (max 500)
        world_height: Height of the world (max 500)
    """
    # Ensure world size doesn't exceed maximum
    world_width = min(world_width, 500)
    world_height = min(world_height, 500)
    
    sim = EvolutionarySimulation(
        initial_ants=num_ants,
        world_width=world_width,
        world_height=world_height
    )
    sim.run()


if __name__ == "__main__":
    # Run simulation with default parameters
    run_simulation()
