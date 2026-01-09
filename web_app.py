"""Flask web application for ant world simulation."""
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from world import World
from ant import Ant
import numpy as np
import threading
import time
import logging

app = Flask(__name__)
CORS(app)

# Disable Flask request logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Global simulation state
simulation_state = {
    'world': None,
    'running': False,
    'step': 0,
    'config': {
        'num_ants': 20,
        'world_width': 500,
        'world_height': 500,
        'speed': 0,  # milliseconds per step (0 = no delay)
        'food_spawn_rate': 0.05,  # 5% chance per step
        'health_decay_rate': 0.5,  # health lost per step
        'max_health': 100,
        'initial_food_count': 50,  # Initial food items
        'visualization_enabled': True  # Toggle visualization
    }
}

simulation_lock = threading.Lock()


@app.route('/')
def index():
    """Render the simulation page."""
    return render_template('simulation.html')


@app.route('/simulation')
def simulation():
    """Render the simulation page."""
    return render_template('simulation.html')


@app.route('/api/config', methods=['GET', 'POST'])
def config():
    """Get or update simulation configuration."""
    with simulation_lock:
        if request.method == 'POST':
            data = request.json
            simulation_state['config'].update(data)
            return jsonify({'status': 'success', 'config': simulation_state['config']})
        else:
            return jsonify(simulation_state['config'])


@app.route('/api/start', methods=['POST'])
def start_simulation():
    """Start or restart the simulation."""
    with simulation_lock:
        config = simulation_state['config']
        
        # Create new world with food spawning
        world = World(
            width=config['world_width'], 
            height=config['world_height'],
            food_spawn_rate=config.get('food_spawn_rate', 0.05),
            initial_food_count=config.get('initial_food_count', 50)
        )
        
        # Create ants at random positions
        for i in range(config['num_ants']):
            x = np.random.randint(5, config['world_width'] - 5)
            y = np.random.randint(5, config['world_height'] - 5)
            ant = Ant(x=x, y=y, max_health=config['max_health'])
            ant.health_decay_rate = config['health_decay_rate']
            world.add_ant(ant)
        
        simulation_state['world'] = world
        simulation_state['running'] = True
        simulation_state['step'] = 0
        
        return jsonify({'status': 'started'})


@app.route('/api/stop', methods=['POST'])
def stop_simulation():
    """Stop the simulation."""
    with simulation_lock:
        simulation_state['running'] = False
        return jsonify({'status': 'stopped'})


@app.route('/api/reset', methods=['POST'])
def reset_simulation():
    """Reset the simulation."""
    with simulation_lock:
        simulation_state['world'] = None
        simulation_state['running'] = False
        simulation_state['step'] = 0
        return jsonify({'status': 'reset'})


@app.route('/api/step', methods=['POST'])
def step_simulation():
    """Advance the simulation by one or more steps."""
    with simulation_lock:
        if simulation_state['world'] and simulation_state['running']:
            try:
                # Get number of steps from request, default to 1
                data = request.get_json() or {}
                num_steps = data.get('steps', 1)
                num_steps = min(max(1, int(num_steps)), 100)  # Limit to 1-100 steps
                
                # Run multiple steps
                for _ in range(num_steps):
                    simulation_state['world'].step()
                    simulation_state['step'] += 1
                
                # Get state and return it (only after all steps)
                world = simulation_state['world']
                ants_data = []
                
                for i, ant in enumerate(world.ants):
                    # Ensure values are valid (not NaN or infinity)
                    x = float(ant.x) if np.isfinite(ant.x) else 0.0
                    y = float(ant.y) if np.isfinite(ant.y) else 0.0
                    distance = float(ant.distance_traveled) if np.isfinite(ant.distance_traveled) else 0.0
                    health = float(ant.health) if np.isfinite(ant.health) else 0.0
                    fitness = float(ant.get_fitness()) if np.isfinite(ant.get_fitness()) else 0.0
                    
                    ants_data.append({
                        'id': i,
                        'x': x,
                        'y': y,
                        'distance_traveled': distance,
                        'steps_taken': ant.steps_taken,
                        'health': health,
                        'max_health': float(ant.max_health),
                        'food_collected': ant.food_collected,
                        'fitness': fitness
                    })
                
                # Get food data
                food_data = []
                for food in world.food:
                    food_x = float(food.x) if np.isfinite(food.x) else 0.0
                    food_y = float(food.y) if np.isfinite(food.y) else 0.0
                    food_data.append({
                        'x': food_x,
                        'y': food_y,
                        'energy': float(food.energy)
                    })
                
                return jsonify({
                    'running': simulation_state['running'],
                    'step': simulation_state['step'],
                    'ants': ants_data,
                    'food': food_data,
                    'world': {
                        'width': world.width,
                        'height': world.height
                    },
                    'config': simulation_state['config']
                })
            except Exception as e:
                log.error(f"Error in simulation step {simulation_state['step']}: {str(e)}")
                # Don't stop the simulation, just log the error and continue
                return jsonify({
                    'running': simulation_state['running'],
                    'step': simulation_state['step'],
                    'ants': [],
                    'food': [],
                    'world': {
                        'width': simulation_state['config']['world_width'],
                        'height': simulation_state['config']['world_height']
                    },
                    'config': simulation_state['config']
                })
        return jsonify({'error': 'Simulation not running'})


@app.route('/api/step_multiple', methods=['POST'])
def step_multiple():
    """Advance the simulation by multiple steps."""
    with simulation_lock:
        if simulation_state['world'] and simulation_state['running']:
            try:
                # Get number of steps from request
                data = request.get_json() or {}
                num_steps = min(int(data.get('steps', 1)), 100)  # Cap at 100 for safety
                
                # Run multiple steps
                for _ in range(num_steps):
                    simulation_state['world'].step()
                    simulation_state['step'] += 1
                
                # Get state and return it (only once after all steps)
                world = simulation_state['world']
                ants_data = []
                
                for i, ant in enumerate(world.ants):
                    # Ensure values are valid (not NaN or infinity)
                    x = float(ant.x) if np.isfinite(ant.x) else 0.0
                    y = float(ant.y) if np.isfinite(ant.y) else 0.0
                    distance = float(ant.distance_traveled) if np.isfinite(ant.distance_traveled) else 0.0
                    health = float(ant.health) if np.isfinite(ant.health) else 0.0
                    fitness = float(ant.get_fitness()) if np.isfinite(ant.get_fitness()) else 0.0
                    
                    ants_data.append({
                        'id': i,
                        'x': x,
                        'y': y,
                        'distance_traveled': distance,
                        'steps_taken': ant.steps_taken,
                        'health': health,
                        'max_health': float(ant.max_health),
                        'food_collected': ant.food_collected,
                        'fitness': fitness
                    })
                
                # Get food data
                food_data = []
                for food in world.food:
                    food_x = float(food.x) if np.isfinite(food.x) else 0.0
                    food_y = float(food.y) if np.isfinite(food.y) else 0.0
                    food_data.append({
                        'x': food_x,
                        'y': food_y,
                        'energy': float(food.energy)
                    })
                
                return jsonify({
                    'running': simulation_state['running'],
                    'step': simulation_state['step'],
                    'ants': ants_data,
                    'food': food_data,
                    'world': {
                        'width': world.width,
                        'height': world.height
                    },
                    'config': simulation_state['config']
                })
            except Exception as e:
                log.error(f"Error in simulation step {simulation_state['step']}: {str(e)}")
                # Don't stop the simulation, just log the error and continue
                return jsonify({
                    'running': simulation_state['running'],
                    'step': simulation_state['step'],
                    'ants': [],
                    'food': [],
                    'world': {
                        'width': simulation_state['config']['world_width'],
                        'height': simulation_state['config']['world_height']
                    },
                    'config': simulation_state['config']
                })
        return jsonify({'error': 'Simulation not running'})


@app.route('/api/state', methods=['GET'])
def get_state():
    """Get current simulation state."""
    with simulation_lock:
        if simulation_state['world'] is None:
            config = simulation_state['config']
            return jsonify({
                'running': False,
                'step': 0,
                'ants': [],
                'food': [],
                'world': {
                    'width': config['world_width'],
                    'height': config['world_height']
                },
                'config': config
            })
        
        world = simulation_state['world']
        ants_data = []
        
        for i, ant in enumerate(world.ants):
            ants_data.append({
                'id': i,
                'x': float(ant.x),
                'y': float(ant.y),
                'distance_traveled': float(ant.distance_traveled),
                'steps_taken': ant.steps_taken,
                'health': float(ant.health),
                'max_health': float(ant.max_health),
                'food_collected': ant.food_collected,
                'fitness': float(ant.get_fitness())
            })
        
        # Get food data
        food_data = []
        for food in world.food:
            food_data.append({
                'x': float(food.x),
                'y': float(food.y),
                'energy': float(food.energy)
            })
        
        return jsonify({
            'running': simulation_state['running'],
            'step': simulation_state['step'],
            'ants': ants_data,
            'food': food_data,
            'world': {
                'width': world.width,
                'height': world.height
            },
            'config': simulation_state['config']
        })


def run_app(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask application."""
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    print("=" * 50)
    print("Ant World Web Simulator")
    print("=" * 50)
    print("\nStarting web server...")
    print("Open your browser to: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 50)
    run_app(debug=False)  # Set debug=False to reduce logging
