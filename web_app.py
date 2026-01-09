"""Flask web application for ant world simulation."""
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from world import World
from ant import Ant
import numpy as np
import threading
import time

app = Flask(__name__)
CORS(app)

# Global simulation state
simulation_state = {
    'world': None,
    'running': False,
    'step': 0,
    'config': {
        'num_ants': 5,
        'world_width': 50,
        'world_height': 50,
        'speed': 50,  # milliseconds per step
        'food_spawn_rate': 0.1,  # 10% chance per step
        'health_decay_rate': 0.5,  # health lost per step
        'max_health': 100
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
            food_spawn_rate=config['food_spawn_rate']
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
    """Advance the simulation by one step."""
    with simulation_lock:
        if simulation_state['world'] and simulation_state['running']:
            simulation_state['world'].step()
            simulation_state['step'] += 1
            # Get state and return it
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
                    'food_collected': ant.food_collected
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
                'food_collected': ant.food_collected
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
    run_app(debug=True)
