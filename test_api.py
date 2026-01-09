#!/usr/bin/env python
"""Test script to verify the web API is working."""
import requests
import json

print("Testing Ant World Web API")
print("=" * 50)

try:
    # Test config endpoint
    print("\n1. Testing /api/config...")
    response = requests.get('http://localhost:5000/api/config')
    config = response.json()
    print(f"   Status: {response.status_code}")
    print(f"   Config: {json.dumps(config, indent=2)}")
    
    # Test state endpoint
    print("\n2. Testing /api/state...")
    response = requests.get('http://localhost:5000/api/state')
    state = response.json()
    print(f"   Status: {response.status_code}")
    print(f"   Running: {state['running']}")
    print(f"   Step: {state['step']}")
    print(f"   Ants: {len(state['ants'])}")
    print(f"   World: {state['world']}")
    if 'config' in state:
        print(f"   Config in state: {state['config']}")
    
    # Test start endpoint
    print("\n3. Testing /api/start...")
    response = requests.post('http://localhost:5000/api/start')
    result = response.json()
    print(f"   Status: {response.status_code}")
    print(f"   Result: {result}")
    
    # Test state after start
    print("\n4. Testing /api/state after start...")
    response = requests.get('http://localhost:5000/api/state')
    state = response.json()
    print(f"   Running: {state['running']}")
    print(f"   Ants: {len(state['ants'])}")
    print(f"   First ant position: ({state['ants'][0]['x']:.1f}, {state['ants'][0]['y']:.1f})")
    
    # Test stop endpoint
    print("\n5. Testing /api/stop...")
    response = requests.post('http://localhost:5000/api/stop')
    result = response.json()
    print(f"   Status: {response.status_code}")
    print(f"   Result: {result}")
    
    print("\n" + "=" * 50)
    print("✓ All API tests passed!")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
