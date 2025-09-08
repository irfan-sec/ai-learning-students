#!/usr/bin/env python3
"""
Week 1: Simple Agent Examples
This module demonstrates basic agent concepts with simple Python implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Any
import random

class Agent(ABC):
    """
    Abstract base class for all agents.
    
    An agent must be able to:
    1. Perceive its environment
    2. Choose actions based on perceptions
    """
    
    def __init__(self, name: str):
        self.name = name
        self.performance_score = 0
    
    @abstractmethod
    def perceive(self, environment_state: Any) -> Any:
        """Perceive the current state of the environment."""
        pass
    
    @abstractmethod
    def act(self, percept: Any) -> Any:
        """Choose an action based on the current percept."""
        pass


class SimpleReflexAgent(Agent):
    """
    A simple reflex agent that responds to current percept only.
    Example: Thermostat that turns on/off based on temperature.
    """
    
    def __init__(self, name: str, target_temp: float = 22.0):
        super().__init__(name)
        self.target_temp = target_temp
    
    def perceive(self, environment_state: dict) -> float:
        """Perceive the current temperature."""
        return environment_state.get('temperature', 20.0)
    
    def act(self, percept: float) -> str:
        """Turn heater on/off based on temperature."""
        if percept < self.target_temp:
            action = "heat_on"
            print(f"{self.name}: Temperature {percept}°C is below target. Turning heater ON.")
        else:
            action = "heat_off"
            print(f"{self.name}: Temperature {percept}°C is at/above target. Turning heater OFF.")
        return action


class RandomAgent(Agent):
    """
    An agent that chooses actions randomly.
    Useful as a baseline for comparison.
    """
    
    def __init__(self, name: str, actions: List[str]):
        super().__init__(name)
        self.actions = actions
    
    def perceive(self, environment_state: Any) -> Any:
        """This agent ignores the environment state."""
        return environment_state
    
    def act(self, percept: Any) -> str:
        """Choose a random action."""
        action = random.choice(self.actions)
        print(f"{self.name}: Randomly chose action '{action}'")
        return action


class VacuumAgent(Agent):
    """
    A simple vacuum cleaner agent that demonstrates goal-based behavior.
    Environment: Two locations (A and B), each can be clean or dirty.
    """
    
    def __init__(self, name: str):
        super().__init__(name)
        self.location = 'A'  # Start at location A
    
    def perceive(self, environment_state: dict) -> dict:
        """Perceive current location and whether it's dirty."""
        current_location = self.location
        is_dirty = environment_state.get(current_location, False)
        
        percept = {
            'location': current_location,
            'dirty': is_dirty,
            'environment': environment_state
        }
        
        print(f"{self.name}: At location {current_location}, dirty: {is_dirty}")
        return percept
    
    def act(self, percept: dict) -> str:
        """
        Simple reflex agent rules:
        1. If current location is dirty, clean it
        2. Otherwise, move to the other location
        """
        if percept['dirty']:
            action = 'suck'
            print(f"{self.name}: Cleaning location {percept['location']}")
            self.performance_score += 10  # Reward for cleaning
        else:
            # Move to the other location
            self.location = 'B' if self.location == 'A' else 'A'
            action = f"move_to_{self.location}"
            print(f"{self.name}: Moving to location {self.location}")
            self.performance_score -= 1  # Small cost for moving
        
        return action


def demonstrate_agents():
    """
    Demonstrate different types of agents in their environments.
    """
    print("=" * 50)
    print("AI Agents Demonstration")
    print("=" * 50)
    
    # 1. Thermostat Agent (Simple Reflex)
    print("\n1. THERMOSTAT AGENT (Simple Reflex)")
    print("-" * 30)
    
    thermostat = SimpleReflexAgent("Smart Thermostat", target_temp=22.0)
    
    # Simulate different temperature readings
    temperature_readings = [18.5, 21.0, 22.5, 23.0, 19.5]
    
    for temp in temperature_readings:
        environment = {'temperature': temp}
        percept = thermostat.perceive(environment)
        action = thermostat.act(percept)
        print()
    
    # 2. Random Agent
    print("\n2. RANDOM AGENT")
    print("-" * 30)
    
    random_robot = RandomAgent("Random Robot", ['move_left', 'move_right', 'pick_up', 'put_down'])
    
    for i in range(3):
        environment = {'step': i}
        percept = random_robot.perceive(environment)
        action = random_robot.act(percept)
        print()
    
    # 3. Vacuum Cleaner Agent
    print("\n3. VACUUM CLEANER AGENT")
    print("-" * 30)
    
    vacuum = VacuumAgent("RoboVac")
    
    # Initial environment: both locations dirty
    environment = {'A': True, 'B': True}  # True = dirty
    
    print(f"Initial environment: {environment}")
    print(f"Starting performance score: {vacuum.performance_score}")
    print()
    
    # Simulate several time steps
    for step in range(6):
        print(f"Step {step + 1}:")
        percept = vacuum.perceive(environment)
        action = vacuum.act(percept)
        
        # Update environment based on action
        if action == 'suck':
            environment[vacuum.location] = False  # Clean the location
        
        print(f"Environment after action: {environment}")
        print(f"Performance score: {vacuum.performance_score}")
        print()


def agent_environment_classification():
    """
    Demonstrate how to classify agent environments.
    """
    print("\n" + "=" * 50)
    print("ENVIRONMENT CLASSIFICATION EXAMPLES")
    print("=" * 50)
    
    environments = [
        {
            'name': 'Chess Game',
            'observable': 'Fully Observable',
            'deterministic': 'Deterministic',
            'episodic': 'Sequential', 
            'static': 'Static',
            'discrete': 'Discrete',
            'agents': 'Multi-agent'
        },
        {
            'name': 'Taxi Driving',
            'observable': 'Partially Observable',
            'deterministic': 'Stochastic',
            'episodic': 'Sequential',
            'static': 'Dynamic', 
            'discrete': 'Continuous',
            'agents': 'Multi-agent'
        },
        {
            'name': 'Email Spam Filter',
            'observable': 'Fully Observable',
            'deterministic': 'Stochastic',
            'episodic': 'Episodic',
            'static': 'Static',
            'discrete': 'Discrete', 
            'agents': 'Single-agent'
        }
    ]
    
    for env in environments:
        print(f"\n{env['name']}:")
        print(f"  Observable:    {env['observable']}")
        print(f"  Deterministic: {env['deterministic']}")
        print(f"  Episodic:      {env['episodic']}")
        print(f"  Static:        {env['static']}")
        print(f"  Discrete:      {env['discrete']}")
        print(f"  Agents:        {env['agents']}")


if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    
    # Run demonstrations
    demonstrate_agents()
    agent_environment_classification()
    
    print("\n" + "=" * 50)
    print("Key Takeaways:")
    print("1. Agents perceive their environment and choose actions")
    print("2. Different agent architectures suit different problems")
    print("3. Performance measurement is crucial for agent design")
    print("4. Environment characteristics affect agent design choices")
    print("=" * 50)