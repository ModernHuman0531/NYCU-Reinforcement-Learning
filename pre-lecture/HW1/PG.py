#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple implementation of PG provided by CGPT
@author: pchsieh
"""

import numpy as np
import matplotlib.pyplot as plt

class PolicyGradientAgent:
    def __init__(self, state_dim, action_dim, lr=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.theta = np.random.rand(state_dim, action_dim)  # Policy parameters
        self.lr = lr
    
    def policy(self, state):
        """ policy parameterization"""
        prefs = np.dot(state, self.theta)
        exp_prefs = np.exp(prefs - np.max(prefs))  # Numerical stability
        return exp_prefs / np.sum(exp_prefs)
    
    def sample_action(self, state):
        """Sample action from the policy"""
        probs = self.policy(state)
        return np.random.choice(len(probs), p=probs)
    
    def update(self, states, actions, rewards):
        """Perform gradient ascent on the policy parameters"""
        G = 0
        gradients = []
        
        # Compute gradients for each time step
        for t in reversed(range(len(states))):
            G = rewards[t] + (0.99 * G)  # Discounted return
            state, action = states[t], actions[t]
            probs = self.policy(state)
            
            grad = np.zeros_like(self.theta)
            grad[:, action] = state - np.dot(state, probs)
            gradients.append((G, grad))
        
        # Update policy parameters
        for G, grad in gradients:
            self.theta += self.lr * G * grad

# Example usage
def main():
    env = {'state_dim': 4, 'action_dim': 2}  # Dummy environment
    agent = PolicyGradientAgent(env['state_dim'], env['action_dim'])
    
    # Simulated training loop
    for episode in range(100):
        states, actions, rewards = [], [], []
        state = np.random.rand(env['state_dim'])
        
        for t in range(10):  # Simulated episode length
            action = agent.sample_action(state)
            reward = np.random.randn()  # Random reward
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = np.random.rand(env['state_dim'])  # Next state
        
        agent.update(states, actions, rewards)
    
    print("Training complete!")

if __name__ == "__main__":
    main()
