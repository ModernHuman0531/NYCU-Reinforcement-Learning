#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple implementation of QAC provided by CGPT
"""
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# =========================
# Hyperparameters
# =========================
ENV_NAME = "CartPole-v1"
GAMMA = 0.99
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
HIDDEN_DIM = 128
NUM_EPISODES = 500
MAX_STEPS = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Networks
# =========================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        # Return action logits
        return self.net(x)


class CriticQ(nn.Module):
    """
    Q(s, a) for all discrete actions.
    Input: state
    Output: vector of Q-values, one per action
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Setup
# =========================
env = gym.make(ENV_NAME)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

actor = Actor(state_dim, action_dim, HIDDEN_DIM).to(device)
critic = CriticQ(state_dim, action_dim, HIDDEN_DIM).to(device)

actor_optimizer = optim.Adam(actor.parameters(), lr=ACTOR_LR)
critic_optimizer = optim.Adam(critic.parameters(), lr=CRITIC_LR)


# =========================
# Training loop
# =========================
for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    episode_reward = 0.0

    for step in range(MAX_STEPS):
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        # ----- Actor: sample action -----
        logits = actor(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # ----- Environment step -----
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        episode_reward += reward

        next_state_t = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
        reward_t = torch.tensor([[reward]], dtype=torch.float32, device=device)

        # ----- Critic: Q(s,a) -----
        q_values = critic(state_t)                      # shape: [1, action_dim]
        q_sa = q_values.gather(1, action.view(1, 1))   # shape: [1, 1]

        with torch.no_grad():
            # Expected next Q under current policy: sum_a pi(a|s') Q(s',a)
            next_logits = actor(next_state_t)
            next_probs = F.softmax(next_logits, dim=-1)         # [1, action_dim]
            next_q_values = critic(next_state_t)                # [1, action_dim]

            expected_next_q = (next_probs * next_q_values).sum(dim=1, keepdim=True)

            target_q = reward_t if done else reward_t + GAMMA * expected_next_q

        # ----- Critic update -----
        critic_loss = F.mse_loss(q_sa, target_q)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # ----- Actor update -----
        # Use critic's estimate Q(s,a) as the policy weight.
        # Detach so actor update does not backprop through critic.
        actor_loss = -(log_prob * q_sa.detach().squeeze())

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        state = next_state

        if done:
            break

    if episode % 20 == 0:
        print(f"Episode {episode:4d} | reward = {episode_reward:6.1f}")

env.close()