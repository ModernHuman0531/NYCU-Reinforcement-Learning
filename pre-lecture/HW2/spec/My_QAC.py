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
        # Return action logits(Means the unsoftmaxed output of the network, which can be used to create a probability distribution over actions)
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
# Create environment, networks, and optimizers

# Note: CartPole-v1 has a discrete action space with 2 actions (left, right)
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
        # Convert the data to tensor for GPU processing, ".unsqueeze(0)" adds a batch dimension so the shape becomes [1, state_dim]
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        # ----- Actor: sample action -----
        # logits are the raw outputs of the actor neural network, which reoresent the unnormalized scores fro each action.
        logits = actor(state_t)
        # dist is a categorical distribution created from the logits, which takes logits and applies a softmax to convert them into probabilities. The Categorical distribution allows us to sample discrete actions based on these probabilities.
        dist = torch.distributions.Categorical(logits=logits)
        # action is sampled from the policy distribution, which means we are selecting an action through policy dist.
        action = dist.sample()
        # log_prob is the log probability of the selected action.
        log_prob = dist.log_prob(action)

        # ----- Environment step -----
        # "env.step(action.item()) is to let the environment execute the selected action and return the next state, erward, and done flag."
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        episode_reward += reward

        # Convert next state and reward to tensors (from 1D array to 2D tensor with batch size of 1)
        next_state_t = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
        reward_t = torch.tensor([[reward]], dtype=torch.float32, device=device)

        # ----- Critic: Q(s,a) -----
        # q_values is the output of the critic network, which gives us the estimated Q-values for all actions in the current state.
        q_values = critic(state_t)                      # shape: [1, action_dim]
        # q_sa is the Q-value corresponding to the action that was actually taken. We use "gather" to select the Q-value for the specific action from the vector of Q-values.
        # And action.view(1, 1) means we are reshaping the action tensor to have a shape of [1, 1], which is necessary for the gather operation to work correctly.
        q_sa = q_values.gather(1, action.view(1, 1))   # shape: [1, 1]

        with torch.no_grad():
            # Expected next Q under current policy: sum_a pi(a|s') Q(s',a)
            next_logits = actor(next_state_t)
            next_probs = F.softmax(next_logits, dim=-1)         # [1, action_dim]
            next_q_values = critic(next_state_t)                # [1, action_dim]

            # expected_next_q is the expectation value of the next Q-values under the current policy. We multiply the probabilities of taking each action (next_probs) by the corresponding Q-values (next_q_values) and sum over all actions to get the expected Q-value for the next state.
            # It's equal to V(s') = sum_a pi(a|s') Q(s',a) in the policy evaluation step of QAC. This is used to compute the target Q-value for the critic update.
            expected_next_q = (next_probs * next_q_values).sum(dim=1, keepdim=True)

            # target_q is Bellman Expectation Equation for QAC. Becaues the next state is terminal, the target Q-value is just the reward. Otherwise, it's the reward plus the discounted expected next Q-value.
            target_q = reward_t if done else reward_t + GAMMA * expected_next_q

        # ----- Critic update -----
        # Use mean squared error loss to update the critic. We want the critic's estimate q_sa to be close to the target_q, which is the reward plus the discounted expected next Q-value.
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