#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: pinghsieh
"""

import sys
import gymnasium as gym
import numpy as np
import os
import random
import time
from collections import namedtuple
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import wandb
from tqdm import tqdm

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

Transition = namedtuple(
    'Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

# Add device configuration for potential GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.8, sigma=1.0):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state * self.scale    

class SeedWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, seed: int):
        super().__init__(env)
        self._seed = seed
        self._used = False  # only apply fixed seed on the very first reset

    def reset(self, seed=None):
        if seed is not None:
            return self.env.reset(seed=seed)
        if not self._used:
            self._used = True
            return self.env.reset(seed=self._seed)
        return self.env.reset()  # subsequent resets: random starts

class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # Define hidden layers sizes
        self.hidden_size1 = 400
        self.hidden_size2 = 300

        # Define max action value for scaling the output of the actor network
        self.max_action = 1.0 # For Cheetah-v5

        ########## YOUR CODE HERE (5~10 lines) ##########
        # Construct your own actor network

        """
        Create a neural network for the actor. Use nn.Seqential to stack layers.
        The structure is:
        num_inputs -> 400 -> 300 -> num_outputs
        - Use ReLU activation for the hidden layers and Tanh for the output layer, because the action space
        is typically bounded between -2 and 2 for Pendlum-v1.
        - Model output the determinstic action with the input state, and the output action should be scaled by 2 to match the action space of Pendulum-v1.
        - Put the scaling part in the foward function is prefered.
        """
        self.actor_model = nn.Sequential(
            nn.Linear(num_inputs, self.hidden_size1),
            nn.ReLU(),
            nn.Linear(self.hidden_size1, self.hidden_size2),
            nn.ReLU(),
            nn.Linear(self.hidden_size2, num_outputs),
            nn.Tanh()
        ).to(device)
        ########## END OF YOUR CODE ##########

    def forward(self, inputs):
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your actor network
        return self.max_action * self.actor_model(inputs) # Because the action space is typically bounded between -2 and 2 for Pendlum-v1, but tanh outputs between -1 and 1, so we scale it by 2.
        
        ########## END OF YOUR CODE ##########

class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # Define hidden layers sizes
        self.hidden_size1 = 400
        self.hidden_size2 = 300

        ########## YOUR CODE HERE (5~10 lines) ##########
        """
        Construct your own critic network that takes both state and action as input and output is the Q-value of the state-action pair.
        For simple problem like Pendulum-v1, we can concatenate the state and action dimensions and feed them into fully connected layers.
        The structure is:
        (num_inputs + num_outputs) -> 400 -> 300 -> num_outputs
        - Use ReLU activation for the hidden layers and no activation for the output layer, since the output is a numeric Q-value that can be unbounded. 
        """
        self.critic_model = nn.Sequential(
            nn.Linear(num_inputs + num_outputs, self.hidden_size1),
            nn.ReLU(),
            nn.Linear(self.hidden_size1, self.hidden_size2),
            nn.ReLU(),
            nn.Linear(self.hidden_size2, 1) # Output is a single Q-value, but in HalfCheeth-v5, action space is 6-dim. So the output is 1-dim, not num_outputs-dim.
        ).to(device)
        ########## END OF YOUR CODE ##########

    def forward(self, inputs, actions):
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define the forward pass your critic network

        # Use torch.cat to concatenate the state and action tensors along the input_feature dimension (dim=1)
        # Since the action shape is (1,) and state shape is (3,), we need to concatenate them along the feature dimension, which is dim=1. 
        # Usually the input is [batch_size, input_feature_dim], so we concatenate along dim=1 to get [batch_size, input_feature_dim + action_dim]
        x = torch.cat([inputs, actions],dim=1)
        return self.critic_model(x)
        
        
        
        ########## END OF YOUR CODE ##########  

class DDPG(object):
    def __init__(self, num_inputs, action_space, gamma=0.99, tau=0.001, hidden_size=256, lr_a=3e-4, lr_c=1e-3):

        self.num_inputs = num_inputs
        self.action_space = action_space

        self.actor = Actor(hidden_size, self.num_inputs, self.action_space).to(device)
        self.actor_target = Actor(hidden_size, self.num_inputs, self.action_space).to(device)
        self.actor_perturbed = Actor(hidden_size, self.num_inputs, self.action_space).to(device)
        self.actor_optim = Adam(self.actor.parameters(), lr=lr_a)

        self.critic = Critic(hidden_size, self.num_inputs, self.action_space).to(device)
        self.critic_target = Critic(hidden_size, self.num_inputs, self.action_space).to(device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr_c)

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)


    def select_action(self, state, action_noise=None):
        self.actor.eval()
        # Ensure state is a float tensor with shape [1, obs_dim]
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(device)
        with torch.no_grad():
            mu = self.actor(state)          # [1, act_dim]
            mu = mu.squeeze(0)             # [act_dim]
 
        ########## YOUR CODE HERE (3~5 lines) ##########
        # Add noise to your action for exploration
        # Clipping might be needed (Use torch clamp to clip the action to the valid range of the action space, in this case is between -2 and 2 for Pendulum-v1)

        if action_noise is not None:
            # Add OU noise to the action(mu) for exploration, action_noise is OUNoise object
            noise = action_noise.noise() # np array, but mu is a torch tensor
            noise = torch.FloatTensor(noise).to(device) # Convert noise to torch tensor to add to mu
            # Add noise to the action
            mu += noise
            # Clip the action to the valid range of the action space, in this case is between -2 and 2
            clip_mu = torch.clamp(mu, min=-1.0, max=1.0)

            # Return the action as a numpy array, and put it back to cpu if it's on GPU
            return clip_mu.cpu().numpy()

        return mu.cpu().numpy()
    

        ########## END OF YOUR CODE ##########

    def update_parameters(self, batch):

        batch = Transition(*zip(*batch))
        state_batch = Variable(torch.cat(batch.state).to(device))
        action_batch = Variable(torch.cat(batch.action).to(device))
        reward_batch = Variable(torch.cat(batch.reward).to(device))
        mask_batch = Variable(torch.cat(batch.mask).to(device))
        next_state_batch = Variable(torch.cat(batch.next_state).to(device))
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)
        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)

        ########## YOUR CODE HERE (10~20 lines) ##########
        # Calculate policy loss and value loss
        # Update the actor and the critic 

        # Part 1. Update critic network
        # Compute the target Q value using the reward and the discounted next state action value
        target_Q = reward_batch + (mask_batch * self.gamma * next_state_action_values)
        # Get current Q estimate from critic network using current states and actions
        current_Q = self.critic(state_batch, action_batch)
        # Compute critic loss as the MSE between target Q and current Q, since we only want to update
        # the critic network, we detach the target_Q from the computation graph to prevent gradients from flowing back to the target networks
        value_loss = F.mse_loss(current_Q, target_Q.detach())

        # Optimize the critic network
        self.critic_optim.zero_grad() # Clear the gradients of the critic optimizer
        value_loss.backward()         # Backpropagate the critic loss
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 2.0) # Clip the gradients for stability, max norm is 2.0
        self.critic_optim.step()     # Update the critic network parameters

        # Part 2. Update actor network
        # Compute the actor loss as the negative mean of the critic's Q value
        # Because we want to maximize the Q value, we take the negative to minimize the loss
        policy_loss = -self.critic(state_batch, self.actor(state_batch)).mean()

        # Optimize the actor network
        self.actor_optim.zero_grad() # Clear the gradients of the actor optimizer
        policy_loss.backward()       # Backpropagate the actor loss
        self.actor_optim.step()      # Update the actor network parameters
        
        ########## END OF YOUR CODE ##########         
        
        # Update the target networks here with soft update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()


    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('preTrained_b/'):
            os.makedirs('preTrained_b/')
 
        if actor_path is None:
            actor_path = "preTrained_b/ddpg_cheetah_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "preTrained_b/ddpg_cheetah_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        return actor_path, critic_path
 
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None: 
            self.critic.load_state_dict(torch.load(critic_path))

def train():    
    num_episodes = 500        # 500k steps budget
    max_steps = 500000        # Max steps for HalfCheetah-v5
    gamma = 0.99
    tau = 0.005               # Target tracking, tau=0.005, best performance is 0.003
    hidden_size = 256         # network hidden size
    noise_scale_start = 0.3   # exploration noise at episode 0, bigger start for cheetah since it's more complex than pendulum
    noise_scale_final = 0.05  # anneal down to this by the last episode
    replay_size = 1000000     # buffer size 
    batch_size = 512          # batch size, 256
    lr_a = 2e-4               # actor lr for stability, 3e-4
    lr_c = 3e-4               # critic lr, 3e-4
    updates_per_step = 2      # gradient steps per env step
    warmup_steps = 10000      # collect experience before any gradient updates
    print_freq = 1
    save_freq = 50
    ewma_reward = 0
    rewards = []
    ewma_reward_history = []
    total_numsteps = 0
    updates = 0

    # ────────── W&B: initialize run ───────────────────────────────────────────────
    wandb.init(
        project="ddpg",
        name=f"{env_name}_seed{random_seed}",
        config={
            "env":                env_name,
            "seed":               random_seed,
            "num_episodes":       num_episodes,
            "gamma":              gamma,
            "tau":                tau,
            "hidden_size":        hidden_size,
            "noise_scale_start":  noise_scale_start,
            "noise_scale_final":  noise_scale_final,
            "replay_size":        replay_size,
            "batch_size":         batch_size,
            "lr_actor":           lr_a,
            "lr_critic":          lr_c,
            "warmup_steps":       warmup_steps,
            "updates_per_step":   updates_per_step,
        },
    )
    # ──────────────────────────────────────────────────────────────────────────────────

    agent = DDPG(env.observation_space.shape[0], env.action_space, gamma, tau, hidden_size, lr_a=lr_a, lr_c=lr_c)
    ounoise = OUNoise(env.action_space.shape[0])
    # ounoise = OUNoise(env.action_space.shape[0], scale=0.2, mu=0, theta=0.15, sigma=0.2)
    memory = ReplayMemory(replay_size)

    # ──────────── W&B: watch actor & critic to log gradients + weights automatically ──
    wandb.watch(agent.actor,  log="all", log_freq=100, idx=0)
    wandb.watch(agent.critic, log="all", log_freq=100, idx=1)
    # ──────────────────────────────────────────────────────────────────────────────────

    i_episode = 0 
    # for i_episode in tqdm(range(num_episodes)):
    while total_numsteps < max_steps:   
        # i_episode is only used for logging and saving models, the actual loop terminated condition is total_numsteps < max_steps to ensure we don't exceed 
        # the total steps budget.
        i_episode += 1

        # Linearly anneal exploration noise: high early, low late
        frac = i_episode / max(num_episodes - 1, 1)
        noise_scale = noise_scale_start + frac * (noise_scale_final - noise_scale_start)
        ounoise.scale = noise_scale
        ounoise.reset()
        
        state, info = env.reset()
        state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)  # convert immediately

        episode_reward = 0
        episode_value_loss  = 0.0   # W&B: accumulate losses for the episode
        episode_policy_loss = 0.0
        episode_updates     = 0

        while True:
            ########## YOUR CODE HERE (15~30 lines) ##########
            # 1. Interact with the env to get new (s,a,r,s') samples 
            # 2. Push the sample to the replay buffer
            # 3. Update the actor and the critic
            
            # Get action from the agent, add exploration noise
            action = agent.select_action(state, ounoise)
            # Step the environment to interact and get next_state, reward, done
            next_state, reward, terminated, truncated, _ = env.step(action)
            # Convert next_state to tensor following the same format as state
            next_state = torch.FloatTensor(np.array(next_state)).unsqueeze(0).to(device)
            done = terminated or truncated
            episode_reward += reward
            # Update total num steps
            total_numsteps += 1

            # Store the transition in the replay buffer using transition namedtuple
            mask = 0 if done else 1 # mask = 1-done
            # Convert the state, action, reward, next_state, mask to torch tensors and push to replay buffer
            # Because we use torch.cat in line 226, the input must be tensors
            # Note: Transition is ('state', 'action', 'mask', 'next_state', 'reward'), so order matters
            memory.push(
                state.to(device),
                torch.FloatTensor(action).unsqueeze(0).to(device), # Unsequeeze action to make it [1, act_dim] for torch.cat in Critic forward pass to match the state shape [1, obs_dim]
                torch.FloatTensor([mask]).to(device),
                next_state.to(device),
                torch.FloatTensor([reward]).to(device)
            )
            # Update state to next_state for the next step
            state = next_state

            # Update the actor and critic network if we collect enough samples for a batch, and we are past the warmup steps
            if memory.__len__() >= batch_size and total_numsteps >= warmup_steps:
                # Sample a batch from the replay buffer
                batch = memory.sample(batch_size)
                # Update the actor and critic network using batch data set, and get the value loss and policy loss for logging
                value_loss, policy_loss = agent.update_parameters(batch)
                # Update the episode loss and update count for logging
                episode_value_loss += value_loss
                episode_policy_loss += policy_loss
                episode_updates += 1
            # End the loop if the episode is done
            if done or total_numsteps >= max_steps:
                break

            ########## END OF YOUR CODE ########## 
      

        rewards.append(episode_reward)
        t = 0
        if i_episode % print_freq == 0:
            state, _ = env.reset()
            state = torch.FloatTensor(np.array(state)).unsqueeze(0).to(device)

            episode_reward = 0
            while True:
                action = agent.select_action(state)

                next_state, reward, terminated, truncated, _  = env.step(action)
                done = terminated or truncated
                #env.render()
                
                episode_reward += reward

                next_state = torch.FloatTensor(np.array(next_state)).unsqueeze(0).to(device)
                state = next_state
                
                t += 1
                if done:
                    break

            rewards.append(episode_reward)
            # update EWMA reward and log the results
            ewma_reward = 0.05 * episode_reward + (1 - 0.05) * ewma_reward
            ewma_reward_history.append(ewma_reward)        
            print("Episode: {}, length: {}, reward: {:.2f}, ewma reward: {:.2f}".format(i_episode, t, rewards[-1], ewma_reward))

            # ────────── W&B: log training metrics every print_freq episode ──────────────
            log_dict = {
                "episode":              i_episode,
                "train/total_steps":    total_numsteps,
                "train/total_updates":  updates,
                "train/ewma_reward":    ewma_reward,
                "train/noise_scale":    noise_scale,
                "replay_buffer/size":   len(memory),
            }
            # Average losses over the updates performed this episode
            if episode_updates > 0:
                log_dict["train/value_loss"]  = episode_value_loss  / episode_updates
                log_dict["train/policy_loss"] = episode_policy_loss / episode_updates
            wandb.log(log_dict, step=i_episode)
            # ────────────────────────────────────────────────────────────────────────────
    
        if (i_episode+1) % save_freq == 0 or ewma_reward > 5000:
            filename = 'ep{}_score{:.1f}.pth'.format(i_episode, ewma_reward)
            actor_path, critic_path = agent.save_model(env_name, filename)
            mean_eval_reward = test(actor_path, critic_path)

            # ──────────── W&B: log evaluation score at save_freq intervals ────────────
            wandb.log({
                "episode":              i_episode,
                "eval/mean_reward":     mean_eval_reward,
                "eval/ewma_reward":     ewma_reward,
            }, step=i_episode)
            # ──────────── W&B: save model checkpoints as artifacts ────────────────────
            artifact = wandb.Artifact(
                name=f"ddpg-{env_name}-checkpoint",
                type="model",
                description=f"Saved at episode {i_episode}, ewma_reward={ewma_reward:.1f}",
            )
            artifact.add_dir("preTrained_b/")
            wandb.log_artifact(artifact)
            # ──────────────────────────────────────────────────────────────────────────

    wandb.finish()   # W&B: cleanly close the run
 
def test(actor_path, critic_path, hidden_size=256, n_episodes=20):
    '''
        Test the learned model (no change needed)
    '''      
    test_env = gym.make(env_name)
    model = DDPG(test_env.observation_space.shape[0], test_env.action_space, hidden_size=hidden_size)
    
    model.load_model(actor_path, critic_path)
    
    render = False
    eval_reward_history = []
 
    for i_episode in range(1, n_episodes+1):
        state, info = test_env.reset()
        running_reward = 0
        t = 0
        while True:
            action = model.select_action(state)
            next_state, reward, terminated, truncated, _  = test_env.step(action)
            done = terminated or truncated
            running_reward += reward
            next_state = torch.FloatTensor(np.array(next_state)).unsqueeze(0)
            state = next_state
            t += 1
            if render:
                 test_env.render()
            if done:
                eval_reward_history.append(running_reward)
                print("Eval Episode: {}, length: {}, reward: {:.2f}".format(i_episode, t, running_reward))
                break
 
    mean_reward = np.mean(eval_reward_history)
    print('Number of Eval Episodes: {}\t; Evaluation Reward: {}'.format(n_episodes, mean_reward))
    test_env.close()
    return mean_reward

def set_seed(env, seed):
    # For reproducibility, fix the random seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    env = SeedWrapper(env=env,seed=random_seed)
    env.action_space.seed(seed)


if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 42
    # env_name = 'Pendulum-v1'
    env_name = 'HalfCheetah-v5'
    env = gym.make(env_name)
    set_seed(env,seed=random_seed)
    # train()
    test(actor_path='preTrained_b/ddpg_cheetah_actor_HalfCheetah-v5_ep499_score4863.5.pth', critic_path='preTrained_b/ddpg_cheetah_critic_HalfCheetah-v5_ep499_score4863.5.pth')