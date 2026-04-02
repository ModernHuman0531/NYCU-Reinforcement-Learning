# Spring 2026, 535510 Reinforcement Learning
# HW1: REINFORCE with baseline and GAE

import os
import gymnasium as gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler
from torch.utils.tensorboard import SummaryWriter

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

# Define a tensorboard writer
writer = SummaryWriter("./tb_record_1")
        
class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the GAE parameters, shared layer(s), the action layer(s), and the value layer(s))
            2. Random weight initialization of each layer
    """
    def __init__(self):
        super(Policy, self).__init__()
        
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        # In Cart-Pole problem, the observation space 4-dim vector(Cart Position, Cart Velocity, Pole Angle, Pole Angular Velocity)
        self.observation_dim = env.observation_space.shape[0]
        # In Cart-Pole problem, action_dim is discrete and dim is 2(left and right)
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size_1 = 128
        self.hidden_size_2 = 64

        
        ########## YOUR CODE HERE (5~10 lines) ##########
        # Define a simple MLP to implement actor and value networks.(4 -> 128 -> 64 -> Each output)

        # Define a share layer for actor and value networks
        self.fc1 = nn.Linear(self.observation_dim, self.hidden_size_1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(self.hidden_size_1, self.hidden_size_2)
        self.relu2 = nn.ReLU()

        # Define layer for action and value output respectively, action_dim is 2, value_dim is 1(current state's value function)
        self.value = nn.Linear(self.hidden_size_2, 1)
        self.actor = nn.Linear(self.hidden_size_2, self.action_dim)

        # Randomize the actor and value layer to avoid them from too confindent of specific action. (Other layer are already nornalize in nn.Linear)
        nn.init.normal_(self.actor.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.value.weight, mean=0.0, std=0.01)
        ########## END OF YOUR CODE ##########
        
        # action & reward memory
        self.saved_actions = []
        self.rewards = []
        
        # self.double() make weight from float32 to float64, so the input state must turn into double type tensor
        self.double()

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding 
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########
        # First, get share layer's output
        x = self.fc1(state)
        x = self.relu1(x)
        x = self.fc2(x) 
        x = self.relu2(x)

        # Then get output of each action and value's layer
        action_prob, state_value = self.actor(x), self.value(x)


        ########## END OF YOUR CODE ##########

        return action_prob, state_value


    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########
        # Because of self.double() for higher precise, we must turn state into double type tensor
        state = torch.from_numpy(state).double().unsqueeze(0)
        # Get the value and prob distribution
        action_prob, state_value = self.forward(state)
        # Based on the stochastic policy, it's like a weighted box to pick action by probability distribution
        m = Categorical(logits=action_prob)
        # Select an action from the box(m)
        action = m.sample()
        ########## END OF YOUR CODE ##########
        
        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()


    def calculate_loss(self, gamma=0.99):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
            TODO:
                1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
                2. Calculate the policy loss using the policy gradient
                3. Calculate the value loss using either MSE loss or smooth L1 loss
        """
        
        # Initialize the lists and variables
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] 
        value_losses = [] 
        returns = []

        ########## YOUR CODE HERE (8-15 lines) ##########
        # Use reverse function is helpful for calculate the Return
        # First calculate the Return and save it into retruns list
        for reward in reversed(self.rewards):
            R = reward + R * gamma
            # Insert into the first place
            returns.insert(0, R)
        # Normalize the return
        returns = torch.tensor(returns).double()
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for i, saved_action in enumerate(saved_actions):
            # Calculate the Vanilla REINFORCE pg, policy_loss = -log_prob * Return
            R = returns[i]
            policy_loss = -R * saved_action.log_prob
            policy_losses.append(policy_loss)

            # Turn R into 2D tensor for MSE calculation
            R_tensor = R.unsqueeze(0).unsqueeze(0)
            # Calculate value loss, because we use Return as estimator to estimate value function
            # So we take MSE function on Value function and Return
            value_loss = F.mse_loss(saved_action.value, R_tensor.detach())
            value_losses.append(value_loss)
        
        # Calculate the loss = value_loss + policy_loss (list stored tensor -> 1D tensor use stack function)
        loss = torch.stack(value_losses).sum() + torch.stack(policy_losses).sum()
            
        ########## END OF YOUR CODE ##########
        
        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

class GAE:
    def __init__(self, gamma, lambda_, num_steps):
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_steps = num_steps          # set num_steps = None to adapt full batch

    def __call__(self, rewards, values, done):
        """
            Implement Generalized Advantage Estimation (GAE) for your value prediction
            TODO (1): Pass correct corresponding inputs (rewards, values, and done) into the function arguments
            TODO (2): Calculate the Generalized Advantage Estimation and return the obtained value
        """

        ########## YOUR CODE HERE (8-15 lines) ##########



        
        ########## END OF YOUR CODE ##########

def train(lr=0.01):
    """
        Train the model using SGD (via backpropagation)
        TODO (1): In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode

        TODO (2): In each episode, 
        1. record all the value you aim to visualize on tensorboard (lr, reward, length, ...)
    """
    
    # Instantiate the policy model and the optimizer
    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    # EWMA reward for tracking the learning progress
    ewma_reward = 0
    
    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state, _ = env.reset()
        ep_reward = 0
        t = 0

        # Uncomment the following line to use learning rate scheduler
        scheduler.step()
        
        # For each episode, only run 9999 steps to avoid entering infinite loop during the learning process
        
        ########## YOUR CODE HERE (10-15 lines) ##########
        for t in range(9999):
            """
            Sampling the trajectory by running the policy in the environment for max_episode_len steps or until the episode ends (whichever comes first)
            1. Select an action from the policy network
            2. Interact with the environment using the selected action and get the next state and reward
            3. Save the reward obtained from the environment into model.rewards
            4. If the terminated or truncated signal is True, break the loop and start to update the policy and value network
            """
            # Select an action
            action = model.select_action(state)
            # Interract with environment
            new_state, reward, terminated, truncated, _ = env.step(action)
            # Save the reward and update the state
            state = new_state
            model.rewards.append(reward)
            # Update the episode reeard
            ep_reward += reward

            # Determine whether the episode ends
            if terminated or truncated:
                break
        # After the epsisode ends, calculate the loss and perform backpropagation to update the parameters of both the policy and value network
        optimizer.zero_grad()
        # Calculate the loss
        loss = model.calculate_loss()
        # Backward pass to calculate loss gradient to each parameter
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name}: grad mean = {param.grad.mean().item():.6f}")
        # Use optimizer to update the parameters based on the calculated gradients
        optimizer.step()
        # Clear the memory of rewards and action buffer for the next episode
        model.clear_memory()
        
        
        ########## END OF YOUR CODE ##########
            
        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        #Try to use Tensorboard to record the behavior of your implementation 
        ########## YOUR CODE HERE (4-5 lines) ##########
        # Record the learning rate, episode reward and episode length into tensorboard for visualization
        writer.add_scalar('lr/episode', lr, i_episode)
        writer.add_scalar('reward/episode', ep_reward, i_episode)
        writer.add_scalar('length/episode', t, i_episode)

        ########## END OF YOUR CODE ##########

        # check if we have "solved" the cart pole problem, use 120 as the threshold in LunarLander-v2
        if ewma_reward > env.spec.reward_threshold:
            if not os.path.isdir("./preTrained"):
                os.mkdir("./preTrained")
            torch.save(model.state_dict(), './preTrained/CartPole_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break


def test(name, env_name, n_episodes=10):
    """
        Test the learned model (no change needed)
    """     
    model = Policy()
    
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    env = gym.make(env_name, render_mode='human')
    max_episode_len = 10000
    
    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, terminations, truncations, _ = env.step(action)
            done = np.logical_or(terminations, truncations)
            running_reward += reward
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    

if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 10  
    lr = 0.001
    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    obs, _ = env.reset(seed=random_seed)
    torch.manual_seed(random_seed)  
    train(lr)
    test(f'CartPole_{lr}.pth', env_name)
