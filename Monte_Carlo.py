######## EXAMPLE IMPLEMENTATION OF AN REINFORCMENT MONTE CARLO PREDICTION AND CONTROL

#AUTHOR: Michael Rogocki

import sys
import gym
import numpy as np
from collections import defaultdict
from plot_utils import plot_blackjack_values, plot_policy


# 1. Create Open AI gym blackjack environment
env = gym.make("Blackjack-v1")

# Examine env
print(env.action_space)
print(env.observation_space)

# 2. Run example games and print outcome

for episodes in range(3):
    state = env.reset()

    # Collect all steps to print out the complete episode as list
    episode = []
    
    while True:
        # Select one action based on the discrete action space of 2
        action = env.action_space.sample()

        # Receive next state, reward, terminal info of the step made
        state, reward, done, info, _ = env.step(action)
        episode.append([state, reward, done])
        # If episode in terminal state return info about outcome
        if done:
            print("End of the blackjack game")
            print("You won!") if reward > 0 else print("You lost!")
            print(f"Complete episode is: {episode}")
            break

####### Generate episode with specific stochastic episode

def generate_stochastic_episode(env):
    state = env.reset()
    episode = []
    
    while True:
        # If card more than 17, than say STICK with 70%, otherwise 30%
        prob_pos = 0.7
        probs = [prob_pos, 1 - prob_pos] if state[0] > 17 else [1 - prob_pos, prob_pos]
        
        # Select action based on probability
        action = np.random.choice(np.arange(2), p=probs)
        next_state, rewards, done, info, _ = env.step(action)
        episode.append([next_state, action, rewards])

        # Move to next state
        state = next_state
        
        # Terminate if done
        if done:
            break

    return episode


####### Generate a Monte Carlo Prediction
# Implementation of a first visit policy prediction

def mc_prediction(env, number_episodes, generate_episode, gamma=1):
    # Initialize visit count
    N = defaultdict(lambda: np.zeros(env.action_space.n))
    # Initialize return sums for all state action pairs
    return_sums = defaultdict(lambda: np.zeros(env.action_space.n))
    # Initialize Q table
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    for index_episode in range(number_episodes+1):
        episode = generate_episode(env)
        # Unpack list
        states, actions, rewards = zip(*episode)
        # Discount the reward with gamma, default = 1
        discount = [gamma ** i for i in range(len(rewards)+1)]
        for index, state in enumerate(states):
            # Update the Q table if first visit
            return_sums[state][actions[index]] += sum(rewards[index:]*discount[:-(1+index)])
            N[state][actions[index]] += 1.0
            Q[state][actions[index]] = return_sums[state][actions[index]]

