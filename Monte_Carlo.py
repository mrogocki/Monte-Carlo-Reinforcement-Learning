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

