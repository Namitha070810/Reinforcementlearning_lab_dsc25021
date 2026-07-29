
# Task 1: Environment Setup

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


print("Task 1: Environment Setup")

print("Gymnasium Version :", gym.__version__)

# Task 2: Create and Initialize Environment

print("Task 2: Create and Initialize RL Environment")


env = gym.make("CartPole-v1")

observation, info = env.reset()

print("\nInitial Observation:")
print(observation)

print("\nEnvironment Information:")
print(info)

# Task 3: Observation & Action Spaces

print("Task 3: Observation and Action Spaces")

print("\nObservation Space:")
print(env.observation_space)

print("\nAction Space:")
print(env.action_space)

print("\nObservation Space Type:")
print(type(env.observation_space))

print("\nNumber of Possible Actions:")
print(env.action_space.n)

# Task 4: Execute Random Agent

print("Task 4: Random Agent Execution")

observation, info = env.reset()

done = False
step = 0
total_reward = 0

while not done:

    # Select random action
    action = env.action_space.sample()

    # Perform action
    observation, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    step += 1
    total_reward += reward

    print("Step :", step)
    print("Selected Action :", action)
    print("Observation :", observation)
    print("Reward :", reward)
    print("Episode Finished :", done)

print("Episode Summary")

print("Total Steps :", step)
print("Cumulative Reward :", total_reward)

env.close()