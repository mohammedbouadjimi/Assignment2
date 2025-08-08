import gymnasium as gym
import aisd_examples
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("aisd_examples/CreateRedBall-v0")

# Q-table: shape [num_observations, num_actions]
q_table = np.zeros((640, 640))

# Hyperparameters
alpha = 0.1        # learning rate
gamma = 0.99       # discount factor
epsilon = 0.1      # exploration rate
episodes = 100     # number of episodes
max_steps = 100    # steps per episode

returns = []       # to track total rewards

for ep in range(episodes):
    obs, _ = env.reset()
    total_reward = 0

    for _ in range(max_steps):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[obs])

        next_obs, reward, terminated, truncated, _ = env.step(action)

        # Q-value update
        q_table[obs, action] += alpha * (reward + gamma * np.max(q_table[next_obs]) - q_table[obs, action])

        obs = next_obs
        total_reward += reward

        if terminated or truncated:
            break

    returns.append(total_reward)
    print(f"Episode {ep + 1}: Return = {total_reward:.2f}")

env.close()

# Plot the learning curve
plt.plot(returns)
plt.title("Original Hyperparameters")
plt.xlabel("Episode")
plt.ylabel("Total Return")
plt.grid()
plt.savefig("qlearning_returns.png")
plt.show()
