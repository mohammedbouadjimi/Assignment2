import gymnasium as gym
import aisd_examples
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("aisd_examples/CreateRedBall-v0")

configs = [
    {"alpha": 0.1, "gamma": 0.99, "epsilon": 0.1},
    {"alpha": 0.5, "gamma": 0.9,  "epsilon": 0.3},
    {"alpha": 0.9, "gamma": 0.8,  "epsilon": 0.05}
]

episodes = 100
max_steps = 100
all_returns = []

for idx, config in enumerate(configs):
    q_table = np.zeros((640, 640))
    returns = []

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0

        for _ in range(max_steps):
            if np.random.rand() < config["epsilon"]:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[obs])

            next_obs, reward, terminated, truncated, _ = env.step(action)

            q_table[obs, action] += config["alpha"] * (
                reward + config["gamma"] * np.max(q_table[next_obs]) - q_table[obs, action]
            )

            obs = next_obs
            total_reward += reward

            if terminated or truncated:
                break

        returns.append(total_reward)
        print(f"Config {idx+1}, Episode {ep+1}: Return = {total_reward:.2f}")

    all_returns.append(returns)

env.close()

# Plot comparison
for i, returns in enumerate(all_returns):
    plt.plot(returns, label=f"Config {i+1}")

plt.title("Q-Learning Variants Comparison")
plt.xlabel("Episode")
plt.ylabel("Total Return")
plt.legend()
plt.grid()
plt.savefig("qlearning_comparison.png")
plt.show()
