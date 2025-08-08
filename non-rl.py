import gymnasium as gym
import aisd_examples
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("aisd_examples/CreateRedBall-v0", render_mode="human")
obs, info = env.reset()

returns = []
steps_per_episode = []
episodes = 10  # or however many you want

for episode in range(episodes):
    total_reward = 0
    steps = 0
    obs, info = env.reset()

    for _ in range(100):  # Each episode = 100 steps
        # Simple rule: if red ball is left of center, turn left; if right, turn right
        redball_x = obs  # This is the pixel position (0-640)
        
        if redball_x < 300:
            action = 280  # turn left
        elif redball_x > 340:
            action = 360  # turn right
        else:
            action = 320  # stay

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    returns.append(total_reward)
    steps_per_episode.append(steps)
    print(f"Episode {episode+1}: Return = {total_reward}, Steps = {steps}")

env.close()

# Plotting
plt.plot(returns, label="Returns")
plt.plot(steps_per_episode, label="Steps")
plt.xlabel("Episode")
plt.ylabel("Score")
plt.title("Rule-Based Agent Performance")
plt.legend()
plt.grid(True)
plt.savefig("rule_based_agent_returns.png")
plt.show()
