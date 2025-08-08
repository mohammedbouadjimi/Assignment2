import gymnasium as gym
import aisd_examples
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

# Create the environment
env = gym.make("aisd_examples/CreateRedBall-v0", render_mode="human")
env = DummyVecEnv([lambda: env])  # Required by Stable-Baselines3

# Create the model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
timesteps = 10000
model.learn(total_timesteps=timesteps)

# Evaluate the model and store rewards
episode_rewards = []
n_eval_episodes = 10

for ep in range(n_eval_episodes):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated

    episode_rewards.append(total_reward)

# Close environment
env.close()

# Plotting the results
plt.plot(episode_rewards)
plt.title("PPO: Returns per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid()
plt.savefig("ppo_rewards.png")  # Saves plot to file
plt.show()
