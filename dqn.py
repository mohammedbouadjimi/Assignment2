import gymnasium as gym
import aisd_examples
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt

# Create the environment
env = gym.make("aisd_examples/CreateRedBall-v0", render_mode="human")
env = DummyVecEnv([lambda: env])  # DQN also needs VecEnv wrapper

# Initialize the model
model = DQN("MlpPolicy", env, verbose=1)

# Train the model
timesteps = 10000
model.learn(total_timesteps=timesteps)

# Evaluate the model
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

# Close the environment
env.close()

# Plotting results
plt.plot(episode_rewards)
plt.title("DQN: Returns per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid()
plt.savefig("dqn_rewards.png")
plt.show()
