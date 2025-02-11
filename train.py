from sac_and_cql import SAC
import gym
import matplotlib.pyplot as plt
import torch 
import numpy as np
from math import atan2


import os
model_path = os.path.join(os.path.dirname(__file__), 'best_model.pt')

epochs = 100

evaluate_steps = 1

agent = SAC()
batchsize = 256

env = gym.make("Pendulum-v1", render_mode='human')
rewards = []
reward_best = -np.inf
for i in range(epochs):
    agent.train()
    agent.update(batchsize)

    if i % evaluate_steps == 0:
        agent.eval()
        with torch.no_grad():
            print(f"Evaluating at iter={i}")
            obs, _ = env.reset()
            obs_torch = torch.tensor(obs).float().unsqueeze(0)
            done = False
            reward_total = 0
            while not done:
                action = agent.get_mean(obs_torch)
                obs, reward, terminated, truncated, _ = env.step(action.numpy()[0])
                done = terminated or truncated
                reward_total += reward
                obs_torch = torch.tensor(obs).float().unsqueeze(0)

            rewards.append(reward_total)
            if reward_total > reward_best:
                reward_best = reward_total
                torch.save(agent, model_path)



plt.plot(np.arange(0, len(rewards), 1), rewards)
plt.savefig(os.path.join(os.path.dirname(__file__), 'reward_evaluation.png'))


