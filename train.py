from sac_and_cql import SAC
import gym
import matplotlib.pyplot as plt
import torch 
import numpy as np
from math import atan2
import matplotlib
matplotlib.use('TKAgg')

import os
model_path = os.path.join(os.path.dirname(__file__), 'best_model.pt')

iternum = 5000

evaluate_steps = 100

agent = SAC()
batchsize = 1000

env = gym.make("Pendulum-v1")
rewards = []
reward_best = -np.inf
for i in range(iternum):
    agent.train()
    agent.update(batchsize)

    if i%evaluate_steps == 0:
        agent.eval()
        with torch.no_grad():
            print(f"Evaluating at iter={i}")
            state = env.reset()
            state = np.array([[np.float32(atan2(state[0][0], state[0][1])), np.float32(state[0][2])]])
            state = torch.tensor(state)
            done = False
            reward_total = 0
            while not done:            
                action = agent.get_mean(state)
                nextstate, reward, terminated, truncated, _ = env.step(action.numpy()[0]) 
                done = terminated or truncated
                reward_total += reward
                state = nextstate
                state = np.array([[np.float32(atan2(state[0], state[1])), np.float32(state[2])]])
                state = torch.tensor(state)
            rewards.append(reward_total)
            if reward_total > reward_best:
                reward_best = reward_total
                torch.save(agent, model_path)



plt.plot(np.arange(0, len(rewards), 1), rewards)
plt.savefig(os.path.join(os.path.dirname(__file__), 'reward_evaluation.png'))


