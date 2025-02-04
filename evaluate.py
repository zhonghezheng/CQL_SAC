import gymnasium as gym
from sac_and_cql import SAC
import torch 
import numpy as np 
from math import atan2

import os
model_path = os.path.join(os.path.dirname(__file__), 'best_model.pt')


# Parallel environments
env = gym.make('Pendulum-v1', g=9.81, render_mode="human")


model = SAC() 
model = torch.load(model_path, weights_only=False)

state = env.reset()
state = state[0]
done = False
while not done:
    print(state)
    state = np.array([[np.float32(atan2(state[0], state[1])), np.float32(state[2])]])
    state = torch.tensor(state)
    action = model.get_action(state)
    state, reward, terminated, truncated, _ = env.step(action.detach().numpy()[0]) 
    done = terminated or truncated

env.close()
