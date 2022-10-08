import gym
import numpy as np


env_name = 'Pendulum-v0'
env = gym.make(env_name)

env.reset()
print(env.observation_space)
print(env.action_space)


done = False

while not done:
    action = 0
    new_state, reward ,done,_ = env.step(action ,6)
    print(new_state)
    env.render()
    
env.close()
    
 