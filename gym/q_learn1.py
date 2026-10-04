import gym
import numpy as np
env = gym.make('MountainCar-v0')

env.reset()

'''
Initially car is at (-0.5) in x axis. The max it can reach i.e. flag is 0.6 in x dirn and the least is -0.12 in x dirn    (-0.12 <----X----> 0.6)

observation space tells us the structure of the observations your environment will be returning

env.observation_space.high is a numpy array
'''

Discrete_os_size = [20] * len(env.observation_space.high)
# as they are continous values they would eat up a lot of space and would take light years to converge on cpu to avoid this problem we use discrete values in ranges. 

discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / Discrete_os_size
# discrete observation space window size 


q_table = np.random.uniform(low = -2 , high = 0 , size = (Discrete_os_size + [env.action_space.n])) 
# low and high need to tinkered and checked according to the env set

print(env.step(0))
'''
done = False
cnt =0
while not done:
    action = 1
    new_state, reward ,done,_ = env.step(action)
    print(new_state)
    env.render()
    
print(cnt)
env.close()
'''


