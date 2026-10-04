import gym
from matplotlib.pyplot import get, table
import numpy as np

env = gym.make('MountainCar-v0')
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 6000  

Discrete_os_size = [20] * len(env.observation_space.high)
# as they are continous values they would eat up a lot of space and would take light years to converge on cpu to avoid this problem we use discrete values in ranges. 

discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / Discrete_os_size


epsilon = 0.5
start_epsilon_decaying = 1
end_epsilon_decaying = EPISODES//2

epsilon_decay_value = epsilon / (end_epsilon_decaying-start_epsilon_decaying)


q_table = np.random.uniform(low = -2 , high = 0 , size = (Discrete_os_size + [env.action_space.n])) 
# low and high need to tinkered and checked according to the env set

def get_discrete_state(state):
    discrete_state = (state- env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    if episode % 200 == 0:
        print("episode is ",episode)
        render = True 
    else:
        render = False
        
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        if(np.random.random()) > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)
        
        # print(action)
        new_state, reward ,done,_ = env.step(action)
        new_discrete_action = get_discrete_state(new_state)
        # print("new ",new_discrete_action)
        if render:
            env.render()
        
        if not done:
            max_future_q = np.max(q_table[new_discrete_action]) #for updation in formula
            current_q = q_table[discrete_state+ (action,)]
            
            # formula for getting q value
            new_q = (1-LEARNING_RATE)*current_q + LEARNING_RATE*(reward + DISCOUNT * max_future_q)
            
            q_table[discrete_state + (action,)] = new_q
        
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0
        
        discrete_state = new_discrete_action  #here we update the state   
    
    if end_epsilon_decaying >= episode >= start_epsilon_decaying:
        epsilon -= epsilon_decay_value
             
env.close()

