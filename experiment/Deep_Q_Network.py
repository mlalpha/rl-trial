
# coding: utf-8

# # Deep Q-Network (DQN)
# ---
# In this notebook, you will implement a DQN agent with OpenAI Gym's LunarLander-v2 environment.
# 
# ### 1. Import the Necessary Packages

# In[1]:

import retro
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tool import preprocess
# get_ipython().magic('matplotlib inline')


# ### 2. Instantiate the Environment and Agent
# 
# Initialize the environment in the code cell below.

# In[2]:

env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', record=False)


# In[3]:

env.seed(0)
state_space = list(env.observation_space.shape)
state_space[2] = 1
action_space = env.action_space.n
print('State shape: ', state_space)
print('Number of actions: ', (1, action_space))


# Please refer to the instructions in `Deep_Q_Network.ipynb` if you would like to write your own DQN agent.  Otherwise, run the code cell below to load the solution files.

# In[4]:

from dqn_agent import Agent

agent = Agent(state_size=state_space, action_size=action_space, seed=0)



# ### 3. Train the Agent with DQN
# 
# Run the code cell below to train the agent from scratch.  You are welcome to amend the supplied values of the parameters in the function, to try to see if you can get better performance!
# 
# Alternatively, you can skip to the next step below (**4. Watch a Smart Agent!**), to load the saved model weights from a pre-trained agent.

# In[ ]:

def dqn(n_episodes=10000, max_t=1000, eps_start=1.0, eps_end=0.0001, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    max_score = 0
    max_t_interval = 250
    max_t_dict = [150+(i+1)*150 for i in range(n_episodes//max_t_interval)]
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        state = preprocess(state).reshape(state_space[2], state_space[0], state_space[1])
        score = 0
        max_t = max_t_dict[i_episode//1000]
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess(next_state).reshape(state_space[2], state_space[0], state_space[1])
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=max_score+0.5:
            max_score = np.mean(scores_window)
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            # break

    return scores

scores = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


# ### 4. Watch a Smart Agent!
# 
# In the next code cell, you will load the trained weights from file to watch a smart agent!

# In[ ]:

# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

for i in range(10):
    state = env.reset()
    for j in range(200):
        state = preprocess(state).reshape(state_space[2], state_space[0], state_space[1])
        action = agent.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            break 
            
env.close()


# ### 5. Explore
# 
# In this exercise, you have implemented a DQN agent and demonstrated how to use it to solve an OpenAI Gym environment.  To continue your learning, you are encouraged to complete any (or all!) of the following tasks:
# - Amend the various hyperparameters and network architecture to see if you can get your agent to solve the environment faster.  Once you build intuition for the hyperparameters that work well with this environment, try solving a different OpenAI Gym task with discrete actions!
# - You may like to implement some improvements such as prioritized experience replay, Double DQN, or Dueling DQN! 
# - Write a blog post explaining the intuition behind the DQN algorithm and demonstrating how to use it to solve an RL environment of your choosing.  

# In[ ]:



