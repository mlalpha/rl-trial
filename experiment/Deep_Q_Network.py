# import retro
# from retro_contest.local import make
from sonic_util import make_env
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tool import preprocess


# Import environment and get env infor
# env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', record=False)
# env = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1')
env = make_env(stack=False, scale_rew=False)
# env.seed(0)
state_space = list(env.observation_space.shape)
action_space = env.action_space.n
print('State shape: ', state_space)
print('Number of actions: ', action_space)


from dqn_agent import Agent
agent = Agent(state_size=state_space, action_size=action_space, seed=0)
weight_fn = 'step_checkpoint.pth'
latest_fn = 'latest_%s'%weight_fn

def dqn(n_episodes=10000, max_t=4500, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    max_t_interval = 250
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=max_t_interval)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    max_mean_score = 0
    # max_t_dict = [3000+(i)*200 for i in range(n_episodes//max_t_interval)]
    # print(max_t_dict)
    # max_t = max_t_dict[0]
    print('\nMax Step updated to: {:d}'.format(max_t))
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        state = state.reshape(state_space[2], state_space[0], state_space[1])
        score = 0
        # if i_episode%max_t_interval == 0:
        #     max_t = max_t_dict[i_episode//max_t_interval]
        #     print('\nMax Step updated to: {:d}'.format(max_t))
        for _ in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.reshape(state_space[2], state_space[0], state_space[1])
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % max_t_interval == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), latest_fn)
        if np.mean(scores_window) >= max_mean_score+500:
            max_mean_score = np.mean(scores_window)
            print('\nEnvironment enhanced in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, max_mean_score))
            torch.save(agent.qnetwork_local.state_dict(), weight_fn)
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

