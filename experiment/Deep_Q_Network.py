# import retro
from retro_contest.local import make
from sonic_util import make_env
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from tool import preprocess


# Import environment and get env infor
# env = retro.make(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act1', record=False)
# env, multi_action = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1'), True
env, multi_action = make_env(stack=False, scale_rew=False), False

# env.seed(0)
state_space = list(env.observation_space.shape)
action_space = env.action_space.n
print('State shape: ', state_space)
print('Number of actions: ', [1, action_space])


from dqn_agent import Agent
agent = Agent(state_size=state_space, action_size=action_space, 
                seed=0, multi_action=multi_action, experience_replay=True)
weight_fn = 'checkpoint/discrete_explore_step'
latest_fn = '%s_epoch_%i.pth'
best_weight_fn = weight_fn+'.pth'

print('-----------Weight name: {}--------------'.format(weight_fn))

def dqn(n_episodes=10000, max_t=4500, eps_start=1.0, eps_end=0.1, eps_decay=0.999, max_t_interval = 100):
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
    scores_window = deque(maxlen=max_t_interval)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    max_mean_score = 1500
    print('\nMax Step updated to: {:d}'.format(max_t))
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        state = state.reshape(state_space[2], state_space[0], state_space[1])
        score = 0
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
            torch.save(agent.qnetwork_local.state_dict(), latest_fn%(weight_fn, i_episode))
        if np.mean(scores_window) >= max_mean_score+500:
            max_mean_score = np.mean(scores_window)
            print('\nEnvironment enhanced in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, max_mean_score))
            torch.save(agent.qnetwork_local.state_dict(), best_weight_fn)
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

