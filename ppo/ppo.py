from sonic_util import make_env
from agent import Agent
from collections import deque
import numpy as np

env = make_env(stack=False, scale_rew=False)

state_space = list(env.observation_space.shape)
action_space = env.action_space.n
print('State shape: ', state_space)
print('Number of actions: ', action_space)

agent = Agent(state_space, action_space)

BATCH_SIZE = 64

def ppo(n_episodes=10000, max_t=4500, max_t_interval = 100):
    scores = []
    scores_window = deque(maxlen=max_t_interval)
    max_mean_score = 500


    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for _ in range(max_t):
            action, actions_prob = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        
        if score > 5000:
            print('\nThis Episode {}\tScore: {:.2f}'.format(i_episode, score), end="\n")
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % max_t_interval == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            # torch.save(agent.qnetwork_local.state_dict(), latest_fn%(weight_fn, i_episode))
        if np.mean(scores_window) >= max_mean_score+500:
            max_mean_score = np.mean(scores_window)
            print('\nEnvironment enhanced in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, max_mean_score))
            # torch.save(agent.qnetwork_local.state_dict(), best_weight_fn)
            # break

    agent.learn(BATCH_SIZE)

ppo()