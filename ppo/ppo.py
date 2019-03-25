import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from sonic_util import make_env
from agent import Agent
from collections import deque
import numpy as np

level_name='GreenHillZone.Act1'
env = make_env(level_name=level_name, \
                stack=False, scale_rew=True)
env.seed = 714

state_space = list(env.observation_space.shape)
action_space = env.action_space.n
print('State shape: ', state_space)
print('Number of actions: ', action_space)

BATCH_SIZE = 3000
EXPERIENCE_REPLAY = False
BUFFER_SIZE = int(9100)

agent = Agent(state_space, action_space, level_name=level_name, \
    param={
        'EXPERIENCE_REPLAY': EXPERIENCE_REPLAY,
        'BUFFER_SIZE': BUFFER_SIZE,
        'BATCH_SIZE': BATCH_SIZE
    })

def add_noise(state):
    row,col,ch= state.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = state + gauss
    return noisy

def ppo(agent, n_episodes=10000, max_t=4500, max_t_interval = 100):
    scores = []
    scores_window = deque(maxlen=max_t_interval)
    target_max_mean_score = 25

    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        negative_reward = 0
        for _ in range(max_t):
            state = state/255.0
            state = add_noise(state)
            action, action_took, actions_prob = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            # env.render()
            if reward < 0:
                negative_reward += reward
                reward = 0
            elif negative_reward < 0:
                negative_reward += reward
                reward = max(0, negative_reward)
                negative_reward = min(0, negative_reward)
            agent.step(state, action_took, actions_prob, reward)
            state = next_state
            score += reward
#            if agent.get_memory_size() >= BATCH_SIZE:
#                agent.learn(BATCH_SIZE, i_episode)
            if done:
                break
        score += negative_reward
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score

        agent.writer.add_scalar('Episode Reward', score, i_episode)

        if score > 55:
            print('\nThis Episode {}\tScore: {:.2f}'.format(i_episode, score), end="\n")
            agent.save_model()
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % max_t_interval == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            # torch.save(agent.qnetwork_local.state_dict(), latest_fn%(weight_fn, i_episode))
        if i_episode > 50 and np.mean(scores_window) >= target_max_mean_score:
            max_mean_score = np.mean(scores_window)
            print('\nEnvironment enhanced in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, max_mean_score))
            agent.save_model()
            target_max_mean_score += 1.5
            # break
        agent.compute_decay_reward()
        #if i_episode % 4 == 0:
        agent.learn(BATCH_SIZE, i_episode)
        if EXPERIENCE_REPLAY is True:
            agent.learn_from_buffer(BATCH_SIZE, i_episode)

ppo(agent)