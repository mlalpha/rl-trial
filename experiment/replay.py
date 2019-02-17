import retro
from retro_contest.local import make
from sonic_util import make_env
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from gym.wrappers import Monitor

def recording(recording_env, recording_agent, weight_name):
    env = Monitor(recording_env, './video_%s'%(weight_name), force=True)
    # watch an trained agent
    window = []
    n_epsiode = 5
    for i_episode in range(n_epsiode):
        # add Recording tigger
        state = env.reset()
        total_score = 0
        for j in range(MAX_STEP):
            state = state.reshape(state_space[2], state_space[0], state_space[1])
            action = recording_agent.act(state)[0]
    #         env.render()
            state, reward, done, _ = env.step(action)
            total_score += reward
            if done:
                break
        window.append(total_score)
        print('Total score for this episode {:.4f}'.format(total_score))
    print('Avg score {}'.format(np.mean(window)))


def checkpoint_generator(epoch, weight_fn = 'checkpoint/discrete_explore_step'):
    latest_fn = '%s_epoch_%i.pth'
    best_weight_fn = weight_fn+'.pth'
    return latest_fn%(weight_fn, epoch)


def build_env():
    #env, multi_action = make(game='SonicTheHedgehog-Genesis', state='LabyrinthZone.Act1'), True
    env, multi_action = make_env(stack=False, scale_rew=False), False
    return env, multi_action

env, multi_action = build_env()

# for normal record speed


state_space = list(env.observation_space.shape)
action_space = env.action_space.n
print('State shape: ', state_space)
print('Number of actions: ', [1, action_space])

MAX_STEP = 4500


from dqn_agent import Agent
agent = Agent(state_size=state_space, action_size=action_space, 
                seed=0, multi_action=multi_action, experience_replay=True)


for i in range(100, 10001, 100):
    wn = checkpoint_generator(i)
    agent.qnetwork_local.load_state_dict(torch.load(wn))
    recording(env, agent, 'agent_%i'%i)
    env.close()
    env, _ = build_env()



# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint/discrete_explore_step.pth'))
recording(env, agent, 'agent_best')
env.close()