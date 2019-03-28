import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from sonic_util import make_env
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from gym.wrappers import Monitor

def recording(recording_env, recording_agent, weight_name):
    env = Monitor(recording_env, './video_%s'%(weight_name), force=True)
    # watch an trained agent
    window = []
    n_epsiode = 10
    for _ in range(n_epsiode):
        # add Recording tigger
        state = env.reset()
        total_score = 0
        for j in range(MAX_STEP):
            state = state/255.0
            action, _, _ = recording_agent.act(state, test=True)
            env.render()
            state, reward, done, _ = env.step(action)
            total_score += reward
            if done:
                break
        window.append(total_score)
        print('Total score for this episode {:.4f}'.format(total_score))
    print('Avg score {}'.format(np.mean(window)))


# def checkpoint_generator(epoch, weight_fn = 'checkpoint/discrete_explore_step'):
#     latest_fn = '%s_epoch_%i.pth'
#     best_weight_fn = weight_fn+'.pth'
#     return latest_fn%(weight_fn, epoch)


def build_env(level_name='LabyrinthZone.Act1'):
    env = make_env(stack=False, scale_rew=True, level_name=level_name)
    return env

# LabyrinthZone.Act1
# GreenHillZone.Act1
# GreenHillZone.Act2
env = build_env(level_name='GreenHillZone.Act2')

state_space = list(env.observation_space.shape)
action_space = env.action_space.n
print('State shape: ', state_space)
print('Number of actions: ', [1, action_space])

MAX_STEP = 4500


from agent import Agent
agent = Agent(state_space, action_space)
agent.load_model(actor_model_fn='ppo_best_actor.h5', critic_model_fn='ppo_best_critic.h5')


# for i in range(100, 10001, 100):
#     wn = checkpoint_generator(i)
#     agent.qnetwork_local.load_state_dict(torch.load(wn))
#     recording(env, agent, 'agent_%i'%i)
#     env.close()
#     env, _ = build_env()


# load the weights from file
recording(env, agent, 'agent_best')
env.close()