import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import tensorflow as tf
# import gym
from sonic_util import make_env
from collections import deque
from memory import ReplayBuffer
import datetime

# np.random.seed(2)
# tf.set_random_seed(2)  # reproducible

# Hyper Parameters
OUTPUT_GRAPH = True
MAX_EPISODE = 6000
DISPLAY_REWARD_THRESHOLD = 3000  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 4500  # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9  # reward discount in TD error
LR_A = 0.000001  # learning rate for actor
LR_C = 0.0001  # learning rate for critic
BUFFER_SIZE = 10000
BATCH_SIZE = 16
UPDATE_EVERY = 50
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)

env = make_env(stack=False, scale_rew=False)
# env.seed(1)  # reproducible
# env = env.unwrapped

state_space = list(env.observation_space.shape)
action_space = env.action_space.n

print('State shape: ', state_space)
print('Number of actions: ', [1, action_space])


def reshape_state(s):
    s = s[np.newaxis, :]
    return s


class Actor(object):
    def __init__(self, sess, state_size, action_size, lr=0.001):
        self.sess = sess

        input_shape = [None, 84, 84, 1]

        self.s = tf.placeholder(tf.float32, input_shape, "state")
        self.a = tf.placeholder(tf.int32, [None, 1], "act")
        self.flatten_act = tf.placeholder(tf.int32, [None], "flatten_act")
        self.td_error = tf.placeholder(tf.float32, [None, 1], "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            conv1 = tf.layers.conv2d(
                inputs=self.s,
                filters=20,
                kernel_size=[2, 2],
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                name='conv1')

            pool1 = tf.layers.average_pooling2d(
                inputs=conv1,
                pool_size=[2, 2],
                strides=1,
                padding='same',
                name='avg_pool1'
            )

            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=20,
                kernel_size=[4, 4],
                padding="same",
                activation=tf.nn.relu,
                name='conv2')

            pool2 = tf.layers.average_pooling2d(
                inputs=conv2,
                pool_size=[2, 2],
                strides=1,
                padding='same',
                name='avg_pool2'
            )

            flattened_vector = tf.layers.flatten(pool2)

            fc1 = tf.layers.dense(
                inputs=flattened_vector,
                units=64,  # number of hidden units
                activation=tf.nn.softmax,
                name='fc1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=fc1,
                units=action_size,  # output units
                activation=tf.nn.softmax,  # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0.1, .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='actions_prob'
            )
            # tf.summary.scalar('max_act_prob', tf.reduce_max(self.acts_prob))
            # for i in range(action_size):
            #     tf.summary.scalar(name='act_%i_prob'%i, tensor=self.acts_prob[i])

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(tf.gather(self.acts_prob, self.flatten_act, axis = 1))
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('actor_train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)


    def learn(self, s, a, td):
        # s = s[np.newaxis, :]
        flatten_act = a.flatten()
        feed_dict = {self.s: s, self.a: a, self.flatten_act: flatten_act, self.td_error: td}
        _, exp_v, _ = self.sess.run(
            [
                self.train_op, self.exp_v, self.acts_prob
            ], feed_dict)
        # print('action', a)
        # print('flatten_act', flatten_act)
        # print('td error', td)
        # print('exp_v', exp_v)
        # print('acts_prob', acts_prob)
        # print('*'*30)
        
        return exp_v

    def choose_action(self, s):
        # s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})  # get probabilities for all actions
        

        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())  # return a int


class Critic(object):
    def __init__(self, sess, state_size, lr=0.01):
        self.sess = sess

        input_shape = [None, 84, 84, 1]

        self.s = tf.placeholder(tf.float32, input_shape, "state")
        self.v_ = tf.placeholder(tf.float32, [None, 1], "v_next")
        self.r = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Critic'):
            conv1 = tf.layers.conv2d(
                inputs=self.s,
                filters=20,
                kernel_size=[2, 2],
                strides=1,
                padding="same",
                activation=tf.nn.relu,
                name='conv1')

            pool1 = tf.layers.average_pooling2d(
                inputs=conv1,
                pool_size=[2, 2],
                strides=1,
                padding='same',
                name='avg_pool1'
            )

            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=20,
                kernel_size=[4, 4],
                padding="same",
                activation=tf.nn.relu,
                name='conv2')

            pool2 = tf.layers.average_pooling2d(
                inputs=conv2,
                pool_size=[2, 2],
                strides=1,
                padding='same',
                name='avg_pool2'
            )

            flattened_vector = tf.layers.flatten(pool2)

            fc1 = tf.layers.dense(
                inputs=flattened_vector,
                units=64,  # number of hidden units
                activation=tf.nn.softmax,
                name='fc1'
            )

            self.v = tf.layers.dense(
                inputs=fc1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .3),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )
        
            tf.summary.scalar('mean_q_value', tf.reduce_mean(self.v))

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            tf.summary.scalar('mean_td_error', tf.reduce_mean(self.td_error))
            self.loss = tf.square(self.td_error)  # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('critic_train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

        self.merged = tf.summary.merge_all()

    def learn(self, s, r, s_):
        # s, s_ = s[np.newaxis, :], s_[np.newaxis, :]


        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _, summary = self.sess.run(
            [
                self.td_error, self.train_op, self.merged
            ],
            {self.s: s, self.v_: v_, self.r: r})


        return td_error, summary


sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



actor = Actor(sess, state_size=state_space, action_size=action_space, lr=LR_A)
critic = Critic(sess, state_size=state_space,
                lr=LR_C)  # we need a good teacher, so the teacher should learn faster than the actor

timestampe = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
writer = tf.summary.FileWriter("./logs/%s"%timestampe, sess.graph)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

max_t_interval = 100
scores = []  # list containing scores from each episode
scores_window = deque(maxlen=max_t_interval)  # last 100 scores

memory = ReplayBuffer(action_space, BUFFER_SIZE, BATCH_SIZE, 714)




# RENDER = False
max_mean_score = 0
total_timestep = 0
for i_episode in range(1, MAX_EPISODE + 1):
    state = env.reset()
    timestep = 1
    track_r = []
    while True:
        # if RENDER: 
        #     env.render()

        action = actor.choose_action(reshape_state(state))

        next_state, reward, done, info = env.step(action)

        memory.add(state, action, reward, next_state, done)
        track_r.append(reward)

        state = next_state
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if timestep % UPDATE_EVERY == 0:
            if len(memory) > BATCH_SIZE:
                experiences = memory.sample()
                td_error, summary = critic.learn(experiences[0], experiences[2],
                                        experiences[3])  # gradient = grad[r + gamma * V(s_) - V(s)]
                writer.add_summary(summary, total_timestep)                     
                _ = actor.learn(experiences[0], experiences[1], td_error)  # true_gradient = grad[logPi(s,a) * td_error]
                # writer.add_summary(summary, total_timestep)
                    

        timestep += 1
        total_timestep += 1

        if done or timestep > MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)
            # if ep_rs_sum > 1500:
            #     RENDER = True
            scores_window.append(ep_rs_sum)  # save most recent score
            scores.append(ep_rs_sum)  # save most recent score
            if i_episode % max_t_interval == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            if len(scores_window) > max_t_interval and np.mean(scores_window) >= max_mean_score + 500:
                max_mean_score = np.mean(scores_window)
                print('\nEnvironment enhanced in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, max_mean_score))
                save_path = saver.save(sess, "./model/model.ckpt")
                print("Model saved in path: %s with reward: %f" % (save_path, ep_rs_sum))

            break