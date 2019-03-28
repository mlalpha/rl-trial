import numpy as np
from actor import Actor
from critic import Critic
from tensorboardX import SummaryWriter
import datetime
from replay_buffer import ReplayBuffer

class Agent():
    
    def __init__(self, state_size, action_size, param={}, level_name='general'):
        self.seed = 714
        np.random.seed(seed=self.seed)
        self.state_size = state_size
        self.action_size = action_size
        self.dummy_adv = np.zeros((1, 1))
        self.dummy_actions_prob = np.zeros((1, action_size))
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)
        self.level_name = level_name
        timestampe = datetime.datetime.now().strftime("%Y_%m_%d_%H%M")
        self.writer = SummaryWriter('logs/%s/%s'%(self.level_name, timestampe), write_graph=True)
        self.best_weight_fn = 'ppo_best_%s.h5'
        self.memory = [[], [], [], []]
        self.update_count = 0
        self.cur_ind = 0
        self.GAMMA = 0.99

        self.EXPERIENCE_REPLAY = param.get('EXPERIENCE_REPLAY', False)
        if self.EXPERIENCE_REPLAY is True:
            self.BUFFER_SIZE = param['BUFFER_SIZE'] 
            self.BATCH_SIZE = param['BATCH_SIZE']
            self.buffer = ReplayBuffer(self.action_size, self.BUFFER_SIZE, self.BATCH_SIZE, self.seed)

    def get_memory_size(self):
        return len(self.memory[0])

    def get_batch(self, batch_size):
        start, end = self.cur_ind, self.cur_ind + batch_size
        self.cur_ind += batch_size
        if end >= self.get_memory_size():
            end = self.get_memory_size()
            self.cur_ind = 0
            batch_size = end - start
        state = np.array(self.memory[0][start:end])
        action_took = np.array(self.memory[1][start:end])
        old_actions_prob = np.array(self.memory[2][start:end])
        reward = np.array(self.memory[3][start:end]).reshape(batch_size, 1)

        if self.cur_ind == 0:
            self.reset_memory()
        
        return state, action_took, old_actions_prob, reward, batch_size
        
    def reset_memory(self):
        del self.memory
        self.memory = [[], [], [], []]

    def step(self, state, action_took, actions_prob, reward):
        self.memory[0].append(state)
        self.memory[1].append(action_took)
        self.memory[2].append(actions_prob)
        self.memory[3].append(reward)

    def act(self, state, test=False):
        actions_prob = None
        if test is True:
            actions_prob = self.actor.model.predict(
                np.array([state])
            )
        else:
            actions_prob = self.actor.model.predict(
                [
                    np.array([state]),
                    self.dummy_adv,
                    self.dummy_actions_prob
                ]
            )
        action = np.random.choice(self.action_size, p=np.nan_to_num(actions_prob[0]))
        action_took = np.zeros(self.action_size)
        action_took[action] = 1
        return action, action_took, actions_prob[0]

    def compute_decay_reward(self):
        memory_size = self.get_memory_size()

        self.memory[0] = self.memory[0]
        self.memory[1] = np.array(self.memory[1])
        self.memory[2] = np.array(self.memory[2])
        self.memory[3] = np.array(self.memory[3])

        for t in range(memory_size - 2, -1, -1):
            # timestep t
            # reward = r(t) + \sum_{t'} 
            self.memory[3][t] = self.memory[3][t] + self.memory[3][t+1] * self.GAMMA

        if self.EXPERIENCE_REPLAY is True:
            self.buffer.adds(self.memory[0], self.memory[1], self.memory[2], self.memory[3])
    

    def learn(self, batch_size, i_epoch):
        """
            batch: state, action, actions_prob, reward
        """

        while self.get_memory_size() != 0:
            state, action_took, old_actions_prob, reward, batch_size = self.get_batch(batch_size)
            if batch_size == 0:
                break
            advantage = self.critic.model.predict(state)
            advantage = reward - advantage
            actor_loss = self.actor.model.fit(
                [
                    state, advantage, old_actions_prob
                ],
                [action_took],
                batch_size=batch_size, shuffle=True, epochs=i_epoch, verbose=False)
            critic_loss = self.critic.model.fit([state], [reward], 
                batch_size=batch_size, shuffle=True, epochs=i_epoch, verbose=False)
            self.writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.update_count)
            self.writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.update_count)
            
            self.update_count += 1

    def learn_from_buffer(self, batch_size, i_epoch):
        """
            batch: state, action, actions_prob, reward
        """

        if self.EXPERIENCE_REPLAY is True and len(self.buffer) > batch_size:
            experiences = self.buffer.sample()
            state, action_took, reward, old_actions_prob = experiences
            advantage = self.critic.model.predict(state)
            advantage = reward - advantage
            actor_loss = self.actor.model.fit(
                [
                    state, advantage, old_actions_prob
                ],
                [action_took],
                batch_size=batch_size, shuffle=True, epochs=i_epoch, verbose=False)
            critic_loss = self.critic.model.fit([state], [reward], 
                batch_size=batch_size, shuffle=True, epochs=i_epoch, verbose=False)
            self.writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.update_count)
            self.writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.update_count)
            
            self.update_count += 1


    def save_model(self, name = None):
        if name is None:
            actor_name = self.best_weight_fn%'actor'
            critic_name = self.best_weight_fn%'critic'
        else:
            actor_name = name + 'actor' + '.h5'
            critic_name = name + 'critic' + '.h5'

        self.actor.save_model(actor_name)
        self.critic.save_model(critic_name)    

    def load_model(self, actor_model_fn, critic_model_fn):
        self.actor.load_model(actor_model_fn)
        self.critic.load_model(critic_model_fn)    
    