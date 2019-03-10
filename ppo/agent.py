import numpy as np
from actor import Actor
from critic import Critic
from tensorboardX import SummaryWriter

class Agent():
    
    def __init__(self, state_size, action_size, param={}):
        self.state_size = state_size
        self.action_size = action_size
        self.dummy_adv = np.zeros((1, 1))
        self.dummy_actions_prob = np.zeros((1, action_size))
        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)
        self.name = 'ppo/'
        weight_fn = 'ppo'
        self.best_weight_fn = weight_fn+'.pth'
        self.writer = SummaryWriter('logs/')
        self.memory = [[], [], [], []]
        self.update_count = 0
        self.cur_ind = 0
        self.GAMMA = 0.99

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
        self.memory = [[], [], [], []]

    def step(self, state, action_took, actions_prob, reward):
        self.memory[0].append(state)
        self.memory[1].append(action_took)
        self.memory[2].append(actions_prob)
        self.memory[3].append(reward)

    def act(self, state):
        actions_prob = self.actor.model.predict(
            [
                state.reshape(1, 84, 84, 1),
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
        for t in range(memory_size - 2, -1, -1):
            # timestep t
            # reward = r(t) + \sum_{t'} 
            self.memory[3][t] = self.memory[3][t] + self.memory[3][t+1] * self.GAMMA

    def learn(self, batch_size, i_epoch):
        """
            batch: state, action, actions_prob, reward
        """
        self.compute_decay_reward()

        while self.get_memory_size() != 0:
            state, action_took, old_actions_prob, reward, batch_size = self.get_batch(batch_size)
            if batch_size == 0:
                break
            advantage = self.critic.model.predict(state)
            advantage = reward - advantage
            actor_loss = self.actor.model.fit(
                [
                    state, advantage, old_actions_prob
                ], [action_took], 
                batch_size=batch_size, shuffle=True, epochs=i_epoch, verbose=False)
            critic_loss = self.critic.model.fit([state], [reward], 
                batch_size=batch_size, shuffle=True, epochs=i_epoch, verbose=False)
            self.writer.add_scalar('Actor loss', actor_loss.history['loss'][-1], self.update_count)
            self.writer.add_scalar('Critic loss', critic_loss.history['loss'][-1], self.update_count)
            
            self.update_count += 1


        
    