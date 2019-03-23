import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, CnnLnLstmPolicy, CnnPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2, A2C
from sonic_util import make_env
from gym.wrappers import Monitor


env = DummyVecEnv([lambda: make_env(level_name='LabyrinthZone.Act1', \
                stack=False, scale_rew=True)])

modelname = 'sonicppo'
model = PPO2(CnnPolicy, env,n_steps=4500, verbose=1)
model.load("./checkpoint" + modelname)

obs = env.reset()
done = False
reward = 0

while not done:
    actions, _ = model.predict(obs)
    obs, rew, done, info = env.step(actions)
    reward += rew
    env.render()
env.close()