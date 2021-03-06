import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, CnnLnLstmPolicy, CnnPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2, A2C
from sonic_util import make_env
from gym.wrappers import Monitor


env = DummyVecEnv([lambda: make_env(level_name='GreenHillZone.Act1', \
                stack=False, scale_rew=True)])

modelname = 'sonicppo'
model = PPO2(CnnPolicy, env,n_steps=3500, verbose=1)
model.learn(total_timesteps=1000000)
model.save("./checkpoint" + modelname)