from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, CnnLnLstmPolicy, CnnPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines import PPO2, A2C
from sonic_util import make_env


env = DummyVecEnv([lambda: make_env(level_name='LabyrinthZone.Act1', \
                stack=False, scale_rew=True, skip_neg_rew=True)])

modelname = 'sonicppo'
model = PPO2(CnnPolicy,env,n_steps=2048, verbose=1)
model.learn(total_timesteps=10000)
model.save("./checkpoint" + modelname)