# Reward engineering
The determine a good action and reinforce it may be important

The reward system will be divided into two parts:
1. exploring reward  
2. mistake avoid reward

This part cannot be saved with model before it has been encaptured into class

## exploring reward
Store states (output of CNN/VAE intermediate result) and use Cosine distance as reward, encourage actor try if new state of environment can been found. To encourage actor try more on new observation, K-Mean was introduced to keep reward for observing states not oftenly seen by actor.

## mistake avoid reward
- Punishment on mistake  
- Reward on avoid mistake  

Train RNN on last 100 states for game play fails  
RNN predict if actor will fail or not base on states  

```
reward_t = reward_(t-1) * 0.9999 + abs(reward_(t-1) - reward_system)
```

### RNN (GRU)
input = [sequence_of_states, reward_t]
output = predict possibility of fail

if RNN expected possible fail, then `reward_t -= possibility_of_fail * FAIL_PUNISH`

if actor did not fail, then `reward_t += possibility_of_fail * MAX_REWARD` or replace `MAX_REWARD` with `2 * mean_reward`?

training data: FAIL_PUNISH as a quadratic function, root from bad end (reward = FAIL_PUNISH) trace backward to the previous state where action != no_action (reward = 0)

https://github.com/keishinkickback/Pytorch-RNN-text-classification

https://morvanzhou.github.io/tutorials/machine-learning/torch/4-03-RNN-regression/

https://github.com/keishinkickback/Pytorch-RNN-text-classification/blob/master/model.py
