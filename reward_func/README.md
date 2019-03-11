# Reward engineering
The determine a good action and reinforce it may be important

The reward system will be divided into two parts:
1. exploring reward  
2. mistake avoid reward

This part cannot be saved with model before it has been encaptured into class

## exploring reward
Store states (output of CNN/VAE intermediate result) and use Cosine distance as reward, encourage actor try if new state of environment can been found.

## mistake avoid reward
- Punishment on mistake  
- Reward on avoid mistake  

Train RNN on last 100 states for game play fails  
RNN predict if actor will fail or not base on states  

```
reward_t = reward_(t-1) * 0.9 + abs(reward_(t-1) - reward_system)
```

### RNN (GRU)
input = [sequence_of_states, reward_t]
output = predict possibility of fail

if RNN expected possible fail, then `reward_t -= possibility_of_fail + FAIL_PUNISH`

if actor did not fail, then `reward_t += possibility_of_fail * MAX_REWARD` or replace `MAX_REWARD` with `2 * mean_reward`?

training data: FAIL_PUNISH linear from bad end backward to last action != no_action
