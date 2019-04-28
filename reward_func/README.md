# Reward engineering
The determine a good action and reinforce it may be important

The reward system will be divided into two parts:
1. exploring reward  
2. mistake avoid reward

This part cannot be saved with model before it has been encaptured into class

## exploring reward
Store states (output of CNN/VAE intermediate result) and use the mean Cosine distance of K nearest as reward, encourage actor to explore new states of the environment.

## mistake avoid reward
- Punishment on mistake  
- Reward on avoid mistake  

### RNN (GRU)
input = [state, reward_t]
output = predict_next_reward

the possibility of loss the game = `predict_next_reward - previous_prediction`

if RNN expected possible fail, then `reward_t += possibility_of_fail * FAIL_PUNISH`

if actor did not fail, then `reward_t += possibility_of_fail * MAX_REWARD`

#### training data: 
`possibility_of_fail = reward_at_next_frame`

`possibility_of_fail` is `slope`, `y` is the final reward (win/loss/foul) and `x` is step in time. For reawrd foul lead to a tangent at the action/step while win/loss lead to slope

https://github.com/keishinkickback/Pytorch-RNN-text-classification

https://morvanzhou.github.io/tutorials/machine-learning/torch/4-03-RNN-regression/

https://github.com/keishinkickback/Pytorch-RNN-text-classification/blob/master/model.py
