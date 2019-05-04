# Reward engineering
The determine a good action and reinforce it may be important

The reward system will be divided into two parts:
1. Exploring reward  
2. Suprise reward

This part cannot be saved with model before it has been encaptured into class

The reward will be prepared after a game finished and the states replay for once to produce reward for offline learning.

`reward = Exploring reward + Suprise reward + End game reward`

`Suprise reward = End game reward - Predicted reward`

## Exploring reward
Store states (output of CNN/VAE intermediate result) and use the mean Cosine distance of K nearest as reward, encourage actor to explore new states of the environment.

## Suprise reward
(Also include mistake avoid reward by its nature)
- Punishment on mistake (made expected mistake -> more negative reward)  
- Reward on avoid mistake (expect a mistake but game goes on -> minus negative reward)  

### Predicted reward (RNN/GRU)
input = [state, reward_t]
output = predict_next_reward

https://morvanzhou.github.io/tutorials/machine-learning/torch/4-03-RNN-regression/
