import gym
from model import *
import numpy as np


if __name__ == '__main__':
    env = gym.make('SpaceInvaders-v0')
    brain = Agent(gamma=0.95, epsilon=1.0,
                  lr=0.003, maxMemorySize=5000,
                  replace=None)
    
    while brain.memCntr < brain.memSize:
        observation = env.reset()
        done=False
        while not done:
            # 0 - no action  1 - fire  2 - move right
            # 3 - move left  4 - move right fire  5 - move left fire
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            if done and info['ale.lives'] == 0:
                reward = -100
            brain.storeTransition(np.mean(observation[15:200, 30:125], axis=2),
                                  action,
                                  reward,
                                  np.mean(observation_[15:200, 30:125], axis=2))
            observation = observation_
    print('Done initializing memory')
    
    scores = []
    eps_hist = []
    numGames = 50
    batch_size = 32

    for i in range(numGames):
        print(f'Start game {i+1} with epsilon {brain.EPSILON}')
        eps_hist.append(brain.EPSILON)
        done = False
        observation = env.reset()
        frames = [np.sum(observation[15:200, 30:125], axis=2)]
        score = 0
        last_action = 0

        while not done:
            if len(frames) == 3:
                action = brain.chooseAction(frames)
                frames = []
            else:
                action = last_action
            
            observation_, reward, done, info = env.step(action)
            score += reward
            frames.append(np.sum(observation_[15:200, 30:125], axis=2))

            if done and info['ale.lives'] == 0:
                reward = -100
            brain.storeTransition(np.mean(observation[15:200, 30:125], axis=2),
                                  action,
                                  reward,
                                  np.mean(observation_[15:200, 30:125], axis=2))
            observation = observation_
            brain.learn(batch_size)
            last_action = action
        scores.append(score)
        print(f'Score: {score}')

