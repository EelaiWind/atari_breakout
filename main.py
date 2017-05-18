import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
from collections import deque, namedtuple

from observation_processor import ObservationProcessor
from dqn import DQN
# Hyper Parameters:
GAMMA = 0.99                        # decay rate of past observations

# Epsilon
INITIAL_EPSILON = 1.0               # 0.01 # starting value of epsilon
FINAL_EPSILON = 0.1                 # 0.001 # final value of epsilon
EXPLORE_STPES = 500000              # frames over which to anneal epsilon

# replay memory
INIT_REPLAY_MEMORY_SIZE = 50000
REPLAY_MEMORY_SIZE = 300000

BATCH_SIZE = 32
FREQ_UPDATE_TARGET_Q = 10000        # Update target network every 10000 steps
TRAINING_EPISODES = 10000

MONITOR_PATH = 'breakout_videos/'

# Valid actions for breakout: ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
VALID_ACTIONS = [0, 1, 2, 3]

def main(_):
    # make game eviornment
    env = gym.envs.make("Breakout-v0")

    # Define Transition tuple
    Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    # The replay memory
    replay_memory = []

    # create a observation processor
    ob_proc = ObservationProcessor()

    # Behavior Network & Target Network
    behavior_Q = DQN()
    target_Q = DQN()

    # tensorflow session
    sess = tf.InteractiveSession()

    # Populate the replay buffer
    observation = env.reset()                       # retrive first env image
    observation = ob_proc.process(sess, observation)        # process the image
    state = np.stack([observation] * 4, axis=2)     # stack the image 4 times
    while len(replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        '''
        *** This part is just pseudo code ***

        action = None
        if random.random() <= epsilon
            action = random_action
        else
            action = DQN_action
        '''

        next_observation, reward, done, _ = env.step(VALID_ACTIONS[action])
        next_observation = ob_proc(next_observation)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
        replay_memory.append(Transition(state, action, reward, next_state, done))

        # Current game episode is over
        if done:
            observation = env.reset()
            observation = ob_proc(sess, observation)
            state = np.stack([observation] * 4, axis=2)

        # Not over yet
        else:
            state = next_state


    # record videos
    env = Monitor(env, directory=MONITOR_PATH, video_callable=lambda count: count % record_video_every == 0, resume=True)

    # total steps
    total_t = 0

    for episode in range(TRAINING_EPISODES):

        # Reset the environment
        observation = env.reset()
        observation = state_processor.process(sess, observation)
        state = np.stack([observation] * 4, axis=2)
        episode_reward = 0                              # store the episode reward
        '''
        How to update episode reward:
        next_observation, reward, done, _ = env.step(VALID_ACTIONS[action])
        episode_reward += reward
        '''

        for t in itertools.count():

            # choose a action

            # execute the action

            # if the size of replay buffer is too big, remove the oldest one. Hint: replay_memory.pop(0)

            # save the transition to replay buffer

            # sample a minibatch from replay buffer. Hint: samples = random.sample(replay_memory, batch_size)

            # calculate target Q values by target network

            # Update network

            # Update target network every FREQ_UPDATE_TARGET_Q steps
            # if total_t % FREQ_UPDATE_TARGET_Q == 0:
            #     update_target_network()

            if done:
                print ("Episode reward: ", episode_reward)
                break

            state = next_state
            total_t += 1


if __name__ == '__main__':
    tf.app.run()