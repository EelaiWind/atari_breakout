import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf
from collections import deque, namedtuple
from datetime import datetime

from observation_processor import ObservationProcessor
from dqn import DQN
# Hyper Parameters:
GAMMA = 0.99                        # decay rate of past observations

# Epsilon
INITIAL_EPSILON = 1.0               # 0.01 # starting value of epsilon
FINAL_EPSILON = 0.1                 # 0.001 # final value of epsilon
EXPLORE_STPES = 1000000              # frames over which to anneal epsilon

# replay memory
INIT_REPLAY_MEMORY_SIZE = 50000
REPLAY_MEMORY_SIZE = 300000

BATCH_SIZE = 32
FREQ_UPDATE_TARGET_Q = 10000        # Update target network every 10000 steps
TRAINING_EPISODES = 10000

PREFIX = 'log'
MONITOR_PATH = os.path.join(PREFIX, 'videos/')
TENSORNOARD_PATH = os.path.join(PREFIX, 'tensorboard/')
CHECKPOINT_PATH = os.path.join(PREFIX, 'checkpoint')
RECORD_VIDEO_EVERY = 500

# Valid actions for breakout: ['NOOP', 'FIRE', 'RIGHT', 'LEFT']
ACTION_SPACE = 4

TERMINATE_REWARD = -10

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
    behavior_Q = DQN("behavior_network", ACTION_SPACE, save_directory=CHECKPOINT_PATH)
    target_Q = DQN("target_network", ACTION_SPACE, save_directory=None)

    # tensorflow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    summary_writer = tf.summary.FileWriter(TENSORNOARD_PATH, sess.graph)
    sess.run(tf.global_variables_initializer())

    # Populate the replay buffer
    observation = env.reset()                       # retrive first env image
    observation = ob_proc.process(sess, observation)        # process the image
    state = np.stack([observation] * 4, axis=2)     # stack the image 4 times
    while len(replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        action = env.action_space.sample()

        next_observation, reward, done, _ = env.step(action)
        if done: reward = TERMINATE_REWARD
        next_observation = ob_proc.process(sess, next_observation)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
        replay_memory.append(Transition(state, action, reward, next_state, int(done)))

        if done:
            observation = env.reset()
            observation = ob_proc.process(sess, observation)
            state = np.stack([observation] * 4, axis=2)
        else:
            state = next_state


    # record videos
    env = Monitor(env, directory=MONITOR_PATH, video_callable=lambda episode: episode % RECORD_VIDEO_EVERY == 0, resume=True)

    # total steps
    total_iteration = 0
    epsilon = INITIAL_EPSILON
    max_episode_reward = -999
    soft_train_loss = 0
    soft_episode_reward = 0
    soft_decay = 0.9
    for episode in xrange(TRAINING_EPISODES):

        # Reset the environment
        observation = env.reset()
        observation = ob_proc.process(sess, observation)
        state = np.stack([observation] * 4, axis=2)

        episode_reward = 0                              # store the episode reward
        for frame in itertools.count():
            # choose a action
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_feature = np.expand_dims(state, axis=0)
                q_value = behavior_Q.forward(sess, state_feature)
                action = np.argmax(q_value[0])

            # execute the action
            next_observation, reward, done, _ = env.step(action)
            episode_reward += reward
            if done: reward = TERMINATE_REWARD

            # if the size of replay buffer is too big, remove the oldest one. Hint: replay_memory.pop(0)
            if len(replay_memory) == REPLAY_MEMORY_SIZE: replay_memory.pop(0)

            # save the transition to replay buffer
            next_observation = ob_proc.process(sess, next_observation)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
            replay_memory.append(Transition(state, action, reward, next_state, int(done)))

            # sample a minibatch from replay buffer. Hint: samples = random.sample(replay_memory, batch_size)
            samples = random.sample(replay_memory, BATCH_SIZE)

            # calculate target Q values by target network
            batch_state_feature, batch_selected_action, batch_reward, batch_next_state_feature, batch_done = map(np.stack, zip(*samples))
            target_q_value = target_Q.forward(sess, batch_next_state_feature)
            target_q_value = batch_reward + GAMMA * (np.max(target_q_value, axis=1) * (1-batch_done))

            # Update network
            loss = behavior_Q.update(sess, batch_state_feature, batch_selected_action, target_q_value)
            soft_train_loss += (1.0 - soft_decay) * (loss - soft_train_loss)

            state = next_state
            total_iteration += 1

            # Update target network every FREQ_UPDATE_TARGET_Q steps
            if total_iteration % FREQ_UPDATE_TARGET_Q == 0:
                 target_Q.copy_parameter_from(sess, behavior_Q)

            if total_iteration <= EXPLORE_STPES:
                epsilon += (FINAL_EPSILON - INITIAL_EPSILON)/EXPLORE_STPES
            else:
                epsilon = FINAL_EPSILON

            if done:
                soft_episode_reward += (1 - soft_decay) * (episode_reward - soft_episode_reward)
                max_episode_reward = max(max_episode_reward, episode_reward)
                print ("[%s] Episode %d, frames = %d, reward = %d (%d)" % (datetime.strftime(datetime.now(), "%Y/%m/%d %H:%M:%S"), episode+1, frame+1, episode_reward, max_episode_reward))
                break

        if episode % 200 == 0:
            train_summary = tf.Summary()
            train_summary.value.add(tag="train_loss", simple_value=soft_train_loss)
            train_summary.value.add(tag="episode_reward", simple_value=soft_episode_reward)
            train_summary.value.add(tag="epsilon", simple_value=epsilon)
            summary_writer.add_summary(train_summary, episode)
            behavior_Q.save_model(sess, episode)
    sess.close()

if __name__ == '__main__':
    tf.app.run()