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
EXPLORE_STPES = 500000              # frames over which to anneal epsilon
epsilon_series = np.linspace(INITIAL_EPSILON, FINAL_EPSILON, EXPLORE_STPES)

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
MAX_LIFE = 5

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
    config.allow_soft_placement = True
    sess = tf.InteractiveSession(config=config)
    summary_writer = tf.summary.FileWriter(TENSORNOARD_PATH, sess.graph)
    sess.run(tf.global_variables_initializer())

    # Populate the replay buffer
    observation = env.reset()                       # retrive first env image
    observation = ob_proc.process(sess, observation)        # process the image
    state = np.stack([observation] * 4, axis=2)     # stack the image 4 times
    pre_life = MAX_LIFE
    while len(replay_memory) < INIT_REPLAY_MEMORY_SIZE:
        action = random.randint(0,ACTION_SPACE-1)

        next_observation, reward, done, info = env.step(action)
        life = info['ale.lives']
        if life != pre_life:
            done = True
            reward = TERMINATE_REWARD
            pre_life = life
        next_observation = ob_proc.process(sess, next_observation)
        next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
        replay_memory.append(Transition(state, action, reward, next_state, done))

        if life == 0:
            observation = env.reset()
            observation = ob_proc.process(sess, observation)
            state = np.stack([observation] * 4, axis=2)
            pre_life = MAX_LIFE
        else:
            state = next_state


    # record videos
    env = Monitor(env, directory=MONITOR_PATH, video_callable=lambda episode: episode % RECORD_VIDEO_EVERY == 0 or (episode+1) == TRAINING_EPISODES, resume=True)

    # total steps
    total_iteration = 0
    max_episode_reward = TERMINATE_REWARD
    loss_record = []
    for episode in xrange(TRAINING_EPISODES):

        # Reset the environment
        observation = env.reset()
        observation = ob_proc.process(sess, observation)
        state = np.stack([observation] * 4, axis=2)
        pre_life = MAX_LIFE

        episode_reward = 0                              # store the episode reward
        del loss_record[:]
        for frame in itertools.count():
            # choose a action
            epsilon = epsilon_series[min(EXPLORE_STPES, total_iteration)]
            if random.random() < epsilon:
                action = random.randint(0,ACTION_SPACE-1)
            else:
                state_feature = np.expand_dims(state, axis=0)
                q_value = behavior_Q.forward(sess, state_feature)
                action = np.argmax(q_value[0])

            # execute the action
            next_observation, reward, done, info = env.step(action)
            episode_reward += reward
            life = info['ale.lives']
            if pre_life != life:
                done = True
                reward = TERMINATE_REWARD
                pre_life = life

            # save the transition to replay buffer
            next_observation = ob_proc.process(sess, next_observation)
            next_state = np.append(state[:,:,1:], np.expand_dims(next_observation, 2), axis=2)
            replay_memory.append(Transition(state, action, reward, next_state, done))

             # if the size of replay buffer is too big, remove the oldest one. Hint: replay_memory.pop(0)
            if len(replay_memory) > REPLAY_MEMORY_SIZE: del replay_memory[0]

            # sample a minibatch from replay buffer. Hint: samples = random.sample(replay_memory, batch_size)
            samples = random.sample(replay_memory, BATCH_SIZE)

            # calculate target Q values by target network
            batch_state_feature, batch_selected_action, batch_reward, batch_next_state_feature, batch_done = map(np.array, zip(*samples))
            target_q_value = target_Q.forward(sess, batch_next_state_feature)
            target_q_value = batch_reward + np.invert(batch_done).astype(np.float32) * GAMMA * np.amax(target_q_value, axis=1)

            # Update network
            loss = behavior_Q.update(sess, batch_state_feature, batch_selected_action, target_q_value)
            loss_record.append(loss)
            state = next_state
            total_iteration += 1

            # Update target network every FREQ_UPDATE_TARGET_Q steps
            if total_iteration % FREQ_UPDATE_TARGET_Q == 0:
                 target_Q.copy_parameter_from(sess, behavior_Q)

            if life == 0: break

        max_episode_reward = max(max_episode_reward, episode_reward)
        print ("[%s] Episode %d, frames = %d, reward = %d (%d)" % (datetime.strftime(datetime.now(), "%Y/%m/%d %H:%M:%S"), episode+1, frame+1, episode_reward, max_episode_reward))
        sys.stdout.flush()
        train_summary = tf.Summary()
        train_summary.value.add(tag="train_loss", simple_value=np.mean(loss_record))
        train_summary.value.add(tag="episode_reward", simple_value=episode_reward)
        train_summary.value.add(tag="epsilon", simple_value=epsilon)
        summary_writer.add_summary(train_summary, episode)
        if episode % 200 == 0:
            behavior_Q.save_model(sess, episode)
    sess.close()

if __name__ == '__main__':
    with tf.device("/gpu:1"):
        tf.app.run()

