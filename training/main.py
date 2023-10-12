
"""Training Loop Script

This script is aimed to execute the training loop of the implemented algorithms.

It loads the training experience buffer, it allows the user to choose which agent he/she wants to train,
and then it performs a random search with different hyperparameters, training the chosen agent until convergence or
until a maximum number of training steps.

After a certain number of training steps it runs some episodes in the environment to evaluate the performance of the agent
(it runs some episodes and saves the mean reward obtained in a list).

To check the convergence: abs(mean(rewards[-CONVERGENCE_NUM_REWARDS*2:-CONVERGENCE_NUM_REWARDS]) - mean(rewards[-CONVERGENCE_NUM_REWARDS:])) < threshold,
where rewards is the list of the mean rewards obtained when evaluating the agent during training.

For each hyperparameter configuration, it logs the evolution of the loss function and the mean reward obtained in each evaluation step,
along with other useful information. These logs can be plotted using tensorboard. Also, the hyperparameter configuration is saved in a txt file.

If the agent is good enough, the trained model is saved.

This file contains the following functions:

    * draw_ambulance() - prints an ambulance in the terminal.
    * check_gpu() - checks if the gpu is detected by tensorflow.
    * save_config(f, hparam) - saves the hyperparameter configuration hparam to the file f.
    * run_episode(agent) - given an agent, it runs an episode in the environment taking the actions chosen by the agent.
    * check_convergence(mean_rewards) - given the list of the mean rewards obtained when evaluating the agent during training, checks if it has converged.
    * agent_evaluation(agent, step) - given an agent and a step, it does an evaluation step of the agent.
"""

import pickle
import tensorflow as tf
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import inspect
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'algorithms'))
sys.path.append(os.path.join(os.getcwd(), 'environment'))

from algorithms.behavioural_cloning import behaviouralCloningAgent
from algorithms.cql import CQLAgent
from algorithms.ac_kl import actorCriticKLAgent
from environment.emergencies_environment import emergenciesEnv
import hyperparameters
from utils import transform_state_to_model_input, IMPOSSIBLE_ACTION_AGENTS
from utils import STATE_SIZE, ACTION_SPACE, IDX_PRIORITY, check_experiences_version

RES_NAMES = pickle.load(open(Path('generated_data/resource_names.pkl'), 'rb'))
"""Resource names.
"""

MAX_NUM_ITERATIONS = 5000
"""Maximum number of training steps.
"""

NUM_RANDOM_SEARCH = 1
"""Number of random searches (number of different hyperparameters configurations to try).
"""

CHECK_CONVERGENCE = False
"""If True, convergence will be checked during the training, otherwise it will not be checked.
"""

CONVERGENCE_NUM_REWARDS = 50
"""To check convergence: abs(mean(rewards[-CONVERGENCE_NUM_REWARDS*2:-CONVERGENCE_NUM_REWARDS]) - mean(rewards[-CONVERGENCE_NUM_REWARDS:])) < threshold.
"""

CONVERGENCE_THRESHOLD = 0.01
"""Convergence threshold.
"""

NUM_TRAIN_STEPS_TO_RUN_EPISODE = 25
"""The number of training steps before performing the evaluation, running some episodes in the environment (used in behavioural cloning and CQL).
"""

THRESHOLD_SAVE_MODEL = 0.8
"""To save a trained model, it has to achieve a mean reward value greater than THRESHOLD_SAVE_MODEL * (maximum possible reward).
"""

NUM_EPISODES_SAVE_MODEL = 20
"""The number of episodes to run, to compute the mean total reward to decide if the trained model/s is/are saved or not.
"""

NUM_EPISODES = 3
"""The number of episodes that will be executed to evaluate the agent after NUM_TRAIN_STEPS_TO_RUN_EPISODE training steps.
"""

EPISODE_LENGTH = 100
"""Episode length.
"""

USE_TRAINING_INCIDENTS_IN_ENV = False
"""Boolean that indicates if the environment uses incidents from the training set (seen incidents during training) or not.
"""

USE_EXPLORATION_EXPERIENCES = False
"""Boolean that indicates if exploration experiences are added to the experience buffer or not.
"""

ENV = emergenciesEnv(episode_length=EPISODE_LENGTH, training_incidents=USE_TRAINING_INCIDENTS_IN_ENV)
"""Instance of the environment, that will be used to evaluate the agents.
"""

def draw_ambulance():
    """Prints an ambulance in the terminal.
    """

    print('   o_______________}o{\n' +
          '   |              |   \\\n' +
          '   |              |____\_____\n' +
          '   | _____        |    |_o__ |\n' +
          '   [/ ___ \       |   / ___ \|\n' +
          '  []_/.-.\_\______|__/_/.-.\_[]\n' +
          '     |(O)|             |(O)|\n' +
          "      '-'               '-'\n" +
          '---   ---   ---   ---   ---   ---\n')

def check_gpu():
    """Checks if the gpu is detected by tensorflow.
    """

    # https://www.tensorflow.org/guide/gpu
    # https://www.tensorflow.org/install/pip
    # https://towardsdatascience.com/how-to-finally-install-tensorflow-gpu-on-windows-10-63527910f255
    # https://docs.nvidia.com/cuda/archive/11.2.0/cuda-installation-guide-microsoft-windows/index.html
    print('\nPython version:', sys.version)
    print('Tensorflow version:', tf.__version__)
    print(tf.config.list_physical_devices(), '\n')

def save_config(f, hparam):
    """Saves the hyperparameter configuration hparam to the file f.

    Parameters
    ----------
    f : file
        The file where the configuration will be saved.
    hparam : dict
        Hyperparameter configuration to save.
    """

    f.write(str(hparam))
    f.write('\nMAX_NUM_ITERATIONS = ' + str(MAX_NUM_ITERATIONS))
    f.write('\nNUM_RANDOM_SEARCH = ' + str(NUM_RANDOM_SEARCH))
    f.write('\nCHECK_CONVERGENCE = ' + str(CHECK_CONVERGENCE))
    f.write('\nCONVERGENCE_NUM_REWARDS = ' + str(CONVERGENCE_NUM_REWARDS))
    f.write('\nCONVERGENCE_THRESHOLD = ' + str(CONVERGENCE_THRESHOLD))
    f.write('\nNUM_TRAIN_STEPS_TO_RUN_EPISODE = ' + str(NUM_TRAIN_STEPS_TO_RUN_EPISODE))
    f.write('\nTHRESHOLD_SAVE_MODEL = ' + str(THRESHOLD_SAVE_MODEL))
    f.write('\nNUM_EPISODES_SAVE_MODEL = ' + str(NUM_EPISODES_SAVE_MODEL))
    f.write('\nNUM_EPISODES = ' + str(NUM_EPISODES))
    f.write('\nEPISODE_LENGTH = ' + str(EPISODE_LENGTH))
    f.write('\nUSE_TRAINING_INCIDENTS_IN_ENV = ' + str(USE_TRAINING_INCIDENTS_IN_ENV))
    f.write('\nUSE_EXPLORATION_EXPERIENCES = ' + str(USE_EXPLORATION_EXPERIENCES))
    f.write('\n' + inspect.getsource(IMPOSSIBLE_ACTION_AGENTS))
    f.write('\nVersion experiences.pkl: ' +
            str(datetime.fromtimestamp(os.path.getmtime(Path('generated_data/experiences/experiences.pkl')))))
    f.write('\nVersion experiences_to_train_model.pkl: ' +
            str(datetime.fromtimestamp(os.path.getmtime(Path('generated_data/experiences_to_train_model/experiences_to_train_model.pkl')))))
    f.close()

def run_episode(agent):
    """Given an agent, it runs an episode in the environment taking the actions chosen by the agent.

    Parameters
    ----------
    agent : agentRL
        The agent to evaluate in the environment.

    Returns
    -------
    float
        Total obtained reward.
    int
        Number of different actions taken by the agent.
    dict
        Information about the obtained immediate rewards for incidents of priority 0.
    dict
        Information about the obtained immediate rewards for incidents of priority 1.
    dict
        Information about the obtained immediate rewards for incidents of priority != 0 and != 1.
    """

    obs, info = ENV.reset()

    terminated = False
    total_reward = 0
    diff_actions = set()

    it_was_p0 = False
    it_was_p1 = False
    it_was_pother = False

    p0_info = dict()
    p1_info = dict()
    pother_info = dict()

    p0_info['mean_normalized_distance'] = 0
    p0_info['mean_immediate_reward'] = 0
    p0_info['num'] = 0

    p1_info['mean_normalized_distance'] = 0
    p1_info['mean_immediate_reward'] = 0
    p1_info['num'] = 0

    pother_info['mean_normalized_distance'] = 0
    pother_info['mean_immediate_reward'] = 0
    pother_info['num'] = 0

    while not terminated:

        if obs[IDX_PRIORITY] == '0':
            it_was_p0 = True
            p0_info['num'] += 1
        elif obs[IDX_PRIORITY] == '1':
            it_was_p1 = True
            p1_info['num'] += 1
        else:
            it_was_pother = True
            pother_info['num'] += 1

        action = agent.choose_action(transform_state_to_model_input(obs))
        diff_actions.add(action)
        obs, reward, terminated, _, info = ENV.step(action)
        total_reward += reward[1]

        if it_was_p0:
            p0_info['mean_normalized_distance'] += reward[0]
            p0_info['mean_immediate_reward'] += reward[1]
            it_was_p0 = False
        elif it_was_p1:
            p1_info['mean_normalized_distance'] += reward[0]
            p1_info['mean_immediate_reward'] += reward[1]
            it_was_p1 = False
        elif it_was_pother:
            pother_info['mean_normalized_distance'] += reward[0]
            pother_info['mean_immediate_reward'] += reward[1]
            it_was_pother = False

    if p0_info['num'] > 0:
        p0_info['mean_normalized_distance'] = p0_info['mean_normalized_distance'] / p0_info['num']
        p0_info['mean_immediate_reward'] = p0_info['mean_immediate_reward'] / p0_info['num']
    if p1_info['num'] > 0:
        p1_info['mean_normalized_distance'] = p1_info['mean_normalized_distance'] / p1_info['num']
        p1_info['mean_immediate_reward'] = p1_info['mean_immediate_reward'] / p1_info['num']
    if pother_info['num'] > 0:
        pother_info['mean_normalized_distance'] = pother_info['mean_normalized_distance'] / pother_info['num']
        pother_info['mean_immediate_reward'] = pother_info['mean_immediate_reward'] / pother_info['num']

    return total_reward, len(diff_actions), p0_info, p1_info, pother_info

def check_convergence(mean_rewards):
    """Given the list of the mean rewards obtained when evaluating the agent during training, checks if it has converged.

    Parameters
    ----------
    mean_rewards : list
        List of the mean rewards obtained when evaluating the agent during training.

    Returns
    -------
    boolean
        If it has converged True, otherwise False.
    """

    if len(mean_rewards) < CONVERGENCE_NUM_REWARDS*2:
        return False
    
    # first mean
    mean1 = sum(mean_rewards[-CONVERGENCE_NUM_REWARDS*2:-CONVERGENCE_NUM_REWARDS])/CONVERGENCE_NUM_REWARDS
                
    # second mean
    mean2 = sum(mean_rewards[-CONVERGENCE_NUM_REWARDS:])/CONVERGENCE_NUM_REWARDS

    return abs(mean1 - mean2) < CONVERGENCE_THRESHOLD

def agent_evaluation(agent, step):
    """Given an agent and a step, it does an evaluation step of the agent.

    Parameters
    ----------
    agent : agentRL
        The agent to evaluate.
    step : int
        The step number.

    Returns
    -------
    float
        The obtained mean reward.
    """

    sum_rewards = 0
    sum_diff_actions = 0

    sum_mean_normalized_distance_p0 = 0
    sum_mean_normalized_distance_p1 = 0
    sum_mean_normalized_distance_pother = 0

    sum_mean_immediate_reward_p0 = 0
    sum_mean_immediate_reward_p1 = 0
    sum_mean_immediate_reward_pother = 0

    for _ in range(NUM_EPISODES):

        total_reward, num_diff_actions, p0_info, p1_info, pother_info = run_episode(agent)

        sum_rewards += total_reward
        sum_diff_actions += num_diff_actions

        sum_mean_normalized_distance_p0 += p0_info['mean_normalized_distance']
        sum_mean_normalized_distance_p1 += p1_info['mean_normalized_distance']
        sum_mean_normalized_distance_pother += pother_info['mean_normalized_distance']

        sum_mean_immediate_reward_p0 += p0_info['mean_immediate_reward']
        sum_mean_immediate_reward_p1 += p1_info['mean_immediate_reward']
        sum_mean_immediate_reward_pother += pother_info['mean_immediate_reward']

    mean_reward = sum_rewards / NUM_EPISODES
    mean_num_diff_actions = sum_diff_actions / NUM_EPISODES

    mean_normalized_distance_p0 = sum_mean_normalized_distance_p0 / NUM_EPISODES
    mean_immediate_reward_p0 = sum_mean_immediate_reward_p0 / NUM_EPISODES

    mean_normalized_distance_p1 = sum_mean_normalized_distance_p1 / NUM_EPISODES
    mean_immediate_reward_p1 = sum_mean_immediate_reward_p1 / NUM_EPISODES

    mean_normalized_distance_pother = sum_mean_normalized_distance_pother / NUM_EPISODES
    mean_immediate_reward_pother = sum_mean_immediate_reward_pother / NUM_EPISODES

    tf.summary.scalar('reward', data=mean_reward, step=step)
    tf.summary.scalar('num_diff_actions', data=mean_num_diff_actions, step=step)

    tf.summary.scalar('mean_normalized_distance_p0', data=mean_normalized_distance_p0, step=step)
    tf.summary.scalar('mean_immediate_reward_p0', data=mean_immediate_reward_p0, step=step)

    tf.summary.scalar('mean_normalized_distance_p1', data=mean_normalized_distance_p1, step=step)
    tf.summary.scalar('mean_immediate_reward_p1', data=mean_immediate_reward_p1, step=step)

    tf.summary.scalar('mean_normalized_distance_pother', data=mean_normalized_distance_pother, step=step)
    tf.summary.scalar('mean_immediate_reward_pother', data=mean_immediate_reward_pother, step=step)

    return mean_reward

if __name__ == '__main__':

    # checking versions of python, tensorflow and if it detects the gpu
    check_gpu()

    print('Loading experiences...\n')

    # loading experiences
    check_experiences_version()
    experience_buffer = pickle.load(open(Path('generated_data/experiences_to_train_model/experiences_to_train_model.pkl'), 'rb'))

    if USE_EXPLORATION_EXPERIENCES:
        print('Adding exploration experiences...\n')
        exploration_experiences = pickle.load(open(Path('generated_data/experiences_to_train_model/exploration_experiences.pkl'), 'rb'))
        experience_buffer = np.concatenate((experience_buffer, exploration_experiences), axis=0)

    print('Number of experiences:', experience_buffer.shape[0])

    # TRAINING MENU
    print('\n\n#### WELCOME TO THE TRAINING MENU ####\n')
    draw_ambulance()
    print('Choose the algorithm (type the option number):\n')
    print('\t0. Behavioural cloning.')
    print('\t1. Conservative Q-Learning (CQL).')
    print('\t2. Actor-Critic with KL-divergence penalty.')
    # user input
    opt = input()

    for t in range(NUM_RANDOM_SEARCH): # RandomSearch

        time = datetime.now().strftime('%Y%m%d-%H%M%S')

        hparam = dict()

        mean_rewards = []

        if opt == '0':
            for p, v in hyperparameters.cloning_hparam.items():
                hparam[p] = random.choice(v)
            # save hyperparameter configuration
            f = open(Path('configs/cloning/' + time + '.txt'), 'w')
            save_config(f, hparam)
            # tensorboard file writer
            # https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.as_posix
            logdir = Path('logs/cloning/' + time + '/scalars').as_posix()
            file_writer = tf.summary.create_file_writer(logdir)
            file_writer.set_as_default()
            # create agent
            agent = behaviouralCloningAgent(state_size=STATE_SIZE, action_space=ACTION_SPACE, train_experiences=experience_buffer, hyperparameters=hparam)

        elif opt == '1':
            for p, v in hyperparameters.cql_hparam.items():
                hparam[p] = random.choice(v)
            # save hyperparameter configuration
            f = open(Path('configs/cql/' + time + '.txt'), 'w')
            save_config(f, hparam)
            # tensorboard file writer
            # https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.as_posix
            logdir = Path('logs/cql/' + time + '/scalars').as_posix()
            file_writer = tf.summary.create_file_writer(logdir)
            file_writer.set_as_default()
            # create agent
            agent = CQLAgent(state_size=STATE_SIZE, action_space=ACTION_SPACE, train_experiences=experience_buffer, hyperparameters=hparam)

        elif opt == '2':
            for p, v in hyperparameters.ac_kl_hparam.items():
                hparam[p] = random.choice(v)
            # save hyperparameter configuration
            f = open(Path('configs/ac_kl/' + time + '.txt'), 'w')
            save_config(f, hparam)
            # tensorboard file writer
            # https://docs.python.org/3/library/pathlib.html#pathlib.PurePath.as_posix
            logdir = Path('logs/ac_kl/' + time + '/scalars').as_posix()
            file_writer = tf.summary.create_file_writer(logdir)
            file_writer.set_as_default()
            # create agent
            agent = actorCriticKLAgent(state_size=STATE_SIZE, action_space=ACTION_SPACE, train_experiences=experience_buffer, hyperparameters=hparam, behaviour_model_v='20230930-115945')

        print('\nRANDOM SEARCH NUMBER: ' + str(t))
        print('Option selected by user:', opt)
        print(hparam, '\n')

        if opt in {'0', '1'}:

            for i in range(MAX_NUM_ITERATIONS):

                if i%NUM_TRAIN_STEPS_TO_RUN_EPISODE == 0:
                    mean_rewards.append(agent_evaluation(agent, i))

                if CHECK_CONVERGENCE and check_convergence(mean_rewards):
                    break

                agent.run_train_step(i)

        elif opt in {'2'}:

            for i in range(int(MAX_NUM_ITERATIONS/agent.get_gradient_steps())):

                mean_rewards.append(agent_evaluation(agent, i*agent.get_gradient_steps()))

                if CHECK_CONVERGENCE and check_convergence(mean_rewards):
                    break

                agent.run_train_step(i) # it will run X gradient steps

        # decide whether the model is saved or not
        rew_save_model = []
        for _ in range(NUM_EPISODES_SAVE_MODEL):
            rew, _, _, _, _ = run_episode(agent)
            rew_save_model.append(rew)

        if sum(rew_save_model)/len(rew_save_model) > THRESHOLD_SAVE_MODEL*EPISODE_LENGTH:
            # max total reward can change if I change the reward function
            agent.save_model(time)
