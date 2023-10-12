
"""Collect Exploration Experiences

This script is aimed to collect exploration experiences to add to the experience buffer.

This file contains the following functions:

    * exploration(agent, number_experiences) - collects the exploration experiences.
"""

import numpy as np
import pickle
from pathlib import Path
import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'algorithms'))
sys.path.append(os.path.join(os.getcwd(), 'environment'))

from algorithms.rand import randomAgent
from environment.emergencies_environment import emergenciesEnv
from utils import IDX_STATE, IDX_ACTION, IDX_REWARD, IDX_NEXT_STATE
from utils import transform_state_to_model_input
from utils import STATE_SIZE, ACTION_SPACE

def exploration(agent, number_experiences):
    """Collects the exploration experiences.

    Parameters
    ----------
    agent : agentRL
        The agent that will collect the experiences in the environment.
    number_experiences : int
        The number of experiences to collect.

    Returns
    -------
    numpy array
        The collected exploration experiences.
    """

    exploration_experiences = []

    env = emergenciesEnv(episode_length=number_experiences, training_incidents=True) # with training incidents

    obs, info = env.reset()
    terminated = False

    obs = transform_state_to_model_input(obs)

    while not terminated:

        experience = [None, None, None, None]
        experience[IDX_STATE] = obs
        action = agent.choose_action(obs)
        experience[IDX_ACTION] = action
        obs, reward, terminated, _, info = env.step(action)
        obs = transform_state_to_model_input(obs)
        experience[IDX_REWARD] = reward[1]
        experience[IDX_NEXT_STATE] = obs

        exploration_experiences.append(experience)

    return np.array(exploration_experiences, dtype=object)

if __name__ == '__main__':

    # exploration experiences
    number_experiences = 10000
    agent = randomAgent(STATE_SIZE, ACTION_SPACE)
    exploration_experiences = exploration(agent, number_experiences)

    pickle.dump(exploration_experiences, open(Path('generated_data/experiences_to_train_model/exploration_experiences.pkl'), 'wb'))
