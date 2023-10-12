
import numpy as np

import folium
import io
from PIL import Image

import pickle
from pathlib import Path

import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'algorithms'))
sys.path.append(os.path.join(os.getcwd(), 'environment'))

from algorithms.behavioural_cloning import behaviouralCloningAgent
from algorithms.cql import CQLAgent
from algorithms.ac_kl import actorCriticKLAgent
from algorithms.rand import randomAgent
from algorithms.greedy import greedyAgent
from environment.emergencies_environment import emergenciesEnv
from build_experiences.resource_manager import resourceManager

from utils import transform_state_to_model_input
from utils import IDX_PRIORITY, IDX_DISTANCES, ACTION_SPACE, IDX_RES_AVAIL, STATE_SIZE, IDX_RES_POS

EPISODE_LENGTH = 1000

USE_TRAINING_INCIDENTS_IN_ENV = False

ACCEPTED_ERROR = 2 # km

ENV = emergenciesEnv(episode_length=EPISODE_LENGTH, training_incidents=USE_TRAINING_INCIDENTS_IN_ENV)

def run_episode(agent):

    obs, info = ENV.reset(seed=0)

    print('Initial incident index:', ENV._incident_idx)

    terminated = False

    total_reward = 0

    p0_info = dict()
    p1_info = dict()
    pother_info = dict()

    p0_info['num_greedy'] = 0
    p0_info['num_almost_greedy'] = 0
    p0_info['num'] = 0

    p1_info['num_greedy'] = 0
    p1_info['num_almost_greedy'] = 0
    p1_info['num'] = 0

    pother_info['num_greedy'] = 0
    pother_info['num_almost_greedy'] = 0
    pother_info['num'] = 0

    diff_actions = set()
    ambu_ini_pos_priority = []

    while not terminated:

        action = agent.choose_action(transform_state_to_model_input(obs))

        diff_actions.add(action)

        ambu_ini_pos_priority.append((obs[IDX_RES_POS][action], obs[IDX_PRIORITY]))

        greedy_action_dist = np.amin(np.array(obs[IDX_DISTANCES])[np.array(obs[IDX_RES_AVAIL]) == 0])

        if obs[IDX_PRIORITY] == '0':
            p0_info['num'] += 1
            if obs[IDX_DISTANCES][action] == greedy_action_dist:
                p0_info['num_greedy'] += 1
            if abs(obs[IDX_DISTANCES][action] - greedy_action_dist) <= ACCEPTED_ERROR:
                p0_info['num_almost_greedy'] += 1
            
        elif obs[IDX_PRIORITY] == '1':
            p1_info['num'] += 1
            if obs[IDX_DISTANCES][action] == greedy_action_dist:
                p1_info['num_greedy'] += 1
            if abs(obs[IDX_DISTANCES][action] - greedy_action_dist) <= ACCEPTED_ERROR:
                p1_info['num_almost_greedy'] += 1
            
        else:
            pother_info['num'] += 1
            if obs[IDX_DISTANCES][action] == greedy_action_dist:
                pother_info['num_greedy'] += 1
            if abs(obs[IDX_DISTANCES][action] - greedy_action_dist) <= ACCEPTED_ERROR:
                pother_info['num_almost_greedy'] += 1
            
        obs, reward, terminated, _, info = ENV.step(action)

        total_reward += reward[1]

    return total_reward, p0_info, p1_info, pother_info, diff_actions, ambu_ini_pos_priority

if __name__ == '__main__':

    rm = resourceManager(precomputed_pos_avail_info=pickle.load(open(Path('generated_data/resource_pos_avail_info.pkl'), 'rb')))

    cloning_agent = behaviouralCloningAgent(action_space=ACTION_SPACE)
    cql_agent = CQLAgent(action_space=ACTION_SPACE)
    ac_kl_agent = actorCriticKLAgent(action_space=ACTION_SPACE)
    rand_agent = randomAgent(state_size=STATE_SIZE, action_space=ACTION_SPACE)
    greedy_agent = greedyAgent(state_size=STATE_SIZE, action_space=ACTION_SPACE)

    cloning_agent.load_model('20230930-115945')
    cql_agent.load_model('20230930-154205')
    ac_kl_agent.load_model('20231001-082017')

    agents = [('CLONING', cloning_agent), ('CQL', cql_agent), ('AC_KL', ac_kl_agent),
              ('RAND', rand_agent), ('GREEDY', greedy_agent)]

    for alg, agent in agents:

        print(alg)

        total_reward, p0_info, p1_info, pother_info, diff_actions, ambu_ini_pos_priority = run_episode(agent)

        if p0_info['num'] > 0:
            perce_greedy_p0 = 100 * p0_info['num_greedy']/p0_info['num']
            perce_almost_greedy_p0 = 100 * p0_info['num_almost_greedy']/p0_info['num']
        if p1_info['num'] > 0:
            perce_greedy_p1 = 100 * p1_info['num_greedy']/p1_info['num']
            perce_almost_greedy_p1 = 100 * p1_info['num_almost_greedy']/p1_info['num']
        if pother_info['num'] > 0:
            perce_greedy_pother = 100 * pother_info['num_greedy']/pother_info['num']
            perce_almost_greedy_pother = 100 * pother_info['num_almost_greedy']/pother_info['num']

        print('P0:')
        print('\tNum P0 = ' + str(p0_info['num']))
        print('\tNum greedy P0 = ' + str( p0_info['num_greedy']))
        print('\tNum almost greedy P0 = ' + str(p0_info['num_almost_greedy']))
        print('\tPercentage greedy P0 = ' + str(perce_greedy_p0))
        print('\tPercentage almost greedy P0 = ' + str(perce_almost_greedy_p0))

        print('P1:')
        print('\tNum P1 = ' + str(p1_info['num']))
        print('\tNum greedy P1 = ' + str(p1_info['num_greedy']))
        print('\tNum almost greedy P1 = ' + str(p1_info['num_almost_greedy']))
        print('\tPercentage greedy P1 = ' + str(perce_greedy_p1))
        print('\tPercentage almost greedy P1 = ' + str(perce_almost_greedy_p1))

        print('Pother:')
        print('\tNum Pother = ' + str(pother_info['num']))
        print('\tNum greedy Pother = ' + str(pother_info['num_greedy']))
        print('\tNum almost greedy Pother = ' + str(pother_info['num_almost_greedy']))
        print('\tPercentage greedy Pother = ' + str(perce_greedy_pother))
        print('\tPercentage almost greedy Pother = ' + str(perce_almost_greedy_pother))

        print('Total reward:', total_reward)
        print('Mean immmediate reward:', total_reward/(EPISODE_LENGTH))
        print('Num of different ambulances:', len(diff_actions))

        print()

        if alg in {'RAND', 'GREEDY', 'CQL', 'AC_KL'}:

            m = folium.Map(location=[46.6739, 6.6830], zoom_start=9, height=680, width=680, zoom_control=False)

            for ambu_ini_pos, priority in ambu_ini_pos_priority:
                if priority == '2':
                    folium.CircleMarker(location=ambu_ini_pos, radius=1, color='black', fill=True).add_to(m)

            #m.save(Path('data_analysis/incident_maps/index.html'))
            img = m._to_png(20)
            img = Image.open(io.BytesIO(img))
            img.save(Path('test/resource_maps/' + alg.lower() + '.png'))

