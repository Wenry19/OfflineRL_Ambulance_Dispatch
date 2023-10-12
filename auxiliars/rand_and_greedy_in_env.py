
import pickle
from pathlib import Path
import sys
import os

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'algorithms'))
sys.path.append(os.path.join(os.getcwd(), 'environment'))

from algorithms.rand import randomAgent
from algorithms.greedy import greedyAgent
from environment.emergencies_environment import emergenciesEnv
from utils import STATE_SIZE, ACTION_SPACE, transform_state_to_model_input, check_experiences_version

if __name__ == '__main__':

    res_names = pickle.load(open(Path('generated_data/resource_names.pkl'), 'rb'))

    env = emergenciesEnv()

    rand = randomAgent(STATE_SIZE, ACTION_SPACE)
    greedy = greedyAgent(STATE_SIZE, ACTION_SPACE)

    rand_rewards = []
    greedy_rewards = []

    for _ in range(50):

        # random
        obs, info = env.reset()
        terminated = False
        total_reward = 0
        while not terminated:
            obs, reward, terminated, _, info = env.step(rand.choose_action(transform_state_to_model_input(obs)))
            total_reward += reward[1]
        rand_rewards.append(total_reward)

        # greedy
        obs, info = env.reset()
        terminated = False
        total_reward = 0
        while not terminated:
            obs, reward, terminated, _, info = env.step(greedy.choose_action(transform_state_to_model_input(obs)))
            total_reward += reward[1]
        greedy_rewards.append(total_reward)

    print('Mean Total Reward Random Agent:', sum(rand_rewards)/len(rand_rewards))
    print('Mean Total Reward Greedy Agent:', sum(greedy_rewards)/len(greedy_rewards))

    # Mean Total Reward Random Agent: 10.889448306397826
    # Mean Total Reward Greedy Agent: 99.74425180462971
