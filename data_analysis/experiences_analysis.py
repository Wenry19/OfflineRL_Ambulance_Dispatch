
from pathlib import Path
import numpy as np
import pickle

import os
import sys
sys.path.append(os.getcwd())
from utils import *

ACCEPTED_ERROR = [0, 2, 3] # km

if __name__ == '__main__':

    check_experiences_version()
    experience_buffer = pickle.load(open(Path('generated_data/experiences/experiences.pkl'), 'rb'))

    print('Shape:', experience_buffer.shape)

    for acc_err in ACCEPTED_ERROR:

        count_p0 = 0
        count_p1 = 0
        count_pother = 0

        count_greedy_p0 = 0
        count_greedy_p1 = 0
        count_greedy_pother = 0

        sum_immediate_reward = 0

        for exp in experience_buffer:

            sum_immediate_reward += exp[IDX_REWARD]

            if exp[IDX_STATE][IDX_PRIORITY] == '0':
                count_p0 += 1
            elif exp[IDX_STATE][IDX_PRIORITY] == '1':
                count_p1 += 1
            else:
                count_pother += 1

            greedy_action_dist = np.amin(np.array(exp[IDX_STATE][IDX_DISTANCES])[np.array(exp[IDX_STATE][IDX_RES_AVAIL]) == 0])

            if abs(exp[IDX_STATE][IDX_DISTANCES][exp[IDX_ACTION]] - greedy_action_dist) <= acc_err:

                if exp[IDX_STATE][IDX_PRIORITY] == '0':
                    count_greedy_p0 += 1
                elif exp[IDX_STATE][IDX_PRIORITY] == '1':
                    count_greedy_p1 += 1
                else:
                    count_greedy_pother += 1

        print('Percentage of greedy actions in P0:', 100*count_greedy_p0/count_p0)
        print('Percentage of greedy actions in P1:', 100*count_greedy_p1/count_p1)
        print('Percentage of greedy actions in Pother:', 100*count_greedy_pother/count_pother)
        print('With ACCEPTED_ERROR =', acc_err)
        print()

    print('Mean immediate reward =', sum_immediate_reward/experience_buffer.shape[0])
