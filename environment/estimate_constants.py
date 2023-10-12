
"""Estimate Environment Constants

This script is aimed to estimate the environment constants using the data.
"""

import pickle
from pathlib import Path

import os
import sys

sys.path.append(os.getcwd())

from utils import calculate_duration

if __name__ == '__main__':

    res_pos_time = pickle.load(open(Path('generated_data/resource_pos_avail_info.pkl'), 'rb'))

    durations = []

    for k, v in res_pos_time.items():

        for x in v:

            diff = calculate_duration(x['itime'], x['ftime'])

            if diff < 0:
                print('DIFF < 0')
                print(k)
                print(x['itime'])
                print(x['ftime'])

            durations.append(diff)

    print('Mean busy time (seconds):', sum(durations)/len(durations)) # 3546.2313052980508 (without negative differences)
