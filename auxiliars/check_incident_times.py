import numpy as np
import pandas as pd
from pathlib import Path

import os
import sys
sys.path.append(os.getcwd())
from utils import read_time

if __name__ == '__main__':

    # first incident time
    inci2014 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2014.csv'), dtype=str)['LocalTime'].to_numpy()

    # transform times to integers in order to sort them by time
    for i in range(inci2014.shape[0]):
        inci2014[i] = read_time(inci2014[i])

    # sort times
    inci2014 = np.sort(inci2014)

    # print first time
    print('First incident time:', inci2014[0])

    # first and last V6 incident time
    inciV6 = pd.read_csv(Path('CHUV/ChuvExportIncidentsV6.csv'), dtype=str)['LocalTime'].to_numpy()

    # transform times to integers in order to sort them by time
    for i in range(inciV6.shape[0]):
        inciV6[i] = read_time(inciV6[i])

    # sort times
    inciV6 = np.sort(inciV6)

    # print first time
    print('First V6 incident time:', inciV6[0])
    print('Last V6 incident time:', inciV6[-1])

'''
First incident time: 20140618152119
First V6 incident time: 20210823223859
Last V6 incident time: 20220323140716
'''
