
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

if __name__ == '__main__':

    incidents_data = pickle.load(open(Path('generated_data/preprocessed_incidents/all_prep_inci.pkl'), 'rb'))

    print(incidents_data.shape)
    print(len(np.unique(incidents_data[:, 0])))

    print('Priorities:', np.unique(incidents_data[:, 2]))

    print(incidents_data[0])

    #print(type(incidents_data))

    #print(type(incidents_data[0, 2]))
    #print(incidents_data[0, 2])

    print('NULLS?', pd.isnull(incidents_data).any())

    #print(incidents_data[0, [0, 1, 2]])
