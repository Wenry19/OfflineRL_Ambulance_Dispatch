
import pandas as pd
from pathlib import Path
import numpy as np

if __name__ == '__main__':

    res_2014_2018 = pd.read_csv(Path('CHUV/ChuvExportResources_V5_2014_2018.csv'), dtype=str)
    res_2019_2021 = pd.read_csv(Path('CHUV/ChuvExportResources_V5_2019_2021.csv'), dtype=str)
    res_V6 = pd.read_csv(Path('CHUV/ChuvExportResourcesV6.csv'), dtype=str)

    data = [('2014_2018', res_2014_2018), ('2019_2021', res_2019_2021), ('V6', res_V6)]

    for id, df in data:

        print('****' + id + '****')

        # number instances/attributes
        print('Columns:', list(df.columns))
        print('Number of columns:', df.shape[1])
        print('Number of rows:', df.shape[0])
        print()

        # unique resType values
        print('unique resType values:', np.unique(df['resType']))
        print()

        # number of vehicles (ambulances) in the data

        print('Number of vehicles:', df[df['resType'] == 'Vehicle'].shape[0])
        print()

        print('#########################################################')

