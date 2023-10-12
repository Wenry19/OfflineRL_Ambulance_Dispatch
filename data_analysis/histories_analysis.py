
import pandas as pd
from pathlib import Path
import numpy as np

def count_E_values_coords(data):
    E_count = 0
    for i in range(data.shape[0]):
        if 'E' in str(data['CoordX'].iloc[i]):
            E_count += 1
    return E_count

if __name__ == '__main__':

    data_2014 = pd.read_csv(Path('CHUV/ChuvExportResourceHistory_V5_2014.csv'), dtype=str)
    data_2015 = pd.read_csv(Path('CHUV/ChuvExportResourceHistory_V5_2015.csv'), dtype=str)
    data_2016 = pd.read_csv(Path('CHUV/ChuvExportResourceHistory_V5_2016.csv'), dtype=str)
    data_2017 = pd.read_csv(Path('CHUV/ChuvExportResourceHistory_V5_2017.csv'), dtype=str)
    data_2018 = pd.read_csv(Path('CHUV/ChuvExportResourceHistory_V5_2018.csv'), dtype=str)
    data_2019 = pd.read_csv(Path('CHUV/ChuvExportResourceHistory_V5_2019.csv'), dtype=str)
    data_2020 = pd.read_csv(Path('CHUV/ChuvExportResourceHistory_V5_2020.csv'), dtype=str)
    data_2021 = pd.read_csv(Path('CHUV/ChuvExportResourceHistory_V5_2021.csv'), dtype=str)
    data_V6 = pd.read_csv(Path('CHUV/ChuvExportResourceHistoryV6.csv'), dtype=str)

    dataframes = [('2014', data_2014), ('2015', data_2015), ('2016', data_2016),
                  ('2017', data_2017), ('2018', data_2018), ('2019', data_2019),
                  ('2020', data_2020), ('2021', data_2021), ('V6', data_V6)]
    
    for id, df in dataframes:

        print('****Year: ' + id + '****')

        # SHAPE AND BASIC INFORMATION
        print('Columns:', list(df.columns))
        print('Number of columns:', df.shape[1])
        print('Number of rows:', df.shape[0])
        print('Number of Demande d\'engagement:', df[df['Text'] == 'Demande d\'engagement'].shape[0])
        print('Number of Terminé (en rue):', df[df['Text'] == 'Terminé (en rue)'].shape[0])
        print('Number of Annulé (en centrale):', df[df['Text'] == 'Annulé (en centrale)'].shape[0])
        print()

        # MISSING VALUES
        print('Missing values (nan):\n' + str(df.isna().sum()))
        print('Missing values (E in coords):', count_E_values_coords(df))
        print('Percentage (%) of rows with missing values in the coordinates:', 100*(df['CoordX'].isna().sum() + count_E_values_coords(df))/df.shape[0])
        if not id in {'2014', '2015'}:
            mask = np.char.find(np.array(df['CoordX'].to_numpy(), dtype=str), 'E') != -1 # https://numpy.org/devdocs/reference/generated/numpy.char.find.html
            resource_names_with_MV = np.unique(df['Ambulance'][mask])
            print('Resource names with missing values:', resource_names_with_MV)
        print()

        # OTHER THINGS TO CHECK

        print('#########################################################')


