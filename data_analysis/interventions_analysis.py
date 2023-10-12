
import pandas as pd
from pathlib import Path
import numpy as np

if __name__ == '__main__':

    inci_2014 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2014.csv'), dtype=str)
    inci_2015 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2015.csv'), dtype=str)
    inci_2016 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2016.csv'), dtype=str)
    inci_2017 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2017.csv'), dtype=str)
    inci_2018 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2018.csv'), dtype=str)
    inci_2019 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2019.csv'), dtype=str)
    inci_2020 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2020.csv'), dtype=str)
    inci_2021 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2021.csv'), dtype=str)
    inci_V6 = pd.read_csv(Path('CHUV/ChuvExportIncidentsV6.csv'), dtype=str)
    
    data_2014 = pd.read_csv(Path('CHUV/ChuvExportInterventions_V5_2014.csv'), dtype=str)
    data_2015 = pd.read_csv(Path('CHUV/ChuvExportInterventions_V5_2015.csv'), dtype=str)
    data_2016 = pd.read_csv(Path('CHUV/ChuvExportInterventions_V5_2016.csv'), dtype=str)
    data_2017 = pd.read_csv(Path('CHUV/ChuvExportInterventions_V5_2017.csv'), dtype=str)
    data_2018 = pd.read_csv(Path('CHUV/ChuvExportInterventions_V5_2018.csv'), dtype=str)
    data_2019 = pd.read_csv(Path('CHUV/ChuvExportInterventions_V5_2019.csv'), dtype=str)
    data_2020 = pd.read_csv(Path('CHUV/ChuvExportInterventions_V5_2020.csv'), dtype=str)
    data_2021 = pd.read_csv(Path('CHUV/ChuvExportInterventions_V5_2021.csv'), dtype=str)
    data_V6 = pd.read_csv(Path('CHUV/ChuvExportInterventionsV6.csv'), dtype=str)

    dataframes = [('2014', data_2014), ('2015', data_2015), ('2016', data_2016),
                  ('2017', data_2017), ('2018', data_2018), ('2019', data_2019),
                  ('2020', data_2020), ('2021', data_2021), ('V6', data_V6)]
    
    incidents = [inci_2014, inci_2015, inci_2016, inci_2017, inci_2018, inci_2019, inci_2020, inci_2021, inci_V6]

    for x, (id, df) in enumerate(dataframes):

        print('****Year: ' + id + '****')

        # SHAPE AND BASIC INFORMATION
        print('Columns:', list(df.columns))
        print('Number of columns:', df.shape[1])
        print('Number of rows:', df.shape[0])
        print()

        # MISSING VALUES
        print('Missing values: ' + str(pd.isnull(df).any().any()))
        print()

        # OTHER THINGS TO CHECK
        print('ResType uniques:', np.unique(df['ResType']))
        print()
        print('Number of rows ResType == Vehicle:', df[df['ResType'] == 'Vehicle'].shape[0])
        print()
        unq_callids, unq_callids_count = np.unique(df['ccCallcardUid'][df['ResType'] == 'Vehicle'], return_counts=True)
        print('Number of unique ccCallcardUid with ResType == Vehicle:', unq_callids.shape[0])
        print('Mean number of ambulances (vehicles) sent per incident:', np.sum(unq_callids_count)/unq_callids_count.shape[0])
        print()
        callid_unq_inci = set(np.unique(incidents[x]['ccCallcardUid']))
        callid_unq_inter = set(np.unique(df['ccCallcardUid']))

        print('There are incidents without interventions:', len(callid_unq_inci) > len(callid_unq_inter))
        print('There are interventions without an associated incident:', not(callid_unq_inter.issubset(callid_unq_inci)))
        print('ccCallcardUid in interventions and not in incidents:', callid_unq_inter - callid_unq_inci)
        print()

        print('#########################################################')


