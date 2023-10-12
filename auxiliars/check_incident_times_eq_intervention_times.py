
from pathlib import Path
import pandas as pd
import numpy as np

if __name__ == '__main__':

    inci_2014 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2014.csv'), dtype=str)[['ccCallcardUid', 'LocalTime']].to_numpy()
    inci_2015 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2015.csv'), dtype=str)[['ccCallcardUid', 'LocalTime']].to_numpy()
    inci_2016 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2016.csv'), dtype=str)[['ccCallcardUid', 'LocalTime']].to_numpy()
    inci_2017 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2017.csv'), dtype=str)[['ccCallcardUid', 'LocalTime']].to_numpy()
    inci_2018 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2018.csv'), dtype=str)[['ccCallcardUid', 'LocalTime']].to_numpy()
    inci_2019 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2019.csv'), dtype=str)[['ccCallcardUid', 'LocalTime']].to_numpy()
    inci_2020 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2020.csv'), dtype=str)[['ccCallcardUid', 'LocalTime']].to_numpy()
    inci_2021 = pd.read_csv(Path('CHUV/ChuvExportIncidents_V5_2021.csv'), dtype=str)[['ccCallcardUid', 'LocalTime']].to_numpy()
    inci_V6 = pd.read_csv(Path('CHUV/ChuvExportIncidentsV6.csv'), dtype=str)[['ccCallcardUid', 'LocalTime']].to_numpy()
    
    inter_2014 = pd.read_csv(Path('CHUV/ChuvExportInterventions_V5_2014.csv'), dtype=str)[['ccCallcardUid', 'CreationUtc']].to_numpy()
    inter_2015 = pd.read_csv(Path('CHUV/ChuvExportInterventions_V5_2015.csv'), dtype=str)[['ccCallcardUid', 'CreationUtc']].to_numpy()
    inter_2016 = pd.read_csv(Path('CHUV/ChuvExportInterventions_V5_2016.csv'), dtype=str)[['ccCallcardUid', 'CreationUtc']].to_numpy()
    inter_2017 = pd.read_csv(Path('CHUV/ChuvExportInterventions_V5_2017.csv'), dtype=str)[['ccCallcardUid', 'CreationUtc']].to_numpy()
    inter_2018 = pd.read_csv(Path('CHUV/ChuvExportInterventions_V5_2018.csv'), dtype=str)[['ccCallcardUid', 'CreationUtc']].to_numpy()
    inter_2019 = pd.read_csv(Path('CHUV/ChuvExportInterventions_V5_2019.csv'), dtype=str)[['ccCallcardUid', 'CreationUtc']].to_numpy()
    inter_2020 = pd.read_csv(Path('CHUV/ChuvExportInterventions_V5_2020.csv'), dtype=str)[['ccCallcardUid', 'CreationUtc']].to_numpy()
    inter_2021 = pd.read_csv(Path('CHUV/ChuvExportInterventions_V5_2021.csv'), dtype=str)[['ccCallcardUid', 'CreationUtc']].to_numpy()
    inter_V6 = pd.read_csv(Path('CHUV/ChuvExportInterventionsV6.csv'), dtype=str)[['ccCallcardUid', 'CreationUtc']].to_numpy()


    inci_data = np.concatenate((inci_2014, inci_2015, inci_2016, inci_2017,\
                                inci_2018, inci_2019, inci_2020, inci_2021, inci_V6), axis=0)
    
    inter_data = np.concatenate((inter_2014, inter_2015, inter_2016, inter_2017,\
                                inter_2018, inter_2019, inter_2020, inter_2021, inter_V6), axis=0)
    
    all_eq = True
    
    for inci in inci_data:

        rel_inter = inter_data[inter_data[:, 0] == inci[0]]

        for inter in rel_inter:

            if inci[1] != inter[1]:
                all_eq = False
                print('incident time != intervention time found!', inci, inter)

    if all_eq:
        print('All incident times equal to their corresponding intervention times.') # TRUE!

