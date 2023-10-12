
"""Main Build Experiences

This script prints a menu with all the different preprocessing steps that have to be executed in order to obtain the experiences from the data.

The user is able to choose which option wants to be executed. Note that some options have dependencies with previous options,
so it is recommended to execute them in the order they appear.

This file contains the following functions:

    * resource_info()
        Saves the positions of the resources during time using a resourceManager instance. It also saves the resource names.
    * preprocess_incidents(year)
        Given a year, it preprocesses and saves all the incidents data of that year using a preprocessIncidents instance.
    * concatenate_preprocessed_incidents()
        Concatenates all the preprocessed incidents data (separated by year), sorts the preprocessed incidents by time,
        deletes incidents with repeated ccCallcardUid and creates a train/environment split of the incidents data.
    * combine_datasets()
        For each preprocessed incident finds its interventions using the ccCallcardUid.
        It saves the final data that will be used to build the experiences.
    * build_experiences()
        Creates the final dataset of experiences (state, action, reward, next state)
        that will be used to train the reinforcement learning agents in an offline setting.
        It saves the experiences in a friendly format and the experiences with the states ready to be the input of the models
        (for instance, deep neural networks).
"""

import pandas as pd
import numpy as np

import pickle
from pathlib import Path
import os
import sys
sys.path.append(os.getcwd())

from utils import get_resource_positions_and_avail_info, reward_function, get_experiences_to_train_model, get_distances, check_experiences_version
from utils import IDX_STATE, IDX_ACTION, IDX_REWARD, IDX_NEXT_STATE, IMPOSSIBLE_ACTION_VALUE_YEAR_AVAILABILITY
from resource_manager import resourceManager
from preprocess_incidents import preprocessIncidents

NUM_INCIDENTS_ENVIRONMENT = 50000
"""Number of incidents reserved for the environment. They will not be used to train the reinforcement learning agents.
"""

def resource_info():
    """Saves the positions of the resources during time using a resourceManager instance. It also saves the resource names.
    """

    res_man = resourceManager()
    pickle.dump(res_man.get_resource_pos_avail_info(), open(Path('generated_data/resource_pos_avail_info.pkl'), 'wb'))
    pickle.dump(res_man.get_resource_names(), open(Path('generated_data/resource_names.pkl'), 'wb'))

def preprocess_incidents(year):
    """Given a year, it preprocesses and saves all the incidents data of that year using a preprocessIncidents instance.
    Each row of the generated dataset will contain: ccCallcardUid, incident time, incident priority, incident coordinates.

    Parameters
    ----------
    year : str
        The year of the incidents to be preprocessed.
    """

    if year == 'V6':
        prep_incidents = preprocessIncidents(Path('CHUV/ChuvExportIncidentsV6.csv')).preprocess()
    else:
        prep_incidents = preprocessIncidents(Path('CHUV/ChuvExportIncidents_V5_' + year + '.csv')).preprocess()
    
    print('Num of preprocessed incidents:', prep_incidents.shape[0])
    
    pickle.dump(prep_incidents, open(Path('generated_data/preprocessed_incidents/my_prep_incidents_' + year + '.pkl'), 'wb'))

def concatenate_preprocessed_incidents():
    """Concatenates all the preprocessed incidents data (separated by year), sorts the preprocessed incidents by time,
    deletes incidents with repeated ccCallcardUid and creates a train/environment split of the incidents data.
    """

    i2016 = pickle.load(open(Path('generated_data/preprocessed_incidents/my_prep_incidents_2016.pkl'), 'rb'))
    i2017 = pickle.load(open(Path('generated_data/preprocessed_incidents/my_prep_incidents_2017.pkl'), 'rb'))
    i2018 = pickle.load(open(Path('generated_data/preprocessed_incidents/my_prep_incidents_2018.pkl'), 'rb'))
    i2019 = pickle.load(open(Path('generated_data/preprocessed_incidents/my_prep_incidents_2019.pkl'), 'rb'))
    i2020 = pickle.load(open(Path('generated_data/preprocessed_incidents/my_prep_incidents_2020.pkl'), 'rb'))
    i2021 = pickle.load(open(Path('generated_data/preprocessed_incidents/my_prep_incidents_2021.pkl'), 'rb'))
    iV6 = pickle.load(open(Path('generated_data/preprocessed_incidents/my_prep_incidents_V6.pkl'), 'rb'))

    all_prep_inci = np.concatenate((i2016, i2017, i2018, i2019, i2020, i2021, iV6), axis=0) # concatenate the data from all the years

    all_prep_inci = all_prep_inci[all_prep_inci[:, 1].argsort()] # sort them by time

    # Delete incidents that have more than one row in the dataset (repeated ccCallcardUid)
    # Since I am not sure what does that mean, I will delete these cases (they are not that much, maybe they are mistakes).

    unq, unq_cnt = np.unique(all_prep_inci[:, 0], return_counts=True)
    mask = unq_cnt > 1
    not_unique_ids = unq[mask]
    mask = np.logical_not(np.isin(all_prep_inci[:, 0], not_unique_ids)) # https://numpy.org/doc/stable/reference/generated/numpy.isin.html
    all_prep_inci = all_prep_inci[mask]

    train_prep_inci = all_prep_inci[0:len(all_prep_inci)-NUM_INCIDENTS_ENVIRONMENT]
    env_prep_inci = all_prep_inci[len(all_prep_inci)-NUM_INCIDENTS_ENVIRONMENT:]

    print('Shape of all preprocessed incidents:', all_prep_inci.shape)
    print('Shape of preprocessed incidents (for training):', train_prep_inci.shape)
    print('Shape of preprocessed incidents (for the environment):', env_prep_inci.shape)
    print('Percentage of repeated ccCallcardUid: ' + str(100*not_unique_ids.shape[0]/unq.shape[0]) + '%')
    
    pickle.dump(all_prep_inci, open(Path('generated_data/preprocessed_incidents/all_prep_inci.pkl'), 'wb'))
    pickle.dump(train_prep_inci, open(Path('generated_data/preprocessed_incidents/train_prep_inci.pkl'), 'wb'))
    pickle.dump(env_prep_inci, open(Path('generated_data/preprocessed_incidents/env_prep_inci.pkl'), 'wb'))

def combine_datasets():
    """For each preprocessed incident finds its interventions using the ccCallcardUid.
    It saves the final data that will be used to build the experiences.
    It does not include the cases when a busy ambulance was chosen for an incident.
    Each row of the generated dataset will contain:
    ccCallcardUid, incident time, incident priority, incident coordinates, intervention resource name.
    """

    incidents_data = pickle.load(open(Path('generated_data/preprocessed_incidents/train_prep_inci.pkl'), 'rb'))

    inter2016 = pd.read_csv(Path('CHUV/ChuvExportInterventions_V5_2016.csv'), dtype=str)[['ccCallcardUid', 'Ambulance', 'ResType']].to_numpy()
    inter2017 = pd.read_csv(Path('CHUV/ChuvExportInterventions_V5_2017.csv'), dtype=str)[['ccCallcardUid', 'Ambulance', 'ResType']].to_numpy()
    inter2018 = pd.read_csv(Path('CHUV/ChuvExportInterventions_V5_2018.csv'), dtype=str)[['ccCallcardUid', 'Ambulance', 'ResType']].to_numpy()
    inter2019 = pd.read_csv(Path('CHUV/ChuvExportInterventions_V5_2019.csv'), dtype=str)[['ccCallcardUid', 'Ambulance', 'ResType']].to_numpy()
    inter2020 = pd.read_csv(Path('CHUV/ChuvExportInterventions_V5_2020.csv'), dtype=str)[['ccCallcardUid', 'Ambulance', 'ResType']].to_numpy()
    inter2021 = pd.read_csv(Path('CHUV/ChuvExportInterventions_V5_2021.csv'), dtype=str)[['ccCallcardUid', 'Ambulance', 'ResType']].to_numpy()
    interV6 = pd.read_csv(Path('CHUV/ChuvExportInterventionsV6.csv'), dtype=str)[['ccCallcardUid', 'Ambulance', 'ResType']].to_numpy()

    # concatenate the data from all the years
    interventions_data = np.concatenate((inter2016, inter2017, inter2018, inter2019, inter2020, inter2021, interV6), axis=0)

    interventions_data = interventions_data[interventions_data[:, 2] == 'Vehicle']
    interventions_data = interventions_data[:, [0, 1]]

    # no missing values in interventions data

    # resource manager
    rm = resourceManager(precomputed_pos_avail_info=pickle.load(open(Path('generated_data/resource_pos_avail_info.pkl'), 'rb')))

    final_data = []

    count_actions_with_busy_res_p0 = 0
    count_actions_with_busy_res_p1 = 0
    count_actions_with_busy_res_pother = 0
    count_not_avail_res_due_to_year_avail = 0

    num_incidents_in_the_final_data = 0

    for incident in incidents_data: # iterate incidents

        callid = incident[0]
        incident_time = incident[1]
        
        # find the interventions with the same callid
        interventions = interventions_data[interventions_data[:, 0] == callid]

        included_incident = False

        for idx in range(interventions.shape[0]):

            if interventions[idx, 1] in rm.get_resource_names(): # we have valid info about this resource
                        
                _, availability = rm.get_resource_state(interventions[idx, 1], incident_time)

                if availability == 0:
                    included_incident = True
                    row = list(incident)
                    row.append(interventions[idx, 1]) # Ambulance
                    final_data.append(row)

                elif availability > 0:
                    if incident[2] == '0':
                        count_actions_with_busy_res_p0 += 1
                    elif incident[2] == '1':
                        count_actions_with_busy_res_p1 += 1
                    else:
                        count_actions_with_busy_res_pother += 1

                elif availability == IMPOSSIBLE_ACTION_VALUE_YEAR_AVAILABILITY: # not possible action
                    # if we enter here, it is bad...
                    count_not_avail_res_due_to_year_avail += 1

        num_incidents_in_the_final_data += int(included_incident)

    final_data = np.array(final_data, dtype=object)

    print('Shape of final data, ready to build experiences:', final_data.shape)
    print('Number of incidents in the final data:', num_incidents_in_the_final_data)
    print('Mean number of interventions per incident in the final data:', final_data.shape[0]/num_incidents_in_the_final_data)
    print('Number of actions with not available resource P0:', count_actions_with_busy_res_p0)
    print('Number of actions with not available resource P1:', count_actions_with_busy_res_p1)
    print('Number of actions with not available resource Pother:', count_actions_with_busy_res_pother)
    print('Number of actions with not available resource TOTAL:', count_actions_with_busy_res_p0 \
          + count_actions_with_busy_res_p1 + count_actions_with_busy_res_pother)
    print('\t- Not available resource due to year availability:', count_not_avail_res_due_to_year_avail)

    pickle.dump(final_data, open(Path('generated_data/data_to_build_experiences.pkl'), 'wb'))

def build_experiences():
    """Creates the final dataset of experiences (state, action, reward, next state)
    that will be used to train the reinforcement learning agents in an offline setting.
    It saves the experiences in a friendly format and the experiences with the states ready to be the input of the models
    (for instance, deep neural networks).
    """

    experiences = []

    # load the data and initialize the resource manager
    data = pickle.load(open(Path('generated_data/data_to_build_experiences.pkl'), 'rb'))
    #data = data[np.random.choice(data.shape[0], replace=False, size=20000)]
    rm = resourceManager(precomputed_pos_avail_info=pickle.load(open(Path('generated_data/resource_pos_avail_info.pkl'), 'rb')))
    res_names = rm.get_resource_names()

    last_next_state = None

    for i in range(data.shape[0]):

        experience = [None, None, None, None] # 0: state, 1: action, 2: reward, 3: next state

        # STATE
        if last_next_state == None:

            res_pos, res_avail_info, min_wtime, max_wtime = get_resource_positions_and_avail_info(data[i, 1], rm)
            distances, min_dist, max_dist = get_distances(data[i, 3], res_pos, res_avail_info)
            experience[IDX_STATE] = (data[i, 1], data[i, 2], data[i, 3], res_pos, distances, res_avail_info,
                               min_dist, max_dist, min_wtime, max_wtime)

        else:
            experience[IDX_STATE] = last_next_state

        # ACTION
        experience[IDX_ACTION] = res_names.index(data[i, 4])

        # REWARD
        _, experience[IDX_REWARD] = reward_function(experience[IDX_STATE], experience[IDX_ACTION])

        # NEXT STATE
        if i+1 < data.shape[0]:

            res_pos, res_avail_info, min_wtime, max_wtime = get_resource_positions_and_avail_info(data[i+1, 1], rm)
            distances, min_dist, max_dist = get_distances(data[i+1, 3], res_pos, res_avail_info)
            experience[IDX_NEXT_STATE] = (data[i+1, 1], data[i+1, 2], data[i+1, 3], res_pos, distances, res_avail_info,
                             min_dist, max_dist, min_wtime, max_wtime)
            last_next_state = experience[IDX_NEXT_STATE]

            experiences.append(experience)

    experiences = np.array(experiences, dtype=object)

    print('Total number of (s, a, r, next_s):', experiences.shape[0])
    print('Total number of instances in the data used to build experiences:', data.shape[0])

    pickle.dump(experiences, open(Path('generated_data/experiences/experiences.pkl'), 'wb'))
    pickle.dump(get_experiences_to_train_model(experiences),
                open(Path('generated_data/experiences_to_train_model/experiences_to_train_model.pkl'), 'wb'))
    
    check_experiences_version()

if __name__ == '__main__':

    print('\nChoose an option (write the number of the desired option):\n')
    print('\t0. Save resource info (resourceManager).')
    print('\t1. Preprocess incidents data.')
    print('\t2. Concatenate preprocessed incidents.')
    print('\t3. Combine datasets (incidents and interventions data).')
    print('\t4. Build experiences.')
    print('\t5. Exit.')
    print('\t6. EXTRA OPTION: Execute options 0, 2, 3, 4 and exit.')
    print('\nNote: execute options in order (there are dependencies).\n')

    opt = int(input())

    while opt != 5:

        if opt == 0:
            resource_info()
            print('Resource info saved.')
            print('Choose the next option:\n')

        elif opt == 1:
            print('Type the year or V6 for the last dataset:')
            year = input()
            print('YEAR:', year)
            preprocess_incidents(year)
            print('Incidents data preprocessed.')
            print('Choose the next option:\n')

        elif opt == 2:
            concatenate_preprocessed_incidents()
            print('Preprocessed incidents concatenated.')
            print('Choose the next option:\n')

        elif opt == 3:
            combine_datasets()
            print('Datasets combined.')
            print('Choose the next option:\n')

        elif opt == 4:
            build_experiences()
            print('Experience buffer constructed.')
            print('Choose the next option:\n')

        elif opt == 6:
            print('\n EXTRA OPTION 6\n')
            print('**********OPTION 0**********')
            resource_info()
            print('Resource info saved.\n\n')

            print('**********OPTION 2**********')
            concatenate_preprocessed_incidents()
            print('Preprocessed incidents concatenated.\n\n')

            print('**********OPTION 3**********')
            combine_datasets()
            print('Datasets combined.\n\n')

            print('**********OPTION 4**********')
            build_experiences()
            print('Experience buffer constructed.\n\n')

            break

        opt = int(input())
