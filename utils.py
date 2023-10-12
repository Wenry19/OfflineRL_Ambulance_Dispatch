
from geopy.distance import geodesic
from datetime import datetime
import os
from pathlib import Path

VERY_SMALL_CONSTANT = 0.00000001
"""Very small constant.
"""

STATE_SIZE = 1041
"""State size.
"""

ACTION_SPACE = 258
"""Action space (number of different actions).
"""

IMPOSSIBLE_ACTION_VALUE_YEAR_AVAILABILITY = -1
"""Impossible action value due to year availability.
"""

IMPOSSIBLE_ACTION_AGENTS = lambda x : x != 0
"""Impossible action check for agents.
"""

IDX_TIME = 0
"""Index of incident time in a state.
"""

IDX_PRIORITY = 1
"""Index of incident priority in a state.
"""

IDX_COORDS = 2
"""Index of incident coordinates in a state.
"""

IDX_RES_POS = 3
"""Index of the resource positions in a state.
"""

IDX_DISTANCES = 4
"""Index of the distance information in a state.
"""

IDX_RES_AVAIL = 5
"""Index of the resource availability information in a state.
"""

IDX_MIN_DIST = 6
"""Index of the minimum distance in a state.
"""

IDX_MAX_DIST = 7
"""Index of the maximum distance in a state.
"""

IDX_MIN_WTIME = 8
"""Index of the minimum waiting time in a state.
"""

IDX_MAX_WTIME = 9
"""Index of the maximum waiting time in a state.
"""

IDX_STATE = 0
"""Index of the state in an experience.
"""

IDX_ACTION = 1
"""Index of the action in an experience.
"""

IDX_REWARD = 2
"""Index of the reward in an experience.
"""

IDX_NEXT_STATE = 3
"""Index of the next state in an experience.
"""

def check_experiences_version():
    print('-----------------------------------------')
    print('Version experiences.pkl:',
          str(datetime.fromtimestamp(os.path.getmtime(Path('generated_data/experiences/experiences.pkl')))))
    print('Version experiences_to_train_model.pkl:',
          str(datetime.fromtimestamp(os.path.getmtime(Path('generated_data/experiences_to_train_model/experiences_to_train_model.pkl')))))
    print('-----------------------------------------')

def get_resource_positions_and_avail_info(time, rm):
    """Given a time and an instance of a resourceManager, returns the estimated position and availability information
    of all the resources at that time. It also returns the minimum waiting time and the maximum waiting time.

    Parameters
    ----------
    time : int
        The time at which you want to know where all the resources were and what was their availability.
    rm : resourceManager instance
        A resourceManager instance.

    Returns
    -------
    list of tuples
        List containing all the resource positions (lat, long).
    list of ints
        List containig the availability information of the resources.
    int
        Minimum waiting time.
    int
        Maximum waiting time.
    """

    res_pos = []
    res_avail_info = []

    min_waiting_time = None
    max_waiting_time = None

    for res_name in rm.get_resource_names():

        pos, avail = rm.get_resource_state(res_name, time)

        res_pos.append(pos)
        res_avail_info.append(avail)

        if avail != IMPOSSIBLE_ACTION_VALUE_YEAR_AVAILABILITY:

            if min_waiting_time == None or avail < min_waiting_time:
                min_waiting_time = avail
            
            if max_waiting_time == None or avail > max_waiting_time:
                max_waiting_time = avail

    return res_pos, res_avail_info, min_waiting_time, max_waiting_time

def get_distances(incident_pos, res_pos, res_avail_info):
    """Given an incident position, the resource positions and the resource availability information,
    for each resource it returns the distance from its position to the incident. It also returns the minimum distance and the maximum distance.

    Parameters
    ----------
    incident_pos : tuple (lat, long)
        Incident coordinates.
    res_pos : list of tuples (lat, long)
        Coordinates of all the resources.
    res_avail_info : list of ints
        Availability information of all the resources.

    Returns
    -------
    list of floats
        List of distances.
    float
        Minimum distance.
    float
        Maximum distance.
    """

    distances = []

    min_dist = None
    max_dist = None

    for res_i, pos in enumerate(res_pos):

        dist = geodesic(incident_pos, pos).km

        distances.append(dist)

        if res_avail_info[res_i] != IMPOSSIBLE_ACTION_VALUE_YEAR_AVAILABILITY: # only taking into account the available resources!

            if min_dist == None or dist < min_dist:
                min_dist = dist
            
            if max_dist == None or dist > max_dist:
                max_dist = dist

    return distances, min_dist, max_dist

def reward_function(state, action):
    """Computes the immediate reward given a state and an action.

    Parameters
    ----------
    state : tuple
        The state.
    action : int
        The action.

    Returns
    -------
    float
        The normalized distance.
    float
        The immediate reward.
    """

    normalized_distance = (state[IDX_DISTANCES][action] - state[IDX_MIN_DIST]) / (state[IDX_MAX_DIST] - state[IDX_MIN_DIST])

    if state[IDX_RES_AVAIL][action] == IMPOSSIBLE_ACTION_VALUE_YEAR_AVAILABILITY:
        reward = -100

    else:
        if state[IDX_PRIORITY] == '0':
            reward = 11*(1-normalized_distance)-10
        elif state[IDX_PRIORITY] == '1':
            reward = 2*(1-normalized_distance)-1
        else:
            reward = 1-normalized_distance

    return normalized_distance, reward

def transform_state_to_model_input(state):
    """Given a state, it transforms it such that it can be the input of a model.

    Remember that each state contains (in a tuple):
        - incident time (int)
        - incident priority (string)
        - incident coordinates (tuple of floats)
        - position of each resource (list of tuples)
        - distances (list of floats)
        - availability information (list of ints)
        - min distance (float)
        - max distance (float)
        - min waiting time (int)
        - max waiting time (int)

    The transformed state will be a list with:
        - incident year min-max normalized.
        - incident month / 12.
        - incident day / 31.
        - incident hour / 23.
        - incident minute / 59.
        - incident second / 59.
        - incident priority / 2.
        - incident latitude / 90.
        - incident longitude / 180.
        for each resource:
            - resource latitude / 90.
            - resource longitude / 180.
            - resource distance (min-max normalized).
            - 0 available, -1 impossible action, > 0 how many time to be available in seconds (min-max normalized).

    Parameters
    ----------
    state : tuple
        The state to transform to model input.

    Returns
    -------
    list
        The transformed state.
    """

    model_input = [(int(str(state[IDX_TIME])[0:4]) - 2016) / (2022 - 2016), # incident year
                    int(str(state[IDX_TIME])[4:6]) / 12, # incident month
                    int(str(state[IDX_TIME])[6:8]) / 31, # incident day
                    int(str(state[IDX_TIME])[8:10]) / 23, # incident hour
                    int(str(state[IDX_TIME])[10:12]) / 59, # incident minute
                    int(str(state[IDX_TIME])[12:14]) / 59, # incident second
                    int(state[IDX_PRIORITY]) / 2, # incident priority
                    state[IDX_COORDS][0] / 90, # incident lat
                    state[IDX_COORDS][1] / 180, # incident lng
                    ]
    
    for res_i in range(len(state[IDX_RES_POS])):
        model_input.append(state[IDX_RES_POS][res_i][0] / 90) # resource lat
        model_input.append(state[IDX_RES_POS][res_i][1] / 180) # resource lng
        model_input.append((state[IDX_DISTANCES][res_i] - state[IDX_MIN_DIST]) / (state[IDX_MAX_DIST] - state[IDX_MIN_DIST])) # distance
        if state[IDX_RES_AVAIL][res_i] != IMPOSSIBLE_ACTION_VALUE_YEAR_AVAILABILITY:
            model_input.append((state[IDX_RES_AVAIL][res_i] - state[IDX_MIN_WTIME]) / (state[IDX_MAX_WTIME] - state[IDX_MIN_WTIME] + VERY_SMALL_CONSTANT)) # time to be available
        else:
            model_input.append(state[IDX_RES_AVAIL][res_i])

    return model_input

def get_experiences_to_train_model(experiences):
    """Given a set of experiences, it transforms the state and next state of each experience,
    such that they can be the input of a model (Neural Network).

    The transformation is in place!

    Parameters
    ----------
    experiences : numpy array
        A numpy array containing experiences to transform.

    Returns
    -------
    numpy array
        The given experiences but, with the states and next states being lists of floats with the values normalized.
        See the documentation of transform_state_to_model_input(state) to see more details about the transformation of the states.
    """

    #experiences_to_train_model = deepcopy(experiences)

    for e in experiences:

        e[IDX_STATE] = transform_state_to_model_input(e[IDX_STATE])
        e[IDX_NEXT_STATE] = transform_state_to_model_input(e[IDX_NEXT_STATE])

    return experiences

def read_time(date):
    """Given a date (string), returns the date as an integer.

    For example:
    Given '8/25/2020 4:16:07 PM' it returns the integer: 20200825161607
    That is computed as year*10000000000 + month*100000000 + day*1000000 + hour*10000 + minute*100 + second.

    Parameters
    ----------
    date : string
        The date to transform.

    Returns
    -------
    int
        The transformed date to an integer.
    """

    # example: 8/25/2020 4:16:07 PM
    aux = date.split('/')
    month = int(aux[0])
    day = int(aux[1])
    aux2 = aux[2].split(' ')
    year = int(aux2[0])
    aux3 = aux2[1].split(':')
    hour = int(aux3[0])
    minute = int(aux3[1])
    second = int(aux3[2])
    ampm = aux2[2]
    if (hour == 12 and ampm == 'AM') or (hour != 12 and ampm == 'PM'):
        hour += 12
        hour = hour % 24
    return year*10000000000 + month*100000000 + day*1000000 + hour*10000 + minute*100 + second

def calculate_duration(itime, ftime):
    """It calculates the duration in seconds between itime and ftime.

    Parameters
    ----------
    itime : int
        Initial time.
    ftime : int
        Final time.

    Returns
    -------
    int
        The number of seconds between itime and ftime.
    """

    # https://stackoverflow.com/questions/4362491/how-do-i-check-the-difference-in-seconds-between-two-dates

    iaux = str(itime)
    iaux = datetime(year=int(iaux[0:4]), month=int(iaux[4:6]), day=int(iaux[6:8]),
                    hour=int(iaux[8:10]), minute=int(iaux[10:12]), second=int(iaux[12:14]))
    
    faux = str(ftime)
    faux = datetime(year=int(faux[0:4]), month=int(faux[4:6]), day=int(faux[6:8]),
                    hour=int(faux[8:10]), minute=int(faux[10:12]), second=int(faux[12:14]))
    
    return int((faux - iaux).total_seconds())

def swiss_coords_tolatlon(E, N):
    """
    specific conversion from swiss system to GPS coordinates
    source : https://www.swisstopo.admin.ch/fr/cartes-donnees-en-ligne/
    calculation-services/navref.html
    :param E: (double) x in meters (0,0) in Bern
    :param N: (double) y in meters (0,0) in Bern
    :return: (double,double) latitude and longitude
    """

    if E == 0 or N == 0:
        return 0, 0
    # look if it is either MN95 or MN03 format
    # i.e. if it is MN03, E has 6 decimal numbers, otherwise 7
    r = E / 100000
    yp = (E - 2600000) / 1000000 if r > 10 else (E - 600000) / 1000000
    xp = (N - 1200000) / 1000000 if r > 10 else (N - 200000) / 1000000
    lambdap = 2.6779094 + 4.728982 * yp + 0.791484 * xp * yp + 0.1306 * yp * xp * xp - 0.0436 * yp * yp * yp
    phip = 16.9023892 + 3.238272 * xp - 0.270978 * yp * yp - 0.002528 * xp * xp - 0.0447 * yp * yp * xp - 0.0140 * xp * xp * xp
    return phip * 100./36, lambdap * 100./36
