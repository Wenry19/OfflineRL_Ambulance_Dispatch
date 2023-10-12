
# https://www.gymlibrary.dev/content/environment_creation/

import gym
from gym import spaces
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from geopy.distance import geodesic

import os
import sys
sys.path.append(os.getcwd())
from build_experiences.resource_manager import resourceManager
from utils import reward_function, calculate_duration, ACTION_SPACE, IDX_STATE, IDX_TIME, IMPOSSIBLE_ACTION_VALUE_YEAR_AVAILABILITY

BUSY_TIME = 3546 # seconds
"""Mean number of seconds that takes an ambulance to be available again after being chosen.
"""

class emergenciesEnv(gym.Env):
    """
    Custom environment, subclass of gym.Env.
    
    The main aim of this class is to implement a custom gym environment and evaluate the trained agents in it.

    Attributes
    ----------
    episode_length : int
        Episode length.
    training_incidents : boolean
        If we use training incidents to generate observations, otherwise the incidents reserved for the environment will be used.
    action_space : gym spaces
        Definition of the action space.
    render_mode : int
        Render mode.
    _env_incidents_data : numpy array
        Incidents data used by the environment.
    _num_incidents : int
        Number of incidents in _env_incidents_data.
    _rm : resourceManager instance
        A resourceManager instance.
    _res_coords_histories : dict
        A dictionary with key = resource and value = its coordinates history, for all the resources (actions).
    _current_state : tuple
        The current state.
    _incident_idx :  int
        The current incident index.
    _num_steps : int
        The number of steps done.
    _busy_resources : dict
        A dictionary with key = a busy resource and value = when it will be available again.
    _res_pos : list
        A list containing the current position of each resource.

    Methods
    -------
    reset(seed=None, options=None)
        Resets the environment.
    step(action)
        Does the action passed as a parameter in the environment.
    render()
        TODO
    """

    def __init__(self, episode_length=100, render_mode=None, training_incidents=False):
        """
        Parameters
        ----------
        episode_length : int, optional
            Episode length.
        render_mode : int, optional
            Render mode.
        training_incidents : boolean, optional
            If we use training incidents to generate observations, otherwise the incidents reserved for the environment will be used.
        """

        self.episode_length = episode_length

        self.training_incidents = training_incidents

        self.action_space = spaces.Discrete(ACTION_SPACE)

        self.render_mode = render_mode # TODO

        if not self.training_incidents:
            self._env_incidents_data = pickle.load(open(Path('generated_data/preprocessed_incidents/env_prep_inci.pkl'), 'rb'))[:, [1, 2, 3]]
            # 0: incident time, 1: incident priority, 2: incident coordinates
        else:
            self._env_incidents_data = pickle.load(open(Path('generated_data/preprocessed_incidents/train_prep_inci.pkl'), 'rb'))[:, [1, 2, 3]]

        self._num_incidents = self._env_incidents_data.shape[0]

        self._rm = resourceManager(precomputed_pos_avail_info=pickle.load(open(Path('generated_data/resource_pos_avail_info.pkl'), 'rb')))

        self._initialize_res_coords_histories() # to sample coordinates for each resource (from resource manager)

    def _initialize_res_coords_histories(self):
        """Initializes _res_coords_histories.
        """

        self._res_coords_histories = dict()

        for res_i, res_n in enumerate(self._rm.get_resource_names()):
            self._res_coords_histories[res_i] = self._rm.get_coordinates_history(res_n)

    def _add_seconds(self, time, seconds_to_add):
        """Given a time and a number of seconds to add, it returns the resulting time after adding the seconds.

        Parameters
        ----------
        time : int
            The time.
        seconds_to_add : int
            The number of seconds to add.

        Returns
        -------
        int
            The resulting time after adding the seconds.
        """

        aux = str(time)

        time2 = datetime(year=int(aux[0:4]), month=int(aux[4:6]), day=int(aux[6:8]),
                         hour=int(aux[8:10]), minute=int(aux[10:12]), second=int(aux[12:14]))
        
        resulting_time = time2 + timedelta(seconds=seconds_to_add)

        resulting_time = resulting_time.year*10000000000 + resulting_time.month*100000000 +\
            resulting_time.day*1000000 + resulting_time.hour*10000 + resulting_time.minute*100 + resulting_time.second

        return resulting_time

    def _get_obs(self):
        """Gets a new observation.

        Returns
        -------
        tuple
            The new observation as the new current state (_current_state).
        """

        # next incident
        self._incident_idx += 1
        incident = self._env_incidents_data[self._incident_idx]
        incident_time = incident[0]

        # resources info
        res_avail_info = []
        distances = []
        min_dist = None
        max_dist = None
        min_wtime = None
        max_wtime = None

        for res_i, res_n in enumerate(self._rm.get_resource_names()):

            if not res_i in self._busy_resources:
                res_avail_info.append(self._rm.year_availability(res_n, incident_time))
            else:
                if incident_time >= self._busy_resources[res_i]:
                    res_avail_info.append(self._rm.year_availability(res_n, incident_time))
                    del self._busy_resources[res_i]
                    # sample a new position for the resource that becomes available again
                    self._res_pos[res_i] = self._res_coords_histories[res_i][self.np_random.integers(0, len(self._res_coords_histories[res_i]), size=1, dtype=int)[0]]
                else:
                    res_avail_info.append(calculate_duration(incident_time, self._busy_resources[res_i]))

            distances.append(geodesic(incident[2], self._res_pos[res_i]).km)

            if res_avail_info[res_i] != IMPOSSIBLE_ACTION_VALUE_YEAR_AVAILABILITY:

                if min_dist == None or distances[res_i] < min_dist:
                    min_dist = distances[res_i]
                
                if max_dist == None or distances[res_i] > max_dist:
                    max_dist = distances[res_i]

                if min_wtime == None or res_avail_info[res_i] < min_wtime:
                    min_wtime = res_avail_info[res_i]
                
                if max_wtime == None or res_avail_info[res_i] > max_wtime:
                    max_wtime = res_avail_info[res_i]

        self._current_state = (incident[0], incident[1], incident[2], self._res_pos, distances, res_avail_info,
                               min_dist, max_dist, min_wtime, max_wtime)

        return self._current_state

    def _get_info(self):
        """TODO
        """

        return None

    def reset(self, seed=None, options=None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Seed for the np_random generator.
        options : int, optional
            Special options (not used).

        Returns
        -------
        tuple
            The starting state.
        tuple
            Extra information.
        
        """

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # choose a random incident to start (eye with episode length)
        self._incident_idx = self.np_random.integers(0, self._num_incidents-self.episode_length, size=1, dtype=int)[0] - 1
        # -1 because when calling self._get_obs() it will take the next incident, which would be self._incident_idx + 1

        self._num_steps = 0

        # not available resources empty
        self._busy_resources = dict()

        # initial resources positions
        self._res_pos = []
        for res_i in range(len(self._rm.get_resource_names())):
            # sample one position from its position history
            self._res_pos.append(self._res_coords_histories[res_i][self.np_random.integers(0, len(self._res_coords_histories[res_i]), size=1, dtype=int)[0]])
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Does the action passed as a parameter in the environment.

        Parameters
        ----------
        action : int
            The action to be done.

        Returns
        -------
        tuple
            The new current state.
        tuple
            The normalized distance and the immediate reward.
        boolean
            If the episode is terminated.
        boolean
            Always False (truncated).
        tuple
            Extra info.
        """

        # check if the resource is busy or not
        if action in self._busy_resources:
            self._busy_resources[action] = self._add_seconds(self._busy_resources[action], BUSY_TIME)
        else:
            self._busy_resources[action] = self._add_seconds(self._current_state[IDX_TIME], BUSY_TIME) # incident time + busy time

        # compute the reward
        reward = reward_function(self._current_state, action)

        # get the next observation
        observation = self._get_obs()

        # get additional info
        info = self._get_info()

        self._num_steps += 1

        # check if the episode is done
        terminated = (self._num_steps == self.episode_length)
        
        return observation, reward, terminated, False, info # next_state, reward, terminated, truncated, info

    def render(self):
        """TODO
        """

        pass
