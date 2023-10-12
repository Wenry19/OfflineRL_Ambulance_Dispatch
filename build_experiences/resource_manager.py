
import pandas as pd
import numpy as np
from pathlib import Path

from utils import swiss_coords_tolatlon, read_time, calculate_duration, IMPOSSIBLE_ACTION_VALUE_YEAR_AVAILABILITY

class resourceManager():
    """
    A class used to answer some queries about the resources.

    Attributes
    ----------
    _resource_pos_avail_info : dict of lists of dicts
        Dictionary with the information of the resource positions during time.
        Each entry of the dictionary has key = (resource name) and
        value = (list of dicts {itime : initial time, ftime : final time, ilat : initial latitude, ilng : initial longitude, flat : final latitude, flng : final longitude}).
        There is an element in the list for each time the resource was used.
    _res_2014_2018 : numpy array
        Resource names available from the year 2014 to the year 2018 (both included).
    _res_2019_2021 : numpy array
        Resource names available from the year 2019 to the year 2021
        (2019 and 2020 included, but 2021 not included completely due to a change of version of the data).
    _res_V6 : numpy array
        Resource names available from the year 2021 to the year 2022
        (the remaining of 2021 and part of the 2022 with the new data version).
    _firstV6time : int
        First incident time that was stored in the new version of data.
    _all_res_hist : numpy array
        Resource histories of all the years preprocessed and concatenated. See the method _initialize() to know the preprocessing process.

    Methods
    -------
    get_resource_pos_avail_info()
        It returns _resource_pos_avail_info.
    year_availability(resource, time)
        It checks, given a resource name and a time, if the resource was available at that time (it does not check if it was busy serving an incident).
    get_resource_state(resource, time)
        Given a resource name and a time, it returns the estimated position of this resource at that time and information about its availability
        (taking into account the year availability of the resource and if it was busy or not).
    get_resource_names()
        Returns all the resource names.
    get_next_departure(resource, time)
        Given a resource name and a time, it returns information about the next departure of this resource from that time.
    get_coordinates_history(resource)
        Given a resource name, it returns the coordinates history of this resource.
    """

    def __init__(self, precomputed_pos_avail_info=None):
        """
        Parameters
        ----------
        precomputed_pos_avail_info : dict of lists of dicts, optional
            If the information is already saved, the class can be constructed with the information to avoid recomputing it again.
        """

        if precomputed_pos_avail_info == None:
            self._initialize()
            self._info_from_history()
        else:
            # in case we run self.estimate() before, just to do not compute it again
            self._resource_pos_avail_info = precomputed_pos_avail_info

        # resources available each year
        self._res_2014_2018 = pd.read_csv(Path('CHUV/ChuvExportResources_V5_2014_2018.csv'), dtype=str)['Name'].to_numpy()
        self._res_2019_2021 = pd.read_csv(Path('CHUV/ChuvExportResources_V5_2019_2021.csv'), dtype=str)['Name'].to_numpy()
        self._res_V6 = pd.read_csv(Path('CHUV/ChuvExportResourcesV6.csv'), dtype=str)
        self._res_V6 = self._res_V6['Name'][self._res_V6['resType'] == 'Vehicle'].to_numpy()

        # first V6 incident time in order to know from which time the available resources are the ones in res_V6, because in V6 there are times in 2021 and 2022
        self._firstV6time = 20210823223859

    def _initialize(self):
        """Preprocess the resource history data.

        To preprocess the data:
            - concatenates all the resource histories (all the years)
            - deletes the missing values (delete all the entries of the resources with some missing values)
            - transforms times to integers in order to sort them by time
            - sorts the entries, first by resource name and second by time
        
        It also initializes the dictionary _resource_pos_avail_info with keys = (all the resource names) and
        associating for each of them an empty list as its value.
        """

        # concatenate all the datasets

        rh2016 = pd.read_csv(Path('CHUV/ChuvExportResourceHistory_V5_2016.csv'), dtype=str).to_numpy()
        rh2017 = pd.read_csv(Path('CHUV/ChuvExportResourceHistory_V5_2017.csv'), dtype=str).to_numpy()
        rh2018 = pd.read_csv(Path('CHUV/ChuvExportResourceHistory_V5_2018.csv'), dtype=str).to_numpy()
        rh2019 = pd.read_csv(Path('CHUV/ChuvExportResourceHistory_V5_2019.csv'), dtype=str).to_numpy()
        rh2020 = pd.read_csv(Path('CHUV/ChuvExportResourceHistory_V5_2020.csv'), dtype=str).to_numpy()
        rh2021 = pd.read_csv(Path('CHUV/ChuvExportResourceHistory_V5_2021.csv'), dtype=str).to_numpy()
        rhV6 = pd.read_csv(Path('CHUV/ChuvExportResourceHistoryV6.csv'), dtype=str)[['Ambulance', 'resResourceUid1', 'LocalTime', 'Text', 'CoordX', 'CoordY']].to_numpy()
                                                                                # Ambulance,resResourceUid1,LocalTime,Text,CoordX,CoordX1 (version V5)

        self._all_res_hist = np.concatenate((rh2016, rh2017, rh2018, rh2019, rh2020, rh2021, rhV6), axis=0)

        # cleaning resources with missing values (values that are very high in the coordinates columns, nonsense values)

        mask = np.char.find(np.array(self._all_res_hist[:, 4], dtype=str), 'E') != -1 # https://numpy.org/devdocs/reference/generated/numpy.char.find.html

        resource_names_with_MV = np.unique(self._all_res_hist[:, 0][mask])

        mask = np.logical_not(np.isin(self._all_res_hist[:, 0], resource_names_with_MV)) # https://numpy.org/doc/stable/reference/generated/numpy.isin.html

        self._all_res_hist = self._all_res_hist[mask]

        # transform times to integers in order to sort them by time, but also by resource name first
        for i in range(self._all_res_hist.shape[0]):
            self._all_res_hist[i, 2] = read_time(self._all_res_hist[i, 2])

        # sort logs, first by resource name and second by time
        idxs = np.lexsort((self._all_res_hist[:, 2], self._all_res_hist[:, 0])) # https://numpy.org/doc/stable/reference/generated/numpy.lexsort.html
        self._all_res_hist = self._all_res_hist[idxs]

        #print(type(self._all_res_hist[0, 0]), type(self._all_res_hist[0, 1]), type(self._all_res_hist[0, 2]),
        #      type(self._all_res_hist[0, 3]), type(self._all_res_hist[0, 4]), type(self._all_res_hist[0, 5]))
        
        # <class 'str'> <class 'str'> <class 'int'> <class 'str'> <class 'str'> <class 'str'>

        # all the different resources that have logs in the resource history
        resource_names = np.unique(self._all_res_hist[:, 0])

        self._resource_pos_avail_info = dict()
        for res in resource_names:
            self._resource_pos_avail_info[res] = [] # list that will contain elements of kind: {itime, ftime, ilat, ilng, flat, flng}

    def _info_from_history(self):
        """Constructs _resource_pos_avail_info.

        Each log (entry in the resource history) starts with a Demande and finishes with Terminé or Annulé.

        It searches, for each Demande, its associated Terminé or Annulé:
            - From Demande the initial time (itime) and the coordinates (ilat, ilng) are extracted.
            - From Terminé or Annulé the final time (ftime) and the coordinates (flat, flng) are extracted.

        Append to the _resource_pos_avail_info[current resource] a dict with that information
        {itime : initial time, ftime : final time, ilat : initial latitude, ilng : initial longitude, flat : final latitude, flng : final longitude}.

        So, we will know that:
            - From itime to ftime the resource was busy.
            - The resource was at (ilat, ilng), waiting to be used.
            - The resource finished at (flat, flng) where it was available again.

        Notice that in the data we have swiss coordinates and they are transformed to (latitude, longitude) coordinates.
        """

        for i in range(self._all_res_hist.shape[0]):

            if 'Demande' in self._all_res_hist[i, 3]:

                res = self._all_res_hist[i, 0]
                itime = self._all_res_hist[i, 2]
                ilat, ilng = swiss_coords_tolatlon(float(self._all_res_hist[i, 4]), float(self._all_res_hist[i, 5]))

                # search its associated Terminé or Annulé
                ftime = None
                flat = None
                flng = None
                for ii in range(i+1, self._all_res_hist.shape[0]):

                    if self._all_res_hist[ii, 0] != res or 'Demande' in self._all_res_hist[ii, 3]:
                        break

                    if 'Terminé' in self._all_res_hist[ii, 3] or 'Annulé' in self._all_res_hist[ii, 3]:
                        ftime = self._all_res_hist[ii, 2]
                        flat, flng = swiss_coords_tolatlon(float(self._all_res_hist[ii, 4]), float(self._all_res_hist[ii, 5]))
                        break

                if ftime != None: # make sure that we found it
                    self._resource_pos_avail_info[res].append({'itime' : itime, 'ftime' : ftime, 'ilat' : ilat, 'ilng' : ilng, 'flat' : flat, 'flng' : flng})

        # cleaning resources without info
        res_to_delete = []
        for k, v in self._resource_pos_avail_info.items():
            if v == []:
                res_to_delete.append(k)
                
        for r in res_to_delete:
            del self._resource_pos_avail_info[r]
    
    def get_resource_pos_avail_info(self):
        """It returns _resource_pos_avail_info.
        """

        return self._resource_pos_avail_info
    
    def year_availability(self, resource, time):
        """It checks, given a resource name and a time, if the resource was available at that time (it does not check if it was busy serving an incident).

        Parameters
        ----------
        resource : str
            The resource.
        time : int
            The time at which you want to know if the resource was available or not.

        Returns
        -------
        int
            0 if it was available, IMPOSSIBLE_ACTION_VALUE_YEAR_AVAILABILITY if it was not available.
        """

        year = int(str(time)[0:4])
        if 2014 <= year and year <= 2018 and resource in self._res_2014_2018:
            return 0
        elif 2019 <= year and time < self._firstV6time and resource in self._res_2019_2021:
            return 0
        elif self._firstV6time <= time and resource in self._res_V6:
            return 0
        return IMPOSSIBLE_ACTION_VALUE_YEAR_AVAILABILITY # not possible action
    
    def get_resource_state(self, resource, time):
        """Given a resource name and a time, it returns the estimated position of this resource at that time and information about its availability
        (taking into account the year availability of the resource and if it was busy or not).

        If it was busy: the departure location of the resource and the remaining time to be available again are returned.

        If it was not busy: the location where the resource was waiting to be used is returned (the location of its next departure).

        If it was not busy, but there was not a next departure: the last departure location is returned.

        Moreover, if it was not busy, it is checked if the resource was available in that year, if it was not, it is set as not available.
        
        Parameters
        ----------
        resource : str
            The resource.
        time : int
            The time at which you want to know what was the state of the resource.

        Returns
        -------
        tuple
            Latitude and longitude (estimated position of the resource).
        int
            If the resource was busy, the remaining time to be available again (in seconds).
            If the resource was available, 0 (meaning that we do not have to wait).
            If the resource was not available, IMPOSSIBLE_ACTION_VALUE_YEAR_AVAILABILITY.
        """
        
        info = self._resource_pos_avail_info[resource] # note that this is ordered by time

        for x in info:

            if time < x['itime']: # not busy, but year availability has to be checked
                return (x['ilat'], x['ilng']), self.year_availability(resource, time)
                
            elif x['itime'] <= time and time < x['ftime']: # busy
                return (x['ilat'], x['ilng']), calculate_duration(time, x['ftime'])
            
        # if we reach here means that the resource was not being used and
        # there is not a next departure after the time parameter
        # so, the last departure location is returned

        # not busy, but year availability has to be checked
        return (info[-1]['ilat'], info[-1]['ilng']), self.year_availability(resource, time)

    def get_resource_names(self):
        """Returns all the resource names in a sorted list.

        Returns
        -------
        list
            Resource names sorted.
        """

        return list(np.sort(list(self._resource_pos_avail_info.keys())))
    
    def get_next_departure(self, resource, time):
        """Given a resource name and a time, it returns information about the next departure of this resource from that time.

        Parameters
        ----------
        resource : str
            The resource.
        time : int
            The time from which the next departure has to be found.

        Returns
        -------
        dict
            A dictionary that contains {itime : initial time, ftime : final time, ilat : initial latitude, ilng : initial longitude,
            flat : final latitude, flng : final longitude} of the next departure.
            It returns None if there is not any next departure from that time.
        """

        info = self._resource_pos_avail_info[resource]

        for x in info:

            if time < x['itime']:
                return x
            
        return None # if there is not any next departure
    
    def get_coordinates_history(self, resource):
        """Given a resource name, it returns the coordinates history (the initial coordinates every time it was used) of this resource.

        Parameters
        ----------
        resource : str
            The resource.

        Returns
        -------
        list of tuples
            A list of tuples, each of them containing coordinates (latitude, longitude).
        """

        info = self._resource_pos_avail_info[resource]

        coordinates_history = []

        for x in info:
            coordinates_history.append((x['ilat'], x['ilng']))

        return coordinates_history
