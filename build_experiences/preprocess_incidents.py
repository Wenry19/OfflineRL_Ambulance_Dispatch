
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim

from utils import read_time

class preprocessIncidents():
    """
    A class in charge of preprocessing the incidents data.

    Attributes
    ----------
    _incidents_df : pandas dataframe
        The original dataset containing all the incidents to be preprocessed.
    _geolocator : Nominatim instance
        An instance of Nominatim to get the coordinates from the addresses that appear in the original incidents dataset.

    Methods
    -------
    preprocess()
        Preprocess the incidents data.
    """

    def __init__(self, incidents_path):
        """
        Parameters
        ----------
        incidents_path : Path
            The path of the original dataset containg all the incidents to be preprocessed.
        """

        self._incidents_df = pd.read_csv(incidents_path, dtype=str)
        self._geolocator = Nominatim(user_agent='myApp')

    def preprocess(self):
        """Preprocess the incidents data.

        To preprocess the data:
            - the incidents with missing values (in 'ccCallcardUid', 'LocalTime', 'Priority') are discarded
            - priority 2 is assigned to incidents with priorities other than 0 and 1
            - the addresses of the incidents are transformed to coordinates
            - the incidents that are not in Switzerland are discarded
            - the incidents with addresses that the geolocator cannot find are discarded
            - the ccCallcardUid, time (transformed to an int), priority and coordinates are saved
            - the incidents are sorted by time

        Returns
        -------
        numpy array
            Incidents data preprocessed.
        """

        incidents_arr = self._incidents_df[['ccCallcardUid', 'LocalTime', 'Priority', 'PlaceName', 'StreetNumber', 'CityName', 'PostalCode']].to_numpy()

        new_incidents_arr = []

        for i in range(incidents_arr.shape[0]):

            print(str(i) + ' of ' + str(incidents_arr.shape[0]))

            if pd.isnull(incidents_arr[i, [0, 1, 2]]).any():
                print('Null value found.')
                continue

            if not incidents_arr[i, 2] in {'0', '1'}:
                incidents_arr[i, 2] = '2'

            row = []

            address = ''
            if not pd.isnull(incidents_arr[i, 3]): # place name
                address += str(incidents_arr[i, 3])
            if not pd.isnull(incidents_arr[i, 4]): # street number
                address += ', ' + str(incidents_arr[i, 4])
            if not pd.isnull(incidents_arr[i, 5]): # city name
                address += ', ' + str(incidents_arr[i, 5])
            if not pd.isnull(incidents_arr[i, 6]): # postal code
                address += ', ' + str(incidents_arr[i, 6])

            try:
                location = self._geolocator.geocode(address)

                if location != None and 'Suisse' in location.address:

                    row.append(incidents_arr[i, 0]) # ccCallcardUid
                    row.append(read_time(incidents_arr[i, 1])) # time
                    row.append(incidents_arr[i, 2]) # priority
                    row.append((location.latitude, location.longitude)) # coords

                    new_incidents_arr.append(row)

                else:

                    print('Location not in Suisse or None:', location)
                    print('with address:', address)

            except:
                print('Error with address: ' + address)

        new_incidents_arr = np.array(new_incidents_arr, dtype=object)

        new_incidents_arr = new_incidents_arr[new_incidents_arr[:, 1].argsort()]

        return new_incidents_arr
