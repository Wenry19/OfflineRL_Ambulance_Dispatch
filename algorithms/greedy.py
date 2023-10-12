
from geopy.distance import geodesic
import numpy as np

from rl_agent import agentRL
from utils import IMPOSSIBLE_ACTION_AGENTS

VERY_BIG_DISTANCE = 1000000 # km
"""A very big distance.
"""

class greedyAgent(agentRL):
    """
    Subclass of agentRL that implements the greedy approach.

    Attributes
    ----------
    _state_size : int
        State size.
    _action_space : int
        Action space.

    Methods
    -------
    predict(state)
        Implements predict(state) of the parent class agentRL.
    choose_action(state)
        Implements choose_action(state) of the parent class agentRL.
    """

    def __init__(self, state_size, action_space):
        """
        Parameters
        ----------
        state_size : int
            State size.
        action_space : int
            Action space.
        """

        self._state_size = state_size
        self._action_space = action_space

    def predict(self, state):
        """Given a state, it calculates the distance between a resource and the incident for all the resources.
        If a resource is an impossible action, it puts a very big distance instead.

        Parameters
        ----------
        state : list
            State.

        Returns
        -------
        list
            The distances.
        """

        distances = []

        for a in range(self._action_space):

            if IMPOSSIBLE_ACTION_AGENTS(state[8+4*a+4]): # impossible action
                distances.append(VERY_BIG_DISTANCE)
            else:
                distances.append(geodesic((state[7]*90, state[8]*180), (state[8+4*a+1]*90, state[8+4*a+2]*180)))

        return distances
    
    def choose_action(self, state):
        """Given a state, and according to the prediction of predict(state) method, it chooses the best action.

        Parameters
        ----------
        state : list
            State.
        
        Returns
        -------
        int
            The chosen action.
        """

        distances = self.predict(state)

        return np.argmin(distances, axis=0)
