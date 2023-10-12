
import numpy as np

from rl_agent import agentRL
from utils import IMPOSSIBLE_ACTION_AGENTS

class randomAgent(agentRL):
    """
    Subclass of agentRL that implements the random approach.

    Attributes
    ----------
    _state_size : int
        State size.
    _action_space : int
        Action space.

    Methods
    -------
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
    
    def choose_action(self, state):
        """Given a state, it chooses a random possible action.

        Parameters
        ----------
        state : list
            State.
        
        Returns
        -------
        int
            The chosen action.
        """

        a = np.random.randint(0, self._action_space)

        while IMPOSSIBLE_ACTION_AGENTS(state[8+4*a+4]): # impossible action
             a = np.random.randint(0, self._action_space)

        return a
