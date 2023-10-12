
import numpy as np

import os
import sys
sys.path.append(os.getcwd())
from utils import IDX_STATE, IDX_ACTION, IDX_REWARD, IDX_NEXT_STATE

class agentRL:
    """
    Parent class for the implemented algorithms.

    Attributes
    ----------
    _state_size : int
        State size.
    _action_space : int
        Action space.
    _hyperparameters : dict
        Dictionary containing the values of the hyperparameters.
    _train_experiences : numpy array
        The train experiences (experience buffer to train the models).

    Methods
    -------
    get_hyperparameters()
        Returns the dictionary with the hyperarameter values.
    run_train_step(step)
        Executes one train step of the algorithm.
    predict(state)
        Given a state, it predicts the Q-value/probability for each action,
        taking into account that some actions are impossible (discarded actions).
    choose_action(state)
        Given a state, and according to the prediction of predict(state) method, it chooses the best action.
    save_model(v)
        Saves the trained model.
    load_model(v)
        Loads a previously trained model.
    """

    def __init__(self, state_size, action_space, train_experiences, hyperparameters):
        """
        Parameters
        ----------
        state_size : int
            State size.
        action_space : int
            Action space.
        train_experiences : numpy array
            The train experiences (experience buffer to train the models).
        hyperparameters : dict
            Dictionary containing the values of the hyperparameters.
        """

        self._state_size = state_size
        self._action_space = action_space

        self._hyperparameters = hyperparameters
        
        # given experiences
        self._train_experiences = train_experiences

    def _get_batch(self, batch_size):
        """Given a batch size, it returns a batch of experiences.

        Parameters
        ----------
        batch_size : int
            Batch size.

        Returns
        -------
        numpy array
            State sample.
        numpy array
            Action sample.
        numpy array
            Reward sample.
        numpy array
            Next state sample.
        """

        sample_batch = self._train_experiences[np.random.choice(len(self._train_experiences), size=batch_size)]

        state_sample = np.array([sample_batch[i][IDX_STATE] for i in range(batch_size)])
        action_sample = np.array([sample_batch[i][IDX_ACTION] for i in range(batch_size)])
        reward_sample = np.array([sample_batch[i][IDX_REWARD] for i in range(batch_size)])
        next_state_sample = np.array([sample_batch[i][IDX_NEXT_STATE] for i in range(batch_size)])
    
        return state_sample, action_sample, reward_sample, next_state_sample
    
    def get_hyperparameters(self):
        """Returns the dictionary with the hyperarameter values.

        Returns
        -------
        dict
            Dictionary containing the values of the hyperparameters.
        """

        return self._hyperparameters
    
    def run_train_step(self, step):
        """Given the step number, it executes one train step of the algorithm.

        The step number is useful to log the obtained loss.

        Parameters
        ----------
        step : int
            The step number.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """

        raise NotImplementedError("Not implemented Error")
    
    def predict(self, state):
        """Given a state, it predicts the Q-value/probability for each action,
        taking into account that some actions are impossible (discarded actions).

        Parameters
        ----------
        state : list
            State.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """

        raise NotImplementedError("Not implemented Error")
    
    def choose_action(self, state):
        """Given a state, and according to the prediction of predict(state) method, it chooses the best action.

        Parameters
        ----------
        state : list
            State.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        
        raise NotImplementedError("Not implemented Error")
    
    def save_model(self, v):
        """ Saves the trained model.

        Parameters
        ----------
        v : str
            Version identifier (time).

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Not implemented Error")
    
    def load_model(self, v):
        """Loads a previously trained model.

        Parameters
        ----------
        v : str
            Version identifier (time).

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError("Not implemented Error")
