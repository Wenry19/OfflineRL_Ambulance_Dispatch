
import keras
from keras import layers
import tensorflow as tf
import numpy as np
from pathlib import Path

from rl_agent import agentRL
from utils import IMPOSSIBLE_ACTION_AGENTS

class behaviouralCloningAgent(agentRL):
    """
    Subclass of agentRL that implements the behavioural cloning approach.

    Attributes
    ----------
    _learning_rate : float
        Learning rate.
    _batch_size : int
        Batch size.
    _net : int
        ID of the network architecture.
    _optimizer : keras optimizer
        Adam optimizer.
    _loss_function : keras losses
        Mean Squared Error loss function.
    _model : keras model
        The model.

    Methods
    -------
    run_train_step(step)
        Implements run_train_step(step) of the parent class agentRL.
    predict(state)
        Implements predict(state) of the parent class agentRL.
    choose_action(state)
        Implements choose_action(state) of the parent class agentRL.
    save_model(v)
        Implements save_model(v) of the parent class agentRL.
    load_model(v)
        Implements load_model(v) of the parent class agentRL.
    """

    def __init__(self, state_size=None, action_space=None, train_experiences=None, hyperparameters=None):
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

        super().__init__(state_size, action_space, train_experiences, hyperparameters)

        if hyperparameters != None:

            # hyperparameters
            self._learning_rate = hyperparameters['lr'] # 0.0001
            self._batch_size = hyperparameters['bs'] # 2048
            self._net = hyperparameters['net']

            # optimizer
            self._optimizer = keras.optimizers.Adam(learning_rate=self._learning_rate)

            # loss function
            self._loss_function = keras.losses.MeanSquaredError()

            # models
            self._model = self._build_model()

    def _build_model(self):
        """Builds the model, the architecture will depend on the _net value.

        Returns
        -------
        keras model
            The model.
        """

        model = keras.Sequential()

        if self._net == 0:
            # input
            model.add(keras.Input(shape=(self._state_size, )))
            # hidden layer 1
            model.add(layers.Dense(512, activation='relu'))
            # hidden layer 2
            model.add(layers.Dense(512, activation='relu'))
            # hidden layer 3
            model.add(layers.Dense(1024, activation='relu'))
            # output
            model.add(layers.Dense(self._action_space, activation='softmax'))

        elif self._net == 1:
            # input
            model.add(keras.Input(shape=(self._state_size, )))
            # hidden layer 1
            model.add(layers.Dense(1024, activation='relu'))
            # hidden layer 2
            model.add(layers.Dense(2048, activation='relu'))
            # hidden layer 3
            model.add(layers.Dense(1024, activation='relu'))
            # hidden layer 4
            model.add(layers.Dense(512, activation='relu'))
            # output
            model.add(layers.Dense(self._action_space, activation='softmax'))

        elif self._net == 2:
            # input
            model.add(keras.Input(shape=(self._state_size, )))
            # hidden layer 1
            model.add(layers.Dense(1024, activation='relu'))
            # hidden layer 2
            model.add(layers.Dense(2048, activation='relu'))
            # hidden layer 3
            model.add(layers.Dense(4096, activation='relu'))
            # hidden layer 4
            model.add(layers.Dense(1024, activation='relu'))
            # output
            model.add(layers.Dense(self._action_space, activation='softmax'))

        return model
    
    def run_train_step(self, step):
        """Given the step number, it executes one train step of the algorithm.

        The step number is useful to log the obtained loss.

        Parameters
        ----------
        step : int
            The step number.
        """

        if self._train_experiences.shape[0] < self._batch_size:
            return
        
        state_sample, action_sample, _, _ = self._get_batch(self._batch_size)
        
        ground_truth = tf.one_hot(action_sample, self._action_space)
        
        with tf.GradientTape() as tape:
            # https://stackoverflow.com/questions/55308425/difference-between-modelx-and-model-predictx-in-keras
            prediction = self._model(state_sample)
            batch_loss = self._loss_function(prediction, ground_truth)

        tf.summary.scalar('batch_loss', data=batch_loss, step=step)

        # Backpropagation
        grads = tape.gradient(batch_loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

    def predict(self, state):
        """Given a state, it predicts the probability for each action,
        taking into account that some actions are impossible (discarded actions).

        Parameters
        ----------
        state : list
            State.

        Returns
        -------
        numpy array
            The predicted probabilities.
        """
        
        aux = np.reshape(state, [1, len(state)])
        pred_probs = self._model(aux).numpy()[0]

        for j in range(pred_probs.shape[0]):
            if IMPOSSIBLE_ACTION_AGENTS(state[8+4*j+4]): # impossible action
                pred_probs[j] = 0

        pred_probs = pred_probs/np.sum(pred_probs) # renormalizing probabilities

        return pred_probs
    
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
        
        pred_probs = self.predict(state)

        return np.random.choice(self._action_space, p=pred_probs)
    
    def save_model(self, v):
        """ Saves the trained model.

        Parameters
        ----------
        v : str
            Version identifier (time).
        """

        self._model.save(Path('models/cloning/' + v))

    def load_model(self, v):
        """Loads a previously trained model.

        Parameters
        ----------
        v : str
            Version identifier (time).
        """

        self._model = keras.models.load_model(Path('models/cloning/' + v))
