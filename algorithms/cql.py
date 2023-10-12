
import keras
from keras import layers
import tensorflow as tf
import numpy as np
from pathlib import Path

from rl_agent import agentRL
from utils import VERY_SMALL_CONSTANT, IMPOSSIBLE_ACTION_AGENTS

class CQLAgent(agentRL):
    """
    Subclass of agentRL that implements the CQL (Conservative Q-Learning) algorithm.

    Attributes
    ----------
    _learning_rate : float
        Learning rate.
    _batch_size : int
        Batch size.
    _discounting_factor : float
        Discounting factor.
    _T : float
        Parameter that regulates the updates of the target network.
    _alpha : float
        Weight of the "penalization".
    _net : int
        ID of the network architecture.
    _optimizer : keras optimizer
        Adam optimizer.
    _loss_function : keras losses
        Mean Squared Error loss function.
    _model : keras model
        The model.
    _target_model : keras model
        The target model.

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
            self._learning_rate = hyperparameters['lr'] # 0.0002
            self._batch_size = hyperparameters['bs'] # 128
            self._discounting_factor = hyperparameters['disc_fact'] # 0.9
            self._T = hyperparameters['T'] # 0.01
            self._alpha = hyperparameters['alpha'] # 0.9
            self._net = hyperparameters['net']

            # optimizer
            self._optimizer = keras.optimizers.Adam(learning_rate=self._learning_rate)

            # loss function
            self._loss_function = keras.losses.MeanSquaredError()

            # models
            self._model = self._build_model()
            self._target_model = self._build_model()
            self._target_model.set_weights(self._model.get_weights())

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
            model.add(layers.Dense(self._action_space, activation='linear'))

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
            model.add(layers.Dense(self._action_space, activation='linear'))

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
            model.add(layers.Dense(self._action_space, activation='linear'))

        return model
    
    def _reduce_max_possible_actions(self, next_state_sample, future_rewards):
        """Given a batch of next states and their corresponding predicted long term rewards (one long term reward per action),
        for each next state it takes the maximum predicted long term reward that corresponds to a possible action.

        Parameters
        ----------
        next_state_sample : numpy array
            A sample of next states.
        future_rewards : numpy array
            The predicted long term rewards for each next state.

        Returns
        -------
        tensor
            For each next state the best possible long term reward.
        """

        aux = np.amin(future_rewards) - 1

        for i in range(future_rewards.shape[0]):
            for j in range(future_rewards.shape[1]):

                if IMPOSSIBLE_ACTION_AGENTS(next_state_sample[i][8+4*j+4]): # impossible action
                    future_rewards[i, j] = aux

        return tf.reduce_max(future_rewards, axis=1)
    
    def run_train_step(self, step):
        """Given the step number, it executes one train step of the algorithm.

        The step number is useful to log the obtained loss.

        Parameters
        ----------
        step : int
            The step number.
        """

        # https://keras.io/examples/rl/deep_q_network_breakout/

        if self._train_experiences.shape[0] < self._batch_size:
            return
        
        # soft update of the target weights
        # self.model.get_weights() returns a python list of numpy arrays

        weights = np.array(self._model.get_weights(), dtype=object) # cast it to numpy array to do element-wise operations
        weights_target = np.array(self._target_model.get_weights(), dtype=object)

        weights_target = list(self._T * weights + (1 - self._T) * weights_target) # apply soft update and cast it to list again

        self._target_model.set_weights(weights_target)

        # get batch
        state_sample, action_sample, reward_sample, next_state_sample = self._get_batch(self._batch_size)

        # compute the updated q values
        future_rewards = self._target_model(next_state_sample).numpy()
        # https://stackoverflow.com/questions/64199384/tf-keras-model-predict-results-in-memory-leak
        updated_q_values = reward_sample + self._discounting_factor * self._reduce_max_possible_actions(next_state_sample, future_rewards)

        # print(type(updated_q_values)) # <class 'tensorflow.python.framework.ops.EagerTensor'>

        # loss
        masks = tf.one_hot(action_sample, self._action_space)

        with tf.GradientTape() as tape:
            # https://stackoverflow.com/questions/55308425/difference-between-modelx-and-model-predictx-in-keras
            q_values = self._model(state_sample)
            q_values_actions = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            ###CQL PENALTY###
            minimization_cql_term = tf.reduce_mean(tf.math.log(VERY_SMALL_CONSTANT + tf.reduce_sum(tf.math.exp(q_values), axis=1)))
            maximization_cql_term = tf.reduce_mean(q_values_actions)

            batch_loss = self._loss_function(updated_q_values, q_values_actions) + self._alpha * (minimization_cql_term - maximization_cql_term)

        tf.summary.scalar('batch_loss', data=batch_loss, step=step)

        # Backpropagation
        grads = tape.gradient(batch_loss, self._model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

    def predict(self, state):
        """Given a state, it predicts the Q-value for each action,
        taking into account that some actions are impossible (discarded actions).

        Parameters
        ----------
        state : list
            State.

         Returns
        -------
        numpy array
            The predicted Q-values.
        """
        
        aux = np.reshape(state, [1, len(state)])
        pred_q_values = self._model(aux).numpy()[0]

        auxx = np.amin(pred_q_values) - 1

        for j in range(pred_q_values.shape[0]):
            if IMPOSSIBLE_ACTION_AGENTS(state[8+4*j+4]): # impossible action
                pred_q_values[j] = auxx

        return pred_q_values

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
        
        pred_q_values = self.predict(state)

        return int(np.argmax(pred_q_values, axis=0))
    
    def save_model(self, v):
        """ Saves the trained model.

        Parameters
        ----------
        v : str
            Version identifier (time).
        """

        self._model.save(Path('models/cql/' + v))

    def load_model(self, v):
        """Loads a previously trained model.

        Parameters
        ----------
        v : str
            Version identifier (time).
        """

        self._model = keras.models.load_model(Path('models/cql/' + v))
        
        if self._hyperparameters != None:
            self._target_model = self._build_model()
            self._target_model.set_weights(self._model.get_weights())
