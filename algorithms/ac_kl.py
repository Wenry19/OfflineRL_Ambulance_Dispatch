
import keras
from keras import layers
import tensorflow as tf
import numpy as np
from pathlib import Path

from rl_agent import agentRL
from utils import VERY_SMALL_CONSTANT, IMPOSSIBLE_ACTION_AGENTS

class actorCriticKLAgent(agentRL):
    """
    Subclass of agentRL that implements the Actor Critic with KL-divergence penalization algorithm.

    Attributes
    ----------
    _learning_rate_actor : float
        Learning rate for the actor.
    _learning_rate_critic : float
        Learning rate for the critic.
    _batch_size_actor : int
        Batch size for the actor.
    _batch_size_critic : int
        Batch size for the critic.
    _gradient_steps : int
        Gradient steps done in each train step.
    _discounting_factor : float
        Discounting factor.
    _alpha : float
         Weight of the "penalization".
    _net_actor : int
        ID of the network architecture (actor).
    _net_critic : int
        ID of the network architecture (critic).
    _optimizer_critic : keras optimizer
        Adam optimizer for the critic.
    _optimizer_actor : keras optimizer
        Adam optimizer for the actor.
    _loss_function_critic : keras losses
        Mean Squared Error loss function for the critic.
    _actor : keras model
        The actor model.
    _critic : keras model
        The critic model.
    _critic_aux : keras model
        The auxiliar critic model that will be updated during a train step for each gradient step.
        The _critic model will be the target one and updated after each train step.
    _behaviour : keras model
        The behaviour model (behavioural cloning) to compute the KL-divergence.

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
    get_gradient_steps()
        Returns the number of gradient steps that are done for each train step.
    """

    def __init__(self, state_size=None, action_space=None, train_experiences=None, hyperparameters=None, behaviour_model_v=None):
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
        behaviour_model_v : str
            The version of the behaviour model that has to be loaded.
        """

        super().__init__(state_size, action_space, train_experiences, hyperparameters)

        if hyperparameters != None:

            # hyperparameters
            self._learning_rate_actor = hyperparameters['lr_a'] # 0.0003
            self._learning_rate_critic = hyperparameters['lr_c'] # 0.0003

            self._batch_size_actor = hyperparameters['bs_a'] # 128
            self._batch_size_critic = hyperparameters['bs_c'] # 128

            self._gradient_steps = hyperparameters['gs'] # 100

            self._discounting_factor = hyperparameters['disc_fact'] # 0.9
            self._alpha = hyperparameters['alpha'] # 0.9

            self._net_actor = hyperparameters['net_actor']
            self._net_critic = hyperparameters['net_critic']

            # optimizer
            self._optimizer_critic = keras.optimizers.Adam(learning_rate=self._learning_rate_critic)
            self._optimizer_actor = keras.optimizers.Adam(learning_rate=self._learning_rate_actor)

            # loss function
            self._loss_function_critic = keras.losses.MeanSquaredError()

            # models
            self._actor = self._build_actor_model()

            self._critic = self._build_critic_model()
            self._critic_aux = self._build_critic_model()
            self._critic_aux.set_weights(self._critic.get_weights())

            try:
                self._behaviour = keras.models.load_model(Path('models/cloning/' + behaviour_model_v))
            except:
                print('Error AC-KL: Behaviour policy not found!')

    def get_gradient_steps(self):
        """Returns the number of gradient steps that are done for each train step.
        """

        return self._gradient_steps

    def _build_actor_model(self):
        """Builds the actor model, the architecture will depend on the _net_actor value.

        Returns
        -------
        keras model
            The actor model.
        """

        model = keras.Sequential()

        if self._net_actor == 0:
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

        elif self._net_actor == 1:
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

        elif self._net_actor == 2:
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

    def _build_critic_model(self):
        """Builds the critic model, the architecture will depend on the _net_critic value.

        Returns
        -------
        keras model
            The critic model.
        """

        model = keras.Sequential()

        if self._net_critic == 0:
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

        elif self._net_critic == 1:
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

        elif self._net_critic == 2:
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
    
    def _kl_divergence(self, Ps, Qs):
        """Computes the KL-divergence between Ps and Qs:
        https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

        Parameters
        ----------
        Ps : tensor
            The probabilities of choosing each action for each state within a batch of states (Actor).
        Qs : tensor
            The probabilities of choosing each action for each state within a batch of states (Behaviour).

        Returns
        -------
        tensor
            KL-divergence for each state in the batch.
        """

        # Ps are from actor policy, Qs are from behaviour policy
        return tf.reduce_sum(tf.multiply(Ps, tf.math.log(VERY_SMALL_CONSTANT + tf.math.divide(Ps, Qs + VERY_SMALL_CONSTANT))), axis=1)
    
    def _calculate_expected_q_values(self, probs, rewards, states):
        """Calculates the expected Q-value for each state, given a batch of states.
        It only takes into account the actions that are possible.

        Parameters
        ----------
        probs : tensor
            The probabilities of choosing each action for each state within a batch of states.
        rewards : tensor
            The Q-value of each action for each state within a batch of states.
        states : numpy array
            The batch of states.

        Returns
        -------
        tensor
            The expected Q-value for each state in the batch.
        """

        masks = np.ones(probs.shape)

        for i in range(probs.shape[0]): # instances
            for j in range(probs.shape[1]): # actions

                if IMPOSSIBLE_ACTION_AGENTS(states[i][8+4*j+4]): # impossible action
                    masks[i, j] = 0

        probs = tf.multiply(probs, masks)
        sums = tf.reduce_sum(probs, axis=1) + VERY_SMALL_CONSTANT # just to avoid division by 0

        #print(sums)
        #print(tf.reshape(sums, (-1, 1)))
        #print(probs)

        # https://www.tensorflow.org/api_docs/python/tf/reshape
        probs = probs/tf.reshape(sums, (-1, 1)) # https://stackoverflow.com/questions/44094046/tensorflow-dividing-each-row-by-a-different-sum

        #print(probs)
        #print(tf.reduce_sum(probs, axis=1))

        expected_q_values = tf.reduce_sum(tf.multiply(probs, rewards), axis=1)
        
        return expected_q_values
        
    def run_train_step(self, step):
        """Given the step number, it executes one train step of the algorithm.

        The step number is useful to log the obtained loss.

        Parameters
        ----------
        step : int
            The step number.
        """

        ####CRITIC####

        if self._train_experiences.shape[0] < self._batch_size_critic:
            return
        
        for gs in range(self._gradient_steps):

            # sample batch

            state_sample, action_sample, reward_sample, next_state_sample = self._get_batch(self._batch_size_critic)

            # estimate error

            future_rewards = self._critic(next_state_sample)
            actor_prediction = self._actor(next_state_sample)
            behaviour_prediction = self._behaviour(next_state_sample)
            penalty = self._kl_divergence(actor_prediction, behaviour_prediction)

            updated_q_values = reward_sample + self._discounting_factor * self._calculate_expected_q_values(actor_prediction, future_rewards, next_state_sample) - \
                                self._alpha * self._discounting_factor * penalty
            
            # print(type(updated_q_values)) # <class 'tensorflow.python.framework.ops.EagerTensor'>

            masks = tf.one_hot(action_sample, self._action_space)

            with tf.GradientTape() as tape:
                q_values = self._critic_aux(state_sample)
                q_values_actions = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                loss_critic = self._loss_function_critic(updated_q_values, q_values_actions)

            tf.summary.scalar('batch_loss_critic', data=loss_critic, step=step*self._gradient_steps+gs)

            # update critic parameters

            # Backpropagation
            grads = tape.gradient(loss_critic, self._critic_aux.trainable_variables)
            self._optimizer_critic.apply_gradients(zip(grads, self._critic_aux.trainable_variables))

        self._critic.set_weights(self._critic_aux.get_weights())

        ####ACTOR####

        if self._train_experiences.shape[0] < self._batch_size_actor:
            return
        
        for gs in range(self._gradient_steps):

            # sample batch of states

            state_sample, _, _, _ = self._get_batch(self._batch_size_actor)

            # estimate error

            with tf.GradientTape() as tape:

                actor_prediction = self._actor(state_sample)
                behaviour_prediction = self._behaviour(state_sample)
                penalty = self._kl_divergence(actor_prediction, behaviour_prediction)

                critic_prediction = self._critic(state_sample)
                loss_actor = -(tf.reduce_mean(self._calculate_expected_q_values(actor_prediction, critic_prediction, state_sample) - self._alpha * penalty, axis=0))

            tf.summary.scalar('batch_loss_actor', data=loss_actor, step=step*self._gradient_steps+gs)

            # update actor parameters

            # Backpropagation
            grads = tape.gradient(loss_actor, self._actor.trainable_variables)
            self._optimizer_actor.apply_gradients(zip(grads, self._actor.trainable_variables))

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
        pred_probs = self._actor(aux).numpy()[0]

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
        """ Saves the trained models.

        Parameters
        ----------
        v : str
            Version identifier (time).
        """
        
        self._critic.save(Path('models/ac_kl/' + v + '/critic'))
        self._actor.save(Path('models/ac_kl/' + v + '/actor'))

    def load_model(self, v):
        """Loads previously trained models.

        Parameters
        ----------
        v : str
            Version identifier (time).
        """

        if self._hyperparameters != None:
            self._critic = keras.models.load_model(Path('models/ac_kl/' + v + '/critic'))
            self._critic_aux = self._build_critic_model()
            self._critic_aux.set_weights(self._critic.get_weights())

        self._actor = keras.models.load_model(Path('models/ac_kl/' + v + '/actor'))
