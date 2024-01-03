import tensorflow as tf
import numpy as np


class Agent:
    def __init__(self, feature_size, window_size=20, epsilon=1, epsilon_decay=0.998, epsilon_end=0.01, gamma=0.99, learning_rate=0.0001):
        self.feature_size = feature_size
        self.window_size = window_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(
            units=32, input_shape=(self.window_size, self.feature_size),
            return_sequences=True))

        model.add(tf.keras.layers.LSTM(units=32))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dense(
            units=3, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate))
        return model

    def get_action(self, state):
        # rand() returns a random value between 0 and 1
        if np.random.rand() <= self.epsilon:
            # Exploration: random action
            action = np.random.randint(0, 3)
        else:
            # Use the model to predict the Q-values (action values) for the given state
            q_values = self.model.predict(state, verbose=0)

            # Select and return the action with the highest Q-value
            # Take the action from the first (and only) entry
            action = np.argmax(q_values[0])

        # Decay the epsilon value to reduce the exploration over time
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return action

    def learn(self, experiences):
        states = np.array([experience.state for experience in experiences])
        states = np.reshape(states, (32, 20, 11))
        actions = np.array([experience.action for experience in experiences])
        rewards = np.array([experience.reward for experience in experiences])
        next_states = np.array(
            [experience.next_state for experience in experiences])
        next_states = np.reshape(next_states, (32, 20, 11))
        dones = np.array([experience.done for experience in experiences])

        # Predict the Q-values (action values) for the given state batch
        current_q_values = self.model.predict(states, verbose=0)

        # Predict the Q-values for the next_state batch
        next_q_values = self.model.predict(next_states, verbose=0)

        # Initialize the target Q-values as the current Q-values
        target_q_values = current_q_values.copy()

        # Loop through each experience in the batch
        for i in range(len(experiences)):
            if dones[i]:
                # If the episode is done, there is no next Q-value
                # [i, actions[i]] is the numpy equivalent of [i][actions[i]]
                target_q_values[i, actions[i]] = rewards[i]
            else:
                # The updated Q-value is the reward plus the discounted max Q-value for the next state
                # [i, actions[i]] is the numpy equivalent of [i][actions[i]]
                target_q_values[i, actions[i]] = rewards[i] + \
                    self.gamma * np.max(next_q_values[i])

        # Train the model
        self.model.fit(states, target_q_values, epochs=1, verbose=0)

    def load(self, file_path):
        self.model = tf.keras.models.load_model(file_path)

    def save(self, file_path):
        self.model.save(file_path)
