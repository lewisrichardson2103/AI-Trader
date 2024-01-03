from sklearn.preprocessing import Normalizer
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange
from ta.momentum import RSIIndicator
from ta.utils import dropna
import random
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


class Environment:
    def __init__(self, file_path, window_size=20, max_steps=200, capital=1000, capital_goal=1010, render_on=False):
        self.window_size = window_size
        self.render_on = render_on
        self.max_steps = max_steps
        self.capital_goal = capital_goal
        self.current_step = 0
        self.total_reward = 0

        self.starting_capital = capital
        self.capital = 0
        self.profit = 0
        # Raw data not normalised, additional columns added
        self.raw_data = self.load_data(file_path)
        self.normaliser = Normalizer('l2')
        self.reset()
        print(self.data)

    def reset(self):
        self.capital = self.starting_capital
        self.profit = 0
        self.positions = []
        self.current_step = 0
        # self.total_reward = 0
        if self.max_steps == len(self.raw_data):
            self.data = self.raw_data
        else:
            start = random.randint(0, (len(self.raw_data) - (self.max_steps)))
            self.data = self.raw_data.iloc[start: start + (self.max_steps)]
        self.state = None

        if self.render_on:
            self.render()

        # Return the initial state of the grid
        return self.get_state(0)

    def load_data(self, file_path):
        # Load the raw data from csv
        data = pd.read_csv(file_path, index_col=0)
        data.index = pd.to_datetime(data.index, dayfirst=True)
        data['time_of_day'] = data.index
        data['time_of_day'] = (
            data['time_of_day'] - data['time_of_day'].dt.normalize()) / pd.Timedelta(seconds=1)  # Seconds since midnight
        data = data.dropna(axis=1)

        # Add indicators here

        # Add state columns here
        data['num_positions'] = 0
        data['position_value'] = 0.0
        data['capital'] = 0.0
        data['profit'] = 0.0
        data['action'] = 0

        return data

    def render(self):
        # print(f'Capital: {self.capital}')
        # print(f'Profit: {self.profit}')
        # print(f'Total Reward for episode: {self.total_reward}')
        print('TODO Render')
        print()

    def get_state(self, action):
        starting_id = (self.current_step - self.window_size) + 1

        self.data['num_positions'].iloc[self.current_step] = len(
            self.positions)
        self.data['position_value'].iloc[self.current_step] = len(
            self.positions) * self.data['close'].iloc[self.current_step]
        self.data['capital'].iloc[self.current_step] = self.capital
        self.data['profit'].iloc[self.current_step] = self.profit
        self.data['action'].iloc[self.current_step] = action

        if starting_id >= 0:
            windowed_data = self.data.iloc[starting_id:self.current_step+1]
        else:
            windowed_data = self.data.iloc[0]
            windowed_data = [windowed_data.copy()
                             for _ in range(abs(starting_id))]
            for i in range(0, self.current_step + 1):
                windowed_data.append(self.data.iloc[i])

            windowed_data = pd.DataFrame(windowed_data)
            # windowed_data = windowed_data.set_index('datetime')

        values = windowed_data.values
        state = self.normaliser.fit_transform(values)
        return np.array([state])

    def perform_action(self, action):
        done = True if self.current_step >= self.max_steps else False
        buy_reward = 0
        sell_reward = 0
        hold_reward = 0
        closed_market_reward = 0

        # Check for a valid move
        if self.is_valid_trade(action):
            reward = 2
            if action == 0:
                hold_reward = 0.01
            elif action == 1:  # Buy
                price = self.data['close'].iloc[self.current_step]
                self.positions.append(price)
                self.capital -= price
                buy_reward += (self.data['close'].iloc[self.current_step] -
                               self.data['close'].iloc[self.current_step - 1])  # Reward for buying when the stock is falling
                if buy_reward < -1:
                    buy_reward = -1
            elif action == 2:  # Sell
                buy_price = self.positions.pop()
                profit = self.data['close'].iloc[self.current_step] - buy_price
                self.profit += profit
                self.capital += self.data['close'].iloc[self.current_step]
                sell_reward += profit  # Reward for positive profit
                if sell_reward < -1:
                    sell_reward = -1

            # We now penalise for having open positions over a certain time (day trading)
            # This is after 14:00 (market closes at 16:00)
            # if self.data['time_of_day'].iloc[self.current_step] >= 50400 and (action == 0 or action == 1):
            #     # Number of hours over
            #     delta = (
            #         self.data['time_of_day'].iloc[self.current_step] - 50400) // 3600
            #     closed_market_reward = (delta * len(self.positions) * 0.01)

            # Reward for moving towards more capital
            goal_reward = (self.capital - self.capital_goal) * 0.01

            reward += (buy_reward + sell_reward + hold_reward +
                       closed_market_reward + goal_reward)

            # Ensure no negative reward when we used a valid trade
            if reward < 0:
                reward = 0
        else:
            # Slightly larger punishment for an invalid move
            reward = -1

        return reward, done

    def is_valid_trade(self, action):
        if action == 0:  # Hold
            return True
        elif action == 1:  # Buy
            if self.capital >= self.data['close'].iloc[self.current_step]:
                return True
            else:
                return False
        elif action == 2:  # Sell
            if len(self.positions) > 0:
                return True
            else:
                return False
        else:
            return False

    def step(self, action, step):
        # Apply the action to the environment, record the observations
        self.current_step = step
        reward, done = self.perform_action(action)
        self.total_reward += reward
        next_state = self.get_state(action)
        if self.current_step == self.max_steps - 1:
            done = True

        # Render the grid at each step
        if self.render_on and self.current_step == self.max_steps - 1:
            self.render()

        return reward, next_state, done
