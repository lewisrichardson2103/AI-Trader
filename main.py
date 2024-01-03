from environment import Environment
from agent import Agent
from experience_replay import ExperienceReplay
import matplotlib.pyplot as plt
import time
import numpy as np


def backtest():
    window_size = 20

    environment = Environment('test_data/Validation_Data.csv', window_size=window_size,
                              max_steps=180, capital=1000, capital_goal=1010, render_on=True)
    environment.max_steps = len(environment.raw_data)
    state = environment.reset()

    features = len(environment.data.columns)
    agent = Agent(feature_size=features, window_size=window_size,
                  epsilon=0.01, epsilon_decay=0.9998, epsilon_end=0.01)
    agent.load(f'models/model_{features}.keras')

    # Loop until the dataset is done
    for step in range(environment.max_steps):
        # print('Step:', step)
        # Get the action choice from the agents policy
        action = agent.get_action(state)

        # Take a step in the environment and save the experience
        _, next_state, done = environment.step(action, step)

        # Set the state to the next_state
        state = next_state

    print(
        f'Finished at step: {step}')
    print(f'Capital: {environment.capital}')
    print('Profit from trades:', environment.profit)
    print(
        f'Position Value: {environment.data.position_value.iloc[-1]}')
    print(
        f'Total Equity: {( environment.capital + environment.data.position_value.iloc[-1])}')

    plt_data = environment.data.iloc[:180]
    plt_data['equity'] = environment.data.capital + \
        environment.data.position_value
    plt_data['buy'] = np.where(plt_data.action == 1, plt_data.close, np.nan)
    plt_data['sell'] = np.where(plt_data.action == 2, plt_data.close, np.nan)

    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    axs[0].plot(plt_data.index, plt_data.close)
    axs[0].scatter(plt_data.index, plt_data.buy, marker="x", color='g')
    axs[0].scatter(plt_data.index, plt_data.sell, marker="x", color='r')
    axs[1].plot(plt_data.index, plt_data.equity)
    plt.show()


if __name__ == '__main__':

    backtesting = True

    window_size = 20
    max_steps = 180
    rewards = []

    environment = Environment('test_data/AAPL_Daily_2018_2019.csv', window_size=window_size,
                              max_steps=max_steps, capital=1000, capital_goal=1010, render_on=True)
    features = len(environment.data.columns)
    agent = Agent(feature_size=features, window_size=window_size,
                  epsilon=1.0, epsilon_decay=0.9998, epsilon_end=0.01, learning_rate=0.0001)
    # agent.load(f'models/model_{features}.keras')
    experience_replay = ExperienceReplay(capacity=10000, batch_size=32)

    # Number of episodes to run before training stops
    episodes = 100

    for episode in range(episodes):
        # Get the initial state of the environment and set done to False
        state = environment.reset()
        print('Episode:', episode)
        # Loop until the episode finishes
        for step in range(max_steps):
            # print('Step:', step)

            # Get the action choice from the agents policy
            action = agent.get_action(state)

            # Take a step in the environment and save the experience
            reward, next_state, done = environment.step(action, step)
            experience_replay.add_experience(
                state, action, reward, next_state, done)

            # If the experience replay has enough memory to provide a sample, train the agent
            if experience_replay.can_provide_sample():
                experiences = experience_replay.sample_batch()
                agent.learn(experiences)

            # Set the state to the next_state
            state = next_state

            if done:
                print(
                    f'Finished at step: {step}. Time range was: {environment.data.time_of_day.iloc[0]} to {environment.data.time_of_day.iloc[-1]}')
                print(f'Capital: {environment.capital}')
                print('Profit from trades:', environment.profit)
                print(
                    f'Position Value: {environment.data.position_value.iloc[-1]}')
                print(
                    f'Total Equity: {( environment.capital + environment.data.position_value.iloc[-1])}')
                print('Average Reward:',
                      environment.total_reward / (episode + 1))
                print('Epsilon:', agent.epsilon)
                rewards.append(environment.total_reward / (episode + 1))
                break

            # Optionally, pause for half a second to evaluate the model
            # time.sleep(0.5)
        if environment.profit > 0:
            agent.save(f'models/model_{features}.keras')
            print('Model Saved')
        if episode % 10 == 0 and episode != 0:
            plt.plot(rewards)
            plt.show()
            if backtesting:
                backtest()
