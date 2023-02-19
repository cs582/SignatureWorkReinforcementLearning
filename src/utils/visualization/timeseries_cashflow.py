import os
import numpy as np
import matplotlib.pyplot as plt


class TradingCycleCashFlow:
    def __init__(self, saving_path):
        """
        Initialize the class by creating a figure and subplot for the real-time plot, 
        and initializing empty lists to store net worth, cash, and asset value history.
        """
        self.saving_path = saving_path

        self.x_train = []
        self.x_test = []

        self.net_worth_hist_train = []
        self.cash_hist_train = []
        self.asset_value_hist_train = []

        self.net_worth_hist_test = []
        self.cash_hist_test = []
        self.asset_value_hist_test = []

    def update(self, curr_cash, curr_asset_val, curr_net_worth, mode):
        """
        Update the real-time plot with the current net worth, cash, and asset value.
        :param (float) curr_cash: Current cash
        :param (float) curr_asset_val: Current asset value
        :param (float) curr_net_worth: Current net worth
        :param (str) mode: evaluating or training
        """
        # Append the current values to their respective lists
        if mode == 'train':
            self.net_worth_hist_train.append(curr_net_worth)
            self.cash_hist_train.append(curr_cash)
            self.asset_value_hist_train.append(curr_asset_val)
        if mode == 'eval':
            self.net_worth_hist_test.append(curr_net_worth)
            self.cash_hist_test.append(curr_cash)
            self.asset_value_hist_test.append(curr_asset_val)

        # Create an x-axis that ranges from 0 to the length of the net worth history
        self.x_train = np.arange(0, len(self.net_worth_hist_train))

        # Create an x-axis that goes from the end of the training history to the end of the eval history
        self.x_test = np.arange(len(self.x_train), len(self.x_train) + len(self.net_worth_hist_test))

    def save_current_cycle_plot(self, epoch):
        plt.title("Cash Flow")

        plt.plot(self.x_train, self.net_worth_hist_train, color='green', label='Net Worth train')
        plt.plot(self.x_train, self.asset_value_hist_train, color='blue', label='Asset Value train')
        plt.plot(self.x_train, self.cash_hist_train, color='black', label='Cash train')

        if len(self.x_test) > 0:
            plt.plot(self.x_test, self.net_worth_hist_test, color='lime', label='Net Worth eval')
            plt.plot(self.x_test, self.asset_value_hist_test, color='cyan', label='Asset Value eval')
            plt.plot(self.x_test, self.cash_hist_test, color='gray', label='Cash eval')

        plt.legend()
        plt.xlabel("Time Step")
        plt.ylabel("Value")

        if os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)

        plt.savefig(f"{self.saving_path}/trading_cycle_epoch{epoch}.png")

    def reset(self):
        """
        Reset the time history of cashflow back.
        WARNING: This will erase the whole previous history.
        """

        # Clear train history
        self.net_worth_hist_train = []
        self.cash_hist_train = []
        self.asset_value_hist_train = []

        # Clear test history
        self.net_worth_hist_test = []
        self.cash_hist_test = []
        self.asset_value_hist_test = []

        # Clear x axis
        self.x_train = []
        self.x_test = []