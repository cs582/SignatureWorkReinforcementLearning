import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output


class RealTimeCashFlow:
    def __init__(self):
        """
        Initialize the class by creating a figure and subplot for the real-time plot, 
        and initializing empty lists to store net worth, cash, and asset value history.
        """
        self.fig = plt.figure(figsize=(15,8))
        self.ax = self.fig.add_subplot(111)

        self.net_worth_hist = []
        self.cash_hist = []
        self.asset_value_hist = []

    def update(self, curr_cash, curr_asset_val, curr_net_worth, mode):
        """
        Update the real-time plot with the current net worth, cash, and asset value.
        :param (float) curr_cash: Current cash
        :param (float) curr_asset_val: Current asset value
        :param (float) curr_net_worth: Current net worth
        :param (str) mode: evaluating or training
        """
        # Append the current values to their respective lists
        self.net_worth_hist.append(curr_net_worth)
        self.cash_hist.append(curr_cash)
        self.asset_value_hist.append(curr_asset_val)

        # Create an x-axis that ranges from 0 to the length of the net worth history
        x = np.arange(0, len(self.net_worth_hist))

        # Clear the current plot and update it with the new values
        self.ax.clear()
        self.ax.plot(x, self.net_worth_hist, color='green' if mode == 'train' else 'g', label='Net Worth')
        self.ax.plot(x, self.asset_value_hist, color='blue' if mode == 'train' else 'cian', label='Asset Value')
        self.ax.plot(x, self.cash_hist, color='black' if mode == 'train' else 'gray', label='Cash')

        self.ax.legend()
        self.ax.set_xlabel("Time Step")
        self.ax.set_ylabel("Value")
        self.ax.set_title("Cash Flow")

        # Display the plot and clear any previous output
        display(self.fig)
        clear_output(wait=True)

    def reset(self):
        """
        Reset the time history of cashflow back.
        WARNING: This will erase the whole previous history.
        """
        self.net_worth_hist = []
        self.cash_hist = []
        self.asset_value_hist = []