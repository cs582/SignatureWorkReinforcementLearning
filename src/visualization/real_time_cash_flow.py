from IPython.display import display, clear_output
import numpy as np
import matplotlib.pyplot as plt


class RealTimeCashFlow:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        self.net_worth_hist = []
        self.cash_hist = []
        self.asset_value_hist = []

    def update(self, curr_net_worth, curr_cash, curr_asset_val):
        self.net_worth_hist.append(curr_net_worth)
        self.cash_hist.append(curr_cash)
        self.asset_value_hist.append(curr_asset_val)

        x = np.arange(0, len(self.net_worth_hist))

        self.ax.clear()
        self.ax.plot(x, self.net_worth_hist, color='green')
        self.ax.plot(x, self.asset_value_hist, color='blue')
        self.ax.plot(x, self.cash_hist, color='black')

        display(self.fig)
        clear_output()