from IPython.display import display, clear_output
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)


class RealTimeCashFlow:
    def __init__(self, ):
        self.net_worth_hist = []
        self.cash_hist = []
        self.asset_value_hist = []

    def update(self, curr_net_worth, curr_cash, curr_asset_val):
        self.net_worth_hist.append(curr_net_worth)
        self.cash_hist.append(curr_cash)
        self.asset_value_hist.append(curr_asset_val)

        x = np.arange(0, len(self.net_worth_hist))

        ax.clear()
        ax.plot(x, self.net_worth_hist, color='green')
        ax.plot(x, self.asset_value_hist, color='blue')
        ax.plot(x, self.cash_hist, color='black')

        display(fig)
        clear_output()