import numpy as np


def show_current_state(day, tokens_value, cash, net_worth):
    tokens_value = np.round(tokens_value, 2)
    cash = np.round(cash, 2)
    net_worth = np.round(net_worth, 2)
    message = f"Trading Day [{day}]: Current Cash {cash}; Current Units Value {tokens_value}. Net Worth {net_worth}"
    print(message)


def show_swap_position_orange(day, prev_action, curr_cation, tokens_value, cash, net_worth):
    tokens_value = np.round(tokens_value, 2)
    cash = np.round(cash, 2)
    net_worth = np.round(net_worth, 2)
    message = f"\033[1;33mTrading Day [{day}]: Swapped from group {prev_action} to group {curr_cation}\n"
    message += f"Current Cash {cash}; Current Units Value {tokens_value}. Net Worth {net_worth}.\033[0m"
    print(message)


def show_buy_position_green(day, tokens_value, cash, net_worth):
    tokens_value = np.round(tokens_value, 2)
    cash = np.round(cash, 2)
    net_worth = np.round(net_worth, 2)
    message = f"\033[1;32mTrading Day [{day}]: Bought units with value {tokens_value}. Remaining Cash {cash}. Net Worth {net_worth}.\033[0m"
    print(message)


def show_sell_position_red(day, tokens_value, cash, net_worth):
    tokens_value = np.round(tokens_value, 2)
    cash = np.round(cash, 2)
    net_worth = np.round(net_worth, 2)
    message = f"\033[1;34mTrading Day [{day}]: Sold units for {tokens_value}. Remaining Cash {cash}. Net Worth {net_worth}.\033[0m"
    print(message)


def show_hold_position_blue(day, tokens_value, cash, net_worth):
    tokens_value = np.round(tokens_value, 2)
    cash = np.round(cash, 2)
    net_worth = np.round(net_worth, 2)
    message = f"\033[1;31mTrading Day [{day}]: Held Units for value {tokens_value}. Remaining Cash {cash}. Net Worth {net_worth}.\033[0m"
    print(message)


def show_neutral_position_gray(day, tokens_value, cash, net_worth):
    tokens_value = np.round(tokens_value, 2)
    cash = np.round(cash, 2)
    net_worth = np.round(net_worth, 2)
    message = f"\033[1;30mTrading Day [{day}]: Neutral Position. Units held with value {tokens_value}. Remaining Cash {cash}. Net Worth {net_worth}.\033[0m"
    print(message)