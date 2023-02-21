import numpy as np
from logs.logger_file import logger_trading_info


def show_rewards(mode, day, roi, sharpe, reward, hist_max):
    message = f"{mode.upper()} >> TD[{day}]: ROI: {np.round(roi, 2)}, SHARPE RATIO: {np.round(sharpe, 2)}, and REWARD: {np.round(reward, 2)}. HIST MAX: {np.round(hist_max, 2)}"
    print(message)

def show_current_state(day, curr_action, tokens_value, cash, net_worth, mode):
    tokens_value = np.round(tokens_value, 2)
    cash = np.round(cash, 2)
    net_worth = np.round(net_worth, 2)
    message = f"{mode.upper()} >> TD[{day}], ACTION[{curr_action}]: Current Cash {cash}; Current Units Value {tokens_value}. Net Worth {net_worth}"
    logger_trading_info.info(message)
    print(message)


def show_swap_position_orange(day, prev_action, curr_action, tokens_value, cash, net_worth, mode):
    tokens_value = np.round(tokens_value, 2)
    cash = np.round(cash, 2)
    net_worth = np.round(net_worth, 2)
    message = f"{mode.upper()} >> \033[1;34mTD[{day}], ACTION[{curr_action}]: Swapped from group {prev_action}. "
    message += f"Current Cash {cash}; Current Units Value {tokens_value}. Net Worth {net_worth}.\033[0m"
    logger_trading_info.info(message)
    print(message)


def show_buy_position_green(day, curr_action, tokens_value, cash, net_worth, mode):
    tokens_value = np.round(tokens_value, 2)
    cash = np.round(cash, 2)
    net_worth = np.round(net_worth, 2)
    message = f"{mode.upper()} >> \033[1;32mTD[{day}], ACTION[{curr_action}]: Bought units with value {tokens_value}. Remaining Cash {cash}. Net Worth {net_worth}.\033[0m"
    logger_trading_info.info(message)
    print(message)


def show_sell_position_red(day, curr_action, tokens_value, cash, net_worth, mode):
    tokens_value = np.round(tokens_value, 2)
    cash = np.round(cash, 2)
    net_worth = np.round(net_worth, 2)
    message = f"{mode.upper()} >> \033[1;33mTD[{day}], ACTION[{curr_action}]: Sold units for {tokens_value}. Cash Earned {cash}. Net Worth {net_worth}.\033[0m"
    logger_trading_info.info(message)
    print(message)


def show_hold_position_blue(day, curr_action, tokens_value, cash, net_worth, mode):
    tokens_value = np.round(tokens_value, 2)
    cash = np.round(cash, 2)
    net_worth = np.round(net_worth, 2)
    message = f"{mode.upper()} >> \033[1;31mTD[{day}], ACTION[{curr_action}]: Held Units for value {tokens_value}. Remaining Cash {cash}. Net Worth {net_worth}.\033[0m"
    logger_trading_info.info(message)
    print(message)


def show_neutral_position_gray(day, curr_action, tokens_value, cash, net_worth, mode):
    tokens_value = np.round(tokens_value, 2)
    cash = np.round(cash, 2)
    net_worth = np.round(net_worth, 2)
    message = f"{mode.upper()} >> \033[1;30mTD[{day}], ACTION[{curr_action}]: Neutral Position. Units held with value {tokens_value}. Remaining Cash {cash}. Net Worth {net_worth}.\033[0m"
    logger_trading_info.info(message)
    print(message)