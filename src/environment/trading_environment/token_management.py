from src.environment.utils import buy_token, sell_token
from src.utils.visualization.custom_messages import show_hold_position_blue, show_buy_position_green, show_sell_position_red, show_neutral_position_gray, show_swap_position_orange

import logging

logger = logging.getLogger("trading_environment/token_management")


def neutral_position(day, curr_action, cash, portfolio, token_prices):
    tokens_new_value = 0
    for token in portfolio.keys():
        tokens_new_value += token_prices[token]
    net_worth = cash + tokens_new_value
    show_neutral_position_gray(day=day, curr_action=curr_action, tokens_value=tokens_new_value, cash=cash, net_worth=net_worth)
    return cash, tokens_new_value, net_worth


def hold_position(day, curr_action, cash, portfolio, token_prices):
    tokens_new_value = 0
    for token in portfolio.keys():
        tokens_new_value += token_prices[token]
    net_worth = cash + tokens_new_value
    show_hold_position_blue(day=day, curr_action=curr_action, tokens_value=tokens_new_value, cash=cash, net_worth=net_worth)
    return cash, tokens_new_value, net_worth


def sell_position(day, curr_action, cash, tokens, base_gas, gas_limit, priority_fee, portfolio, token_prices):
    tokens_new_value = 0
    cash_earned = 0
    for token in tokens:
        token_price = token_prices[token]
        eth_price = token_prices['ETH']
        current_tokens = portfolio[token]

        rem_tokens, rem_cash, rem_tokens_value = sell_token(base_gas=base_gas, gas_limit=gas_limit, priority_fee=priority_fee, available_tokens=current_tokens, token_price=token_price, eth_price=eth_price, token_name=token)

        portfolio[token] = rem_tokens
        cash_earned += rem_cash
        tokens_new_value += rem_tokens_value

    net_worth = cash_earned + tokens_new_value
    show_sell_position_red(day=day,  curr_action=curr_action, tokens_value=tokens_new_value, cash=cash_earned, net_worth=net_worth)
    return cash_earned, tokens_new_value, net_worth, portfolio


def buy_position(day, curr_action, cash, base_gas, gas_limit, priority_fee, tokens, portfolio, token_prices):
    tokens_new_value = 0
    remaining_cash = 0
    cash_per_token = cash / len(tokens)
    for token in tokens:
        token_price = token_prices[token]
        eth_price = token_prices['ETH']

        tokens_bought, rem_cash, tokens_bought_value = buy_token(cash=cash_per_token, base_gas=base_gas, gas_limit=gas_limit, priority_fee=priority_fee, token_price=token_price, eth_price=eth_price, token_name=token)

        portfolio[token] = tokens_bought
        remaining_cash += rem_cash
        tokens_new_value += tokens_bought_value

    net_worth = remaining_cash + tokens_new_value
    show_buy_position_green(day=day, curr_action=curr_action, tokens_value=tokens_new_value, cash=remaining_cash, net_worth=net_worth)

    return remaining_cash, tokens_new_value, net_worth, portfolio


def swap_position(day, prev_action, curr_action, cash, base_gas, gas_limit, priority_fee, tokens_to_sell, tokens_to_buy, portfolio, token_prices):
    cash, tokens_value, net_worth, portfolio = sell_position(day=day, curr_action=curr_action, cash=cash, tokens=tokens_to_sell, base_gas=base_gas, gas_limit=gas_limit, priority_fee=priority_fee, portfolio=portfolio, token_prices=token_prices)
    cash, tokens_value, net_worth, portfolio = buy_position(day=day, curr_action=curr_action, cash=cash, tokens=tokens_to_buy, base_gas=base_gas, gas_limit=gas_limit, priority_fee=priority_fee, portfolio=portfolio, token_prices=token_prices)
    show_swap_position_orange(day=day, prev_action=prev_action, curr_action=curr_action, tokens_value=tokens_value, cash=cash, net_worth=net_worth)
    return cash, tokens_value, net_worth, portfolio