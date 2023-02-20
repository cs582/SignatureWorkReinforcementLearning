from src.environment.utils import buy_token, sell_token
from src.utils.visualization.custom_messages import show_hold_position_blue, show_buy_position_green, show_sell_position_red, show_neutral_position_gray, show_swap_position_orange

from logs.logger_file import logger_trading_info, logger_detailed


def neutral_position(day, curr_action, cash, portfolio, token_prices):
    tokens_new_value = 0
    for token in portfolio.keys():
        tokens_held = portfolio[token]
        tokens_new_value += token_prices[token] * tokens_held
        logger_detailed.info(f"DAY[{day}] TOKEN[{token}] POSITION[NEUTRAL]: {tokens_held} Units Valued at {tokens_held*token_prices[token]}.")
    net_worth = cash + tokens_new_value
    show_neutral_position_gray(day=day, curr_action=curr_action, tokens_value=tokens_new_value, cash=cash, net_worth=net_worth)
    return cash, tokens_new_value, net_worth


def hold_position(day, curr_action, cash, tokens_to_hold, portfolio, token_prices):
    tokens_new_value = 0
    for token in tokens_to_hold:
        tokens_held = portfolio[token]
        tokens_new_value += token_prices[token] * tokens_held
        logger_detailed.info(f"DAY[{day}] TOKEN[{token}] POSITION[HOLD]: {tokens_held} Units Valued at {tokens_held*token_prices[token]}.")
    net_worth = cash + tokens_new_value
    show_hold_position_blue(day=day, curr_action=curr_action, tokens_value=tokens_new_value, cash=cash, net_worth=net_worth)
    return cash, tokens_new_value, net_worth


def sell_position(day, curr_action, tokens, base_gas, gas_limit, priority_fee, portfolio, token_prices):
    tokens_new_value = 0
    cash_earned = 0
    for token in tokens:
        token_price = token_prices[token]
        eth_price = token_prices['ETH']
        current_tokens = portfolio[token]

        logger_detailed.info(f"DAY[{day}] TOKEN[{token}] ACTION[SELL]:")
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

        logger_detailed.info(f"DAY[{day}] TOKEN[{token}] ACTION[SELL]:")
        tokens_bought, rem_cash, tokens_bought_value = buy_token(cash=cash_per_token, base_gas=base_gas, gas_limit=gas_limit, priority_fee=priority_fee, token_price=token_price, eth_price=eth_price, token_name=token)

        portfolio[token] = tokens_bought
        remaining_cash += rem_cash
        tokens_new_value += tokens_bought_value

    net_worth = remaining_cash + tokens_new_value
    show_buy_position_green(day=day, curr_action=curr_action, tokens_value=tokens_new_value, cash=remaining_cash, net_worth=net_worth)

    return remaining_cash, tokens_new_value, net_worth, portfolio


def swap_position(day, prev_action, curr_action, cash, base_gas, gas_limit, priority_fee, tokens_to_sell, tokens_to_buy, portfolio, token_prices):
    logger_detailed.info(f"DAY[{day}] POSITION[SWAP] {prev_action} -> {curr_action}:")

    # Selling current tokens to collect money
    cash_earned, tokens_value, net_worth, portfolio = sell_position(day=day, curr_action=curr_action, tokens=tokens_to_sell, base_gas=base_gas, gas_limit=gas_limit, priority_fee=priority_fee, portfolio=portfolio, token_prices=token_prices)
    total_cash_to_buy = cash_earned + cash

    # Buying tokens
    total_cash, tokens_value, net_worth, portfolio = buy_position(day=day, curr_action=curr_action, cash=total_cash_to_buy, tokens=tokens_to_buy, base_gas=base_gas, gas_limit=gas_limit, priority_fee=priority_fee, portfolio=portfolio, token_prices=token_prices)

    # Show swap changes
    show_swap_position_orange(day=day, prev_action=prev_action, curr_action=curr_action, tokens_value=tokens_value, cash=cash, net_worth=net_worth)
    return total_cash, tokens_value, net_worth, portfolio