from src.environment.trading_environment.token_management import hold_position, sell_position, buy_position, neutral_position, swap_position

import logging

logger = logging.getLogger("trading_environment/portfolio_management")


def portfolio_management(day, prev_action, curr_action, position, cash, token_portfolio, current_token_prices, current_gas_price, priority_fee, gas_limit, tokens_to_buyorhold, tokens_to_sell):
    """
    Manage the portfolio, buying and selling tokens depending on the actions given.

    :param day                      --int, current trading day
    :param prev_action              --int, previous day action
    :param curr_action              --int, current day action
    :param position                 --str, position to take today
    :param cash:                    --float, current available cash
    :param token_portfolio:         --dictionary, map of available tokens
    :param current_token_prices:    --dictionary, map of prices of all tokens
    :param current_gas_price:       --float, current gas price in Gwei
    :param priority_fee:            --int, priority fee in Gwei
    :param gas_limit:              --int, gas limit in units
    :param tokens_to_buyorhold:     --list, list of tokens to trade
    """
    logger.info(f"Calling Portfolio Management Function with {cash} USD available cash.")

    curr_total_net_worth = 0
    total_cash_remaining = 0
    total_tokens_value_remaining = 0

    if position == "Hold":
        logger.info("Hold Position.")
        total_cash_remaining, total_tokens_value_remaining, curr_total_net_worth = hold_position(day=day, curr_action=curr_action, cash=cash, portfolio=token_portfolio, token_prices=current_token_prices)
    if position == "Buy":
        logger.info("Buy Position.")
        total_cash_remaining, total_tokens_value_remaining, curr_total_net_worth, token_portfolio = buy_position(day=day, curr_action=curr_action, cash=cash, base_gas=current_gas_price, gas_limit=gas_limit, priority_fee=priority_fee, tokens=tokens_to_buyorhold, portfolio=token_portfolio, token_prices=current_token_prices)
    if position == "Sell":
        logger.info("Sell Position.")
        total_cash_remaining, total_tokens_value_remaining, curr_total_net_worth, token_portfolio = sell_position(day=day, curr_action=curr_action, cash=cash, tokens=tokens_to_sell, base_gas=current_gas_price, gas_limit=gas_limit, priority_fee=priority_fee, portfolio=token_portfolio, token_prices=current_token_prices)
    if position == "Swap":
        logger.info("Neutral Position.")
        total_cash_remaining, total_tokens_value_remaining, curr_total_net_worth, token_portfolio = swap_position(day=day, prev_action=prev_action, curr_action=curr_action, cash=cash, base_gas=current_gas_price, gas_limit=gas_limit, priority_fee=priority_fee, tokens_to_sell=tokens_to_sell, tokens_to_buy=tokens_to_buyorhold, portfolio=token_portfolio, token_prices=current_token_prices)
    if position == "Neutral":
        logger.info("Neutral Position.")
        total_cash_remaining, total_tokens_value_remaining, curr_total_net_worth = neutral_position(day=day, cash=cash, curr_action=curr_action, portfolio=token_portfolio, token_prices=current_token_prices)
    if position not in ['Hold', 'Buy', 'Sell', 'Swap', 'Neutral']:
        logger.warn(f"UNKNOWN POSITION! AT DAY {day}. Previous Action: {prev_action} and Curr Action: {curr_action}.")
    return token_portfolio, curr_total_net_worth, total_cash_remaining, total_tokens_value_remaining
