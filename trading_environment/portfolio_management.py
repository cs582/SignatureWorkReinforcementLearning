from trading_environment.token_management import trade_token
import logging

logger = logging.getLogger("trading_environment/portfolio_management")


def portfolio_management(cash, token_portfolio, current_token_prices, current_gas_price,
                         priority_fee, gas_limit, actions, buy_limit, sell_limit):
    """
    Manage the portfolio, buying and selling tokens depending on the actions given.

    :param cash:                    --float, current available cash
    :param token_portfolio:         --dictionary, map of available tokens
    :param current_token_prices:    --dictionary, map of prices of all tokens
    :param current_gas_price:       --float, current gas price in Gwei
    :param priority_fee:            --int, priority fee in Gwei
    :param gas_limit:              --int, gas limit in units
    :param actions:                 --numpy.array, array of actions to perform for each token
    :param buy_limit:               --float, limit of units to buy per transaction
    :param sell_limit:              --float, limit of units to sell per transaction
    :return: net worth              --float, total cash and value of units held
    """
    logger.info(f"Calling Portfolio Management Function with {cash}USD available cash")

    # Map actions to their corresponding token name
    logger.debug("Mapping actions to their corresponding token name")
    action_map = {tkn: actions[i] for i, tkn in enumerate(token_portfolio.keys())}
    logger.debug(f"action_map = {action_map}")

    # Get the names of all tokens in the portfolio
    logger.debug("getting the names of all tokens that need to take action in the action map")
    tokens = [x for x, y in action_map.items()]
    logger.debug(f"tokens to be traded: {tokens}")

    # Calculate cash per token
    cash_ptoken = cash / len([x for x, y in action_map.items() if y == 1.0])
    logger.debug(f"Cash per token: {cash_ptoken}")

    curr_total_net_worth = 0
    curr_total_units_value = 0
    curr_total_cash = 0

    # Perform the corresponding action for each token
    for i, token in enumerate(tokens):
        # Trade and get current tokens and cash
        logger.info(f"Transaction #{i+1}. Trading token {token} with action {action_map[token]}.")
        token_portfolio[token], curr_cash = trade_token(
            cash=cash_ptoken if action_map[token]==1.0 else 0.0,
            base_gas=current_gas_price,
            gas_limit=gas_limit,
            available_units=token_portfolio[token],
            token_name=token,
            token_price=current_token_prices[token],
            eth_price=current_token_prices['ETH'],
            priority_fee=priority_fee,
            action=action_map[token],
            buy_limit=buy_limit,
            sell_limit=sell_limit
        )
        logger.info(f"Finished trading token {token}.")

        curr_total_cash += curr_cash
        logger.debug(f"curr_total_cash = {curr_total_cash}, added curr_cash = {curr_cash}")

        # Calculate current value of units held
        curr_units_value = token_portfolio[token] * current_token_prices[token]
        curr_total_units_value += curr_units_value
        logger.debug(f"curr_total_assets_value = {curr_total_units_value}, added curr_assets_value = {curr_units_value}, number of these assets = {token_portfolio[token]}")

        # Calculate current net worth i.e. cash + units value
        curr_net_worth = curr_cash + curr_units_value
        curr_total_net_worth += curr_net_worth
        logger.debug(f"current total net worth = {curr_total_net_worth}, added curr_net_worth = {curr_net_worth}")

    return token_portfolio, curr_total_net_worth, curr_total_cash, curr_total_units_value
