from trading_environment.token_management import trade_token
import logging


def portfolio_management(cash, token_portfolio, current_token_prices, current_gas_price, actions, buy_limit,
                         sell_limit, print_transactions=True):
    """
    Manage the portfolio, buying and selling tokens depending on the actions given.

    :param print_transactions:      --boolean, true to print every single transaction taking place, false otherwise
    :param cash:                    --float, current available cash
    :param token_portfolio:         --dictionary, map of available tokens
    :param current_token_prices:    --dictionary, map of prices of available tokens
    :param current_gas_price:       --float, current gas price in Gwei
    :param actions:                 --numpy.array, array of actions to perform for each token
    :param buy_limit:               --float, limit of units to buy per transaction
    :param sell_limit:              --float, limit of units to sell per transaction
    :return: net worth              --float, total cash and value of units held
    """
    logging.info("Portfolio Management Function")

    # Ensure portfolio and available prices have the same number of tokens
    length_portfolio = len(token_portfolio)
    length_token_prices = len(current_token_prices)
    logging.debug(f"Portfolio length = {length_portfolio}, Length Token Prices = {length_token_prices}")

    # Map actions to their corresponding token name
    logging.debug("Mapping actions to their corresponding token name")
    action_map = {tkn: actions[i] for i, tkn in enumerate(token_portfolio.keys())}
    logging.debug(f"action_map = {action_map}")

    assert length_portfolio == length_token_prices, f"Error: token_portfolio and current_token_prices must have same length, got {length_portfolio} and {length_token_prices}"

    # Ensure portfolio and available prices have the exact same tokens
    logging.debug("Ensuring token_portfolio and current_token_prices have the exact same tokens.")
    portfolio_tokens = sorted([x for x in token_portfolio.keys()])
    token_prices_available = sorted([x for x in current_token_prices.keys()])

    assert portfolio_tokens == token_prices_available, f"Error: portfolio tokens and tokens in available prices must match."

    # Get the names of all tokens in the portfolio
    logging.debug("getting the names of all tokens that need to take action in the action map")
    tokens = [x for x, y in action_map.items() if y != 1.0]
    logging.debug(f"tokens to be traded: {tokens}")

    # Calculate cash per token
    cash_ptoken = cash / len(tokens)
    logging.debug(f"Cash per token: {cash_ptoken}")

    curr_total_net_worth = 0
    curr_total_units_value = 0
    curr_total_cash = 0

    # Perform the corresponding action for each token
    for i, token in enumerate(tokens):
        # Trade and get current tokens and cash
        logging.info(f"Transaction #{i+1}. Trading token {token} with action {action_map[token]}.")
        token_portfolio[token], curr_cash = trade_token(
            cash=cash_ptoken,
            gas=current_gas_price,
            available_tokens=token_portfolio[token],
            price=current_token_prices[token],
            action=action_map[token],
            buy_limit=buy_limit,
            sell_limit=sell_limit,
            token_name=token,
            print_transaction=print_transactions
        )
        logging.info(f"Finished trading token {token}.")

        curr_total_cash += curr_cash
        logging.debug(f"curr_total_cash = {curr_total_cash}, added curr_cash = {curr_cash}")

        # Calculate current value of units held
        curr_units_value = token_portfolio[token] * current_token_prices[token]
        curr_total_units_value += curr_units_value
        logging.debug(f"curr_total_assets_value = {curr_total_units_value}, added curr_assets_value = {curr_units_value}, number of these assets = {token_portfolio[token]}")

        # Calculate current net worth i.e. cash + units value
        curr_net_worth = curr_cash + curr_units_value
        curr_total_net_worth += curr_net_worth
        logging.debug(f"current total net worth = {curr_total_net_worth}, added curr_net_worth = {curr_net_worth}")

    return token_portfolio, curr_total_net_worth, curr_total_cash, curr_total_units_value
