from transaction_functions.token_management import trade_token


def portfolio_management(cash, token_portfolio, current_token_prices, current_gas_price, actions, buy_limit, sell_limit):
    """
    Manage the portfolio, buying and selling tokens depending on the actions given.

    :param cash:                    --float, current available cash
    :param token_portfolio:         --dictionary, map of available tokens
    :param current_token_prices:    --dictionary, map of prices of available tokens
    :param current_gas_price:       --float, current gas price in Gwei
    :param actions:                 --dictionary, map of actions to perform for each token
    :param buy_limit:               --float, limit of units to buy per transaction
    :param sell_limit:              --float, limit of units to sell per transaction
    :return: net worth              --float, total cash and value of units held
    """

    # Ensure portfolio and available prices have the same number of tokens
    length_portfolio = len(token_portfolio.values())
    length_token_prices = len(current_token_prices.values())

    assert (len(length_portfolio) == len(length_token_prices), f"Error: token_portfolio and current_token_prices must have same length, got {length_portfolio} and {length_token_prices}")

    # Ensure portfolio and available prices have the exact same tokens
    portfolio_tokens = sorted([x for x in token_portfolio.keys()])
    token_prices_available = sorted([x for x in current_token_prices.keys()])

    assert (portfolio_tokens == token_prices_available, f"Error: portfolio tokens and tokens in available prices must match.")

    # Get the names of all tokens in the portfolio
    tokens = [x for x in token_portfolio.keys()]

    # Calculate cash per token
    cash_ptoken = cash/len(tokens)

    curr_net_worth = 0

    # Perform the corresponding action for each token
    for token in tokens:
        # Trade and get current tokens and cash
        token_portfolio[token], curr_cash = trade_token(
            cash=cash_ptoken,
            gas=current_gas_price,
            available_tokens=token_portfolio[token],
            price=current_token_prices[token],
            action=actions[token],
            buy_limit=buy_limit,
            sell_limit=sell_limit,
            token_name=token
        )

        # Calculate current value of units held
        curr_units_value = token_portfolio[token]*current_token_prices[token]

        # Calculate current net worth i.e. cash + units value
        curr_net_worth += curr_cash + curr_units_value

    return curr_net_worth, curr_cash, curr_units_value