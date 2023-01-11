import logging


def trade_token(cash, gas, available_tokens, price, action, sell_limit, buy_limit, token_name=None, print_transaction=False):
    """
    Performs the corresponding transaction for the given token. It returns the remaining cash and tokens.
    :param print_transaction:   --boolean, decide whether to print the transaction or not
    :param cash:                --float, cash available for transaction
    :param gas:                 --float, gas price in Gwei
    :param available_tokens:    --float, units available of the given token
    :param price:               --float, current price of the given token
    :param action:              --int, action to take, it can be one of 1 (buy), 0 (nothing), -1 (sell)
    :param sell_limit:          --float, limit of units to sell per transaction
    :param buy_limit:           --float, limit of units to buy per transaction
    :param token_name:          --string, name of the traded token
    :return:                    --tuple(float, float), returns the units and cash remaining, respectively
    """

    # Initialize cash flow variables
    units_to_buy = 0
    units_to_sell = 0
    cash_earned = 0
    cash_spent = 0

    # Calculate the gas price per unit in ETH
    gas_per_unit = price * gas / 1e-9

    # Calculate the price per unit after gas fee
    price_per_unit = price + gas_per_unit

    # If buy and there is available money, then buy
    if action == 2 and cash > 0:
        units_to_buy = cash/price_per_unit if cash/price_per_unit <= buy_limit else buy_limit
        cash_spent = units_to_buy * price_per_unit

        if print_transaction:
            if token_name is not None:
                logging.info(
                    f"Bought {units_to_buy} units of {token_name} at {price_per_unit} per unit. Total cash spent {cash_spent}.")
            else:
                logging.info(
                    f"Bought {units_to_buy} units at {price_per_unit} per unit. Total cash spent {cash_spent}.")

    # If sell and there is available tokens, then buy
    if action == 0 and available_tokens > 0:
        units_to_sell = available_tokens if available_tokens <= sell_limit else sell_limit
        cash_earned = units_to_sell*price - gas_per_unit*units_to_sell

        if print_transaction:
            if token_name is not None:
                logging.info(
                    f"Sold {units_to_sell} units of {token_name} at {price} per unit with {gas_per_unit} gas. Total cash earned {cash_earned}.")
            else:
                logging.info(
                    f"Sold {units_to_sell} units at {price} per unit with {gas_per_unit} gas. Total cash earned {cash_earned}.")

    # Calculate the total remaining tokens and total remaining cash for given token
    remaining_tokens = available_tokens + units_to_buy - units_to_sell
    remaining_cash = cash + cash_earned - cash_spent

    logging.info(f"remaining tokens: {remaining_tokens}, remaining cash: {remaining_cash}")

    return remaining_tokens, remaining_cash
