import logging

logger = logging.getLogger("trading_environment/token_management")


def trade_token(cash, base_gas, gas_limit, priority_fee, available_units, token_price, eth_price, action, sell_limit, buy_limit, token_name=None, terminal_state=False):
    """
    Performs the corresponding transaction for the given token. It returns the remaining cash and tokens.
    :param cash:                --float, cash available for transaction
    :param base_gas:            --float, base gas price in Gwei
    :param gas_limit:           --int, gas limit in units
    :param priority_fee:        --floar, priority fee (tip per transaction) in Gwei
    :param available_units:     --float, units available of the given token
    :param token_price:         --float, current price of the given token
    :param eth_price:           --float, current price in USD of ETH
    :param action:              --int, action to take, it can be one of 1 (buy), 0 (nothing), -1 (sell)
    :param sell_limit:          --float, limit of units to sell per transaction
    :param buy_limit:           --float, limit of units to buy per transaction
    :param token_name:          --string, name of the traded token
    :param terminal_state:      --boolean, terminal state i.e. last day of trading
    :return:                    --tuple(float, float), returns the units and cash remaining, respectively
    """

    # Initialize cash flow variables
    units_to_buy = 0
    units_to_sell = 0
    cash_earned = 0
    cash_spent = 0

    # Calculate the gas price per unit in ETH
    gas_per_transaction_eth = 1e-9 * gas_limit * (base_gas + priority_fee)
    gas_per_transaction = gas_per_transaction_eth * eth_price

    # Adjust available cash after expected transaction fee
    available_cash = cash - gas_per_transaction

    # If price drops to 0, then sell
    action = action if token_price != 0 else 0

    # If buy and there is available money, then buy
    if action == 1 and available_cash > 0:
        units_to_buy = available_cash/token_price if available_cash/token_price <= buy_limit else buy_limit
        cash_spent = units_to_buy * token_price + gas_per_transaction

        logger.info(f"Token: {token_name}, close price: {token_price}")
        logger.info(f"Bought {units_to_buy} units at price {token_price} per unit after gas.")
        logger.info(f"With a {gas_per_transaction} gas per unit. Total cash spent {cash_spent}.")

    # If sell and there is available tokens, then sell
    if action == 0 and available_units > 0:
        if gas_per_transaction > available_units*token_price:
            logger.info(f"gas price {gas_per_transaction} per transaction is too high compared to unit current value {available_units*token_price}")

        if gas_per_transaction <= available_units*token_price:
            units_to_sell = available_units if available_units <= sell_limit else sell_limit
            cash_earned = units_to_sell * token_price - gas_per_transaction

            logger.info(f"Token {token_name}, close price: {token_price}")
            logger.info(f"Sold {units_to_sell} units at price {token_price} per unit after gas")
            logger.info(f"With a {gas_per_transaction} gas per unit. Total cash earned {cash_earned}.")

    # Calculate the total remaining tokens and total remaining cash for given token
    remaining_tokens = available_units + units_to_buy - units_to_sell
    remaining_cash = cash + cash_earned - cash_spent

    logger.info(f"remaining tokens: {remaining_tokens} with value {remaining_tokens*token_price}, remaining cash: {remaining_cash}")

    return remaining_tokens, remaining_cash
