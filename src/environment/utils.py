import logging

logger = logging.getLogger("trading_environment/utils")


def buy_token(cash, base_gas, gas_limit, priority_fee, token_price, eth_price, token_name=None):
    """
    Performs the corresponding transaction for the given token. It returns the remaining cash and tokens.
    :param cash:                --float, cash available for transaction
    :param base_gas:            --float, base gas price in Gwei
    :param gas_limit:           --int, gas limit in units
    :param priority_fee:        --floar, priority fee (tip per transaction) in Gwei
    :param token_price:         --float, current price of the given token
    :param eth_price:           --float, current price in USD of ETH
    :param token_name:          --string, name of the traded token
    :return:                    --tuple(float, float), returns the units and cash remaining, respectively
    """

    # Calculate the gas price per unit in ETH
    gas_per_transaction_eth = 1e-9 * gas_limit * (base_gas + priority_fee)
    gas_per_transaction = gas_per_transaction_eth * eth_price

    # Adjust available cash after expected transaction fee
    available_cash = cash - gas_per_transaction

    if available_cash > 0:
        tokens_to_buy = available_cash/token_price if available_cash/token_price <= gas_limit else gas_limit
        cash_spent = tokens_to_buy * token_price + gas_per_transaction

        logger.info(f"Token: {token_name}, close price: {token_price}")
        logger.info(f"Bought {tokens_to_buy} units at price {token_price} per unit after gas.")
        logger.info(f"With a {gas_per_transaction} gas per unit. Total cash spent {cash_spent}.")

        bought_tokens = tokens_to_buy
        remaining_cash = cash - cash_spent
        bought_tokens_value = tokens_to_buy*token_price

        logger.info(f"Remaining tokens: {bought_tokens} with value {bought_tokens_value}, remaining cash: {remaining_cash}")

        return bought_tokens, remaining_cash, bought_tokens_value

    return 0, cash, 0


def sell_token(base_gas, gas_limit, priority_fee, available_tokens, token_price, eth_price, token_name=None, terminal_state=False):
    """
    Performs the corresponding transaction for the given token. It returns the remaining cash and tokens.
    :param base_gas:            --float, base gas price in Gwei
    :param gas_limit:           --int, gas limit in units
    :param priority_fee:        --floar, priority fee (tip per transaction) in Gwei
    :param available_tokens:     --float, units available of the given token
    :param token_price:         --float, current price of the given token
    :param eth_price:           --float, current price in USD of ETH
    :param token_name:          --string, name of the traded token
    :param terminal_state:      --boolean, terminal state i.e. last day of trading
    :return:                    --tuple(float, float), returns the units and cash remaining, respectively
    """

    # Calculate the gas price per unit in ETH
    gas_per_transaction_eth = 1e-9 * gas_limit * (base_gas + priority_fee)
    gas_per_transaction = gas_per_transaction_eth * eth_price

    # If sell and there is available tokens, then sell
    if available_tokens > 0:
        if gas_per_transaction > available_tokens*token_price:
            logger.info(f"gas price {gas_per_transaction} per transaction is too high compared to unit current value {available_tokens*token_price}")
            return available_tokens, 0, available_tokens*token_price

        if gas_per_transaction <= available_tokens*token_price:
            tokens_to_sell = available_tokens if available_tokens <= gas_limit else gas_limit
            cash_earned = tokens_to_sell * token_price - gas_per_transaction

            logger.info(f"Token {token_name}, close price: {token_price}")
            logger.info(f"Sold {tokens_to_sell} units at price {token_price} per unit after gas")
            logger.info(f"With a {gas_per_transaction} gas per unit. Total cash earned {cash_earned}.")

            tokens_sold = tokens_to_sell
            remaining_tokens = available_tokens - tokens_sold
            remaining_tokens_value = remaining_tokens*token_price

            logger.info(f"Remaining tokens: {remaining_tokens} with value {remaining_tokens_value}. Cash earned: {cash_earned}")

            return remaining_tokens, cash_earned, remaining_tokens_value

    return available_tokens, 0, available_tokens*token_price
