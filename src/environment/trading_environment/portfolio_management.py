from src.environment.trading_environment.token_management import trade_token
import logging

logger = logging.getLogger("trading_environment/portfolio_management")


def performing_actions(cash, tokens, portfolio, gas_price, gas_limit, action_code, token_prices, priority_fee,
                       buy_limit, sell_limit):
    action = 1 if action_code == "Buy" else -1

    cash_p_token = cash / len(tokens) if action == 1 and len(tokens) > 0 else 0.0

    total_token_value = 0
    total_cash = 0

    for i, token in enumerate(tokens):
        # Trade and get current tokens and cash
        logger.info(f"Transaction #{i + 1}. {action} {token}.")
        portfolio[token], cash_earned, remaining_token_value = trade_token(
            cash=cash_p_token,
            base_gas=gas_price,
            gas_limit=gas_limit,
            available_units=portfolio[token],
            token_name=token,
            token_price=token_prices[token],
            eth_price=token_prices['ETH'],
            priority_fee=priority_fee,
            action=action,
            buy_limit=buy_limit,
            sell_limit=sell_limit
        )
        logger.info(f"Finished {action}: {token}.")

        # Cash earned
        logger.debug(f"Cash earned in {action}ing {token}: {cash_earned}.")
        total_cash += cash_earned

        # tokens remaining
        logger.debug(f"Remaining value in {token}: {remaining_token_value}.")
        total_token_value += remaining_token_value

    return total_cash, total_token_value


def portfolio_management(cash, token_portfolio, current_token_prices, current_gas_price,
                         priority_fee, gas_limit, tokens, buy_limit, sell_limit):
    """
    Manage the portfolio, buying and selling tokens depending on the actions given.

    :param cash:                    --float, current available cash
    :param token_portfolio:         --dictionary, map of available tokens
    :param current_token_prices:    --dictionary, map of prices of all tokens
    :param current_gas_price:       --float, current gas price in Gwei
    :param priority_fee:            --int, priority fee in Gwei
    :param gas_limit:              --int, gas limit in units
    :param tokens:                 --list, list of tokens to trade
    :param buy_limit:               --float, limit of units to buy per transaction
    :param sell_limit:              --float, limit of units to sell per transaction
    """
    logger.info(f"Calling Portfolio Management Function with {cash}USD available cash")

    # Getting all tokens to sell
    logger.debug("Getting the names of all tokens to sell")
    tokens_to_sell = [k for k, v in token_portfolio.items() if token_portfolio[k] > 0 and k not in tokens]

    # Selling
    cash_after_selling, tokens_value_after_selling = performing_actions(
        cash=cash,
        tokens=tokens_to_sell,
        portfolio=token_portfolio,
        gas_price=current_gas_price,
        gas_limit=gas_limit,
        action_code="Sell",
        token_prices=current_token_prices,
        priority_fee=priority_fee,
        buy_limit=buy_limit,
        sell_limit=sell_limit
    )

    # buying
    logger.debug("Getting the names of all tokens to buy")
    tokens_to_buy = [k for k, v in token_portfolio.items() if token_portfolio[k] == 0 and k in tokens]

    # Buying
    cash_after_buying, tokens_value_after_buying = performing_actions(
        cash=cash,
        tokens=tokens_to_buy,
        portfolio=token_portfolio,
        gas_price=current_gas_price,
        gas_limit=gas_limit,
        action_code="Buy",
        token_prices=current_token_prices,
        priority_fee=priority_fee,
        buy_limit=buy_limit,
        sell_limit=sell_limit
    )

    # Add up all cash
    total_cash_remaining = cash_after_buying + cash_after_selling
    total_tokens_value_remaining = tokens_value_after_buying + tokens_value_after_selling

    curr_total_net_worth = total_cash_remaining + total_tokens_value_remaining
    return token_portfolio, curr_total_net_worth, total_cash_remaining, total_tokens_value_remaining
