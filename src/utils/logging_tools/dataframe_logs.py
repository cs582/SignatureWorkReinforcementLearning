def prices_and_gas_preview(logger, token_prices, gas_prices):
    logger.debug(f"""
    Head of the token prices:
    {token_prices[:5]}
    Tail of the token prices:
    {token_prices[-5:]}
    """)

    logger.debug(f"""
    Head of the gas prices:
    {gas_prices.head(5)}
    Tail of the gas prices:
    {gas_prices.tail(5)}
    """)


def images_preview(logger, images):
    logger.debug(f"""
    Head of the images:
    {images.head(5)}
    Tail of the images:
    {images.tail(5)}
    """)