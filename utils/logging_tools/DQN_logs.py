import logging


def check_experience(cur_state, cur_action, cur_reward, next_image):
    if cur_state.isnan().any() or cur_state.isinf().any():
        logging.debug(f"current state has {cur_state.isnan().sum()} nans")
        logging.debug(f"current state has {cur_state.isinf().sum()} infs")
    if next_image.isnan().any() or next_image.isinf().any():
        logging.debug(f"next image has {next_image.isnan().sum()} nans")
        logging.debug(f"next image has {next_image.isinf().sum()} infs")