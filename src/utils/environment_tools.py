def map_actions_to_tokens(action, action_map):
    if action != 0:
        return action_map[str(action)]
    return []

