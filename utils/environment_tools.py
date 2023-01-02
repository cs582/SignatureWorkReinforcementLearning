def map_actions(actions, token_names):
    action_map = {}
    for i, name in enumerate(token_names):
        action_map[name] = actions[i]

    return action_map
