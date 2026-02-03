import numpy as np 

def find_evader(observation, aim_distance):
    evader_layer = observation[:, :, 2]
    positions = np.argwhere(evader_layer > 0)

    # no evader in sight
    if len(positions) == 0:
        return None

    # getting the closest evader
    center = np.array([aim_distance, aim_distance])
    dists = np.sum((np.abs(positions - center)), axis=1)

    return positions[np.argmin(dists)]


def chase_evader(evader, aim_distance):
    cx, cy = aim_distance, aim_distance
    tx, ty = evader
    dx = tx - cx
    dy = ty - cy

    if abs(dx) > abs(dy):
        if dx < 0:
            return 1   # up
        else:
            return 2   # down
    else:
        if dy < 0:
            return 3   # left
        else:
            return 4   # right


def find_ally(observation, aim_distance):
    ally_layer = observation[:, :, 1]
    positions = np.argwhere(ally_layer > 0)

    # no ally in sight
    if len(positions) == 0:
        return None

    # getting the closest ally
    center = np.array([aim_distance, aim_distance])
    dists = np.sum((np.abs(positions - center)), axis=1)

    return positions[np.argmin(dists)]


def follow_ally(ally, aim_distance, threshold):
    cx, cy = aim_distance, aim_distance
    tx, ty = ally
    dx = tx - cx
    dy = ty - cy

    # threshold has to be in manhattan distance
    # close enough not to lose ally
    if abs(dx) + abs(dy) <= threshold:
        return 0

    if abs(dx) > abs(dy):
        if dx < 0:
            return 1   # up
        else:
            return 2   # down
    else:
        if dy < 0:
            return 3   # left
        else:
            return 4   # right


def find_borders(observation):
    border_layer = observation[:, :, 0]
    positions = np.argwhere(border_layer > 0)

    # no border in sight
    if len(positions) == 0:
        return None

    return positions


def follow_path(border, aim_distance):
    cx, cy = aim_distance, aim_distance
    center = np.array([cx, cy])

    positions = center - border
    dists = np.sum((np.abs(positions)), axis=1)
    closest = positions[np.argmin(dists)]

    dx, dy = closest[0], closest[1]

    if abs(dx) > abs(dy):
        return 3 if dx < 0 else 4  # Left/Right
    else:
        return 1 if dy < 0 else 2  # Up/Down