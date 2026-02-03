import utils
import numpy as np

def frantic_chase(agent, observations, aim_distance=3):
    observation = observations[agent]
    evader = utils.find_evader(observation, aim_distance)
    # random move when no observable evader
    if evader is None:
        return np.random.randint(5)
    return utils.chase_evader(evader, aim_distance)


def coward_chase(agent, observations, aim_distance=3, buddy_distance=3):
    observation = observations[agent]
    ally = utils.find_ally(observation, aim_distance)
    # solo random march
    if ally is None:
        return np.random.randint(5)
    # team random march
    if ally is not None:
        follow_move = utils.follow_ally(ally, aim_distance, buddy_distance)
        if follow_move == 0:
            return np.random.randint(5)
        return follow_move


def group_chase(agent, observations, aim_distance=3, buddy_distance=3):
    observation = observations[agent]
    ally = utils.find_ally(observation, aim_distance)
    evader = utils.find_evader(observation, aim_distance)
    # lonely agent: no allies around and no evader found
    if ally is None and evader is None:
        return np.random.randint(5)
    # team scounting: allies around but no evader found
    if ally is not None and evader is None:
        follow_move = utils.follow_ally(ally, aim_distance, buddy_distance)
        if follow_move == 0:
            return np.random.randint(5)
        return follow_move
    # team effort: allies around and evader found
    if ally is not None and evader is not None:
        return utils.chase_evader(evader, aim_distance)
    # calling backup: no allies, invader found
    return utils.chase_evader(evader, aim_distance)


def border_guided_group_chase(agent, observations, aim_distance=3, buddy_distance=3):
    observation = observations[agent]
    ally = utils.find_ally(observation, aim_distance)
    evader = utils.find_evader(observation, aim_distance)
    # lonely agent: no allies around and no evader found
    if ally is None and evader is None:
        path = utils.find_borders(observation)
        if path is None:
            return np.random.randint(5)
        return np.random.choice([np.random.randint(5), utils.follow_path(path, aim_distance)])
    # team scounting: allies around but no evader found
    if ally is not None and evader is None:
        follow_move = utils.follow_ally(ally, aim_distance, buddy_distance)
        if follow_move == 0:
            path = utils.find_borders(observation)
            if path is None:
                return np.random.randint(5)
            return np.random.choice([np.random.randint(5), utils.follow_path(path, aim_distance)])
        return follow_move
    # team effort: allies around and evader found
    if ally is not None and evader is not None:
        return utils.chase_evader(evader, aim_distance)
    # calling backup: no allies, invader found
    return utils.chase_evader(evader, aim_distance)