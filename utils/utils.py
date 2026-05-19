from gymnasium.spaces import Tuple, Discrete, Box

def get_space_size(space):

    if isinstance(space, Discrete):
        return space.n

    elif isinstance(space, Tuple):
        n_states = 1
        for s in space.spaces:
            n_states *= s.n
        return n_states

    elif isinstance(space, Box):
        n_states = 1
        for s in space.shape:
            n_states *= s
        return n_states
