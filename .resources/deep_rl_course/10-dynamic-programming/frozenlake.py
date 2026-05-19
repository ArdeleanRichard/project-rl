import sys
from io import StringIO

import numpy as np
import gymnasium as gym
from gymnasium import spaces

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}


class FrozenLakeEnv(gym.Env):
    """
    Frozen Lake environment compatible with Gymnasium.

    Actions:
        0 = LEFT
        1 = DOWN
        2 = RIGHT
        3 = UP
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, desc=None, map_name="4x4", is_slippery=True, render_mode=None):
        if desc is None and map_name is None:
            raise ValueError("Must provide either desc or map_name")
        elif desc is None:
            desc = MAPS[map_name]

        self.render_mode = render_mode
        self.is_slippery = is_slippery

        # Use string dtype to simplify comparisons and rendering.
        desc = [row.decode("utf-8") if isinstance(row, bytes) else row for row in desc]
        self.desc = np.array([list(row) for row in desc], dtype="U1")
        self.nrow, self.ncol = self.desc.shape

        self.nA = 4
        self.nS = self.nrow * self.ncol

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        isd = (self.desc == "S").astype(np.float64).ravel()
        isd /= isd.sum()
        self.initial_state_distrib = isd

        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}

        def to_s(row, col):
            return row * self.ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, self.nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, self.ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return row, col

        for row in range(self.nrow):
            for col in range(self.ncol):
                s = to_s(row, col)
                letter = self.desc[row, col]

                for a in range(self.nA):
                    li = self.P[s][a]

                    if letter in {"G", "H"}:
                        li.append((1.0, s, 0.0, True))
                    else:
                        if self.is_slippery:
                            for b in ((a - 1) % 4, a, (a + 1) % 4):
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = self.desc[newrow, newcol]
                                terminated = newletter in {"G", "H"}
                                reward = 1.0 if newletter == "G" else 0.0
                                li.append((1.0 / 3.0, newstate, reward, terminated))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = self.desc[newrow, newcol]
                            terminated = newletter in {"G", "H"}
                            reward = 1.0 if newletter == "G" else 0.0
                            li.append((1.0, newstate, reward, terminated))

        self.s = None
        self.lastaction = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.s = int(self.np_random.choice(self.nS, p=self.initial_state_distrib))
        self.lastaction = None

        if self.render_mode == "human":
            self.render()

        return self.s, {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in [0, {self.nA - 1}].")

        action = int(action)
        transitions = self.P[self.s][action]
        probs = [t[0] for t in transitions]
        idx = self.np_random.choice(len(transitions), p=probs)

        _, next_state, reward, terminated = transitions[idx]
        self.s = int(next_state)
        self.lastaction = action

        if self.render_mode == "human":
            self.render()

        return self.s, float(reward), bool(terminated), False, {}

    def render(self):
        if self.s is None:
            return None

        outfile = sys.stdout if self.render_mode == "human" else StringIO()

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()

        # Mark the agent position.
        if self.render_mode == "human":
            desc[row][col] = f"\x1b[41m{desc[row][col]}\x1b[0m"
        else:
            desc[row][col] = "A"

        if self.lastaction is not None:
            action_names = ["Left", "Down", "Right", "Up"]
            outfile.write(f"  ({action_names[self.lastaction]})\n")
        else:
            outfile.write("\n")

        outfile.write("\n".join("".join(line) for line in desc) + "\n")

        if self.render_mode != "human":
            return outfile.getvalue()

    def close(self):
        pass