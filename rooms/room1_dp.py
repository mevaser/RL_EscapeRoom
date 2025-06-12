import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from environment.grid_world import GridWorldEnv

class DPRoom(GridWorldEnv):
    """
    Room 1: Dynamic Programming implementation using Value Iteration with snitch collection
    """
    def __init__(self, size: int = 10, gamma: float = 0.99, theta: float = 1e-6):
        super().__init__(size=size)
        self._setup_room()
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros((size, size, 1 << self.num_snitches))
        self.policy = np.zeros((size, size, 1 << self.num_snitches), dtype=int)


    def _setup_room(self):
        # Obstacles
        for x in [2, 3, 4]:
            self.add_special_tile('obstacles', (x, 2))
        for y in [4, 5, 6]:
            self.add_special_tile('obstacles', (6, y))

        # Slippery tiles
        for x in [1, 2, 3]:
            for y in [7, 8]:
                self.add_special_tile('slippery', (x, y))

        # Prisons
        self.add_special_tile('prison', (7, 3))
        self.add_special_tile('prison', (8, 7))

        # Goal
        self.add_special_tile('goal', (9, 9))

        # Snitches
        self.add_special_tile('snitch', (1, 1))
        self.add_special_tile('snitch', (3, 5))
        self.add_special_tile('snitch', (6, 8))

        # Snitch mapping for bitmask
        self.snitch_positions = list(self.special_tiles['snitch'])
        self.snitch_positions.sort()
        self.snitch_indices = {pos: idx for idx, pos in enumerate(self.snitch_positions)}
        self.num_snitches = len(self.snitch_positions)
        self.full_snitch_mask = (1 << self.num_snitches) - 1

    def _get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        direction = {
            0: (-1, 0),  # LEFT
            1: (1, 0),   # RIGHT
            2: (0, -1),  # UP
            3: (0, 1)    # DOWN
        }[action]

        next_state = (state[0] + direction[0], state[1] + direction[1])

        if (0 <= next_state[0] < self.size and
            0 <= next_state[1] < self.size and
            next_state not in self.special_tiles['obstacles']):
            return next_state
        return state

    def value_iteration(self) -> Tuple[np.ndarray, np.ndarray]:
        V = np.zeros((self.size, self.size, 1 << self.num_snitches))
        policy = np.zeros((self.size, self.size, 1 << self.num_snitches), dtype=int)

        while True:
            delta = 0
            for x in range(self.size):
                for y in range(self.size):
                    for mask in range(1 << self.num_snitches):
                        if (x, y) in self.special_tiles['obstacles']:
                            continue

                        best_value = -np.inf
                        best_action = 0

                        for action in range(4):
                            next_x, next_y = self._get_next_state((x, y), action)
                            next_mask = mask
                            reward = -0.1

                            if (next_x, next_y) in self.special_tiles['prison']:
                                reward = -5.0

                            if (next_x, next_y) in self.special_tiles['snitch']:
                                idx = self.snitch_indices[(next_x, next_y)]
                                if not (mask & (1 << idx)):
                                    next_mask = mask | (1 << idx)
                                    reward = 1.0

                            if (next_x, next_y) in self.special_tiles['goal']:
                                reward = 10.0 if next_mask == self.full_snitch_mask else -1.0

                            v = reward + self.gamma * V[next_x, next_y, next_mask]

                            if v > best_value:
                                best_value = v
                                best_action = action

                        delta = max(delta, abs(V[x, y, mask] - best_value))
                        V[x, y, mask] = best_value
                        policy[x, y, mask] = best_action

            if delta < self.theta:
                break

        self.V = V
        self.policy = policy
        return V, policy

    def plot_value_function(self):
        plt.figure(figsize=(8, 8))
        # נציג את value בפורמט mask אפס (בהתחלה)
        plt.imshow(self.V[:, :, 0].T, origin='upper')
        plt.colorbar(label='Value')
        plt.title('Value Function (initial state)')
        plt.show()

    def plot_policy(self):
        plt.figure(figsize=(8, 8))

        X, Y = np.meshgrid(np.arange(self.size), np.arange(self.size))
        U = np.zeros((self.size, self.size))
        V = np.zeros((self.size, self.size))

        for x in range(self.size):
            for y in range(self.size):
                if (x, y) in self.special_tiles['obstacles']:
                    continue

                action = self.policy[x, y, 0]  # נציג policy עבור mask=0 (התחלה)

                if action == 0:    # LEFT
                    U[x, y] = -1
                elif action == 1:  # RIGHT
                    U[x, y] = 1
                elif action == 2:  # UP
                    V[x, y] = 1
                else:              # DOWN
                    V[x, y] = -1

        plt.quiver(X, Y, U.T, -V.T)
        plt.gca().invert_yaxis() 
        plt.title('Policy (initial mask state)')
        plt.show()

