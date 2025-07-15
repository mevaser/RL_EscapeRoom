import numpy as np
from typing import Dict, Tuple, List, Any
import matplotlib.pyplot as plt
from environment.grid_world import GridWorldEnv


class SARSARoom(GridWorldEnv):
    """
    Room 2: SARSA (On-policy) implementation - WALL-E Charging Room
    """

    def __init__(
        self,
        size: int = 10,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        """Initialize the SARSA room with charging task."""
        super(SARSARoom, self).__init__(size=size)

        self.goal_position = (9, 9)
        self._setup_room()

        self.fixed_charging_cells = {(3, 3), (6, 6)}
        self.charging_cells = set()
        self.collected_batteries = set()
        self._place_charging_cells()

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q = {(x, y): np.zeros(4) for x in range(size) for y in range(size)}
        self.episode_rewards = []
        self.current_episode = 0

    def _setup_room(self):
        """Setup the room with obstacles, goal, and charging cells."""
        for x in [2, 3, 4, 5]:
            self.add_special_tile("obstacles", (x, 4))
        for y in [1, 2, 3]:
            self.add_special_tile("obstacles", (5, y))
        for x in [7, 8]:
            for y in [5, 6]:
                self.add_special_tile("slippery", (x, y))
        self.add_special_tile("prison", (3, 7))
        self.add_special_tile("prison", (7, 2))
        self.add_special_tile("goal", self.goal_position)

        # Add teleport portals
        self.portal1 = (1, 2)  # row 1, column 2
        self.portal2 = (8, 7)  # row 8, column 7
        self.add_special_tile("portal", self.portal1)
        self.add_special_tile("portal", self.portal2)

    def _place_charging_cells(self):
        """Place charging cells in the room."""
        self.charging_cells = set(self.fixed_charging_cells)
        while len(self.charging_cells) < 5:
            cell = (np.random.randint(0, self.size), np.random.randint(0, self.size))
            if (
                cell not in self.special_tiles["obstacles"]
                and cell not in self.special_tiles["goal"]
                and cell not in self.special_tiles["prison"]
                and cell not in self.charging_cells
            ):
                self.charging_cells.add(cell)

    def get_action(self, state: Tuple[int, int], training: bool = True) -> int:
        """Select an action based on the current state using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            return int(np.random.choice(4))
        return int(np.random.choice(4))

    def train_episode(self) -> float:
        """Run a single training episode."""
        obs, _ = self.reset()
        state = tuple(obs)
        total_reward = 0

        action = self.get_action(state, training=True)
        done = False

        while not done:
            next_obs, reward, terminated, truncated, info = self.step(action)
            next_state = tuple(next_obs)
            done = terminated
            total_reward += reward

            next_action = self.get_action(next_state, training=True)

            if not done:
                self.Q[state][action] += self.alpha * (
                    reward
                    + self.gamma * self.Q[next_state][next_action]
                    - self.Q[state][action]
                )
            else:
                self.Q[state][action] += self.alpha * (reward - self.Q[state][action])

            state = next_state
            action = next_action

        self.episode_rewards.append(total_reward)
        self.current_episode += 1

        # prevent policy oscillation on goal
        self.Q[self.goal_position] = np.zeros(4)

        return total_reward

    def train(self, num_episodes: int = 3000):
        """Train the agent for a specified number of episodes."""
        for _ in range(num_episodes):
            self.train_episode()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        obs, reward, terminated, truncated, info = super().step(action)
        position = tuple(obs)

        # Handle teleportation
        if position == self.portal1:
            # Teleport to portal2
            self.agent_position = self.portal2
            obs = np.array(self.portal2)
            position = tuple(obs)
        elif position == self.portal2:
            # Teleport to portal1
            self.agent_position = self.portal1
            obs = np.array(self.portal1)
            position = tuple(obs)

        if position in self.charging_cells:
            reward += 5
            self.charging_cells.remove(position)
            self.collected_batteries.add(position)

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """Reset the environment to an initial state."""
        obs, info = super().reset(seed=seed, options=options)
        self._place_charging_cells()
        self.collected_batteries.clear()
        return obs, info

    def plot_training_progress(self):
        """Plot the training progress over episodes."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards)
        plt.title("Training Progress")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.show()

    def plot_q_values(self):
        """Plot the Q-value function as a heatmap."""
        plt.figure(figsize=(10, 10))
        value_matrix = np.zeros((self.size, self.size))
        for x in range(self.size):
            for y in range(self.size):
                if (x, y) in self.special_tiles["obstacles"]:
                    value_matrix[x, y] = float("nan")
                else:
                    value_matrix[x, y] = np.max(self.Q[(x, y)])
        plt.imshow(value_matrix.T, origin="upper")
        plt.colorbar(label="Max Q-Value")
        plt.title("Q-Value Function")
        plt.show()

    def plot_policy(self):
        """Plot the learned policy as a vector field."""
        plt.figure(figsize=(10, 10))
        X, Y = np.meshgrid(np.arange(self.size), np.arange(self.size))
        U, V = np.zeros((self.size, self.size)), np.zeros((self.size, self.size))

        for x in range(self.size):
            for y in range(self.size):
                if (x, y) in self.special_tiles["obstacles"] or (
                    x,
                    y,
                ) in self.special_tiles["goal"]:
                    continue
                action = np.argmax(self.Q[(x, y)])
                if action == 0:
                    U[x, y] = -1
                elif action == 1:
                    U[x, y] = 1
                elif action == 2:
                    V[x, y] = 1
                else:
                    V[x, y] = -1

        plt.quiver(X, Y, U.T, -V.T)
        plt.gca().invert_yaxis()
        plt.title("Learned Policy")
        plt.show()
