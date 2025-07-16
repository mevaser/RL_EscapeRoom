import numpy as np
from typing import Optional, Dict, Tuple, Any
import matplotlib.pyplot as plt
from environment.grid_world import GridWorldEnv

# Mapping from action index to (dx, dy) direction
ACTION_TO_VECTOR = {
    0: (0, -1),  # LEFT
    1: (0, 1),  # RIGHT
    2: (-1, 0),  # UP
    3: (1, 0),  # DOWN
}


class QLearningRoom(GridWorldEnv):
    """
    Room 3 — Q-Learning with an ordered-shape-collection task.
    The agent must collect Circle → Square → Triangle and then reach the goal.
    """

    def __init__(
        self,
        size: int = 10,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
    ):
        super().__init__(size=size)

        # Goal (appears only after all shapes are collected)
        self.goal_position = (9, 9)
        self.goal_active = False

        # Shape locations (row, col) — order matters
        self.shapes_positions = {
            "circle": (2, 3),
            "square": (5, 6),
            "triangle": (4, 2),
        }
        self.shape_list = sorted(self.shapes_positions.items())
        self.shape_indices = {name: i for i, (name, _) in enumerate(self.shape_list)}
        self.num_shapes = len(self.shape_list)
        self.full_shape_mask = (1 << self.num_shapes) - 1  # e.g. 0b111 for 3 shapes

        self._setup_room()

        # Q-Learning hyper-parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Initialize Q-table: (x, y, mask) → Q-values for 4 actions
        self.Q = {
            (x, y, mask): np.zeros(4)
            for x in range(size)
            for y in range(size)
            for mask in range(1 << self.num_shapes)
        }

        self.epsilon_history = []  # track epsilon per episode
        self.episode_rewards = []  # track reward per episode
        self.current_episode = 0

    # --------------------------------------------------------------------- #
    # Room layout
    # --------------------------------------------------------------------- #
    def _setup_room(self):
        """Build static walls, slippery cells, prisons, buttons, and obstacles."""

        # Store obstacles that will be removed upon yellow button press
        self.button_removable_obstacles = set()

        # Vertical wall at column 7 (except position (8, 7))
        for x in range(1, 9):
            if x != 8:  # leave (8, 7) open
                pos = (x, 7)
                self.add_special_tile("obstacles", pos)
                self.button_removable_obstacles.add(pos)

        # Horizontal wall at row 7 (except position (7, 1))
        for y in range(3, 9):
            if y != 1:  # leave (7, 1) open
                pos = (7, y)
                self.add_special_tile("obstacles", pos)
                self.button_removable_obstacles.add(pos)

        # Track if the button was already pressed
        self.obstacles_removed = False

        # Slippery tiles near the button
        for x in (7, 8):
            for y in (1, 2):
                self.add_special_tile("slippery", (x, y))

        # Prison cells
        self.add_special_tile("prison", (4, 6))
        self.add_special_tile("prison", (8, 4))

        # Yellow button that removes specific obstacles
        self.yellow_button = (6, 6)
        self.add_special_tile("yellow_button", self.yellow_button)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one action and return (obs, reward, terminated, truncated, info).
        Includes task-specific reward shaping and button / shape handling.
        """
        obs, reward, terminated, truncated, info = super().step(action)
        state = tuple(obs)

        # ----------------------------------------------------------------- #
        # 1. Yellow button: remove static and dynamic obstacles and give bonus
        # ----------------------------------------------------------------- #
        if state == self.yellow_button and self.obstacles_active:
            # Remove static obstacles that block the path
            for x in range(1, 9):
                if x != 8:
                    self.remove_special_tile("obstacles", (x, 7))
            for y in range(2, 9):
                if y != 1:
                    self.remove_special_tile("obstacles", (7, y))

            # Remove all removable obstacles
            for obstacle in self.button_removable_obstacles:
                self.remove_special_tile("obstacles", obstacle)

            print(
                f"[DEBUG] Removed {len(self.button_removable_obstacles)} obstacles via yellow button."
            )

            self.obstacles_active = False
            reward += 5.0  # Bonus for using the button

        # ----------------------------------------------------------------- #
        # 2. Shape collection logic
        # ----------------------------------------------------------------- #
        for shape_name, pos in self.shape_list:
            if state == pos:
                shape_idx = int(self.shape_indices[shape_name])
                if not (self.collected_mask & (1 << shape_idx)):
                    self.collected_mask |= 1 << shape_idx
                    reward += 10.0  # New shape collected
                else:
                    reward -= 0.05  # Penalty for revisiting

        # ----------------------------------------------------------------- #
        # 3. Unlock the goal once all shapes are collected
        # ----------------------------------------------------------------- #
        if self.collected_mask == self.full_shape_mask and not self.goal_active:
            self.add_special_tile("goal", self.goal_position)
            self.goal_active = True

        # ----------------------------------------------------------------- #
        # 4. Goal handling
        # ----------------------------------------------------------------- #
        if state == self.goal_position and self.goal_active:
            reward += 15.0
            terminated = True
            info["success"] = True
        elif state == self.goal_position:
            reward -= 1.0  # Attempted to finish too early
            info["success"] = False
        else:
            info["success"] = False

        return obs, reward, terminated, truncated, info

    # --------------------------------------------------------------------- #
    # Q-Learning helpers
    # --------------------------------------------------------------------- #
    def get_q_state(self, state: Tuple[int, int]) -> Tuple[int, int, int]:
        """Return the Q-table key (x, y, mask)."""
        return (state[0], state[1], self.collected_mask)

    def get_action(self, state: Tuple[int, int], training: bool = True) -> int:
        """
        Epsilon-greedy choice.
        During evaluation (training=False) epsilon is ignored.
        """
        q_state = self.get_q_state(state)
        if q_state not in self.Q:
            self.Q[q_state] = np.zeros(4)

        if training and np.random.random() < self.epsilon:
            return int(np.random.choice(4))
        return int(np.argmax(self.Q[q_state]))

    # --------------------------------------------------------------------- #
    # Training loop
    # --------------------------------------------------------------------- #
    def train_episode(self) -> float:
        """Run a single training episode and return the total reward."""
        obs, _ = self.reset()
        state = tuple(obs)
        total_reward = 0.0
        max_steps = 300
        steps = 0
        done = False
        info: Dict[str, Any] = {}

        while not done:
            q_state = self.get_q_state(state)
            if q_state not in self.Q:
                self.Q[q_state] = np.zeros(4)

            action = self.get_action(state, training=True)
            next_obs, reward, terminated, truncated, info = self.step(action)
            next_state = tuple(next_obs)
            next_q_state = self.get_q_state(next_state)
            if next_q_state not in self.Q:
                self.Q[next_q_state] = np.zeros(4)

            total_reward += reward

            # Standard Q-Learning update
            if not terminated:
                td_target = reward + self.gamma * np.max(self.Q[next_q_state])
            else:
                td_target = reward
            td_error = td_target - self.Q[q_state][action]
            self.Q[q_state][action] += self.alpha * td_error

            state = next_state
            steps += 1
            done = terminated or steps >= max_steps

        # Book-keeping
        self.episode_rewards.append(total_reward)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
        self.current_episode += 1
        return total_reward

    def train(self, num_episodes: int = 1500):
        """Train the agent and track reward/epsilon per episode for plotting."""
        self.episode_rewards = []
        self.epsilon_history = []

        for _ in range(num_episodes):
            total_reward = self.train_episode()
            self.episode_rewards.append(total_reward)
            self.epsilon_history.append(self.epsilon)

        # Display plots only once after training
        self.plot_training_progress()
        self.plot_epsilon_curve()
        self.plot_policy()
        self.plot_q_values()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to the initial state, including agent location,
        collected shapes, active obstacles, and goal availability.
        """
        # Reset base GridWorld state (agent position etc.)
        obs, info = super().reset(seed=seed, options=options)

        # Reset collected shapes bitmask
        self.collected_mask = 0

        # Remove goal if it was previously added
        self.goal_active = False
        if self.goal_position in self.special_tiles["goal"]:
            self.special_tiles["goal"].remove(self.goal_position)

        # Reset obstacles — ensure all removable ones are active again
        for pos in self.button_removable_obstacles:
            self.add_special_tile("obstacles", pos)

        self.obstacles_active = True

        return obs, info

    # ------------------------------------------------------------------ #
    # Plotting utilities
    # ------------------------------------------------------------------ #
    def plot_training_progress(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards)
        plt.title("Training Progress")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.show()

    def plot_epsilon_curve(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.epsilon_history)
        plt.title("Epsilon Decay")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.grid(True)
        plt.show()

    def plot_policy(self, mask: int = 0):
        """Plot the greedy policy for a given shape-collection bitmask."""
        plt.figure(figsize=(8, 8))
        X, Y = np.meshgrid(np.arange(self.size), np.arange(self.size))
        U = np.zeros_like(X, dtype=float)
        V = np.zeros_like(Y, dtype=float)

        for x in range(self.size):
            for y in range(self.size):
                state = (x, y, mask)
                if state not in self.Q:
                    continue
                action = np.argmax(self.Q[state])
                if action == 0:  # up
                    U[y, x], V[y, x] = 0, 1
                elif action == 1:  # down
                    U[y, x], V[y, x] = 0, -1
                elif action == 2:  # left
                    U[y, x], V[y, x] = -1, 0
                elif action == 3:  # right
                    U[y, x], V[y, x] = 1, 0

        plt.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=1, color="black")
        plt.title(f"Greedy Policy (Mask = {mask})")
        plt.xticks(np.arange(self.size))
        plt.yticks(np.arange(self.size))
        plt.xlim(-0.5, self.size - 0.5)
        plt.ylim(-0.5, self.size - 0.5)
        plt.grid(True)
        plt.gca().set_aspect("equal")
        plt.show()

    def plot_q_values(self):
        """Heat-map of max-Q for each state (mask = 0)."""
        value_matrix = np.zeros((self.size, self.size))
        for x in range(self.size):
            for y in range(self.size):
                state = (x, y, 0)
                value_matrix[x, y] = (
                    np.max(self.Q[state]) if state in self.Q else np.nan
                )

        plt.figure(figsize=(8, 8))
        plt.imshow(value_matrix.T, origin="lower", cmap="viridis")
        plt.colorbar(label="Max Q-Value")
        plt.title("Q-Value Surface (Mask = 0)")
        plt.show()
