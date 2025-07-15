import numpy as np
from typing import Optional, Dict, Tuple, Any
import matplotlib.pyplot as plt
from environment.grid_world import GridWorldEnv


class QLearningRoom(GridWorldEnv):
    """
    Room 3: Q-Learning with shapes task (collect shapes in order)
    """

    def __init__(
        self,
        size: int = 10,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
    ):
        """Initialize the Q-Learning room with shapes task."""
        super().__init__(size=size)

        self.goal_position = (9, 9)
        self.goal_active = False

        # Define shapes positions (row, col)
        self.shapes_positions = {"circle": (2, 3), "square": (5, 6), "triangle": (4, 2)}

        self.shape_order = ["circle", "square", "triangle"]
        self.current_stage = 0

        self.collected_shapes = set()

        self._setup_room()

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.Q = {
            (x, y, stage): np.zeros(4)
            for x in range(size)
            for y in range(size)
            for stage in range(4)
        }
        self.episode_rewards = []
        self.current_episode = 0

    def _setup_room(self):
        """Setup the room with obstacles, goal, and shapes."""
        # Add static obstacles to force path through green button
        for x in [1, 2, 3, 4, 5, 6, 7, 8]:
            if x != 8:
                self.add_special_tile(
                    "obstacles", (x, 7)
                )  # Block row 7 except at (8,7)
        for y in [2, 3, 4, 5, 6, 7, 8]:
            if y != 1:
                self.add_special_tile(
                    "obstacles", (7, y)
                )  # Block col 7 except at (7,1)
        for x in [7, 8]:
            for y in [1, 2]:
                self.add_special_tile("slippery", (x, y))
        self.add_special_tile("prison", (4, 7))
        self.add_special_tile("prison", (7, 4))

        # Add interactive buttons
        self.red_button = (1, 8)  # Red button for adding obstacles
        self.green_button = (8, 1)  # Green button for removing obstacles
        self.add_special_tile("red_button", self.red_button)
        self.add_special_tile("green_button", self.green_button)

        # Dynamic obstacles that can be added/removed
        self.dynamic_obstacles = {(3, 3), (4, 4), (5, 5), (6, 6)}
        self.obstacles_active = False

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to an initial state."""
        obs, info = super().reset(seed=seed, options=options)
        self.current_stage = 0
        self.collected_shapes = set()
        self.goal_active = False

        if self.goal_position in self.special_tiles["goal"]:
            self.special_tiles["goal"].remove(self.goal_position)

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        obs, reward, terminated, truncated, info = super().step(action)
        state = tuple(obs)

        # Handle interactive buttons
        if state == self.red_button and not self.obstacles_active:
            # Activate dynamic obstacles
            for obstacle in self.dynamic_obstacles:
                self.add_special_tile("obstacles", obstacle)
            self.obstacles_active = True
            reward += 1.0  # Small reward for using the button
        elif state == self.green_button and self.obstacles_active:
            # Remove ALL dynamic obstacles
            for obstacle in list(self.dynamic_obstacles):
                self.remove_special_tile("obstacles", obstacle)
            self.obstacles_active = False
            reward += 1.0  # Small reward for using the button

        # Check if standing on shape
        shape_here = None
        for shape, pos in self.shapes_positions.items():
            if state == pos:
                shape_here = shape
                break

        if shape_here:
            if self.current_stage < len(self.shape_order):
                expected_shape = self.shape_order[self.current_stage]
                if (
                    shape_here == expected_shape
                    and shape_here not in self.collected_shapes
                ):
                    reward += 3.0
                    self.collected_shapes.add(shape_here)
                    self.current_stage += 1
                else:
                    reward -= 10.0
            else:
                reward -= 10.0

        if self.current_stage == 3 and not self.goal_active:
            self.add_special_tile("goal", self.goal_position)
            self.goal_active = True

        if state == self.goal_position and self.goal_active:
            reward += 10.0
            terminated = True

        return obs, reward, terminated, truncated, info

    def get_q_state(self, state):
        """Convert the state to a Q-learning state representation."""
        return (state[0], state[1], self.current_stage)

    def get_action(self, state: Tuple[int, int], training: bool = True) -> int:
        """Select an action based on the current state using epsilon-greedy policy."""
        q_state = self.get_q_state(state)
        if training and np.random.random() < self.epsilon:
            return int(np.random.choice(4))
        return int(np.argmax(self.Q[q_state]))

    def train_episode(self) -> float:
        """Run a single training episode."""
        obs, _ = self.reset()
        state = tuple(obs)
        total_reward = 0
        done = False

        while not done:
            q_state = self.get_q_state(state)
            action = self.get_action(state, training=True)
            next_obs, reward, terminated, truncated, info = self.step(action)
            next_state = tuple(next_obs)
            done = terminated
            total_reward += reward

            next_q_state = self.get_q_state(next_state)

            if not done:
                self.Q[q_state][action] += self.alpha * (
                    reward
                    + self.gamma * np.max(self.Q[next_q_state])
                    - self.Q[q_state][action]
                )
            else:
                self.Q[q_state][action] += self.alpha * (
                    reward - self.Q[q_state][action]
                )

            state = next_state

        self.episode_rewards.append(total_reward)
        self.current_episode += 1
        # Update epsilon after each episode
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        return total_reward

    def train(self, num_episodes: int = 1500):
        """Train the agent for a specified number of episodes."""
        for _ in range(num_episodes):
            self.train_episode()

    def plot_training_progress(self):
        """Plot the training progress over episodes."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards)
        plt.title("Training Progress")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.show()

    def plot_epsilon_curve(self):
        """Plot the epsilon decay curve."""
        plt.figure(figsize=(10, 5))
        epsilons = [
            max(self.min_epsilon, self.epsilon * (self.epsilon_decay**i))
            for i in range(len(self.episode_rewards))
        ]
        plt.plot(epsilons)
        plt.title("Epsilon Decay Over Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.grid(True)
        plt.show()
