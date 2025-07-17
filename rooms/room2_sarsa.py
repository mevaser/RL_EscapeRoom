import numpy as np
from typing import Dict, Tuple, List, Any
import matplotlib.pyplot as plt
from environment.grid_world import GridWorldEnv
import json
import os


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
        """Setup the room with obstacles, prison, and teleport portals. Goal is added later."""
        for x in [2, 3, 4, 5]:
            self.add_special_tile("obstacles", (x, 4))
        for y in [1, 2, 3]:
            self.add_special_tile("obstacles", (5, y))
        for x in [7, 8]:
            for y in [5, 6]:
                self.add_special_tile("slippery", (x, y))

        self.add_special_tile("prison", (3, 7))
        self.add_special_tile("prison", (7, 2))

        # 🚫 Goal will be added dynamically after collecting all batteries
        # self.add_special_tile("goal", self.goal_position)

        # Add teleport portals
        self.portal1 = (1, 2)
        self.portal2 = (8, 7)
        self.add_special_tile("portal", self.portal1)
        self.add_special_tile("portal", self.portal2)

    def _place_charging_cells(self):
        """Place charging cells in the room."""
        self.charging_cells = set(self.fixed_charging_cells)
        while len(self.charging_cells) < 5:
            cell = (
                np.random.randint(0, self.size),
                np.random.randint(0, self.size),
            )
            if (
                cell not in self.special_tiles["obstacles"]
                and cell not in self.special_tiles["goal"]
                and cell not in self.special_tiles["prison"]
                and cell not in self.charging_cells
            ):
                self.charging_cells.add(cell)

    def get_action(self, state, training=True):
        """Choose an action based on epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            # During training: explore with probability epsilon
            return np.random.choice(4)
        else:
            # During training (1 - epsilon) or during inference: exploit learned Q-values
            return int(np.argmax(self.Q[state]))

    def train_episode(self) -> float:
        """
        Run a single SARSA episode.
        * epsilon-greedy policy
        * SARSA update
        * epsilon decay
        * hard cap on steps to avoid infinite loops
        """
        obs, _ = self.reset()
        state = tuple(obs)
        total_reward = 0.0

        action = self.get_action(state, training=True)
        done = False
        step_count = 0
        max_steps = 500  # safety cap ─ prevents endless episodes

        while not done and step_count < max_steps:
            # environment transition
            next_obs, reward, terminated, truncated, info = self.step(action)
            next_state = tuple(next_obs)
            done = terminated  # episode ends only when terminated == True
            total_reward += reward

            # choose next action (ε-greedy)
            next_action = self.get_action(next_state, training=True)

            # SARSA update
            target = reward
            if not done:  # bootstrap if episode not finished
                target += self.gamma * self.Q[next_state][next_action]
            td_error = target - self.Q[state][action]
            self.Q[state][action] += self.alpha * td_error

            # move to next transition
            state = next_state
            action = next_action
            step_count += 1

        # keep track of episode statistics
        self.episode_rewards.append(total_reward)
        self.current_episode += 1

        # epsilon decay (minimum 0.01)
        self.epsilon = max(0.01, self.epsilon * 0.995)

        # keep Q-values at goal zero to avoid oscillation
        self.Q[self.goal_position] = np.zeros(4)

        return total_reward

    def train(self, num_episodes: int = 2000):
        self.episode_rewards = []
        self.current_episode = 0

        for i in range(num_episodes):
            reward = self.train_episode()
            print(
                f"Episode {i + 1:>4}/{num_episodes} | Reward: {reward:6.2f} | Epsilon: {self.epsilon:.3f}"
            )

        models_dir = os.path.join(os.path.dirname(__file__), "..", "saved_models")
        os.makedirs(models_dir, exist_ok=True)

        save_path = os.path.join(models_dir, "room2_rewards.npy")
        np.save(save_path, np.array(self.episode_rewards))

        print("[INFO] Saved episode rewards to saved_models/room2_rewards.npy")
        print(f"[DEBUG][SAVE] Saved {len(self.episode_rewards)} rewards to {save_path}")
        print(f"[DEBUG][SAVE] Last 5 rewards: {self.episode_rewards[-5:]}")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # -------------------------------------------------------------------------
        STEP_PENALTY = -1  # קל יותר
        BATTERY_REWARD = 10.0
        GOAL_BONUS = 60.0
        EARLY_EXIT_PEN = -100.0
        TIMEOUT_PENALTY = -100.0
        MAX_STEPS = 300
        # -------------------------------------------------------------------------
        obs, _, terminated, truncated, info = super().step(action)
        position = tuple(obs)
        reward = STEP_PENALTY
        info["success"] = False

        # ---------- 1. Teleporters ------------------------------------------------
        if position == self.portal1:
            self.agent_position = self.portal2
            position = self.portal2
            obs = np.array(position)
        elif position == self.portal2:
            self.agent_position = self.portal1
            position = self.portal1
            obs = np.array(position)

        # ---------- 2. Battery collection ----------------------------------------
        if position in self.charging_cells:
            reward += BATTERY_REWARD
            self.charging_cells.remove(position)
            self.collected_batteries.add(position)

            # goal מופיע מיד אחרי הבטרייה הראשונה
            if "goal" not in self.special_tiles:
                self.add_special_tile("goal", self.goal_position)
                print("[DEBUG] Goal activated")

        goal_is_active = "goal" in self.special_tiles

        # ---------- 3. הגיע לשער לפני בטרייה → כשלון מיידי -----------------------
        if position == self.goal_position and not goal_is_active:
            reward += EARLY_EXIT_PEN
            terminated = True  # ←  מסיים את האפיזודה
            info["success"] = False
            # אין reason להשאיר את השער על הלוח
            self.special_tiles["goal"].clear()
            print("[WARN] Early exit – no battery collected")
            return obs, reward, terminated, truncated, info

        # ---------- 4. הצלחה ------------------------------------------------------
        if position == self.goal_position and goal_is_active:
            reward += GOAL_BONUS
            terminated = True
            info["success"] = True

        # ---------- 5. קאט-אוף ----------------------------------------------------
        if self.steps >= MAX_STEPS and not terminated:
            reward += TIMEOUT_PENALTY
            truncated = True
            info["timeout"] = True

        self.last_reward = reward  # להצגה ב-UI
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """Reset the environment to an initial state."""
        obs, info = super().reset(seed=seed, options=options)
        self.start_position = tuple(obs)
        self.special_tiles["goal"].clear()

        self._place_charging_cells()
        self.collected_batteries.clear()
        return obs, info

    def plot_training_progress(self):
        """Plot the training progress over episodes."""
        print(f"[DEBUG] Plotting {len(self.episode_rewards)} rewards")
        plt.close("all")
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

    def load_rewards_from_file(self, path: str | None = None):
        if path is None:
            path = os.path.join(
                os.path.dirname(__file__),  # ← מיקום הקובץ room2_sarsa.py
                "..",  # ← חזרה לשורש הפרויקט
                "saved_models",
                "room2_rewards.npy",
            )

        if os.path.exists(path):
            arr = np.load(path)
            print(f"[DEBUG] Loaded {len(arr)} rewards from {path}")
            print(f"[DEBUG] Last 5 rewards: {arr[-5:]}")
            self.episode_rewards = arr.tolist()
        else:
            print(f"[WARN] Reward file not found: {path}")
            self.episode_rewards = []
