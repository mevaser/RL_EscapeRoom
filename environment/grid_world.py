import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any
from gymnasium.utils import seeding


class GridWorldEnv(gym.Env):
    """
    Custom GridWorld environment that follows gym interface.
    This is the base environment for all rooms in the escape room game.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size: int = 10, render_mode: Optional[str] = None):
        super().__init__()

        self.size = size  # Size of the square grid (e.g. 10x10)

        # Observation space: agent's position as (x, y) coordinates
        self.observation_space = spaces.Box(
            low=0, high=size - 1, shape=(2,), dtype=np.int32
        )

        # Action space: 0 = LEFT, 1 = RIGHT, 2 = UP, 3 = DOWN
        self.action_space = spaces.Discrete(4)

        # Special tiles configuration
        self.special_tiles = {
            "slippery": set(),
            "prison": set(),
            "goal": set(),
            "obstacles": set(),
            "snitch": set(),
        }

        # Rendering configuration (used by external renderer)
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Agent's current state
        self.agent_position = None
        self.prison_countdown = 0

        # Episode tracking
        self.steps = 0
        self.max_steps = 300
        self.collected_snitch = 0
        self.total_snitch = 0

    def _get_obs(self) -> np.ndarray:
        """Get the current observation."""
        return np.array(self.agent_position, dtype=np.int32)

    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment."""
        goal_list = list(self.special_tiles["goal"])
        if goal_list:
            distance = np.linalg.norm(
                np.array(self.agent_position) - np.array(goal_list[0])
            )
        else:
            distance = None  # או np.nan אם תרצה לשמור פורמט מספרי

        return {"distance_to_goal": distance, "steps": self.steps}

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        # Ensure consistent seeding
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)

        # Initialize agent position (randomly, avoiding special tiles)
        valid_positions = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
            if (x, y) not in self.special_tiles["obstacles"]
            and (x, y) not in self.special_tiles["goal"]
        ]

        chosen_index = self.np_random.integers(0, len(valid_positions))
        self.agent_position = valid_positions[chosen_index]
        self.steps = 0
        self.prison_countdown = 0

        self.collected_snitch = 0
        self.total_snitch = len(self.special_tiles["snitch"])

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        if self.agent_position is None:
            raise RuntimeError("Environment must be reset before calling step()")

        self.steps += 1
        terminated = False
        truncated = False

        # If in prison, stay there and lose reward
        if self.prison_countdown > 0:
            self.prison_countdown -= 1
            return self._get_obs(), -1.0, terminated, truncated, self._get_info()

        # Determine direction
        direction = {
            0: (-1, 0),  # LEFT
            1: (1, 0),  # RIGHT
            2: (0, -1),  # UP
            3: (0, 1),  # DOWN
        }[action]

        new_position = (
            self.agent_position[0] + direction[0],
            self.agent_position[1] + direction[1],
        )

        # Check if move is valid
        if (
            0 <= new_position[0] < self.size
            and 0 <= new_position[1] < self.size
            and new_position not in self.special_tiles["obstacles"]
        ):
            # Handle slippery tiles
            if self.agent_position in self.special_tiles["slippery"]:
                if self.np_random.random() < 0.3:
                    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    directions.remove(direction)
                    slip_direction = self.np_random.choice(directions)
                    new_slip_position = (
                        self.agent_position[0] + slip_direction[0],
                        self.agent_position[1] + slip_direction[1],
                    )
                    if (
                        0 <= new_slip_position[0] < self.size
                        and 0 <= new_slip_position[1] < self.size
                        and new_slip_position not in self.special_tiles["obstacles"]
                    ):
                        new_position = new_slip_position

            self.agent_position = new_position

        # Check special tiles
        reward = -0.1

        if self.agent_position in self.special_tiles["prison"]:
            self.prison_countdown = 5
            reward = -5.0

        if self.agent_position in self.special_tiles["snitch"]:
            self.collected_snitch += 1
            reward = 1.0
            self.special_tiles["snitch"].remove(self.agent_position)

        if self.agent_position in self.special_tiles["goal"]:
            if self.collected_snitch == self.total_snitch:
                reward = 20.0
                terminated = True
            else:
                reward = -5.0

        if self.steps >= self.max_steps:
            reward = -20.0
            truncated = True

        info = self._get_info()
        info["timeout"] = truncated

        return self._get_obs(), reward, terminated, truncated, info

    def add_special_tile(self, tile_type: str, position: Tuple[int, int]) -> None:
        """Add a special tile to the environment."""
        if tile_type in self.special_tiles:
            self.special_tiles[tile_type].add(position)

    def remove_special_tile(self, tile_type: str, position: Tuple[int, int]) -> None:
        """Remove a special tile from the environment."""
        if (
            tile_type in self.special_tiles
            and position in self.special_tiles[tile_type]
        ):
            self.special_tiles[tile_type].remove(position)
