import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any

class GridWorldEnv(gym.Env):
    """
    Custom GridWorld environment that follows gym interface.
    This is the base environment for all rooms in the escape room game.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size: int = 10, render_mode: Optional[str] = None):
        super().__init__()
        
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        
        # Observation space: (x, y) coordinates
        self.observation_space = spaces.Box(
            low=0, high=size-1, shape=(2,), dtype=np.int32
        )
        
        # Action space: 0: LEFT, 1: RIGHT, 2: UP, 3: DOWN
        self.action_space = spaces.Discrete(4)
        
        # Dictionary to store special tiles
        self.special_tiles = {
            'slippery': set(),
            'prison': set(),
            'goal': set(),
            'obstacles': set(),
            'snitch': set()
        }
        
        # Rendering setup
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Current state
        self.agent_position = None
        self.prison_countdown = 0
        
        # Episode info
        self.steps = 0
        self.max_steps = 300

        self.collected_snitch = 0
        self.total_snitch = 0  


        
    def _get_obs(self) -> np.ndarray:
        """Get the current observation."""
        return np.array(self.agent_position, dtype=np.int32)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment."""
        goal_list = list(self.special_tiles['goal'])
        if goal_list:
            distance = np.linalg.norm(
                np.array(self.agent_position) - np.array(goal_list[0])
            )
        else:
            distance = None  # או np.nan אם תרצה לשמור פורמט מספרי

        return {
            "distance_to_goal": distance,
            "steps": self.steps
        }


    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize agent position (randomly, avoiding special tiles)
        valid_positions = [(x, y) for x in range(self.size) for y in range(self.size)
                         if (x, y) not in self.special_tiles['obstacles'] and
                            (x, y) not in self.special_tiles['goal']]
        
        self.agent_position = self.np_random.choice(len(valid_positions))
        self.agent_position = valid_positions[self.agent_position]
        self.steps = 0
        self.prison_countdown = 0
        
        observation = self._get_obs()
        info = self._get_info()

        self.collected_snitch = 0
        self.total_snitch = len(self.special_tiles['snitch'])

        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one time step within the environment."""
        self.steps += 1
        
        # If in prison, stay there and lose reward
        if self.prison_countdown > 0:
            self.prison_countdown -= 1
            return self._get_obs(), -1.0, False, self._get_info()
        
        # Calculate new position
        direction = {
            0: (-1, 0),  # LEFT
            1: (1, 0),   # RIGHT
            2: (0, -1),  # UP
            3: (0, 1)    # DOWN
        }[action]
        
        new_position = (
            self.agent_position[0] + direction[0],
            self.agent_position[1] + direction[1]
        )
        
        # Check if the move is valid
        if (0 <= new_position[0] < self.size and
            0 <= new_position[1] < self.size and
            new_position not in self.special_tiles['obstacles']):
            
            # Handle slippery tiles
            if self.agent_position in self.special_tiles['slippery']:
                if self.np_random.random() < 0.3:  # 30% chance to slip
                    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # All possible directions
                    directions.remove(direction)  # Remove current direction
                    slip_direction = self.np_random.choice(directions)  # Choose random direction
                    new_slip_position = (
                        self.agent_position[0] + slip_direction[0],
                        self.agent_position[1] + slip_direction[1]
                    )
                    # Only apply slip if it's a valid move
                    if (0 <= new_slip_position[0] < self.size and
                        0 <= new_slip_position[1] < self.size and
                        new_slip_position not in self.special_tiles['obstacles']):
                        new_position = new_slip_position
            
            self.agent_position = new_position
        
        # Check for special tiles
        reward = -0.1  # Small negative reward for each step
        terminated = False
        
        if self.agent_position in self.special_tiles['prison']:
            self.prison_countdown = 5  # Stuck for 5 steps
            reward = -5.0
        
        # בדיקה אם השחקן דורך על snitch
        if self.agent_position in self.special_tiles['snitch']:
            self.collected_snitch += 1
            reward = 1.0  # אפשר לשחק עם גובה התגמול
            self.special_tiles['snitch'].remove(self.agent_position)

        # בדיקה אם השחקן דורך על goal
        if self.agent_position in self.special_tiles['goal']:
            if self.collected_snitch == self.total_snitch:
                reward = 20.0   # פרס מוגבר להצלחה
                terminated = True
            else:
                reward = -5.0   # עונש חזק יותר על נסיון כושל


        if self.steps >= self.max_steps:
            reward = -20.0  # עונש כבד על מריחה
            terminated = True

        info = self._get_info()
        info["timeout"] = (self.steps >= self.max_steps and not terminated)
        return self._get_obs(), reward, terminated, info
    
    def add_special_tile(self, tile_type: str, position: Tuple[int, int]) -> None:
        """Add a special tile to the environment."""
        if tile_type in self.special_tiles:
            self.special_tiles[tile_type].add(position)
    
    def remove_special_tile(self, tile_type: str, position: Tuple[int, int]) -> None:
        """Remove a special tile from the environment."""
        if tile_type in self.special_tiles and position in self.special_tiles[tile_type]:
            self.special_tiles[tile_type].remove(position) 