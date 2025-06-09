
import pygame
import numpy as np
from typing import Tuple, Optional, Dict

class GridWorldRenderer:
    """Renderer for the GridWorld environment using Pygame."""

    def __init__(self, size: int = 10, window_size: int = 512, background_path: Optional[str] = None):
        self.size = size
        self.window_size = (window_size // size) * size
        self.cell_size = self.window_size // self.size

        self.colors = {
            'background': (255, 255, 255),
            'grid': (200, 200, 200),
            'goal': (0, 255, 0),
            'obstacle': (128, 128, 128),
            'slippery': (0, 255, 255),
            'prison': (255, 0, 0),
            'text': (0, 0, 0)
        }

        pygame.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("RL Escape Room")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 32)

        self.agent_image = pygame.image.load("assets/images/einstein.png").convert_alpha()
        self.agent_image = pygame.transform.scale(self.agent_image, (self.cell_size, self.cell_size))

        self.background_path = background_path
        if self.background_path:
            self.background_image = pygame.image.load(self.background_path).convert()
            self.background_image = pygame.transform.smoothscale(self.background_image, (self.window_size, self.window_size))
        else:
            self.background_image = None

    def render_with_shapes(self, agent_position: Tuple[int, int], special_tiles: Dict[str, set],
               info: Optional[Dict] = None, shapes_positions=None, collected_shapes=None,
               current_task: Optional[str] = None) -> np.ndarray:

        if self.background_image:
            self.window.blit(self.background_image, (0, 0))
        else:
            self.window.fill(self.colors['background'])

        for i in range(self.size + 1):
            pygame.draw.line(self.window, self.colors['grid'], (0, i * self.cell_size), (self.window_size, i * self.cell_size))
            pygame.draw.line(self.window, self.colors['grid'], (i * self.cell_size, 0), (i * self.cell_size, self.window_size))

        for tile_type, positions in special_tiles.items():
            for pos in positions:
                color = self.colors.get(tile_type, self.colors['grid'])
                pygame.draw.rect(self.window, color,
                    (pos[1] * self.cell_size, pos[0] * self.cell_size, self.cell_size, self.cell_size))

        if shapes_positions:
            for shape, pos in shapes_positions.items():
                if collected_shapes and shape in collected_shapes:
                    continue  # אל תצייר צורות שכבר נאספו

                center = (pos[1]*self.cell_size + self.cell_size//2, pos[0]*self.cell_size + self.cell_size//2)
                color = (255, 255, 255)  # לבן
                if shape == "circle":
                    pygame.draw.circle(self.window, color, center, self.cell_size//3, width=4)
                elif shape == "square":
                    square_rect = pygame.Rect(0, 0, self.cell_size//2, self.cell_size//2)
                    square_rect.center = center
                    pygame.draw.rect(self.window, color, square_rect, width=4)
                elif shape == "triangle":
                    triangle = [
                        (center[0], center[1] - self.cell_size//3),
                        (center[0] - self.cell_size//3, center[1] + self.cell_size//3),
                        (center[0] + self.cell_size//3, center[1] + self.cell_size//3)
                    ]
                    pygame.draw.polygon(self.window, color, triangle, width=4)



        self.window.blit(self.agent_image, (agent_position[1] * self.cell_size, agent_position[0] * self.cell_size))

        pygame.display.flip()
        self.clock.tick(4)

        return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))

    def close(self):
        pygame.quit()
