import pygame
import numpy as np
from typing import Tuple, Optional, Dict

class GridWorldRenderer:
    """Renderer for the GridWorld environment using Pygame."""

    def __init__(self, size: int = 10, window_size: int = 512, background_path: Optional[str] = None):
        self.size = size
        self.window_size = window_size
        self.cell_size = window_size // size

        # Colors
        self.colors = {
            'background': (255, 255, 255),
            'grid': (200, 200, 200),
            'agent': (0, 0, 255),
            'goal': (0, 255, 0),
            'obstacle': (128, 128, 128),
            'slippery': (0, 255, 255),
            'prison': (255, 0, 0),
            'text': (0, 0, 0)
        }

        pygame.init()
        self.window = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("RL Escape Room")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

        # Load images from assets/images/ directory
        self.snitch_image = pygame.image.load("assets/images/snitch_icon.png").convert_alpha()
        self.snitch_image = pygame.transform.scale(self.snitch_image, (self.cell_size, self.cell_size))

        self.battery_image = pygame.image.load("assets/images/battery.jpg").convert_alpha()
        self.battery_image = pygame.transform.scale(self.battery_image, (self.cell_size, self.cell_size))

        self.agent_image = pygame.image.load("assets/images/eve.png").convert_alpha()
        self.agent_image = pygame.transform.scale(self.agent_image, (self.cell_size, self.cell_size))

        self.yellow_car_image = pygame.image.load("assets/images/yellow_car.png").convert_alpha()
        self.yellow_car_image = pygame.transform.scale(self.yellow_car_image, (self.cell_size, self.cell_size))

        self.red_car_image = pygame.image.load("assets/images/red_car.png").convert_alpha()
        self.red_car_image = pygame.transform.scale(self.red_car_image, (self.cell_size, self.cell_size))

        self.background_image = pygame.image.load("assets/images/room4_background.jpg").convert()
        self.background_image = pygame.transform.scale(self.background_image, (self.window_size, self.window_size))

        self.harry_image = pygame.image.load("assets/images/harry_potter.png").convert_alpha()
        self.harry_image = pygame.transform.scale(self.harry_image, (self.cell_size, self.cell_size))



        self.background_path = background_path

        if self.background_path:
            self.background_image = pygame.image.load(self.background_path).convert()
            self.background_image = pygame.transform.smoothscale(self.background_image, (self.window_size, self.window_size))
        else:
            self.background_image = None

    def render(self, agent_position: Tuple[int, int], special_tiles: Dict[str, set],
            info: Optional[Dict] = None, moving_cars: Optional[list] = None, charging_cells: Optional[set] = None) -> np.ndarray:
        if self.background_image:
            self.window.blit(self.background_image, (0, 0))
        else:
            self.window.fill(self.colors['background'])

        for i in range(self.size + 1):
            pygame.draw.line(
                self.window,
                self.colors['grid'],
                (0, i * self.cell_size),
                (self.window_size, i * self.cell_size)
            )
            pygame.draw.line(
                self.window,
                self.colors['grid'],
                (i * self.cell_size, 0),
                (i * self.cell_size, self.window_size)
            )

        for tile_type, positions in special_tiles.items():
            for pos in positions:
                if tile_type == 'snitch':
                    self.window.blit(
                        self.snitch_image,
                        (pos[0] * self.cell_size, pos[1] * self.cell_size)
                    )
                else:
                    normalized_tile_type = tile_type.rstrip("s")  # מסיר s מסוף מילה
                    color = self.colors.get(normalized_tile_type, self.colors['grid'])

                    if normalized_tile_type in ['goal', 'obstacle', 'slippery', 'prison']:

                        surface = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                        surface.fill((*color, 150))  # 150/255 = שקיפות חלקית
                        self.window.blit(surface, (pos[0] * self.cell_size, pos[1] * self.cell_size))
                    else:
                        pygame.draw.rect(
                            self.window,
                            color,
                            (pos[0] * self.cell_size, pos[1] * self.cell_size, self.cell_size, self.cell_size)
                        )

        if charging_cells:
            for pos in charging_cells:
                self.window.blit(
                    self.battery_image,
                    (pos[0] * self.cell_size, pos[1] * self.cell_size)
                )
        
        if moving_cars:
            for car in moving_cars:
                pos = car["position"]
                if self.background_path and "room4_background.jpg" in self.background_path:
                    self.window.blit(
                        self.yellow_car_image,
                        (pos[0] * self.cell_size, pos[1] * self.cell_size)
                    )
                else:
                    pygame.draw.rect(
                        self.window,
                        (255, 165, 0),
                        (pos[0] * self.cell_size, pos[1] * self.cell_size, self.cell_size, self.cell_size)
                    )



        # Draw agent using EVE image
        if self.background_path and "room4_background.jpg" in self.background_path:
            self.window.blit(
                self.red_car_image,
                (agent_position[0] * self.cell_size, agent_position[1] * self.cell_size)
            )
        
        elif self.background_path and "room1_background.jpg" in self.background_path:
            self.window.blit(
                self.harry_image,
                (agent_position[0] * self.cell_size, agent_position[1] * self.cell_size)
            )
        else:
            self.window.blit(
                self.agent_image,
                (agent_position[0] * self.cell_size, agent_position[1] * self.cell_size)
            )



        if info:
            overlay_width = 140
            overlay_height = 90
            overlay_surface = pygame.Surface((overlay_width, overlay_height), pygame.SRCALPHA)
            overlay_surface.fill((255, 255, 255, 180))
            self.window.blit(overlay_surface, (10, 10))

            y_offset = 20
            for key, value in info.items():
                if key not in ['snitch_collected', 'snitch_total', 'Training']:
                    text = f"{key}: {value}"
                    text_surface = self.font.render(text, True, self.colors['text'])
                    self.window.blit(text_surface, (20, y_offset))
                    y_offset += 25

            if 'snitch_collected' in info and 'snitch_total' in info:
                snitch_text = f"Snitch: {info['snitch_collected']}/{info['snitch_total']}"
                text_surface = self.font.render(snitch_text, True, self.colors['text'])
                self.window.blit(text_surface, (20, y_offset))
                y_offset += 25

        pygame.display.flip()
        self.clock.tick(4)

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window)),
            axes=(1, 0, 2)
        )

    def close(self):
        pygame.quit()