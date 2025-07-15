import pygame
import numpy as np
from typing import Tuple, Optional, Dict


class GridWorldRenderer:
    """Renderer for the GridWorld environment using Pygame."""

    def __init__(
        self,
        size: int = 10,
        window: Optional[pygame.Surface] = None,
        window_size: int = 800,
        background_path: Optional[str] = None,
    ):
        self.size = size
        self.window_size = window_size
        self.cell_size = window_size // size
        self.window = window  # Use shared window from main app

        # Color scheme
        self.colors = {
            "background": (255, 255, 255),
            "grid": (200, 200, 200),
            "agent": (0, 0, 255),
            "goal": (0, 255, 0),
            "obstacle": (128, 128, 128),
            "slippery": (0, 255, 255),
            "prison": (255, 0, 0),
            "portal": (153, 0, 255),
            "red_button": (255, 68, 68),
            "green_button": (68, 255, 68),
            "text": (0, 0, 0),
        }

        pygame.init()
        pygame.display.set_caption("RL Escape Room")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

        # Load images
        self.snitch_image = pygame.image.load(
            "assets/images/snitch_icon.png"
        ).convert_alpha()
        self.battery_image = pygame.image.load(
            "assets/images/battery.jpg"
        ).convert_alpha()
        self.agent_image = pygame.image.load("assets/images/eve.png").convert_alpha()
        self.yellow_car_image = pygame.image.load(
            "assets/images/yellow_car.png"
        ).convert_alpha()
        self.red_car_image = pygame.image.load(
            "assets/images/red_car.png"
        ).convert_alpha()
        self.harry_image = pygame.image.load(
            "assets/images/harry_potter.png"
        ).convert_alpha()

        # Load background
        self.background_path = background_path
        if self.background_path:
            self.background_image = pygame.image.load(self.background_path).convert()

            grid_height = (
                self.cell_size * self.size
            )  # exact pixel height of the grid (not full window)

            # Scale only to grid area, not the full window size
            self.background_image = pygame.transform.smoothscale(
                self.background_image, (self.window_size, grid_height)
            )
        else:
            self.background_image = None

    def render(
        self,
        agent_position: Tuple[int, int],
        special_tiles: Dict[str, set],
        info: Optional[Dict] = None,
        moving_cars: Optional[list] = None,
        charging_cells: Optional[set] = None,
    ) -> np.ndarray:
        if self.window is None:
            raise ValueError(
                "Renderer window is not initialized. Make sure to set it before calling render."
            )

        # Fixed panel height at the top
        info_panel_height = int(self.window_size * 0.2)  # 15% מהמסך

        # Ensure square grid cells based on window width and number of cells
        adjusted_cell_size = self.window_size // self.size

        # Grid starts just below the info panel
        grid_start_y = info_panel_height

        # Actual height the grid will occupy (square grid!)
        grid_height = adjusted_cell_size * self.size

        # Adjust total window height accordingly
        total_height = info_panel_height + grid_height

        # Optional: If you're dynamically creating the window (once), set its height here
        # self.window = pygame.display.set_mode((self.window_size, total_height))

        # Draw info panel
        if info:
            info_panel_rect = pygame.Rect(0, 0, self.window_size, info_panel_height)
            overlay_surface = pygame.Surface(
                (self.window_size, info_panel_height), pygame.SRCALPHA
            )
            overlay_surface.fill((255, 255, 255, 200))
            self.window.blit(overlay_surface, (0, 0))

            pygame.draw.rect(self.window, (100, 100, 100), info_panel_rect, 2)

            y_offset = 15
            x_offset = 20

            for key, value in info.items():
                if key not in ["snitch_collected", "snitch_total", "Training"]:
                    text = f"{key}: {value}"
                    text_surface = self.font.render(text, True, self.colors["text"])
                    self.window.blit(text_surface, (x_offset, y_offset))
                    y_offset += 25
                    if y_offset > info_panel_height - 25:
                        y_offset = 15
                        x_offset = self.window_size // 2 + 20

            if "snitch_collected" in info and "snitch_total" in info:
                snitch_text = (
                    f"Snitch: {info['snitch_collected']}/{info['snitch_total']}"
                )
                text_surface = self.font.render(snitch_text, True, self.colors["text"])
                self.window.blit(text_surface, (x_offset, y_offset))

        # Compute grid height based on adjusted_cell_size
        grid_height = adjusted_cell_size * self.size

        # Draw background only in the grid area (excluding info panel)
        if self.background_image:
            # Just blit the scaled background image at the grid's starting Y
            self.window.blit(self.background_image, (0, grid_start_y))
        else:
            # Fill grid area with solid background color
            background_rect = pygame.Rect(
                0, grid_start_y, self.window_size, grid_height
            )
            pygame.draw.rect(self.window, self.colors["background"], background_rect)

        # Draw grid lines
        for i in range(self.size + 1):
            # Horizontal lines
            y = grid_start_y + i * adjusted_cell_size
            pygame.draw.line(
                self.window, self.colors["grid"], (0, y), (self.window_size, y)
            )

            # Vertical lines
            x = i * adjusted_cell_size
            pygame.draw.line(
                self.window,
                self.colors["grid"],
                (x, grid_start_y),
                (x, grid_start_y + grid_height),
            )

        # Draw special tiles (adjusted for info panel)
        for tile_type, positions in special_tiles.items():
            for idx, pos in enumerate(positions):
                x_pix = pos[0] * adjusted_cell_size
                y_pix = grid_start_y + pos[1] * adjusted_cell_size

                if tile_type == "snitch":
                    scaled_snitch = pygame.transform.scale(
                        self.snitch_image, (adjusted_cell_size, adjusted_cell_size)
                    )
                    self.window.blit(scaled_snitch, (x_pix, y_pix))

                elif tile_type == "portal":
                    center_x = x_pix + adjusted_cell_size // 2
                    center_y = y_pix + adjusted_cell_size // 2

                    pygame.draw.circle(
                        self.window,
                        self.colors["portal"],
                        (center_x, center_y),
                        adjusted_cell_size // 3,
                    )
                    pygame.draw.circle(
                        self.window,
                        (255, 255, 255),
                        (center_x, center_y),
                        adjusted_cell_size // 3,
                        4,
                    )
                    pygame.draw.circle(
                        self.window,
                        (255, 255, 255),
                        (center_x, center_y),
                        adjusted_cell_size // 6,
                    )

                    label = f"P{(idx % 2) + 1}"
                    font = pygame.font.Font(None, adjusted_cell_size // 2)
                    text = font.render(label, True, (0, 0, 0))
                    text_rect = text.get_rect(center=(center_x, center_y))
                    self.window.blit(text, text_rect)

                elif tile_type in ["red_button", "green_button"]:
                    color = self.colors[tile_type]
                    rect = pygame.Rect(
                        x_pix, y_pix, adjusted_cell_size, adjusted_cell_size
                    )
                    pygame.draw.rect(self.window, color, rect, border_radius=8)
                    pygame.draw.rect(self.window, (0, 0, 0), rect, 2, border_radius=8)

                    label = "R" if tile_type == "red_button" else "G"
                    font = pygame.font.Font(None, adjusted_cell_size // 2)
                    text = font.render(label, True, (255, 255, 255))
                    text_rect = text.get_rect(center=rect.center)
                    self.window.blit(text, text_rect)

                else:
                    normalized_type = tile_type.rstrip("s")
                    color = self.colors.get(normalized_type, self.colors["grid"])
                    surface = pygame.Surface(
                        (adjusted_cell_size, adjusted_cell_size), pygame.SRCALPHA
                    )

                    if normalized_type in ["goal", "obstacle", "slippery", "prison"]:
                        surface.fill((*color, 150))
                    else:
                        surface.fill(color)

                    self.window.blit(surface, (x_pix, y_pix))

        # Draw charging cells (adjusted for info panel)
        if charging_cells:
            for pos in charging_cells:
                x_pix = pos[0] * adjusted_cell_size
                y_pix = grid_start_y + pos[1] * adjusted_cell_size
                scaled_battery = pygame.transform.scale(
                    self.battery_image, (adjusted_cell_size, adjusted_cell_size)
                )
                self.window.blit(scaled_battery, (x_pix, y_pix))

        # Draw moving cars (adjusted for info panel)
        if moving_cars:
            for car in moving_cars:
                pos = car["position"]
                if (
                    self.background_path
                    and "room4_background.jpg" in self.background_path
                ):
                    scaled_yellow_car = pygame.transform.scale(
                        self.yellow_car_image, (adjusted_cell_size, adjusted_cell_size)
                    )
                    self.window.blit(
                        scaled_yellow_car,
                        (
                            pos[0] * adjusted_cell_size,
                            grid_start_y + pos[1] * adjusted_cell_size,
                        ),
                    )
                else:
                    pygame.draw.rect(
                        self.window,
                        (255, 165, 0),
                        (
                            pos[0] * adjusted_cell_size,
                            grid_start_y + pos[1] * adjusted_cell_size,
                            adjusted_cell_size,
                            adjusted_cell_size,
                        ),
                    )

        # Draw agent (adjusted for info panel)
        if self.background_path and "room4_background.jpg" in self.background_path:
            scaled_red_car = pygame.transform.scale(
                self.red_car_image, (adjusted_cell_size, adjusted_cell_size)
            )
            self.window.blit(
                scaled_red_car,
                (
                    agent_position[0] * adjusted_cell_size,
                    grid_start_y + agent_position[1] * adjusted_cell_size,
                ),
            )
        elif self.background_path and "room1_background.jpg" in self.background_path:
            scaled_harry = pygame.transform.scale(
                self.harry_image, (adjusted_cell_size, adjusted_cell_size)
            )
            self.window.blit(
                scaled_harry,
                (
                    agent_position[0] * adjusted_cell_size,
                    grid_start_y + agent_position[1] * adjusted_cell_size,
                ),
            )
        else:
            scaled_agent = pygame.transform.scale(
                self.agent_image, (adjusted_cell_size, adjusted_cell_size)
            )
            self.window.blit(
                scaled_agent,
                (
                    agent_position[0] * adjusted_cell_size,
                    grid_start_y + agent_position[1] * adjusted_cell_size,
                ),
            )

        # pygame.display.flip()
        self.clock.tick(4)

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
        )

    def close(self):
        pygame.quit()
