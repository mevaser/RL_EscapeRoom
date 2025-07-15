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
            self.background_image = pygame.transform.smoothscale(
                self.background_image, (self.window_size, self.window_size)
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
        # Calculate grid area (leaving space for info panel at top)
        grid_start_y = 130  # Leave 120 pixels at top for info panel
        grid_size = self.window_size - grid_start_y
        adjusted_cell_size = grid_size // self.size

        # Draw info panel above the grid
        if info:
            # Draw info panel background
            info_panel_height = 130
            info_panel_rect = pygame.Rect(0, 0, self.window_size, info_panel_height)
            overlay_surface = pygame.Surface(
                (self.window_size, info_panel_height), pygame.SRCALPHA
            )
            overlay_surface.fill((255, 255, 255, 200))
            self.window.blit(overlay_surface, (0, 0))

            # Calculate grid area (leaving space for info panel at top)
            grid_start_y = info_panel_height  # Leave 120 pixels at top for info panel
            grid_size = self.window_size - grid_start_y

            # Draw info panel border
            pygame.draw.rect(self.window, (100, 100, 100), info_panel_rect, 2)

            # Display info text
            y_offset = 15
            x_offset = 20
            for key, value in info.items():
                if key not in ["snitch_collected", "snitch_total", "Training"]:
                    text = f"{key}: {value}"
                    text_surface = self.font.render(text, True, self.colors["text"])
                    self.window.blit(text_surface, (x_offset, y_offset))
                    y_offset += 25

                    # Start new column if we've used half the width
                    if y_offset > info_panel_height - 25:
                        y_offset = 15
                        x_offset = self.window_size // 2 + 20

            if "snitch_collected" in info and "snitch_total" in info:
                snitch_text = (
                    f"Snitch: {info['snitch_collected']}/{info['snitch_total']}"
                )
                text_surface = self.font.render(snitch_text, True, self.colors["text"])
                self.window.blit(text_surface, (x_offset, y_offset))
                y_offset += 25

        if self.background_image:
            # רקע יוצג רק באזור הגריד (לא כולל האינפו העליון)
            grid_start_y = 130
            grid_area = pygame.Rect(
                0, grid_start_y, self.window_size, self.window_size - grid_start_y
            )
            self.window.blit(self.background_image, grid_area, area=grid_area)
        else:
            self.window.fill(self.colors["background"])

        # Draw grid lines (adjusted for info panel)
        for i in range(self.size + 1):
            # Horizontal lines
            y_pos = grid_start_y + i * adjusted_cell_size
            pygame.draw.line(
                self.window, self.colors["grid"], (0, y_pos), (self.window_size, y_pos)
            )
            # Vertical lines
            x_pos = i * adjusted_cell_size
            pygame.draw.line(
                self.window,
                self.colors["grid"],
                (x_pos, grid_start_y),
                (x_pos, self.window_size),
            )

        # Draw special tiles (adjusted for info panel)
        for tile_type, positions in special_tiles.items():
            for idx, pos in enumerate(positions):
                if tile_type == "snitch":
                    scaled_snitch = pygame.transform.scale(
                        self.snitch_image, (adjusted_cell_size, adjusted_cell_size)
                    )
                    self.window.blit(
                        scaled_snitch,
                        (
                            pos[0] * adjusted_cell_size,
                            grid_start_y + pos[1] * adjusted_cell_size,
                        ),
                    )
                elif tile_type == "portal":
                    center_x = pos[0] * adjusted_cell_size + adjusted_cell_size // 2
                    center_y = (
                        grid_start_y
                        + pos[1] * adjusted_cell_size
                        + adjusted_cell_size // 2
                    )
                    # Draw portal with bold purple color and white border
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
                    # Add inner circle for portal effect
                    pygame.draw.circle(
                        self.window,
                        (255, 255, 255),
                        (center_x, center_y),
                        adjusted_cell_size // 6,
                    )
                    # Label P1/P2 with bold text
                    label = "P1" if idx == 0 else "P2"
                    font = pygame.font.Font(None, adjusted_cell_size // 2)
                    text = font.render(label, True, (0, 0, 0))
                    text_rect = text.get_rect(center=(center_x, center_y))
                    self.window.blit(text, text_rect)
                elif tile_type in ["red_button", "green_button"]:
                    color = self.colors[tile_type]
                    pygame.draw.rect(
                        self.window,
                        color,
                        (
                            pos[0] * adjusted_cell_size,
                            grid_start_y + pos[1] * adjusted_cell_size,
                            adjusted_cell_size,
                            adjusted_cell_size,
                        ),
                        border_radius=8,
                    )
                    pygame.draw.rect(
                        self.window,
                        (0, 0, 0),
                        (
                            pos[0] * adjusted_cell_size,
                            grid_start_y + pos[1] * adjusted_cell_size,
                            adjusted_cell_size,
                            adjusted_cell_size,
                        ),
                        2,
                        border_radius=8,
                    )
                    # Draw icon/label
                    font = pygame.font.Font(None, adjusted_cell_size // 2)
                    label = "R" if tile_type == "red_button" else "G"
                    text = font.render(label, True, (255, 255, 255))
                    text_rect = text.get_rect(
                        center=(
                            pos[0] * adjusted_cell_size + adjusted_cell_size // 2,
                            grid_start_y
                            + pos[1] * adjusted_cell_size
                            + adjusted_cell_size // 2,
                        )
                    )
                    self.window.blit(text, text_rect)
                else:
                    normalized_tile_type = tile_type.rstrip("s")
                    color = self.colors.get(normalized_tile_type, self.colors["grid"])

                    if normalized_tile_type in [
                        "goal",
                        "obstacle",
                        "slippery",
                        "prison",
                    ]:
                        surface = pygame.Surface(
                            (adjusted_cell_size, adjusted_cell_size), pygame.SRCALPHA
                        )
                        surface.fill((*color, 150))
                        self.window.blit(
                            surface,
                            (
                                pos[0] * adjusted_cell_size,
                                grid_start_y + pos[1] * adjusted_cell_size,
                            ),
                        )
                    else:
                        pygame.draw.rect(
                            self.window,
                            color,
                            (
                                pos[0] * adjusted_cell_size,
                                grid_start_y + pos[1] * adjusted_cell_size,
                                adjusted_cell_size,
                                adjusted_cell_size,
                            ),
                        )

        # Draw charging cells (adjusted for info panel)
        if charging_cells:
            for pos in charging_cells:
                scaled_battery = pygame.transform.scale(
                    self.battery_image, (adjusted_cell_size, adjusted_cell_size)
                )
                self.window.blit(
                    scaled_battery,
                    (
                        pos[0] * adjusted_cell_size,
                        grid_start_y + pos[1] * adjusted_cell_size,
                    ),
                )

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

        pygame.display.flip()
        self.clock.tick(4)

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
        )

    def close(self):
        pygame.quit()
