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
        self.window_size = (window_size // size) * size  # Ensure divisible by grid size
        self.cell_size = self.window_size // self.size
        self.header_height = 0
        self.render_height = self.window_size
        self.total_height = self.render_height + self.header_height
        self.window = window  # Use shared window from main app

        # Define color scheme
        self.colors = {
            "background": (255, 255, 255),
            "grid": (200, 200, 200),
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
        self.font = pygame.font.Font(None, 32)

        # Load agent image
        self.agent_image = pygame.image.load(
            "assets/images/einstein.png"
        ).convert_alpha()

        # Load background if available
        self.background_path = background_path
        if self.background_path:
            self.background_image = pygame.image.load(self.background_path).convert()
            self.background_image = pygame.transform.smoothscale(
                self.background_image, (self.window_size, self.total_height)
            )
        else:
            self.background_image = None

    def render_with_shapes(
        self,
        agent_position: Tuple[int, int],
        special_tiles: Dict[str, set],
        info: Optional[Dict] = None,
        shapes_positions=None,
        collected_shapes=None,
        current_task: Optional[str] = None,
    ) -> np.ndarray:
        # Calculate grid area (leaving space for info panel at top)
        grid_start_y = 130  # Leave 120 pixels at top for info panel
        grid_size = self.window_size - grid_start_y
        adjusted_cell_size = grid_size // self.size

        if self.background_image:
            self.window.blit(self.background_image, (0, 0))
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
            for pos in positions:
                if tile_type == "portal":
                    # Draw portal as a circle with portal color
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
                    # Draw buttons as filled rectangles with button color
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
                    # Add a border to make buttons more visible
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
                    color = self.colors.get(tile_type, self.colors["grid"])
                    tile_surface = pygame.Surface(
                        (adjusted_cell_size, adjusted_cell_size), pygame.SRCALPHA
                    )
                    tile_surface.fill((*color, 150))
                    self.window.blit(
                        tile_surface,
                        (
                            pos[0] * adjusted_cell_size,
                            grid_start_y + pos[1] * adjusted_cell_size,
                        ),
                    )

        # Draw shapes (adjusted for info panel)
        if shapes_positions:
            for shape, pos in shapes_positions.items():
                if collected_shapes and shape in collected_shapes:
                    continue  # Don't draw collected shapes

                center = (
                    pos[0] * adjusted_cell_size + adjusted_cell_size // 2,
                    grid_start_y
                    + pos[1] * adjusted_cell_size
                    + adjusted_cell_size // 2,
                )
                color = (255, 255, 255)  # White
                if shape == "circle":
                    pygame.draw.circle(
                        self.window, color, center, adjusted_cell_size // 3, width=4
                    )
                elif shape == "square":
                    square_rect = pygame.Rect(
                        0, 0, adjusted_cell_size // 2, adjusted_cell_size // 2
                    )
                    square_rect.center = center
                    pygame.draw.rect(self.window, color, square_rect, width=4)
                elif shape == "triangle":
                    triangle = [
                        (center[0], center[1] - adjusted_cell_size // 3),
                        (
                            center[0] - adjusted_cell_size // 3,
                            center[1] + adjusted_cell_size // 3,
                        ),
                        (
                            center[0] + adjusted_cell_size // 3,
                            center[1] + adjusted_cell_size // 3,
                        ),
                    ]
                    pygame.draw.polygon(self.window, color, triangle, width=4)

        # Draw agent (adjusted for info panel)
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

        # Draw info panel above the grid
        if info:
            # Draw info panel background
            info_panel_height = 100
            info_panel_rect = pygame.Rect(0, 0, self.window_size, info_panel_height)
            overlay_surface = pygame.Surface(
                (self.window_size, info_panel_height), pygame.SRCALPHA
            )
            overlay_surface.fill((255, 255, 255, 200))
            self.window.blit(overlay_surface, (0, 0))

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

        pygame.display.flip()
        self.clock.tick(4)

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
        )

    def close(self):
        pygame.quit()
