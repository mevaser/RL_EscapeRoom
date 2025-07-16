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
            "yellow_button": (255, 255, 0),
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
        collected_mask: Optional[int] = None,
        shape_indices: Optional[Dict[str, int]] = None,
        current_task: Optional[str] = None,
    ) -> np.ndarray:

        if self.window is None:
            raise ValueError(
                "Renderer window is not initialized. Make sure to set it before calling render."
            )

        # === Unified layout setup ===
        info_panel_height = int(self.window_size * 0.2)  # 20% מהמסך - כמו בשאר החדרים
        grid_start_y = info_panel_height
        adjusted_cell_size = self.window_size // self.size
        grid_height = adjusted_cell_size * self.size
        total_height = info_panel_height + grid_height

        # === Draw background ===
        if self.background_image:
            self.background_image = pygame.transform.smoothscale(
                self.background_image, (self.window_size, grid_height)
            )
            self.window.blit(self.background_image, (0, grid_start_y))
        else:
            grid_area = pygame.Rect(0, grid_start_y, self.window_size, grid_height)
            pygame.draw.rect(self.window, self.colors["background"], grid_area)

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

        # Draw grid lines (adjusted for info panel)
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

        # --------------------------------------------------------------------
        # Render special tiles (grid offset already accounted for)
        # --------------------------------------------------------------------
        for tile_type, positions in special_tiles.items():
            for idx, pos in enumerate(positions):
                x_pix = pos[0] * adjusted_cell_size
                y_pix = grid_start_y + pos[1] * adjusted_cell_size

                # --- portals --------------------------------------------------
                if tile_type == "portal":
                    center_x = x_pix + adjusted_cell_size // 2
                    center_y = y_pix + adjusted_cell_size // 2

                    # Outer filled circle
                    pygame.draw.circle(
                        self.window,
                        self.colors["portal"],
                        (center_x, center_y),
                        adjusted_cell_size // 3,
                    )
                    # White ring
                    pygame.draw.circle(
                        self.window,
                        (255, 255, 255),
                        (center_x, center_y),
                        adjusted_cell_size // 3,
                        4,
                    )
                    # Inner white dot
                    pygame.draw.circle(
                        self.window,
                        (255, 255, 255),
                        (center_x, center_y),
                        adjusted_cell_size // 6,
                    )

                    # Portal label (“P1”, “P2”, …)
                    label = f"P{(idx % 2) + 1}"
                    font = pygame.font.Font(None, adjusted_cell_size // 2)
                    text = font.render(label, True, (0, 0, 0))
                    text_rect = text.get_rect(center=(center_x, center_y))
                    self.window.blit(text, text_rect)

                # --- yellow button -------------------------------------------
                elif tile_type == "yellow_button":
                    color = self.colors[tile_type]
                    rect = pygame.Rect(
                        x_pix, y_pix, adjusted_cell_size, adjusted_cell_size
                    )

                    pygame.draw.rect(self.window, color, rect, border_radius=8)
                    pygame.draw.rect(self.window, (0, 0, 0), rect, 2, border_radius=8)

                    label = "C"
                    font = pygame.font.Font(None, adjusted_cell_size // 2)
                    text = font.render(label, True, (0, 0, 0))
                    text_rect = text.get_rect(center=rect.center)
                    self.window.blit(text, text_rect)

                # --- generic semi-transparent tiles (obstacles, slippery, …) --
                else:
                    color = self.colors.get(tile_type, self.colors["grid"])
                    surface = pygame.Surface(
                        (adjusted_cell_size, adjusted_cell_size), pygame.SRCALPHA
                    )
                    surface.fill((*color, 150))  # 150 = ~60 % opacity
                    self.window.blit(surface, (x_pix, y_pix))

        # --------------------------------------------------------------------
        # Draw shapes (diamonds, etc.) – single cyan outline + drop-shadow
        # --------------------------------------------------------------------
        shape_color = (0, 255, 255)  # bright cyan outline
        shadow_color = (0, 0, 0, 180)  # semi-transparent black
        shadow_offset = 2  # pixels

        if shapes_positions:
            for shape, pos in shapes_positions.items():
                if collected_mask is not None and shape_indices is not None:
                    shape_idx = shape_indices[shape]
                    if collected_mask & (1 << shape_idx):
                        continue  # skip shapes already collected

                center = (
                    pos[0] * adjusted_cell_size + adjusted_cell_size // 2,
                    grid_start_y
                    + pos[1] * adjusted_cell_size
                    + adjusted_cell_size // 2,
                )

                if shape == "circle":
                    # Shadow
                    pygame.draw.circle(
                        self.window,
                        shadow_color,
                        (center[0] + shadow_offset, center[1] + shadow_offset),
                        adjusted_cell_size // 3,
                    )
                    # Outline circle
                    pygame.draw.circle(
                        self.window,
                        shape_color,
                        center,
                        adjusted_cell_size // 3,
                        width=4,
                    )

                elif shape == "square":
                    square_rect = pygame.Rect(
                        0, 0, adjusted_cell_size // 2, adjusted_cell_size // 2
                    )
                    square_rect.center = center

                    shadow_rect = square_rect.copy()
                    shadow_rect.move_ip(shadow_offset, shadow_offset)

                    shadow_surface = pygame.Surface(square_rect.size, pygame.SRCALPHA)
                    shadow_surface.fill(shadow_color)
                    self.window.blit(shadow_surface, shadow_rect)

                    pygame.draw.rect(self.window, shape_color, square_rect, width=4)

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
                    # Shadow points
                    shadow_triangle = [
                        (x + shadow_offset, y + shadow_offset) for x, y in triangle
                    ]

                    # Shadow (fills slightly larger area)
                    pygame.draw.polygon(self.window, shadow_color, shadow_triangle)

                    # Outline triangle
                    pygame.draw.polygon(self.window, shape_color, triangle, width=4)

        # --------------------------------------------------------------------
        # Draw the agent sprite
        # --------------------------------------------------------------------
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

        self.clock.tick(4)

        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
        )

    def close(self):
        pygame.quit()
