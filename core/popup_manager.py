import pygame
import sys


def show_not_trained_popup(self):
    """
    Display a popup if the agent has not yet been trained.
    """
    popup_width, popup_height = 500, 200
    popup_x = (self.window_size - popup_width) // 2
    popup_y = (self.window_size - popup_height) // 2
    popup_rect = pygame.Rect(popup_x, popup_y, popup_width, popup_height)

    button_width, button_height = 120, 50
    button_x = (self.window_size - button_width) // 2
    button_y = popup_y + popup_height - 70
    button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    running = False
                elif not popup_rect.collidepoint(event.pos):
                    running = False  # Clicked outside

        # Dark transparent background
        overlay = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.window.blit(overlay, (0, 0))

        # Popup background
        pygame.draw.rect(self.window, (255, 255, 255), popup_rect)
        pygame.draw.rect(self.window, (0, 0, 0), popup_rect, 3)

        # Popup message
        text_surface = self.font_medium.render(
            "Agent has not been trained yet.", True, (0, 0, 0)
        )
        text_rect = text_surface.get_rect(center=(self.window_size // 2, popup_y + 60))
        self.window.blit(text_surface, text_rect)

        subtext_surface = self.font_small.render(
            "Please train the agent first.", True, (80, 80, 80)
        )
        subtext_rect = subtext_surface.get_rect(
            center=(self.window_size // 2, popup_y + 100)
        )
        self.window.blit(subtext_surface, subtext_rect)

        # OK button
        pygame.draw.rect(self.window, (0, 128, 0), button_rect)
        button_text = self.font_small.render("OK", True, (255, 255, 255))
        self.window.blit(button_text, button_text.get_rect(center=button_rect.center))

        pygame.display.flip()
        self.clock.tick(30)


def show_room_completed_popup(self):
    """
    Show a popup when the agent successfully completes the room.
    """
    popup_running = True
    pygame_surface = self.capture_background_snapshot()

    popup_width, popup_height = 400, 250
    popup_x = (self.window_size - popup_width) // 2
    popup_y = (self.window_size - popup_height) // 2
    popup_rect = pygame.Rect(popup_x, popup_y, popup_width, popup_height)

    button_width, button_height = 150, 50
    button_x = (self.window_size - button_width) // 2
    button_y = popup_y + popup_height - 80
    button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

    font_large = pygame.font.Font(None, 40)
    font_small = pygame.font.Font(None, 30)

    while popup_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    popup_running = False

        self.window.blit(pygame_surface, (0, 0))

        pygame.draw.rect(self.window, (255, 255, 255), popup_rect)
        pygame.draw.rect(self.window, (0, 0, 0), popup_rect, 3)

        text_surface = font_large.render("Room completed!", True, (0, 0, 0))
        self.window.blit(
            text_surface,
            text_surface.get_rect(center=(self.window_size // 2, popup_y + 60)),
        )

        subtext_surface = font_small.render(
            "Press OK to return to menu", True, (80, 80, 80)
        )
        self.window.blit(
            subtext_surface,
            subtext_surface.get_rect(center=(self.window_size // 2, popup_y + 110)),
        )

        pygame.draw.rect(self.window, (0, 128, 0), button_rect)
        button_text = font_small.render("OK", True, (255, 255, 255))
        self.window.blit(button_text, button_text.get_rect(center=button_rect.center))

        pygame.display.flip()
        pygame.time.Clock().tick(30)


def show_summary_popup(self, summary):
    """
    Show a popup with a final summary when stopping the game manually.
    """
    popup_width, popup_height = 500, 200
    popup_x = (self.window_size - popup_width) // 2
    popup_y = (self.window_size - popup_height) // 2
    popup_rect = pygame.Rect(popup_x, popup_y, popup_width, popup_height)

    button_width, button_height = 120, 50
    button_x = (self.window_size - button_width) // 2
    button_y = popup_y + popup_height - 70
    button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    running = False

        overlay = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        self.window.blit(overlay, (0, 0))

        pygame.draw.rect(self.window, (255, 255, 255), popup_rect)
        pygame.draw.rect(self.window, (0, 0, 0), popup_rect, 3)

        text_surface = self.font_medium.render(summary, True, (0, 0, 0))
        self.window.blit(
            text_surface,
            text_surface.get_rect(center=(self.window_size // 2, popup_y + 60)),
        )

        pygame.draw.rect(self.window, (0, 128, 0), button_rect)
        button_text = self.font_small.render("OK", True, (255, 255, 255))
        self.window.blit(button_text, button_text.get_rect(center=button_rect.center))

        pygame.display.flip()
        self.clock.tick(30)
