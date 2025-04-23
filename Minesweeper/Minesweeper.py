import pygame
import random
import sys
import time

# Инициализация Pygame
pygame.init()

# Константы
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 550
GRID_SIZE = 12  # Количество клеток по горизонтали/вертикали
CELL_SIZE = 40  # Размер клетки в пикселях
MINE_COUNT = 20  # Количество мин
POWERUP_COUNT = 5  # Количество бонусов

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (150, 150, 150)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
COLORS = [BLUE, GREEN, RED, (0, 0, 128), (128, 0, 0), (0, 128, 128), BLACK, GRAY]

# Настройка экрана
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Сапёр с бонусами")
font = pygame.font.SysFont('Arial', 24)
small_font = pygame.font.SysFont('Arial', 16)
timer_font = pygame.font.SysFont('Arial', 20)


class Minesweeper:
    def __init__(self):
        self.reset_game()

    def reset_game(self):
        self.grid = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.revealed = [[False for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.flagged = [[False for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.powerups = [[0 for _ in range(GRID_SIZE)] for _ in
                         range(GRID_SIZE)]  # 0 - нет, 1 - подсказка, 2 - защита, 3 - детонатор
        self.game_over = False
        self.win = False
        self.first_click = True
        self.mines_left = MINE_COUNT
        self.hints_left = 0
        self.protection_active = False
        self.protection_time = 0
        self.start_time = 0
        self.time_elapsed = 0
        self.score = 0

        # Размещение мин и бонусов
        self.place_mines()
        self.place_powerups()

    def place_mines(self):
        mines_placed = 0
        while mines_placed < MINE_COUNT:
            x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
            if self.grid[y][x] != -1:
                self.grid[y][x] = -1
                mines_placed += 1

        # Подсчет чисел вокруг мин
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.grid[y][x] == -1:
                    continue
                count = 0
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and self.grid[ny][nx] == -1:
                            count += 1
                self.grid[y][x] = count

    def place_powerups(self):
        powerups_placed = 0
        while powerups_placed < POWERUP_COUNT:
            x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
            if self.grid[y][x] != -1 and self.powerups[y][x] == 0:
                self.powerups[y][x] = random.randint(1, 3)
                powerups_placed += 1

    def reveal(self, x, y):
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE) or self.revealed[y][x] or self.flagged[y][x]:
            return

        # Активация защиты
        if self.protection_active and self.grid[y][x] == -1:
            self.grid[y][x] = 0  # Обезвреживаем мину
            # Пересчитываем числа вокруг
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and self.grid[ny][nx] != -1:
                        self.grid[ny][nx] -= 1
            self.protection_active = False
            self.score += 10  # Бонус за обезвреживание

        self.revealed[y][x] = True

        # Проверка бонусов
        if self.powerups[y][x] > 0:
            self.activate_powerup(x, y)

        if self.grid[y][x] == -1:  # Наступили на мину
            if not self.protection_active:
                self.game_over = True
                self.reveal_all_mines()
            return

        if self.grid[y][x] == 0:  # Пустая клетка - открываем соседей
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    self.reveal(x + dx, y + dy)

        # Начисление очков
        self.score += 1

        # Проверка победы
        self.check_win()

    def activate_powerup(self, x, y):
        powerup_type = self.powerups[y][x]
        self.powerups[y][x] = 0  # Бонус использован

        if powerup_type == 1:  # Подсказка
            self.hints_left += 1
            self.score += 5
        elif powerup_type == 2:  # Защита
            self.protection_active = True
            self.protection_time = time.time()
            self.score += 10
        elif powerup_type == 3:  # Детонатор
            self.detonate_nearby_mines(x, y)
            self.score += 15

    def detonate_nearby_mines(self, x, y):
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and self.grid[ny][nx] == -1:
                    self.grid[ny][nx] = 0  # Обезвреживаем мину
                    self.revealed[ny][nx] = True
                    # Пересчитываем числа вокруг
                    for ddy in [-1, 0, 1]:
                        for ddx in [-1, 0, 1]:
                            if ddy == 0 and ddx == 0:
                                continue
                            nnx, nny = nx + ddx, ny + ddy
                            if 0 <= nnx < GRID_SIZE and 0 <= nny < GRID_SIZE and self.grid[nny][nnx] != -1:
                                self.grid[nny][nnx] -= 1

    def use_hint(self):
        if self.hints_left <= 0:
            return

        # Находим неоткрытые безопасные клетки
        safe_cells = []
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if not self.revealed[y][x] and not self.flagged[y][x] and self.grid[y][x] != -1:
                    safe_cells.append((x, y))

        if not safe_cells:
            return

        # Выбираем случайную безопасную клетку
        x, y = random.choice(safe_cells)
        self.revealed[y][x] = True
        self.hints_left -= 1

        # Если это пустая клетка, открываем соседей
        if self.grid[y][x] == 0:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    self.reveal(x + dx, y + dy)

    def toggle_flag(self, x, y):
        if not self.revealed[y][x]:
            self.flagged[y][x] = not self.flagged[y][x]
            self.mines_left = MINE_COUNT - sum(sum(row) for row in self.flagged)

    def reveal_all_mines(self):
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.grid[y][x] == -1:
                    self.revealed[y][x] = True

    def check_win(self):
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if self.grid[y][x] != -1 and not self.revealed[y][x]:
                    return
        self.win = True
        self.game_over = True
        # Бонус за быстрый проход
        time_bonus = max(0, 100 - int(self.time_elapsed))
        self.score += time_bonus

    def update_timer(self):
        if not self.game_over and not self.first_click:
            self.time_elapsed = time.time() - self.start_time

    def draw(self):
        # Отрисовка верхней панели
        pygame.draw.rect(screen, DARK_GRAY, (0, 0, SCREEN_WIDTH, 70))

        # Очки
        score_text = font.render(f"Очки: {self.score}", True, WHITE)
        screen.blit(score_text, (10, 10))

        # Мины
        mines_text = font.render(f"Мины: {self.mines_left}", True, WHITE)
        screen.blit(mines_text, (10, 40))

        # Таймер
        timer_text = timer_font.render(f"Время: {int(self.time_elapsed)} сек", True, WHITE)
        screen.blit(timer_text, (SCREEN_WIDTH - 120, 10))

        # Подсказки
        hints_text = timer_font.render(f"Подсказки: {self.hints_left}", True, YELLOW)
        screen.blit(hints_text, (SCREEN_WIDTH - 120, 40))

        # Защита
        if self.protection_active:
            protection_text = timer_font.render("ЗАЩИТА!", True, GREEN)
            screen.blit(protection_text, (SCREEN_WIDTH // 2 - 40, 40))

        if self.game_over:
            status_text = "Вы проиграли!" if not self.win else "ПОБЕДА!"
            status = font.render(status_text, True, RED if not self.win else GREEN)
            screen.blit(status, (SCREEN_WIDTH // 2 - status.get_width() // 2, 10))

        # Отрисовка сетки
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE + 70, CELL_SIZE, CELL_SIZE)

                if self.revealed[y][x]:
                    pygame.draw.rect(screen, WHITE, rect)
                    pygame.draw.rect(screen, GRAY, rect, 1)

                    if self.grid[y][x] == -1:  # Мина
                        pygame.draw.circle(screen, BLACK, rect.center, CELL_SIZE // 3)
                    elif self.grid[y][x] > 0:  # Число
                        text = font.render(str(self.grid[y][x]), True, COLORS[self.grid[y][x] - 1])
                        screen.blit(text, (x * CELL_SIZE + CELL_SIZE // 2 - text.get_width() // 2,
                                           y * CELL_SIZE + 70 + CELL_SIZE // 2 - text.get_height() // 2))

                    # Бонусы (только если клетка открыта и бонус не использован)
                    if self.powerups[y][x] > 0:
                        self.draw_powerup(x, y, rect)
                else:
                    pygame.draw.rect(screen, GRAY, rect)
                    pygame.draw.rect(screen, DARK_GRAY, rect, 1)

                    if self.flagged[y][x]:  # Флажок
                        pygame.draw.polygon(screen, RED, [
                            (x * CELL_SIZE + 10, y * CELL_SIZE + 70 + 30),
                            (x * CELL_SIZE + 30, y * CELL_SIZE + 70 + 20),
                            (x * CELL_SIZE + 10, y * CELL_SIZE + 70 + 10)
                        ])

        # Кнопки
        restart_rect = pygame.Rect(SCREEN_WIDTH - 100, 10, 80, 25)
        pygame.draw.rect(screen, WHITE, restart_rect)
        pygame.draw.rect(screen, BLACK, restart_rect, 2)
        restart_text = small_font.render("Новая игра", True, BLACK)
        screen.blit(restart_text, (SCREEN_WIDTH - 90, 15))

        hint_rect = pygame.Rect(SCREEN_WIDTH - 100, 40, 80, 25)
        pygame.draw.rect(screen, YELLOW, hint_rect)
        pygame.draw.rect(screen, BLACK, hint_rect, 2)
        hint_text = small_font.render("Подсказка", True, BLACK)
        screen.blit(hint_text, (SCREEN_WIDTH - 90, 45))

        return restart_rect, hint_rect

    def draw_powerup(self, x, y, rect):
        powerup_type = self.powerups[y][x]
        if powerup_type == 1:  # Подсказка
            pygame.draw.circle(screen, YELLOW, rect.center, CELL_SIZE // 4)
            text = small_font.render("?", True, BLACK)
            screen.blit(text, (x * CELL_SIZE + CELL_SIZE // 2 - text.get_width() // 2,
                               y * CELL_SIZE + 70 + CELL_SIZE // 2 - text.get_height() // 2))
        elif powerup_type == 2:  # Защита
            pygame.draw.circle(screen, GREEN, rect.center, CELL_SIZE // 4)
            text = small_font.render("Щ", True, BLACK)
            screen.blit(text, (x * CELL_SIZE + CELL_SIZE // 2 - text.get_width() // 2,
                               y * CELL_SIZE + 70 + CELL_SIZE // 2 - text.get_height() // 2))
        elif powerup_type == 3:  # Детонатор
            pygame.draw.circle(screen, ORANGE, rect.center, CELL_SIZE // 4)
            text = small_font.render("Д", True, BLACK)
            screen.blit(text, (x * CELL_SIZE + CELL_SIZE // 2 - text.get_width() // 2,
                               y * CELL_SIZE + 70 + CELL_SIZE // 2 - text.get_height() // 2))


def main():
    game = Minesweeper()
    clock = pygame.time.Clock()

    while True:
        game.update_timer()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                grid_x, grid_y = x // CELL_SIZE, (y - 70) // CELL_SIZE

                # Проверка клика по кнопкам
                restart_rect, hint_rect = game.draw()
                if restart_rect.collidepoint(event.pos):
                    game.reset_game()
                    continue
                if hint_rect.collidepoint(event.pos) and not game.game_over:
                    game.use_hint()
                    continue

                if y < 70 or game.game_over:
                    continue

                if 0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE:
                    if event.button == 1:  # Левая кнопка мыши
                        if game.first_click:
                            # Убедимся, что первая клетка не мина и не бонус
                            while game.grid[grid_y][grid_x] == -1 or game.powerups[grid_y][grid_x] > 0:
                                game.reset_game()
                            game.first_click = False
                            game.start_time = time.time()
                        game.reveal(grid_x, grid_y)
                    elif event.button == 3:  # Правая кнопка мыши
                        game.toggle_flag(grid_x, grid_y)

        screen.fill(WHITE)
        game.draw()
        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()