import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PyQt5 import QtWidgets, QtGui
import sys
import random
import os
from collections import deque

# Модель нейросети
class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.fc1 = nn.Linear(9, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Класс игры для обучения
class TicTacToeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = np.zeros(9)
        return self.board.copy()

    def get_valid_moves(self):
        return np.where(self.board == 0)[0]

    def make_move(self, move, player):
        self.board[move] = player
        return self.board.copy()

    def check_winner(self, player):
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for condition in win_conditions:
            if np.all(self.board[condition] == player):
                return True
        return False

    def is_done(self):
        return len(self.get_valid_moves()) == 0 or self.check_winner(1) or self.check_winner(-1)

    def get_reward(self, player):
        if self.check_winner(player):
            return 1
        if self.check_winner(-player):
            return -1
        if len(self.get_valid_moves()) == 0:
            return 0
        return 0

# Обучение нейросети
def train_model(model, device, episodes=10000, lr=0.0001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, model_path="tictactoe_model.pth", log_path="training_log.txt", batch_size=32):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    env = TicTacToeEnv()
    epsilon = epsilon_start
    memory = deque(maxlen=5000)  # Уменьшенный буфер

    # Статистика
    recent_results = []
    recent_lengths = []

    # Открываем файл для логов
    with open(log_path, "w") as log_file:
        log_file.write("Начало обучения\n")
        log_file.write("-" * 50 + "\n")

        for episode in range(episodes):
            state = env.reset()
            done = False
            total_loss = 0
            steps = 0

            while not done:
                valid_moves = env.get_valid_moves()
                if valid_moves.size == 0:
                    break

                # ε-greedy выбор хода
                if random.random() < epsilon:
                    action = random.choice(valid_moves)
                else:
                    with torch.no_grad():
                        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                        q_values = model(state_tensor)[0].cpu().numpy()
                        valid_q_values = [q_values[i] if i in valid_moves else -float('inf') for i in range(9)]
                        action = np.argmax(valid_q_values)

                # Выполнить ход
                next_state = env.make_move(action, 1)
                reward = env.get_reward(1)
                done = env.is_done()

                # Ход противника
                if not done:
                    valid_moves = env.get_valid_moves()
                    if valid_moves.size == 0:
                        break
                    opponent_action = random.choice(valid_moves)
                    next_state = env.make_move(opponent_action, -1)
                    done = env.is_done()

                # Сохраняем переход
                memory.append((state, action, reward, next_state, done))
                state = next_state
                steps += 1

                # Обучение на батче
                if len(memory) >= batch_size:
                    batch = random.sample(memory, batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)

                    states = torch.tensor(states, dtype=torch.float32, device=device)
                    actions = torch.tensor(actions, dtype=torch.long, device=device)
                    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                    next_states = torch.tensor(next_states, dtype=torch.float32, device=device)
                    dones = torch.tensor(dones, dtype=torch.float32, device=device)

                    q_values = model(states)
                    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                    with torch.no_grad():
                        next_q_values = model(next_states)
                        max_next_q, _ = next_q_values.max(dim=1)
                        targets = rewards + gamma * max_next_q * (1 - dones)

                    loss = criterion(q_values, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Клиппинг градиентов
                    optimizer.step()

                    total_loss += loss.item()

            # Сохранение результата
            if env.check_winner(1):
                recent_results.append(1)
            elif env.check_winner(-1):
                recent_results.append(-1)
            else:
                recent_results.append(0)
            recent_lengths.append(steps)

            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            # Логи каждые 100 эпизодов
            if (episode + 1) % 100 == 0:
                avg_loss = total_loss / (steps * batch_size) if steps > 0 else 0
                last_100_results = recent_results[-100:] if len(recent_results) >= 100 else recent_results
                wins = last_100_results.count(1)
                draws = last_100_results.count(0)
                losses = last_100_results.count(-1)
                avg_length = sum(recent_lengths[-100:]) / len(recent_lengths[-100:]) if recent_lengths else 0

                log_message = (
                    f"Эпизод {episode + 1}/{episodes}\n"
                    f"  Средняя потеря: {avg_loss:.4f}\n"
                    f"  Epsilon: {epsilon:.4f}\n"
                    f"  Победы/Ничьи/Поражения (последние 100): {wins}/{draws}/{losses}\n"
                    f"  Средняя длина игры: {avg_length:.2f} ходов\n"
                    f"{'-' * 50}\n"
                )
                print(log_message, end="")
                with open(log_path, "a") as log_file:
                    log_file.write(log_message)

        # Сохранение модели
        torch.save(model.state_dict(), model_path)
        final_message = f"Модель сохранена в {model_path}\n"
        print(final_message, end="")
        with open(log_path, "a") as log_file:
            log_file.write(final_message)

# Загрузка или обучение модели
def load_or_train_model(model, device, model_path="tictactoe_model.pth", log_path="training_log.txt"):
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        print(f"Модель загружена из {model_path}")
    else:
        print("Модель не найдена, начинаем обучение...")
        model.to(device)
        train_model(model, device, model_path=model_path, log_path=log_path)

# Класс игры с интерфейсом
class TicTacToe(QtWidgets.QMainWindow):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.setWindowTitle("Крестики-Нолики")
        self.setFixedSize(300, 300)
        self.board = np.zeros(9)
        self.buttons = []
        self.init_ui()

    def init_ui(self):
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QGridLayout()
        central_widget.setLayout(layout)

        for i in range(3):
            for j in range(3):
                button = QtWidgets.QPushButton("")
                button.setFont(QtGui.QFont("Arial", 20))
                button.setFixedSize(100, 100)
                idx = i * 3 + j
                button.clicked.connect(lambda _, x=idx: self.player_move(x))
                layout.addWidget(button, i, j)
                self.buttons.append(button)

    def player_move(self, idx):
        if self.board[idx] == 0:
            self.board[idx] = 1
            self.buttons[idx].setText("X")
            self.buttons[idx].setEnabled(False)
            if self.check_winner(1):
                QtWidgets.QMessageBox.information(self, "Победа!", "Игрок X выиграл!")
                self.reset()
                return
            if np.all(self.board != 0):
                QtWidgets.QMessageBox.information(self, "Ничья!", "Игра окончена вничью!")
                self.reset()
                return
            self.ai_move()

    def ai_move(self):
        with torch.no_grad():
            state_tensor = torch.tensor(self.board, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.model(state_tensor)[0].cpu().numpy()
            valid_moves = np.where(self.board == 0)[0]
            valid_q_values = [q_values[i] if i in valid_moves else -float('inf') for i in range(9)]
            move = np.argmax(valid_q_values)

        self.board[move] = -1
        self.buttons[move].setText("O")
        self.buttons[move].setEnabled(False)
        if self.check_winner(-1):
            QtWidgets.QMessageBox.information(self, "Поражение!", "ИИ (O) выиграл!")
            self.reset()
            return
        if np.all(self.board != 0):
            QtWidgets.QMessageBox.information(self, "Ничья!", "Игра окончена вничью!")
            self.reset()

    def check_winner(self, player):
        win_conditions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for condition in win_conditions:
            if np.all(self.board[condition] == player):
                return True
        return False

    def reset(self):
        self.board = np.zeros(9)
        for button in self.buttons:
            button.setText("")
            button.setEnabled(True)

# Запуск приложения
if __name__ == "__main__":
    # Устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Инициализация модели
    model = TicTacToeNet()
    load_or_train_model(model, device)

    # Запуск игры
    app = QtWidgets.QApplication(sys.argv)
    game = TicTacToe(model, device)
    game.show()
    sys.exit(app.exec_())