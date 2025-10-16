import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGame, Direction, Point
from q_model import DQN, DQNTrainer
from plot_helper import plot_graph

MEMORY_CAPACITY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001


class SnakeAgent:

    def __init__(self):
        self.num_games = 0
        self.exploration_rate = 0  # randomness factor
        self.discount_rate = 0.9  # discount factor for future rewards
        self.memory = deque(maxlen=MEMORY_CAPACITY)  # memory buffer
        self.model = DQN(19, 256, 3)
        self.trainer = DQNTrainer(self.model, lr=LEARNING_RATE, gamma=self.discount_rate)

    def get_state(self, game):
        head = game.snake[0]
        points = {
            'left': Point(head.x - 20, head.y),
            'right': Point(head.x + 20, head.y),
            'up': Point(head.x, head.y - 20),
            'down': Point(head.x, head.y + 20)
        }

        directions = {
            'left': game.direction == Direction.LEFT,
            'right': game.direction == Direction.RIGHT,
            'up': game.direction == Direction.UP,
            'down': game.direction == Direction.DOWN
        }

        state = [
            # Danger straight
            (directions['right'] and game.check_collision(points['right'])) or
            (directions['left'] and game.check_collision(points['left'])) or
            (directions['up'] and game.check_collision(points['up'])) or
            (directions['down'] and game.check_collision(points['down'])),

            # Danger right
            (directions['up'] and game.check_collision(points['right'])) or
            (directions['down'] and game.check_collision(points['left'])) or
            (directions['left'] and game.check_collision(points['up'])) or
            (directions['right'] and game.check_collision(points['down'])),

            # Danger left
            (directions['down'] and game.check_collision(points['right'])) or
            (directions['up'] and game.check_collision(points['left'])) or
            (directions['right'] and game.check_collision(points['up'])) or
            (directions['left'] and game.check_collision(points['down'])),

            # Movement direction
            directions['left'],
            directions['right'],
            directions['up'],
            directions['down'],

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y,  # food down
        ]

        for pos in points.values():
            state.append(pos in game.snake)
            state.append(pos == game.food)

        return np.array(state, dtype=int)

    def store_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))  # append memory

    def train_memory_batch(self):
        if len(self.memory) > BATCH_SIZE:
            mini_batch = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_batch = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_single_step(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def select_action(self, state):
        self.exploration_rate = 80 - self.num_games
        action = [0, 0, 0]
        if random.randint(0, 200) < self.exploration_rate:
            move = random.randint(0, 2)
            action[move] = 1
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move = torch.argmax(prediction).item()
            action[move] = 1

        return action


def train_agent():
    scores = []
    mean_scores = []
    total_score = 0
    record_score = 0
    agent = SnakeAgent()
    game = SnakeGame()

    while True:
        old_state = agent.get_state(game)
        action = agent.select_action(old_state)
        reward, game_over, score = game.play_step(action)
        new_state = agent.get_state(game)
        agent.train_single_step(old_state, action, reward, new_state, game_over)
        agent.store_memory(old_state, action, reward, new_state, game_over)

        if game_over:
            game.reset()
            agent.num_games += 1
            agent.train_memory_batch()

            if score > record_score:
                record_score = score
                agent.model.save_model()

            print(f'Game {agent.num_games} | Score: {score} | Record: {record_score}')

            scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            mean_scores.append(mean_score)
            plot_graph(scores, mean_scores)


if __name__ == '__main__':
    train_agent()