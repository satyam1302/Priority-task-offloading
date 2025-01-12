import random
from src.task import Task

class Car:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
        self.tasks = []

    def generate_tasks(self, num_tasks):
        for i in range(num_tasks):
            task = Task(f"Task_{self.id}_{i}", self.x, self.y, random.randint(1, 8), random.randint(10, 50), random.randint(1, 5))
            self.tasks.append(task)