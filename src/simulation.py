import random
import pandas as pd
from src.car import Car
from src.server import Server
from src.task import Task
from src.q_learning import QLearning
from src.utils import calculate_distance

class Simulation:
    def __init__(self):
        self.cars = []
        self.servers = []
        self.q_learning = QLearning(num_states=10000, num_actions=10)  # Increased state space

    def create_cars(self, num_cars):
        for i in range(num_cars):
            car = Car(i, random.randint(0, 100), random.randint(0, 100))
            self.cars.append(car)

    def create_servers(self, num_servers):
        for i in range(num_servers):
            server = Server(i, random.randint(0, 100), random.randint(0, 100), random.randint(8, 32), random.randint(100, 500))
            self.servers.append(server)
            print(f"Server {server.id} is at location ({server.x}, {server.y})")

    def load_tasks(self, file_path):
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            car_id = row['car_id']
            task = Task(row['task_id'], row['car_x'], row['car_y'], row['task_ram'], row['task_rom'], row['priority'])
            self.cars[car_id].tasks.append(task)

    def offload_tasks(self):
        for car in self.cars:
            print(f"\nCar {car.id} at location ({car.x}, {car.y}):")
            for task in car.tasks:
                print(f"\n  Task {task.id} (RAM: {task.ram}, ROM: {task.rom}, Priority: {task.priority})")
                state = self.get_state(car, task)
                action = self.q_learning.choose_action(state)
                server = self.servers[action]
                if server.can_handle_task(task):
                    print(f"    Attempting to offload to Server {server.id} (Available RAM: {server.available_ram}, Available ROM: {server.available_rom})")
                    server.available_ram -= task.ram
                    server.available_rom -= task.rom
                    reward = 1  # Reward for successful offloading
                    print(f"    Successfully offloaded to Server {server.id} (Remaining RAM: {server.available_ram}, Remaining ROM: {server.available_rom})")
                else:
                    # Try to find the nearest server that can handle the task
                    nearest_server = self.find_nearest_server(car, task)
                    if nearest_server:
                        print(f"    Server {server.id} cannot handle the task. Trying nearest Server {nearest_server.id} (Available RAM: {nearest_server.available_ram}, Available ROM: {nearest_server.available_rom})")
                        nearest_server.available_ram -= task.ram
                        nearest_server.available_rom -= task.rom
                        reward = 1  # Reward for successful offloading
                        print(f"    Successfully offloaded to nearest Server {nearest_server.id} (Remaining RAM: {nearest_server.available_ram}, Remaining ROM: {nearest_server.available_rom})")
                    else:
                        reward = -1  # Penalty for unsuccessful offloading
                        print(f"    Could not offload to any server - Insufficient resources")
                next_state = self.get_state(car, task)
                self.q_learning.update_q_table(state, action, reward, next_state)

    def find_nearest_server(self, car, task):
        suitable_servers = [server for server in self.servers if server.can_handle_task(task)]
        if not suitable_servers:
            return None
        nearest_server = min(suitable_servers, key=lambda server: calculate_distance(car.x, car.y, server.x, server.y))
        return nearest_server

    def get_state(self, car, task):
        # State representation includes car coordinates, task requirements, and server resources
        state = (car.x + car.y + task.ram + task.rom) % 100
        for server in self.servers:
            state = (state + server.available_ram + server.available_rom + calculate_distance(car.x, car.y, server.x, server.y)) % 10000
        return int(state)  # Ensure state is an integer

    @staticmethod
    def calculate_distance(car, server):
        return calculate_distance(car.x, car.y, server.x, server.y)
