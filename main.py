from src.simulation import Simulation

def main():
    sim = Simulation()
    sim.create_cars(10)
    sim.create_servers(10)
    sim.load_tasks('data/task_data.csv')
    sim.offload_tasks()

if __name__ == "__main__":
    main()