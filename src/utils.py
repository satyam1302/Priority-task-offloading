import math

def calculate_distance(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two points (x1, y1) and (x2, y2).
    """
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def find_nearest_server(car, servers):
    """
    Find the nearest server to the given car that can handle the task.
    """
    nearest_server = None
    min_distance = float('inf')
    for server in servers:
        distance = calculate_distance(car.x, car.y, server.x, server.y)
        if distance < min_distance:
            min_distance = distance
            nearest_server = server
    return nearest_server