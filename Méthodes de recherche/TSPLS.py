import numpy as np
import random

def TSP_Local_Search(distance_matrix, starting_city=0):
    # Initialize the route with the starting city
    current_route = [starting_city]
    unvisited_cities = set(range(distance_matrix.shape[0]))
    unvisited_cities.remove(starting_city)
    
    while unvisited_cities:
        last_city = current_route[-1]
        next_city = min(unvisited_cities, key=lambda city: distance_matrix[last_city, city])
        current_route.append(next_city)
        unvisited_cities.remove(next_city)
        
        for i in range(1, len(current_route) - 1):
            for j in range(i + 1, len(current_route)):
                new_route = current_route[:]
                new_route[i:j] = reversed(new_route[i:j])
                new_distance = sum(distance_matrix[current_route[i - 1], current_route[i]] for i in range(1, len(current_route)))
                if new_distance < current_distance:
                    current_route = new_route
                    current_distance = new_distance
    
    return current_route
