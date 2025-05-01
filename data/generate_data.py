#!/usr/bin/env python3
"""
Data generation module for food delivery RL simulation.
Generates synthetic data for restaurants, orders, drivers, and road network.
"""

import os
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import networkx as nx
import argparse

np.random.seed(42)

def generate_restaurants(n_restaurants=200, city_size=10):
    """
    Generate restaurant data with clustering in commercial zones.
    
    Args:
        n_restaurants: Number of restaurants to generate
        city_size: Size of the city grid in km
        
    Returns:
        DataFrame with restaurant data
    """
    # Create commercial zones (clusters)
    n_zones = 5
    zone_centers = np.random.uniform(1, city_size-1, size=(n_zones, 2))
    
    restaurants = []
    for i in range(n_restaurants):
        # Select a random zone
        zone = np.random.randint(0, n_zones)
        # Generate location with clustering around zone center
        lat = np.random.normal(zone_centers[zone, 0], 0.5)
        lng = np.random.normal(zone_centers[zone, 1], 0.5)
        # Ensure within city bounds
        lat = max(0, min(city_size, lat))
        lng = max(0, min(city_size, lng))
        # Average preparation time (minutes)
        avg_prep_time = np.random.uniform(5, 20)
        
        restaurants.append({
            'restaurant_id': i,
            'lat': lat,
            'lng': lng,
            'avg_prep_time': avg_prep_time
        })
    
    return pd.DataFrame(restaurants)

def generate_drivers(n_drivers=50, city_size=10, day_start=10, day_end=22):
    """
    Generate driver data with shift patterns.
    
    Args:
        n_drivers: Number of drivers to generate
        city_size: Size of the city grid in km
        day_start: Starting hour of the day (24h format)
        day_end: Ending hour of the day (24h format)
        
    Returns:
        DataFrame with driver data
    """
    drivers = []
    
    # Generate more drivers during peak hours
    morning_shift = int(n_drivers * 0.4)  # 40% morning/afternoon
    evening_shift = int(n_drivers * 0.4)  # 40% evening
    full_day = n_drivers - morning_shift - evening_shift  # 20% full day
    
    driver_id = 0
    
    # Morning/afternoon shift (10:00-16:00)
    for _ in range(morning_shift):
        lat = np.random.uniform(0, city_size)
        lng = np.random.uniform(0, city_size)
        shift_start = day_start
        shift_duration = np.random.uniform(4, 6)  # 4-6 hour shifts
        shift_end = min(shift_start + shift_duration, day_end)
        
        drivers.append({
            'driver_id': driver_id,
            'home_lat': lat,
            'home_lng': lng,
            'shift_start': shift_start,
            'shift_end': shift_end,
            'capacity': 1  # Single order capacity for now
        })
        driver_id += 1
    
    # Evening shift (16:00-22:00)
    for _ in range(evening_shift):
        lat = np.random.uniform(0, city_size)
        lng = np.random.uniform(0, city_size)
        shift_start = 16
        shift_end = day_end
        
        drivers.append({
            'driver_id': driver_id,
            'home_lat': lat,
            'home_lng': lng,
            'shift_start': shift_start,
            'shift_end': shift_end,
            'capacity': 1
        })
        driver_id += 1
    
    # Full day shift (10:00-22:00 with breaks)
    for _ in range(full_day):
        lat = np.random.uniform(0, city_size)
        lng = np.random.uniform(0, city_size)
        shift_start = day_start
        shift_end = day_end
        
        drivers.append({
            'driver_id': driver_id,
            'home_lat': lat,
            'home_lng': lng,
            'shift_start': shift_start,
            'shift_end': shift_end,
            'capacity': 1
        })
        driver_id += 1
    
    return pd.DataFrame(drivers)

def generate_orders(restaurants_df, n_orders=1000, day_start=10, day_end=22, city_size=10):
    """
    Generate order data using inhomogeneous Poisson process for time distribution.
    
    Args:
        restaurants_df: DataFrame containing restaurant information
        n_orders: Number of orders to generate
        day_start: Starting hour of the day (24h format)
        day_end: Ending hour of the day (24h format)
        city_size: Size of the city grid in km
        
    Returns:
        DataFrame with order data
    """
    orders = []
    total_hours = day_end - day_start
    
    # Create rate function for inhomogeneous Poisson process
    # Higher rates during lunch (12-14) and dinner (18-20) hours
    def rate_function(hour):
        # Base rate
        rate = 1.0
        # Lunch peak
        if 12 <= hour < 14:
            rate = 3.0
        # Dinner peak
        elif 18 <= hour < 20:
            rate = 4.0
        return rate
    
    # Generate order times using rejection sampling
    max_rate = 4.0  # Maximum rate
    order_times = []
    
    while len(order_times) < n_orders:
        candidate_time = np.random.uniform(day_start, day_end)
        if np.random.uniform(0, max_rate) <= rate_function(candidate_time):
            order_times.append(candidate_time)
    
    # Sort times
    order_times.sort()
    
    # Generate orders
    for i in range(n_orders):
        # Random restaurant
        restaurant_idx = np.random.randint(0, len(restaurants_df))
        restaurant = restaurants_df.iloc[restaurant_idx]
        
        # Customer location (random in city)
        customer_lat = np.random.uniform(0, city_size)
        customer_lng = np.random.uniform(0, city_size)
        
        # Order time (hour of day)
        order_time = order_times[i]
        
        # Preparation time (based on restaurant's average with some variance)
        prep_time = np.random.normal(restaurant['avg_prep_time'], 2)
        prep_time = max(5, min(20, prep_time))  # Bound between 5-20 minutes
        
        # Promised delivery window (30-45 minutes from order time)
        promised_window = 45  # minutes from order time
        
        orders.append({
            'order_id': i,
            'restaurant_id': restaurant['restaurant_id'],
            'customer_lat': customer_lat,
            'customer_lng': customer_lng,
            'order_time': order_time,
            'prep_time': prep_time,
            'promised_window': promised_window
        })
    
    return pd.DataFrame(orders)

def generate_road_network(city_size=10, grid_size=20):
    """
    Generate a grid-based road network with time-of-day speed multipliers.
    
    Args:
        city_size: Size of the city in km
        grid_size: Number of grid points in each dimension
        
    Returns:
        Dictionary containing road network information
    """
    G = nx.grid_2d_graph(grid_size, grid_size)
    
    # Convert to more usable format with real coordinates
    scale_factor = city_size / (grid_size - 1)
    
    nodes = {}
    for node in G.nodes():
        x, y = node
        nodes[f"node_{x}_{y}"] = {
            "id": f"node_{x}_{y}",
            "lat": x * scale_factor,
            "lng": y * scale_factor
        }
    
    edges = []
    for edge in G.edges():
        (x1, y1), (x2, y2) = edge
        # Calculate base travel time (proportional to Manhattan distance)
        distance = abs(x2 - x1) + abs(y2 - y1)
        base_time = distance * scale_factor * 3  # 3 minutes per km
        
        # Add time-of-day multipliers
        # Morning rush hour: 8-9am
        # Evening rush hour: 5-6pm
        time_multipliers = {
            # Hour: multiplier
            8: 1.5,  # Morning rush hour
            9: 1.3,
            10: 1.0,
            11: 1.0,
            12: 1.2,  # Lunch time
            13: 1.2,
            14: 1.0,
            15: 1.0,
            16: 1.1,
            17: 1.5,  # Evening rush hour
            18: 1.3,
            19: 1.1,
            20: 1.0,
            21: 0.9,
            22: 0.8,
        }
        
        edges.append({
            "from": f"node_{x1}_{y1}",
            "to": f"node_{x2}_{y2}",
            "distance": distance * scale_factor,
            "base_time": base_time,
            "time_multipliers": time_multipliers
        })
    
    return {
        "nodes": nodes,
        "edges": edges
    }

def main(args):
    """Main function to generate and save all datasets."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Generating restaurants...")
    restaurants = generate_restaurants(n_restaurants=args.n_restaurants, city_size=args.city_size)
    restaurants.to_csv(os.path.join(args.output_dir, 'restaurants.csv'), index=False)
    
    print("Generating drivers...")
    drivers = generate_drivers(n_drivers=args.n_drivers, city_size=args.city_size, 
                              day_start=args.day_start, day_end=args.day_end)
    drivers.to_csv(os.path.join(args.output_dir, 'drivers.csv'), index=False)
    
    if args.generate_all:
        print("Generating orders...")
        orders = generate_orders(restaurants, n_orders=args.n_orders, day_start=args.day_start, 
                              day_end=args.day_end, city_size=args.city_size)
        orders.to_csv(os.path.join(args.output_dir, 'orders.csv'), index=False)
        
        print("Generating road network...")
        road_network = generate_road_network(city_size=args.city_size, grid_size=args.grid_size)
        with open(os.path.join(args.output_dir, 'road_network.json'), 'w') as f:
            json.dump(road_network, f)
    else:
        # For large-scale simulation, generate orders in smaller batches
        print(f"Generating {args.n_orders} orders in batches...")
        batch_size = min(100000, args.n_orders)
        num_batches = (args.n_orders + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches)):
            current_batch_size = min(batch_size, args.n_orders - i * batch_size)
            orders = generate_orders(restaurants, n_orders=current_batch_size, 
                                   day_start=args.day_start, day_end=args.day_end, 
                                   city_size=args.city_size)
            
            # For the first batch, write headers
            if i == 0:
                orders.to_csv(os.path.join(args.output_dir, 'orders.csv'), index=False)
            else:
                # Append without headers
                orders.to_csv(os.path.join(args.output_dir, 'orders.csv'), 
                            mode='a', header=False, index=False)
        
        print("Generating road network...")
        road_network = generate_road_network(city_size=args.city_size, grid_size=args.grid_size)
        with open(os.path.join(args.output_dir, 'road_network.json'), 'w') as f:
            json.dump(road_network, f)
    
    print("Data generation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data for food delivery simulation")
    parser.add_argument("--output_dir", type=str, default="../data", help="Output directory for data files")
    parser.add_argument("--n_restaurants", type=int, default=200, help="Number of restaurants")
    parser.add_argument("--n_drivers", type=int, default=50, help="Number of drivers")
    parser.add_argument("--n_orders", type=int, default=10000, help="Number of orders")
    parser.add_argument("--city_size", type=float, default=10.0, help="City size in km")
    parser.add_argument("--grid_size", type=int, default=20, help="Road network grid size")
    parser.add_argument("--day_start", type=int, default=10, help="Start hour of the day (24h format)")
    parser.add_argument("--day_end", type=int, default=22, help="End hour of the day (24h format)")
    parser.add_argument("--generate_all", action="store_true", help="Generate all data at once (not in batches)")
    
    args = parser.parse_args()
    main(args) 