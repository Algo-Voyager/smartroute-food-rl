#!/usr/bin/env python3
"""
Baseline agents for the food delivery RL system.

This module implements baseline heuristic policies for comparison with RL:
1. Nearest-driver heuristic: assign to closest free driver
2. Fixed-cutoff policy: reject if estimated delivery time > 45 min
"""

import numpy as np
import math

class NearestDriverAgent:
    """
    Nearest-driver heuristic policy.
    
    This agent assigns each order to the closest available driver.
    If no drivers are available, it rejects the order.
    """
    
    def __init__(self, env):
        """
        Initialize the nearest-driver agent.
        
        Args:
            env: DeliveryEnv instance
        """
        self.env = env
        self.max_drivers = env.max_drivers
        self.max_orders = env.max_orders
    
    def act(self, observation):
        """
        Choose an action based on the nearest-driver heuristic.
        
        Args:
            observation: Environment observation
            
        Returns:
            action: Action to take
        """
        time = observation['time'][0]
        orders = observation['orders']
        drivers = observation['drivers']
        
        # Find active orders (status 0=prepping or 1=ready)
        active_order_indices = []
        for i in range(len(orders)):
            status = orders[i, 6]
            if status in [0, 1]:  # prepping or ready
                active_order_indices.append(i)
        
        # If no active orders, return a no-op action
        if not active_order_indices:
            return 0  # No-op (will be handled as invalid by the environment)
        
        # Find free drivers (status 1=free)
        free_driver_indices = []
        for i in range(len(drivers)):
            status = drivers[i, 2]
            # Check if driver row is valid (non-zero) and status is free
            if status == 1 and np.any(drivers[i]):
                free_driver_indices.append(i)
        
        # If no free drivers, reject the first active order
        if not free_driver_indices:
            # Calculate the action index for rejecting the first active order
            reject_action = self.max_drivers * self.max_orders + self.max_drivers * 4 + active_order_indices[0]
            return reject_action
        
        # For each active order, find the nearest free driver
        best_order_idx = None
        best_driver_idx = None
        min_distance = float('inf')
        
        for order_idx in active_order_indices:
            restaurant_lat = orders[order_idx, 0]
            restaurant_lng = orders[order_idx, 1]
            
            for driver_idx in free_driver_indices:
                driver_lat = drivers[driver_idx, 0]
                driver_lng = drivers[driver_idx, 1]
                
                # Calculate Manhattan distance
                distance = abs(driver_lat - restaurant_lat) + abs(driver_lng - restaurant_lng)
                
                if distance < min_distance:
                    min_distance = distance
                    best_order_idx = order_idx
                    best_driver_idx = driver_idx
        
        # Calculate action index for assigning this order to this driver
        assign_action = best_driver_idx * self.max_orders + best_order_idx
        
        return assign_action

class FixedCutoffAgent:
    """
    Fixed-cutoff policy.
    
    This agent:
    1. Rejects orders if estimated delivery time > 45 minutes
    2. Otherwise assigns to the nearest driver
    3. Drivers return to depot after each delivery
    """
    
    def __init__(self, env, cutoff_minutes=45, avg_speed=30):
        """
        Initialize the fixed-cutoff agent.
        
        Args:
            env: DeliveryEnv instance
            cutoff_minutes: Reject threshold in minutes
            avg_speed: Average vehicle speed in km/h
        """
        self.env = env
        self.max_drivers = env.max_drivers
        self.max_orders = env.max_orders
        self.cutoff_minutes = cutoff_minutes
        self.avg_speed = avg_speed
        
        # Define depot location (center of the city)
        self.depot_lat = env.city_size / 2
        self.depot_lng = env.city_size / 2
    
    def act(self, observation):
        """
        Choose an action based on the fixed-cutoff policy.
        
        Args:
            observation: Environment observation
            
        Returns:
            action: Action to take
        """
        time = observation['time'][0]
        orders = observation['orders']
        drivers = observation['drivers']
        
        # Find active orders (status 0=prepping or 1=ready)
        active_order_indices = []
        for i in range(len(orders)):
            status = orders[i, 6]
            if status in [0, 1] and np.any(orders[i]):  # Check if row is valid
                active_order_indices.append(i)
        
        # If no active orders, reposition a driver toward the depot if needed
        if not active_order_indices:
            return self._reposition_driver_to_depot(drivers)
        
        # Find free drivers (status 1=free)
        free_driver_indices = []
        for i in range(len(drivers)):
            status = drivers[i, 2]
            # Check if driver row is valid (non-zero) and status is free
            if status == 1 and np.any(drivers[i]):
                free_driver_indices.append(i)
        
        # If no free drivers, reject the first active order
        if not free_driver_indices:
            # Calculate the action index for rejecting the first active order
            reject_action = self.max_drivers * self.max_orders + self.max_drivers * 4 + active_order_indices[0]
            return reject_action
        
        # For each active order, estimate delivery time
        valid_assignments = []
        
        for order_idx in active_order_indices:
            restaurant_lat = orders[order_idx, 0]
            restaurant_lng = orders[order_idx, 1]
            customer_lat = orders[order_idx, 2]
            customer_lng = orders[order_idx, 3]
            time_to_ready = orders[order_idx, 4]  # in hours
            
            for driver_idx in free_driver_indices:
                driver_lat = drivers[driver_idx, 0]
                driver_lng = drivers[driver_idx, 1]
                
                # Calculate Manhattan distances
                to_restaurant_distance = abs(driver_lat - restaurant_lat) + abs(driver_lng - restaurant_lng)
                to_customer_distance = abs(restaurant_lat - customer_lat) + abs(restaurant_lng - customer_lng)
                
                # Time multiplier based on time of day
                hour = int(time)
                time_multiplier = 1.0
                if hour in [8, 9]:  # Morning rush hour
                    time_multiplier = 1.5
                elif hour in [17, 18]:  # Evening rush hour
                    time_multiplier = 1.5
                elif hour in [12, 13]:  # Lunch time
                    time_multiplier = 1.2
                
                # Calculate travel times in hours
                to_restaurant_time = (to_restaurant_distance / self.avg_speed) * time_multiplier
                to_customer_time = (to_customer_distance / self.avg_speed) * time_multiplier
                
                # Total estimated delivery time in minutes
                # Time to reach restaurant + max(0, time until order is ready) + time to reach customer
                wait_time = max(0, time_to_ready)
                estimated_delivery_time = (to_restaurant_time + wait_time + to_customer_time) * 60
                
                # If within cutoff, add to valid assignments
                if estimated_delivery_time <= self.cutoff_minutes:
                    valid_assignments.append((order_idx, driver_idx, estimated_delivery_time))
        
        # If no valid assignments, reject the first order
        if not valid_assignments:
            reject_action = self.max_drivers * self.max_orders + self.max_drivers * 4 + active_order_indices[0]
            return reject_action
        
        # Find the assignment with the shortest estimated delivery time
        best_assignment = min(valid_assignments, key=lambda x: x[2])
        order_idx, driver_idx, _ = best_assignment
        
        # Calculate action index for assigning this order to this driver
        assign_action = driver_idx * self.max_orders + order_idx
        
        return assign_action
    
    def _reposition_driver_to_depot(self, drivers):
        """
        Choose a free driver to reposition toward the depot.
        
        Args:
            drivers: Driver observation array
            
        Returns:
            action: Reposition action or no-op
        """
        # Find free drivers (status 1=free)
        free_driver_indices = []
        for i in range(len(drivers)):
            status = drivers[i, 2]
            # Check if driver row is valid (non-zero) and status is free
            if status == 1 and np.any(drivers[i]):
                free_driver_indices.append(i)
        
        if not free_driver_indices:
            return 0  # No-op (will be handled as invalid by the environment)
        
        # Find the driver furthest from depot
        max_distance = -1
        furthest_driver = None
        
        for driver_idx in free_driver_indices:
            driver_lat = drivers[driver_idx, 0]
            driver_lng = drivers[driver_idx, 1]
            
            # Calculate Manhattan distance to depot
            distance = abs(driver_lat - self.depot_lat) + abs(driver_lng - self.depot_lng)
            
            if distance > max_distance:
                max_distance = distance
                furthest_driver = driver_idx
        
        # If all drivers are at the depot, return no-op
        if max_distance < 0.1:
            return 0  # No-op
        
        # Determine direction to move (toward depot)
        driver_lat = drivers[furthest_driver, 0]
        driver_lng = drivers[furthest_driver, 1]
        
        # Choose direction (N=0, E=1, S=2, W=3)
        lat_diff = self.depot_lat - driver_lat
        lng_diff = self.depot_lng - driver_lng
        
        # Move in the direction of largest difference
        if abs(lat_diff) > abs(lng_diff):
            direction = 0 if lat_diff > 0 else 2  # North or South
        else:
            direction = 1 if lng_diff > 0 else 3  # East or West
        
        # Calculate action index for repositioning
        reposition_action = self.max_drivers * self.max_orders + furthest_driver * 4 + direction
        
        return reposition_action 