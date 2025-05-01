#!/usr/bin/env python3
"""
Delivery Environment for Food Delivery RL System.

This module implements an OpenAI Gym-compatible environment that simulates
a food delivery system with orders, drivers, and restaurants.
"""

import os
import numpy as np
import pandas as pd
import json
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
from collections import deque, defaultdict
import heapq
import math

class DeliveryEnv(gym.Env):
    """
    Food Delivery Environment implementing the OpenAI Gym interface.
    
    State:
        - Current timestamp
        - List of "ready" versus "prepping" orders (location, time-to-ready, deadline)
        - Each driver's location, load, busy/unbusy status, shift end
    
    Action space:
        - At each order-arrival or driver-free event, choose "assign order → driver_i,"
          "reject order," or "reposition driver"
    
    Dynamics:
        - Orders join "prep queue" until prep_time expires
        - Travel times computed via road graph or Manhattan distance + time-of-day multiplier
        - Drivers carry one order at a time; shifts enforced
        - Episodes run one "day" (e.g. 10:00–22:00)
    
    Reward:
        - On delivery: 45 min − actual_delivery_time (negative if >45)
        - Reject penalty = –10
        - Idle/reposition penalty = –1 per move
        - Invalid action penalty = –5
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, 
                 data_dir="../data",
                 restaurants_file="restaurants.csv",
                 drivers_file="drivers.csv",
                 orders_file="orders.csv",
                 road_network_file="road_network.json",
                 day_start=10,
                 day_end=22,
                 city_size=10,
                 use_road_network=True,
                 avg_speed=30,  # km/h
                 max_orders=100,
                 max_drivers=50,
                 render_mode=None):
        """
        Initialize the delivery environment.
        
        Args:
            data_dir: Directory containing data files
            restaurants_file: CSV file with restaurant data
            drivers_file: CSV file with driver data
            orders_file: CSV file with order data
            road_network_file: JSON file with road network data
            day_start: Starting hour of the day (24h format)
            day_end: Ending hour of the day (24h format)
            city_size: Size of the city grid in km
            use_road_network: Whether to use road network for travel times
            avg_speed: Average vehicle speed in km/h
            max_orders: Maximum number of orders in the environment at once
            max_drivers: Maximum number of drivers in the environment
            render_mode: Rendering mode ("human" or "rgb_array")
        """
        super().__init__()
        
        # Environment parameters
        self.data_dir = data_dir
        self.day_start = day_start
        self.day_end = day_end
        self.city_size = city_size
        self.use_road_network = use_road_network
        self.avg_speed = avg_speed  # km/h
        self.max_orders = max_orders
        self.max_drivers = max_drivers
        self.render_mode = render_mode
        
        # Load data
        self.restaurants_df = pd.read_csv(os.path.join(data_dir, restaurants_file))
        self.drivers_df = pd.read_csv(os.path.join(data_dir, drivers_file))
        self.orders_df = pd.read_csv(os.path.join(data_dir, orders_file))
        
        # Sort orders by order time
        self.orders_df = self.orders_df.sort_values(by='order_time')
        
        # Load road network if available and enabled
        self.road_network = None
        self.road_graph = None
        if use_road_network and os.path.exists(os.path.join(data_dir, road_network_file)):
            with open(os.path.join(data_dir, road_network_file), 'r') as f:
                self.road_network = json.load(f)
            
            # Create networkx graph from road network
            self.road_graph = nx.DiGraph()
            
            # Add nodes
            for node_id, node_data in self.road_network['nodes'].items():
                self.road_graph.add_node(node_id, 
                                        lat=node_data['lat'], 
                                        lng=node_data['lng'])
            
            # Add edges
            for edge in self.road_network['edges']:
                self.road_graph.add_edge(edge['from'], edge['to'], 
                                        distance=edge['distance'],
                                        base_time=edge['base_time'],
                                        time_multipliers=edge['time_multipliers'])
        
        # Define action and observation spaces
        
        # Action space:
        # - For each driver (max_drivers), we can assign any order (max_orders)
        # - For each driver, we can reposition to any of 4 directions (N, E, S, W)
        # - For each order, we can reject it
        # Total: max_drivers * max_orders (assign) + max_drivers * 4 (reposition) + max_orders (reject)
        self.action_space = spaces.Discrete(
            max_drivers * max_orders +  # Assign order to driver
            max_drivers * 4 +           # Reposition driver (N, E, S, W)
            max_orders                   # Reject order
        )
        
        # Observation space:
        # - Current time (1)
        # - For each order (max_orders):
        #   - Restaurant location (2: lat, lng)
        #   - Customer location (2: lat, lng)
        #   - Time to ready (1)
        #   - Deadline (1)
        #   - Status (1: prepping, ready, assigned, delivered, rejected)
        # - For each driver (max_drivers):
        #   - Location (2: lat, lng)
        #   - Status (1: busy, free)
        #   - Current order id (1: -1 if none)
        #   - Shift end (1)
        
        order_features = 7  # restaurant_loc(2) + customer_loc(2) + time_to_ready(1) + deadline(1) + status(1)
        driver_features = 5  # location(2) + status(1) + current_order_id(1) + shift_end(1)
        
        self.observation_space = spaces.Dict({
            'time': spaces.Box(low=day_start, high=day_end, shape=(1,), dtype=np.float32),
            'orders': spaces.Box(low=-np.inf, high=np.inf, shape=(max_orders, order_features), dtype=np.float32),
            'drivers': spaces.Box(low=-np.inf, high=np.inf, shape=(max_drivers, driver_features), dtype=np.float32),
        })
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to the beginning of the day.
        
        Returns:
            obs: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        self.current_time = self.day_start
        self.orders = {}  # order_id -> order_info
        self.drivers = {}  # driver_id -> driver_info
        self.active_orders = []  # orders that need action (assignment/rejection)
        self.prepping_orders = []  # orders in preparation
        self.ready_orders = []  # orders ready for pickup
        self.assigned_orders = {}  # order_id -> driver_id
        self.completed_orders = []  # orders that have been delivered
        self.rejected_orders = []  # orders that have been rejected
        
        # Event queue for simulation
        self.event_queue = []  # (time, event_type, event_data)
        
        # Load initial drivers (those available at start time)
        for _, driver_row in self.drivers_df.iterrows():
            if driver_row['shift_start'] <= self.day_start:
                driver_id = int(driver_row['driver_id'])
                self.drivers[driver_id] = {
                    'id': driver_id,
                    'location': (driver_row['home_lat'], driver_row['home_lng']),
                    'status': 'free',
                    'current_order_id': -1,
                    'shift_start': driver_row['shift_start'],
                    'shift_end': driver_row['shift_end'],
                    'capacity': driver_row['capacity']
                }
        
        # Schedule driver shift starts and ends
        for _, driver_row in self.drivers_df.iterrows():
            driver_id = int(driver_row['driver_id'])
            
            # Schedule shift start if after day_start
            if driver_row['shift_start'] > self.day_start:
                heapq.heappush(self.event_queue, 
                              (driver_row['shift_start'], 'driver_shift_start', driver_id))
            
            # Schedule shift end
            heapq.heappush(self.event_queue, 
                          (driver_row['shift_end'], 'driver_shift_end', driver_id))
        
        # Schedule order arrivals
        for _, order_row in self.orders_df.iterrows():
            order_time = order_row['order_time']
            if self.day_start <= order_time < self.day_end:
                order_id = int(order_row['order_id'])
                heapq.heappush(self.event_queue, (order_time, 'order_arrival', order_id))
        
        # Metrics
        self.total_delivery_time = 0
        self.total_deliveries = 0
        self.total_rejects = 0
        self.total_repositions = 0
        self.total_reward = 0
        
        # Process events until the first action-requiring event
        obs, _, _, _ = self._process_events_until_action_required()
        
        info = {}
        return obs, info
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take:
                - 0 to max_drivers*max_orders-1: Assign order to driver
                - max_drivers*max_orders to max_drivers*max_orders+max_drivers*4-1: Reposition driver
                - max_drivers*max_orders+max_drivers*4 to max_drivers*max_orders+max_drivers*4+max_orders-1: Reject order
        
        Returns:
            obs: New observation
            reward: Reward from the action
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Decode action
        reward = 0
        
        # Assign order to driver
        if action < self.max_drivers * self.max_orders:
            driver_idx = action // self.max_orders
            order_idx = action % self.max_orders
            
            # Validate driver index
            if driver_idx >= len(self.drivers):
                reward = -5  # Invalid action penalty
            else:
                driver_id = list(self.drivers.keys())[driver_idx]
                
                # Validate order index
                if order_idx >= len(self.active_orders):
                    reward = -5  # Invalid action penalty
                else:
                    order_id = self.active_orders[order_idx]
                    reward = self._assign_order_to_driver(order_id, driver_id)
        
        # Reposition driver
        elif action < self.max_drivers * self.max_orders + self.max_drivers * 4:
            action_offset = action - self.max_drivers * self.max_orders
            driver_idx = action_offset // 4
            direction = action_offset % 4  # 0: North, 1: East, 2: South, 3: West
            
            # Validate driver index
            if driver_idx >= len(self.drivers):
                reward = -5  # Invalid action penalty
            else:
                driver_id = list(self.drivers.keys())[driver_idx]
                reward = self._reposition_driver(driver_id, direction)
        
        # Reject order
        else:
            action_offset = action - (self.max_drivers * self.max_orders + self.max_drivers * 4)
            
            # Validate order index
            if action_offset >= len(self.active_orders):
                reward = -5  # Invalid action penalty
            else:
                order_id = self.active_orders[action_offset]
                reward = self._reject_order(order_id)
        
        # Process events until the next action-requiring event
        obs, terminated, truncated, info = self._process_events_until_action_required()
        
        self.total_reward += reward
        
        return obs, reward, terminated, truncated, info
    
    def _assign_order_to_driver(self, order_id, driver_id):
        """Assign an order to a driver."""
        if order_id not in self.orders or driver_id not in self.drivers:
            return -5  # Invalid order or driver
        
        order = self.orders[order_id]
        driver = self.drivers[driver_id]
        
        if driver['status'] != 'free':
            return -5  # Driver not available
        
        if order['status'] not in ['ready', 'prepping']:
            return -5  # Order not ready or prepping
        
        # Mark the driver as busy
        driver['status'] = 'busy'
        driver['current_order_id'] = order_id
        
        # Mark the order as assigned
        order['status'] = 'assigned'
        order['driver_id'] = driver_id
        self.assigned_orders[order_id] = driver_id
        
        # Remove from active orders if it was there
        if order_id in self.active_orders:
            self.active_orders.remove(order_id)
        
        # Calculate pickup and delivery times
        restaurant_location = (order['restaurant_lat'], order['restaurant_lng'])
        customer_location = (order['customer_lat'], order['customer_lng'])
        
        # Time to pickup
        pickup_travel_time = self._calculate_travel_time(
            driver['location'], restaurant_location, self.current_time)
        
        pickup_time = self.current_time + pickup_travel_time
        
        # Time to delivery depends on whether order is ready
        if order['status'] == 'ready':
            # Order is ready, driver can pick up right away
            ready_time = self.current_time
        else:
            # Order is still prepping, driver must wait
            ready_time = order['ready_time']
        
        # Actual pickup time is the maximum of when driver arrives and when food is ready
        actual_pickup_time = max(pickup_time, ready_time)
        
        # Time from restaurant to customer
        delivery_travel_time = self._calculate_travel_time(
            restaurant_location, customer_location, actual_pickup_time)
        
        delivery_time = actual_pickup_time + delivery_travel_time
        
        # Schedule pickup and delivery events
        heapq.heappush(self.event_queue, 
                      (pickup_time, 'driver_arrives_restaurant', {'driver_id': driver_id, 'order_id': order_id}))
        
        heapq.heappush(self.event_queue, 
                      (delivery_time, 'order_delivered', {'driver_id': driver_id, 'order_id': order_id}))
        
        return 0  # No immediate reward, reward will be given at delivery time
    
    def _reposition_driver(self, driver_id, direction):
        """Reposition a driver in the given direction."""
        if driver_id not in self.drivers:
            return -5  # Invalid driver
        
        driver = self.drivers[driver_id]
        
        if driver['status'] != 'free':
            return -5  # Driver not available
        
        # Get current location
        lat, lng = driver['location']
        
        # Define step size (in km)
        step_size = 1.0
        
        # Update location based on direction
        if direction == 0:  # North
            new_lat = min(lat + step_size, self.city_size)
            new_lng = lng
        elif direction == 1:  # East
            new_lat = lat
            new_lng = min(lng + step_size, self.city_size)
        elif direction == 2:  # South
            new_lat = max(lat - step_size, 0)
            new_lng = lng
        else:  # West
            new_lat = lat
            new_lng = max(lng - step_size, 0)
        
        # Calculate travel time
        travel_time = self._calculate_travel_time(
            (lat, lng), (new_lat, new_lng), self.current_time)
        
        # Update driver location
        driver['location'] = (new_lat, new_lng)
        
        # Schedule driver arrival at new location
        heapq.heappush(self.event_queue, 
                      (self.current_time + travel_time, 'driver_repositioned', driver_id))
        
        # Mark the driver as busy during repositioning
        driver['status'] = 'busy'
        
        self.total_repositions += 1
        
        return -1  # Repositioning penalty
    
    def _reject_order(self, order_id):
        """Reject an order."""
        if order_id not in self.orders:
            return -5  # Invalid order
        
        order = self.orders[order_id]
        
        if order['status'] not in ['ready', 'prepping']:
            return -5  # Order not available for rejection
        
        # Mark the order as rejected
        order['status'] = 'rejected'
        self.rejected_orders.append(order_id)
        
        # Remove from relevant lists
        if order_id in self.active_orders:
            self.active_orders.remove(order_id)
        
        if order_id in self.prepping_orders:
            self.prepping_orders.remove(order_id)
        
        if order_id in self.ready_orders:
            self.ready_orders.remove(order_id)
        
        self.total_rejects += 1
        
        return -10  # Rejection penalty
    
    def _process_events_until_action_required(self):
        """
        Process events from the event queue until an action is required.
        
        Returns:
            obs: Current observation
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        terminated = False
        truncated = False
        info = {}
        
        while not (self.active_orders or terminated):
            # If event queue is empty or time exceeds day_end, end episode
            if not self.event_queue or self.event_queue[0][0] >= self.day_end:
                terminated = True
                break
            
            # Get next event
            event_time, event_type, event_data = heapq.heappop(self.event_queue)
            
            # Update current time
            self.current_time = event_time
            
            # Process event
            if event_type == 'order_arrival':
                self._handle_order_arrival(event_data)
            
            elif event_type == 'order_ready':
                self._handle_order_ready(event_data)
            
            elif event_type == 'driver_arrives_restaurant':
                self._handle_driver_arrives_restaurant(event_data)
            
            elif event_type == 'order_delivered':
                self._handle_order_delivered(event_data)
            
            elif event_type == 'driver_repositioned':
                self._handle_driver_repositioned(event_data)
            
            elif event_type == 'driver_shift_start':
                self._handle_driver_shift_start(event_data)
            
            elif event_type == 'driver_shift_end':
                self._handle_driver_shift_end(event_data)
        
        # Get current observation
        obs = self._get_observation()
        
        return obs, terminated, truncated, info
    
    def _handle_order_arrival(self, order_id):
        """Handle an order arrival event."""
        # Get order data from orders_df
        order_row = self.orders_df[self.orders_df['order_id'] == order_id].iloc[0]
        
        # Get restaurant data
        restaurant_id = order_row['restaurant_id']
        restaurant_row = self.restaurants_df[self.restaurants_df['restaurant_id'] == restaurant_id].iloc[0]
        
        # Create order object
        order = {
            'id': order_id,
            'restaurant_id': restaurant_id,
            'restaurant_lat': restaurant_row['lat'],
            'restaurant_lng': restaurant_row['lng'],
            'customer_lat': order_row['customer_lat'],
            'customer_lng': order_row['customer_lng'],
            'order_time': order_row['order_time'],
            'prep_time': order_row['prep_time'],
            'ready_time': self.current_time + order_row['prep_time'] / 60,  # Convert minutes to hours
            'promised_window': order_row['promised_window'],
            'deadline': self.current_time + order_row['promised_window'] / 60,  # Convert minutes to hours
            'status': 'prepping',
            'driver_id': None
        }
        
        # Add to orders dictionary
        self.orders[order_id] = order
        
        # Add to prepping orders
        self.prepping_orders.append(order_id)
        
        # Schedule order ready event
        heapq.heappush(self.event_queue, 
                      (order['ready_time'], 'order_ready', order_id))
        
        # If there are free drivers, add to active orders for possible assignment
        if any(driver['status'] == 'free' for driver in self.drivers.values()):
            self.active_orders.append(order_id)
    
    def _handle_order_ready(self, order_id):
        """Handle an order becoming ready for pickup."""
        if order_id not in self.orders:
            return
        
        order = self.orders[order_id]
        
        # Update order status if it's still prepping
        if order['status'] == 'prepping':
            order['status'] = 'ready'
            
            # Move from prepping to ready list
            if order_id in self.prepping_orders:
                self.prepping_orders.remove(order_id)
            self.ready_orders.append(order_id)
            
            # Add to active orders if not already assigned
            if order_id not in self.assigned_orders and order_id not in self.active_orders:
                self.active_orders.append(order_id)
    
    def _handle_driver_arrives_restaurant(self, data):
        """Handle a driver arriving at a restaurant."""
        driver_id = data['driver_id']
        order_id = data['order_id']
        
        if driver_id not in self.drivers or order_id not in self.orders:
            return
        
        driver = self.drivers[driver_id]
        order = self.orders[order_id]
        
        # Update driver location to restaurant
        driver['location'] = (order['restaurant_lat'], order['restaurant_lng'])
        
        # No action needed, delivery is scheduled elsewhere
    
    def _handle_order_delivered(self, data):
        """Handle an order being delivered."""
        driver_id = data['driver_id']
        order_id = data['order_id']
        
        if driver_id not in self.drivers or order_id not in self.orders:
            return
        
        driver = self.drivers[driver_id]
        order = self.orders[order_id]
        
        # Update driver
        driver['status'] = 'free'
        driver['current_order_id'] = -1
        driver['location'] = (order['customer_lat'], order['customer_lng'])
        
        # Update order
        order['status'] = 'delivered'
        order['delivery_time'] = self.current_time
        
        # Calculate delivery time in minutes
        delivery_time_minutes = (self.current_time - order['order_time']) * 60
        
        # Calculate reward
        reward = 45 - delivery_time_minutes
        
        # Add to completed orders
        self.completed_orders.append(order_id)
        
        # Remove from assigned orders
        if order_id in self.assigned_orders:
            del self.assigned_orders[order_id]
        
        # Update metrics
        self.total_delivery_time += delivery_time_minutes
        self.total_deliveries += 1
        
        # Add free driver to active orders list if there are orders waiting
        if self.ready_orders or self.prepping_orders:
            self.active_orders.append(driver_id)
    
    def _handle_driver_repositioned(self, driver_id):
        """Handle a driver finishing repositioning."""
        if driver_id not in self.drivers:
            return
        
        driver = self.drivers[driver_id]
        
        # Update driver status
        driver['status'] = 'free'
        
        # Add free driver to active orders list if there are orders waiting
        if self.ready_orders or self.prepping_orders:
            self.active_orders.append(driver_id)
    
    def _handle_driver_shift_start(self, driver_id):
        """Handle a driver starting their shift."""
        # Get driver data from drivers_df
        driver_row = self.drivers_df[self.drivers_df['driver_id'] == driver_id].iloc[0]
        
        # Create driver object
        self.drivers[driver_id] = {
            'id': driver_id,
            'location': (driver_row['home_lat'], driver_row['home_lng']),
            'status': 'free',
            'current_order_id': -1,
            'shift_start': driver_row['shift_start'],
            'shift_end': driver_row['shift_end'],
            'capacity': driver_row['capacity']
        }
        
        # Add free driver to active orders list if there are orders waiting
        if self.ready_orders or self.prepping_orders:
            self.active_orders.append(driver_id)
    
    def _handle_driver_shift_end(self, driver_id):
        """Handle a driver ending their shift."""
        if driver_id not in self.drivers:
            return
        
        # Only end shift if driver is free
        driver = self.drivers[driver_id]
        if driver['status'] == 'free':
            del self.drivers[driver_id]
        else:
            # Reschedule shift end after the current delivery
            heapq.heappush(self.event_queue, 
                          (self.current_time + 0.5, 'driver_shift_end', driver_id))
    
    def _calculate_travel_time(self, from_location, to_location, current_time):
        """
        Calculate travel time between two locations.
        
        Args:
            from_location: (lat, lng) of start location
            to_location: (lat, lng) of end location
            current_time: Current time (hour of day)
            
        Returns:
            Travel time in hours
        """
        hour = int(current_time)
        
        if self.use_road_network and self.road_graph:
            # Find nearest nodes to locations
            from_node = self._find_nearest_node(from_location)
            to_node = self._find_nearest_node(to_location)
            
            # Find shortest path
            try:
                path = nx.shortest_path(self.road_graph, from_node, to_node, weight='base_time')
                
                # Calculate total travel time with time-of-day multipliers
                travel_time = 0
                for i in range(len(path) - 1):
                    edge_data = self.road_graph.get_edge_data(path[i], path[i+1])
                    base_time = edge_data['base_time']
                    
                    # Apply time multiplier for current hour if available
                    time_multiplier = edge_data['time_multipliers'].get(str(hour), 1.0)
                    travel_time += base_time * time_multiplier
                
                return travel_time / 60  # Convert minutes to hours
            
            except nx.NetworkXNoPath:
                # Fallback to Manhattan distance if no path found
                pass
        
        # Manhattan distance (in km)
        from_lat, from_lng = from_location
        to_lat, to_lng = to_location
        distance = abs(to_lat - from_lat) + abs(to_lng - from_lng)
        
        # Time multiplier based on hour of day
        time_multiplier = 1.0
        if hour in [8, 9]:  # Morning rush hour
            time_multiplier = 1.5
        elif hour in [17, 18]:  # Evening rush hour
            time_multiplier = 1.5
        elif hour in [12, 13]:  # Lunch time
            time_multiplier = 1.2
        
        # Travel time in hours
        return (distance / self.avg_speed) * time_multiplier
    
    def _find_nearest_node(self, location):
        """Find the nearest node in the road network to the given location."""
        lat, lng = location
        nearest_node = None
        min_distance = float('inf')
        
        for node_id, node_data in self.road_network['nodes'].items():
            node_lat = node_data['lat']
            node_lng = node_data['lng']
            
            distance = (lat - node_lat) ** 2 + (lng - node_lng) ** 2
            
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id
        
        return nearest_node
    
    def _get_observation(self):
        """
        Get the current observation.
        
        Returns:
            Dictionary with current state information
        """
        # Current time
        time_obs = np.array([self.current_time], dtype=np.float32)
        
        # Orders
        orders_obs = np.zeros((self.max_orders, 7), dtype=np.float32)
        
        order_idx = 0
        for order_id in list(self.active_orders) + list(self.prepping_orders) + list(self.ready_orders):
            if order_idx >= self.max_orders:
                break
                
            if order_id in self.orders:
                order = self.orders[order_id]
                
                # Convert status to numeric value
                status_map = {'prepping': 0, 'ready': 1, 'assigned': 2, 'delivered': 3, 'rejected': 4}
                status_value = status_map.get(order['status'], 0)
                
                # Time to ready in hours (0 if already ready)
                time_to_ready = max(0, order['ready_time'] - self.current_time)
                
                # Time remaining until deadline in hours (negative if past deadline)
                time_to_deadline = order['deadline'] - self.current_time
                
                orders_obs[order_idx] = [
                    order['restaurant_lat'], 
                    order['restaurant_lng'],
                    order['customer_lat'], 
                    order['customer_lng'],
                    time_to_ready,
                    time_to_deadline,
                    status_value
                ]
                
                order_idx += 1
        
        # Drivers
        drivers_obs = np.zeros((self.max_drivers, 5), dtype=np.float32)
        
        driver_idx = 0
        for driver_id, driver in self.drivers.items():
            if driver_idx >= self.max_drivers:
                break
                
            # Convert status to numeric value
            status_value = 1 if driver['status'] == 'free' else 0
            
            # Current order id (-1 if none)
            current_order_id = driver['current_order_id']
            
            # Time remaining in shift in hours
            shift_remaining = driver['shift_end'] - self.current_time
            
            drivers_obs[driver_idx] = [
                driver['location'][0],  # lat
                driver['location'][1],  # lng
                status_value,
                current_order_id,
                shift_remaining
            ]
            
            driver_idx += 1
        
        return {
            'time': time_obs,
            'orders': orders_obs,
            'drivers': drivers_obs
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return
        
        # Implementation for rendering would go here
        # For now, just print basic info
        if self.render_mode == "human":
            print(f"Time: {self.current_time:.2f}")
            print(f"Active orders: {len(self.active_orders)}")
            print(f"Prepping orders: {len(self.prepping_orders)}")
            print(f"Ready orders: {len(self.ready_orders)}")
            print(f"Drivers: {len(self.drivers)}")
            print(f"Completed deliveries: {self.total_deliveries}")
            print(f"Rejected orders: {self.total_rejects}")
            print(f"Total reward: {self.total_reward:.2f}")
            print("-" * 40)
            
            return None
    
    def close(self):
        """Close the environment."""
        pass 