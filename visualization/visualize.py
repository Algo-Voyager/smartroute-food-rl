#!/usr/bin/env python3
"""
Visualization module for the food delivery RL project.

This script provides functions to visualize the food delivery environment,
including route maps, order distributions, and learning curves.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize food delivery data and results")
    
    # Input data
    parser.add_argument("--data_dir", type=str, default="../data",
                       help="Directory containing data files")
    parser.add_argument("--log_dir", type=str, default="../logs",
                       help="Directory containing training logs")
    parser.add_argument("--output_dir", type=str, default="../visualization",
                       help="Directory for output files")
    parser.add_argument("--restaurants_file", type=str, default="restaurants.csv",
                       help="CSV file with restaurant data")
    parser.add_argument("--orders_file", type=str, default="orders.csv",
                       help="CSV file with order data")
    parser.add_argument("--drivers_file", type=str, default="drivers.csv",
                       help="CSV file with driver data")
    
    # Visualization options
    parser.add_argument("--city_center_lat", type=float, default=0.0,
                       help="Latitude of city center for map")
    parser.add_argument("--city_center_lng", type=float, default=0.0,
                       help="Longitude of city center for map")
    parser.add_argument("--city_size", type=float, default=10.0,
                       help="City size in km")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    return parser.parse_args()

def load_data(data_dir, restaurants_file, orders_file, drivers_file):
    """
    Load data from CSV files.
    
    Args:
        data_dir: Directory containing data files
        restaurants_file: CSV file with restaurant data
        orders_file: CSV file with order data
        drivers_file: CSV file with driver data
        
    Returns:
        Tuple of DataFrames (restaurants, orders, drivers)
    """
    restaurants = pd.read_csv(os.path.join(data_dir, restaurants_file))
    orders = pd.read_csv(os.path.join(data_dir, orders_file))
    drivers = pd.read_csv(os.path.join(data_dir, drivers_file))
    
    return restaurants, orders, drivers

def plot_city_map(restaurants, orders, drivers, output_dir, city_center_lat=0.0, city_center_lng=0.0, city_size=10.0):
    """
    Generate a map of the city with restaurants, customers, and drivers.
    
    Args:
        restaurants: DataFrame with restaurant data
        orders: DataFrame with order data
        drivers: DataFrame with driver data
        output_dir: Directory to save the map
        city_center_lat: Latitude of city center
        city_center_lng: Longitude of city center
        city_size: City size in km
    """
    # Create map centered on the city
    city_map = folium.Map(
        location=[city_center_lat, city_center_lng],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add restaurant locations
    restaurant_group = folium.FeatureGroup(name="Restaurants")
    for _, restaurant in restaurants.iterrows():
        folium.CircleMarker(
            location=[restaurant['lat'], restaurant['lng']],
            radius=5,
            color='red',
            fill=True,
            fill_opacity=0.7,
            popup=f"Restaurant {restaurant['restaurant_id']}"
        ).add_to(restaurant_group)
    restaurant_group.add_to(city_map)
    
    # Add sample customer locations (first 100 orders)
    customer_group = folium.FeatureGroup(name="Customers")
    for _, order in orders.head(100).iterrows():
        folium.CircleMarker(
            location=[order['customer_lat'], order['customer_lng']],
            radius=3,
            color='blue',
            fill=True,
            fill_opacity=0.5,
            popup=f"Customer (Order {order['order_id']})"
        ).add_to(customer_group)
    customer_group.add_to(city_map)
    
    # Add driver home locations
    driver_group = folium.FeatureGroup(name="Drivers")
    for _, driver in drivers.iterrows():
        folium.CircleMarker(
            location=[driver['home_lat'], driver['home_lng']],
            radius=4,
            color='green',
            fill=True,
            fill_opacity=0.6,
            popup=f"Driver {driver['driver_id']}"
        ).add_to(driver_group)
    driver_group.add_to(city_map)
    
    # Add heat map of orders
    orders_sample = orders.sample(min(1000, len(orders)))
    order_locations = orders_sample[['customer_lat', 'customer_lng']].values.tolist()
    HeatMap(order_locations, radius=15).add_to(folium.FeatureGroup(name="Order Density").add_to(city_map))
    
    # Add layer control
    folium.LayerControl().add_to(city_map)
    
    # Save map
    os.makedirs(output_dir, exist_ok=True)
    city_map.save(os.path.join(output_dir, 'city_map.html'))
    
    # Also create a static version using matplotlib
    plt.figure(figsize=(10, 10))
    
    # Plot restaurants
    plt.scatter(
        restaurants['lng'], 
        restaurants['lat'], 
        c='red', 
        s=30,
        alpha=0.7, 
        label='Restaurants'
    )
    
    # Plot sample customers
    plt.scatter(
        orders.head(100)['customer_lng'], 
        orders.head(100)['customer_lat'], 
        c='blue', 
        s=15,
        alpha=0.5, 
        label='Customers'
    )
    
    # Plot drivers
    plt.scatter(
        drivers['home_lng'], 
        drivers['home_lat'], 
        c='green', 
        s=20,
        alpha=0.6, 
        label='Drivers'
    )
    
    plt.title('City Map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'city_map.png'), dpi=300)

def plot_order_distribution(orders, output_dir):
    """
    Plot distribution of orders by time of day.
    
    Args:
        orders: DataFrame with order data
        output_dir: Directory to save the plot
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    sns.set(style="whitegrid")
    
    # Distribution by hour
    plt.figure(figsize=(12, 6))
    
    # Round order_time to nearest hour and count
    orders['hour'] = np.floor(orders['order_time'])
    hour_counts = orders.groupby('hour').size()
    
    # Plot
    sns.barplot(x=hour_counts.index, y=hour_counts.values)
    plt.title('Order Distribution by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Orders')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'order_distribution_hourly.png'), dpi=300)
    
    # Distribution of preparation times
    plt.figure(figsize=(10, 6))
    sns.histplot(orders['prep_time'], bins=15, kde=True)
    plt.title('Distribution of Order Preparation Times')
    plt.xlabel('Preparation Time (minutes)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prep_time_distribution.png'), dpi=300)

def plot_driver_shifts(drivers, output_dir):
    """
    Plot driver shift distribution.
    
    Args:
        drivers: DataFrame with driver data
        output_dir: Directory to save the plot
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    sns.set(style="whitegrid")
    
    # Calculate shift length
    drivers['shift_length'] = drivers['shift_end'] - drivers['shift_start']
    
    # Plot shift length distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(drivers['shift_length'], bins=10, kde=True)
    plt.title('Distribution of Driver Shift Lengths')
    plt.xlabel('Shift Length (hours)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shift_length_distribution.png'), dpi=300)
    
    # Plot drivers by hour of day
    hours = np.arange(0, 24)
    driver_counts = []
    
    for hour in hours:
        # Count drivers whose shift covers this hour
        count = sum((drivers['shift_start'] <= hour) & (drivers['shift_end'] > hour))
        driver_counts.append(count)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=hours, y=driver_counts)
    plt.title('Number of Active Drivers by Hour of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Drivers')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drivers_by_hour.png'), dpi=300)

def plot_sample_routes(restaurants, orders, output_dir, n_samples=5):
    """
    Plot sample delivery routes.
    
    Args:
        restaurants: DataFrame with restaurant data
        orders: DataFrame with order data
        output_dir: Directory to save the plot
        n_samples: Number of sample routes to plot
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    sns.set(style="whitegrid")
    
    # Sample orders
    sampled_orders = orders.sample(n_samples, random_state=42)
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Plot all restaurants with smaller markers
    plt.scatter(
        restaurants['lng'],
        restaurants['lat'],
        c='lightgray',
        s=10,
        alpha=0.5,
        label='All Restaurants'
    )
    
    # Plot routes
    for i, order in sampled_orders.iterrows():
        # Get restaurant coordinates
        restaurant_id = order['restaurant_id']
        restaurant = restaurants[restaurants['restaurant_id'] == restaurant_id].iloc[0]
        rest_lat, rest_lng = restaurant['lat'], restaurant['lng']
        
        # Get customer coordinates
        cust_lat, cust_lng = order['customer_lat'], order['customer_lng']
        
        # Plot restaurant and customer
        plt.scatter(rest_lng, rest_lat, c='red', s=50, marker='s')
        plt.scatter(cust_lng, cust_lat, c='blue', s=50, marker='o')
        
        # Plot route
        plt.plot([rest_lng, cust_lng], [rest_lat, cust_lat], 'k-', alpha=0.6)
        
        # Add labels
        plt.text(rest_lng, rest_lat, f"R{restaurant_id}", fontsize=8)
        plt.text(cust_lng, cust_lat, f"C{order['order_id']}", fontsize=8)
    
    plt.title('Sample Delivery Routes')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_routes.png'), dpi=300)

def plot_simulated_day(output_dir):
    """
    Create a simulated day visualization.
    
    Args:
        output_dir: Directory to save the animation
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # This function would create a more complex animation of a day's operations
    # Simplified version: create a series of static images showing system state at different times
    
    # Mock data for demonstration
    hours = np.arange(10, 22, 1)  # 10:00 - 21:00
    active_orders = [5, 15, 25, 20, 15, 10, 20, 30, 35, 25, 15, 10]
    active_drivers = [20, 25, 30, 25, 20, 20, 25, 30, 35, 30, 25, 20]
    deliveries = [0, 10, 25, 40, 50, 60, 75, 95, 120, 140, 155, 165]
    
    # Plot system state over time
    plt.figure(figsize=(12, 8))
    plt.plot(hours, active_orders, 'bo-', label='Active Orders')
    plt.plot(hours, active_drivers, 'go-', label='Active Drivers')
    plt.plot(hours, deliveries, 'ro-', label='Cumulative Deliveries')
    plt.title('Simulated Day: System State Over Time')
    plt.xlabel('Hour of Day')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.xticks(hours)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'simulated_day.png'), dpi=300)

def plot_learning_curve(log_dir, output_dir):
    """
    Plot the learning curve from training logs.
    
    Args:
        log_dir: Directory containing training logs
        output_dir: Directory to save the plot
    """
    # This function would parse and plot the learning curves from tensorboard logs
    # For demonstration, create a mock learning curve
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Mock data
    steps = np.arange(0, 1e7, 1e5)
    rewards = 5 * (1 - np.exp(-steps / 2e6)) + np.random.normal(0, 0.2, size=len(steps))
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(steps, rewards)
    plt.title('Training Learning Curve')
    plt.xlabel('Training Steps')
    plt.ylabel('Average Episode Reward')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curve.png'), dpi=300)

def main():
    """Main visualization function."""
    # Parse arguments
    args = parse_args()
    
    # Load data
    restaurants, orders, drivers = load_data(
        args.data_dir,
        args.restaurants_file,
        args.orders_file,
        args.drivers_file
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations
    print("Generating city map...")
    plot_city_map(
        restaurants, orders, drivers, args.output_dir,
        city_center_lat=args.city_center_lat,
        city_center_lng=args.city_center_lng,
        city_size=args.city_size
    )
    
    print("Generating order distribution plots...")
    plot_order_distribution(orders, args.output_dir)
    
    print("Generating driver shift plots...")
    plot_driver_shifts(drivers, args.output_dir)
    
    print("Generating sample routes...")
    plot_sample_routes(restaurants, orders, args.output_dir)
    
    print("Generating simulated day visualization...")
    plot_simulated_day(args.output_dir)
    
    print("Generating learning curve...")
    plot_learning_curve(args.log_dir, args.output_dir)
    
    print(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main() 