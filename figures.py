import pandas as pd
import matplotlib.pyplot as plt
import os

# Step 1: Load the CSV file
csv_file = "predicted_resource_usage.csv"

# Ensure the file exists
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"The file {csv_file} does not exist.")

# Read the CSV file into a DataFrame
data = pd.read_csv(csv_file)

# Step 2: Convert the timestamp column to datetime format for better plotting
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

# Step 3: Group the data by service
grouped_data = data.groupby('service')

# Step 4: Create a directory to save the plots
output_dir = "./"
os.makedirs(output_dir, exist_ok=True)

# Step 5: Generate and save plots for each service
for service, group in grouped_data:
    # Sort the data by timestamp for proper plotting
    group = group.sort_values(by='timestamp')
    
    # Create a new figure
    plt.figure(figsize=(10, 6))
    
    # Plot CPU usage
    plt.plot(group['timestamp'], group['cpu_usage'], label='CPU Usage', color='blue', marker='o')
    
    # Plot Memory usage
    plt.plot(group['timestamp'], group['memory_usage'], label='Memory Usage', color='green', marker='x')
    
    # Add labels, title, and legend
    plt.title(f"Resource Usage for {service}")
    plt.xlabel("Timestamp")
    plt.ylabel("Usage (normalized)")
    plt.legend()
    plt.grid(True)
    
    # Save the plot as an image file
    output_file = os.path.join(output_dir, f"{service}_resource_usage.png")
    plt.savefig(output_file)
    plt.close()  # Close the figure to free memory

print(f"Plots have been saved in the '{output_dir}' directory.")