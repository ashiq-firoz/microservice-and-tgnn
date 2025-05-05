import pandas as pd
import numpy as np
import torch

try:
    from torch_geometric_temporal.signal import DynamicGraphTemporalSignal
except ImportError:
    print("Please install dependencies using:")
    print("pip install torch==1.12.0")
    print("pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.12.0+cpu.html")
    print("pip install torch-sparse==0.6.15 -f https://pytorch-geometric.com/whl/torch-1.12.0+cpu.html")
    print("pip install torch-geometric==2.1.0")
    print("pip install torch-geometric-temporal==0.54.0")
    exit(1)

# Load CSV file
df = pd.read_csv("your_file.csv")

# Convert Unix Timestamp to seconds (if not already in seconds)
df["Unix Timestamp"] = pd.to_datetime(df["Unix Timestamp"], unit="s").astype(int) / 10**9

# Encode services as unique node indices
service_to_index = {service: i for i, service in enumerate(set(df["Source Service"]).union(set(df["Destination Service"])))}
df["Source Service"] = df["Source Service"].map(service_to_index)
df["Destination Service"] = df["Destination Service"].map(service_to_index)

# Normalize Request Count
df["Request Count"] = (df["Request Count"] - df["Request Count"].mean()) / df["Request Count"].std()

# Sort by timestamp to maintain time order
df = df.sort_values(by="Unix Timestamp")

# Define time steps (assuming we split the data into 10 time intervals)
num_time_steps = 10
time_splits = np.array_split(df, num_time_steps)

# Create lists for graph snapshots
edge_indices = []
edge_weights = []
timestamps = []

for time_df in time_splits:
    # Create edge index (source, destination)
    edge_index = torch.tensor([time_df["Source Service"].values, time_df["Destination Service"].values], dtype=torch.long)
    
    # Edge attributes (request count as weight)
    edge_attr = torch.tensor(time_df["Request Count"].values, dtype=torch.float32).view(-1, 1)
    
    edge_indices.append(edge_index)
    edge_weights.append(edge_attr)
    timestamps.append(time_df["Unix Timestamp"].values[0])  # Store the earliest timestamp in each interval

# Create a DynamicGraphTemporalSignal dataset
dataset = DynamicGraphTemporalSignal(edge_indices, edge_weights, timestamps)

print(f"Number of time steps: {len(dataset.features)}")
