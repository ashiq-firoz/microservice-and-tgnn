import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import datetime

# Load dataset with your logic
def load_dataset(service_requests_path, cpu_usage_path, memory_usage_path):
    """Load and preprocess the dataset using your logic."""
    # Load CSV files
    service_requests = pd.read_csv(service_requests_path)
    cpu_usage = pd.read_csv(cpu_usage_path)
    memory_usage = pd.read_csv(memory_usage_path)
    
    # Normalize function
    def normalize(series):
        if series.max() == series.min():
            return series * 0
        return (series - series.min()) / (series.max() - series.min())
    
    # Normalize CPU and memory usage
    cpu_usage['Cpu usage'] = normalize(cpu_usage['Cpu usage'])
    memory_usage['Memory Usage'] = normalize(memory_usage['Memory Usage'])
    
    # Merge dataframes on Pod and timestamp
    merged_usage = pd.merge(cpu_usage, memory_usage, on=['Pod', 'Unix Timestamp'], how='outer').fillna(0)
    
    # Extract service name from pod name
    def extract_service_name(pod_name):
        parts = pod_name.split('-')
        if len(parts) > 2:
            return '-'.join(parts[:2])
        return parts[0]
    merged_usage['Service'] = merged_usage['Pod'].apply(extract_service_name)
    
    # Create a common timestamp format for all datasets
    merged_usage['timestamp'] = merged_usage['Unix Timestamp']
    
    # Process service requests
    service_requests['timestamp'] = service_requests.get('Unix Timestamp', service_requests.get('timestamp'))
    
    # Group by service, pod, and timestamp
    service_cpu = merged_usage.groupby(['Service', 'Pod', 'timestamp'])['Cpu usage'].mean().reset_index()
    service_memory = merged_usage.groupby(['Service', 'Pod', 'timestamp'])['Memory Usage'].mean().reset_index()
    
    # Merge service metrics while retaining the Pod column
    service_metrics = pd.merge(service_cpu, service_memory, on=['Service', 'Pod', 'timestamp'], how='outer').fillna(0)
    
    # Rename columns for consistency with TGNN model
    service_metrics.rename(columns={
        'Service': 'service',
        'Cpu usage': 'cpu_usage',
        'Memory Usage': 'memory_usage',
        'Pod': 'pod'  # Retain the Pod column
    }, inplace=True)
    
    # Rename columns for consistency with TGNN model
    service_requests_renamed = service_requests.rename(columns={
        'Source Service': 'source_service',
        'Destination Service': 'target_service',
        'Request Count': 'requests_per_second'
    })
    
    # Ensure all required columns exist
    if 'requests_per_second' not in service_requests_renamed.columns:
        if 'Request Count' in service_requests.columns:
            service_requests_renamed['requests_per_second'] = service_requests['Request Count']
        else:
            service_requests_renamed['requests_per_second'] = 1.0  # Default value
    
    return service_requests_renamed, service_metrics

class TemporalGNN(nn.Module):
    def __init__(self, node_features, hidden_channels, output_features):
        super(TemporalGNN, self).__init__()
        self.conv1 = GCNConv(node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lstm = nn.LSTM(hidden_channels, hidden_channels, batch_first=True)
        self.linear = nn.Linear(hidden_channels, output_features)
        
    def forward(self, x, edge_index, batch=None):
        # Graph convolution layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Reshape for LSTM layer
        x = x.unsqueeze(0)  # Add batch dimension
        
        # LSTM layer
        x, _ = self.lstm(x)
        
        # Output layer
        x = self.linear(x.squeeze(0))
        
        return x

def create_graph_from_dataframe(df_requests, df_resources, timestamp):
    """Create a graph representation of the microservice system at a given timestamp."""
    G = nx.DiGraph()
    
    # Filter data for the specific timestamp
    requests = df_requests[df_requests['timestamp'] == timestamp]
    resources = df_resources[df_resources['timestamp'] == timestamp]
    
    # Add nodes for each service
    services = set(requests['source_service'].unique()) | set(requests['target_service'].unique())
    services.update(resources['service'].unique())
    
    for service in services:
        # Get resource usage for this service
        service_resources = resources[resources['service'] == service]
        
        if not service_resources.empty:
            # Using mean values for multiple pods
            cpu = service_resources['cpu_usage'].mean()
            memory = service_resources['memory_usage'].mean()
            G.add_node(service, cpu=cpu, memory=memory)
        else:
            # Default values if no resource data
            G.add_node(service, cpu=0.0, memory=0.0)
    
    # Add edges for each request
    for _, row in requests.iterrows():
        source = row['source_service']
        target = row['target_service']
        rps = row['requests_per_second']
        
        # Add edge with request rate as weight
        if G.has_edge(source, target):
            G[source][target]['weight'] += rps
        else:
            G.add_edge(source, target, weight=rps)
    
    return G

def prepare_data(df_requests, df_resources, sequence_length=10, prediction_horizon=5):
    """Prepare sequential graph data for training."""
    # Ensure timestamp is properly formatted
    if df_requests['timestamp'].dtype == 'object':
        try:
            df_requests['timestamp'] = pd.to_datetime(df_requests['timestamp'])
        except:
            # If conversion fails, keep as is
            pass
            
    if df_resources['timestamp'].dtype == 'object':
        try:
            df_resources['timestamp'] = pd.to_datetime(df_resources['timestamp'])
        except:
            # If conversion fails, keep as is
            pass
    
    # Get unique timestamps
    all_timestamps = sorted(set(df_requests['timestamp'].unique()) | set(df_resources['timestamp'].unique()))
    
    if len(all_timestamps) < sequence_length + prediction_horizon:
        raise ValueError(f"Not enough timestamps in the dataset. Need at least {sequence_length + prediction_horizon}, but got {len(all_timestamps)}")
    
    features_list = []
    targets_list = []
    edge_indices_list = []
    
    for i in range(len(all_timestamps) - sequence_length - prediction_horizon + 1):
        # Create sequence of graphs
        sequence_graphs = []
        for j in range(sequence_length):
            timestamp = all_timestamps[i + j]
            G = create_graph_from_dataframe(df_requests, df_resources, timestamp)
            sequence_graphs.append(G)
        
        # Create target graphs
        target_graphs = []
        for j in range(prediction_horizon):
            timestamp = all_timestamps[i + sequence_length + j]
            G = create_graph_from_dataframe(df_requests, df_resources, timestamp)
            target_graphs.append(G)
        
        # Extract features and edge indices
        # We'll use the last graph in the sequence for the structure
        last_graph = sequence_graphs[-1]
        
        # Skip if the graph is empty
        if len(last_graph.nodes()) == 0 or len(last_graph.edges()) == 0:
            continue
        
        # Node mapping
        node_mapping = {node: idx for idx, node in enumerate(last_graph.nodes())}
        
        # Feature matrix
        features = []
        for node in last_graph.nodes():
            node_features = []
            # Collect features across the sequence
            for graph in sequence_graphs:
                if node in graph.nodes():
                    node_features.append([
                        graph.nodes[node]['cpu'],
                        graph.nodes[node]['memory']
                    ])
                else:
                    node_features.append([0.0, 0.0])
            
            # Flatten features
            features.append(np.array(node_features).flatten())
        
        # Edge index
        edge_index = []
        for source, target in last_graph.edges():
            edge_index.append([node_mapping[source], node_mapping[target]])
        
        if not edge_index:  # Skip if no edges
            continue
            
        edge_index = np.array(edge_index).T
        
        # Target values
        targets = []
        for node in last_graph.nodes():
            node_targets = []
            for graph in target_graphs:
                if node in graph.nodes():
                    node_targets.append([
                        graph.nodes[node]['cpu'],
                        graph.nodes[node]['memory']
                    ])
                else:
                    node_targets.append([0.0, 0.0])
            
            # Flatten targets
            targets.append(np.array(node_targets).flatten())
        
        features_list.append(np.array(features))
        targets_list.append(np.array(targets))
        edge_indices_list.append(edge_index)
    
    if not features_list:
        raise ValueError("No valid sequences found in the dataset.")
        
    return features_list, targets_list, edge_indices_list

def train_model(df_requests, df_resources, sequence_length=10, prediction_horizon=5,
                hidden_channels=64, epochs=100, learning_rate=0.001):
    """Train the Temporal GNN model."""
    print("Preparing data for training...")
    # Prepare data
    features_list, targets_list, edge_indices_list = prepare_data(
        df_requests, df_resources, sequence_length, prediction_horizon
    )
    
    print(f"Created {len(features_list)} training sequences")
    
    # Split data
    indices = list(range(len(features_list)))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    # Create model
    node_features = features_list[0].shape[1]
    output_features = targets_list[0].shape[1]
    model = TemporalGNN(node_features, hidden_channels, output_features)
    
    print(f"Model created with {node_features} input features and {output_features} output features")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop
    print("Starting training...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for idx in train_indices:
            features = torch.tensor(features_list[idx], dtype=torch.float)
            targets = torch.tensor(targets_list[idx], dtype=torch.float)
            edge_index = torch.tensor(edge_indices_list[idx], dtype=torch.long)
            
            optimizer.zero_grad()
            out = model(features, edge_index)
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss/len(train_indices) if train_indices else 0
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}')
    
    # Evaluation
    print("Evaluating model...")
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for idx in test_indices:
            features = torch.tensor(features_list[idx], dtype=torch.float)
            targets = torch.tensor(targets_list[idx], dtype=torch.float)
            edge_index = torch.tensor(edge_indices_list[idx], dtype=torch.long)
            
            out = model(features, edge_index)
            loss = criterion(out, targets)
            test_loss += loss.item()
    
    avg_test_loss = test_loss/len(test_indices) if test_indices else 0
    print(f'Test Loss: {avg_test_loss:.6f}')
    
    return model

def predict_future(model, df_requests, df_resources, last_n_timestamps=10, prediction_steps=5):
    """Predict future microservice behavior."""
    # Get unique timestamps
    all_timestamps = sorted(set(df_requests['timestamp'].unique()) | set(df_resources['timestamp'].unique()))
    latest_timestamps = all_timestamps[-last_n_timestamps:]
    
    # Create sequence of latest graphs
    sequence_graphs = []
    for timestamp in latest_timestamps:
        G = create_graph_from_dataframe(df_requests, df_resources, timestamp)
        sequence_graphs.append(G)
    
    # Extract features and edge indices
    last_graph = sequence_graphs[-1]
    
    # Check if the graph is empty
    if len(last_graph.nodes()) == 0 or len(last_graph.edges()) == 0:
        raise ValueError("The latest graph is empty. Cannot make predictions.")
    
    node_mapping = {node: idx for idx, node in enumerate(last_graph.nodes())}
    inverse_mapping = {idx: node for node, idx in node_mapping.items()}
    
    # Feature matrix
    features = []
    for node in last_graph.nodes():
        node_features = []
        for graph in sequence_graphs:
            if node in graph.nodes():
                node_features.append([
                    graph.nodes[node]['cpu'],
                    graph.nodes[node]['memory']
                ])
            else:
                node_features.append([0.0, 0.0])
        
        features.append(np.array(node_features).flatten())
    
    # Edge index
    edge_index = []
    for source, target in last_graph.edges():
        edge_index.append([node_mapping[source], node_mapping[target]])
    
    if not edge_index:
        raise ValueError("No edges found in the last graph")
    
    edge_index = np.array(edge_index).T
    
    # Convert to PyTorch tensors
    features_tensor = torch.tensor(np.array(features), dtype=torch.float)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
    
    # Predict
    model.eval()
    with torch.no_grad():
        predictions = model(features_tensor, edge_index_tensor)
    
    # Reshape predictions
    predictions = predictions.numpy()
    
    # Organize predictions
    predicted_data = []
    
    # Get the last timestamp
    last_timestamp = latest_timestamps[-1]
    
    # Calculate timestamp interval
    if len(latest_timestamps) > 1:
        if isinstance(latest_timestamps[0], (int, float, np.number)):
            # For numeric timestamps
            avg_interval = np.mean(np.diff(latest_timestamps))
        else:
            # For datetime timestamps
            try:
                time_diffs = [(latest_timestamps[i+1] - latest_timestamps[i]).total_seconds() 
                              for i in range(len(latest_timestamps)-1)]
                avg_interval = np.mean(time_diffs)
            except:
                # Fallback to index-based intervals
                avg_interval = 1
    else:
        avg_interval = 1
    
    for i, node_idx in enumerate(range(len(last_graph.nodes()))):
        service_name = inverse_mapping[node_idx]
        node_predictions = predictions[i]
        
        # Reshape to get time steps
        node_predictions = node_predictions.reshape(prediction_steps, 2)
        
        for step in range(prediction_steps):
            if isinstance(last_timestamp, (int, float, np.number)):
                # For numeric timestamps
                pred_timestamp = last_timestamp + (step + 1) * avg_interval
            else:
                # For datetime timestamps
                try:
                    pred_timestamp = last_timestamp + datetime.timedelta(seconds=(step + 1) * avg_interval)
                except:
                    # Fallback to a string representation
                    pred_timestamp = f"predicted_{step+1}"
            
            predicted_data.append({
                'timestamp': pred_timestamp,
                'service': service_name,
                'cpu_usage': node_predictions[step, 0],
                'memory_usage': node_predictions[step, 1]
            })
    
    # Create predicted request data
    predicted_requests = []
    for source, target in last_graph.edges():
        source_idx = node_mapping[source]
        target_idx = node_mapping[target]
        
        # Use the average of source and target CPU as a simple proxy for request rate
        for step in range(prediction_steps):
            source_cpu = predictions[source_idx].reshape(prediction_steps, 2)[step, 0]
            target_cpu = predictions[target_idx].reshape(prediction_steps, 2)[step, 0]
            
            # Use the original weight as a base
            original_weight = last_graph[source][target]['weight']
            
            # Calculate predicted rate (simple model)
            source_cpu_original = last_graph.nodes[source]['cpu'] + 0.001  # Avoid division by zero
            predicted_rate = original_weight * (source_cpu / source_cpu_original)
            
            if isinstance(last_timestamp, (int, float, np.number)):
                # For numeric timestamps
                pred_timestamp = last_timestamp + (step + 1) * avg_interval
            else:
                # For datetime timestamps
                try:
                    pred_timestamp = last_timestamp + datetime.timedelta(seconds=(step + 1) * avg_interval)
                except:
                    # Fallback to a string representation
                    pred_timestamp = f"predicted_{step+1}"
            
            predicted_requests.append({
                'timestamp': pred_timestamp,
                'source_service': source,
                'target_service': target,
                'requests_per_second': max(0, predicted_rate)  # Ensure non-negative
            })
    
    return pd.DataFrame(predicted_data), pd.DataFrame(predicted_requests)

def visualize_predictions(original_resources, predicted_resources, service_name):
    """Visualize resource usage predictions for a specific service."""
    # Filter data for the specific service
    original = original_resources[original_resources['service'] == service_name].copy()
    predicted = predicted_resources[predicted_resources['service'] == service_name].copy()
    
    if original.empty or predicted.empty:
        print(f"No data available for service: {service_name}")
        return
    
    # Handle different timestamp formats
    if isinstance(original['timestamp'].iloc[0], (int, float, np.number)):
        # For numeric timestamps - convert to datetime
        original['timestamp'] = pd.to_datetime(original['timestamp'], unit='s')
        predicted['timestamp'] = pd.to_datetime(predicted['timestamp'], unit='s')
    else:
        # Try to ensure datetime format
        try:
            original['timestamp'] = pd.to_datetime(original['timestamp'])
            predicted['timestamp'] = pd.to_datetime(predicted['timestamp'])
        except:
            # If conversion fails, keep as is
            pass
    
    # Set up plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot CPU usage
    ax1.plot(original['timestamp'], original['cpu_usage'], 'b-', label='Historical')
    ax1.plot(predicted['timestamp'], predicted['cpu_usage'], 'r--', label='Predicted')
    ax1.set_title(f'CPU Usage for {service_name}')
    ax1.set_ylabel('CPU Usage (Normalized)')
    ax1.legend()
    
    # Plot Memory usage
    ax2.plot(original['timestamp'], original['memory_usage'], 'b-', label='Historical')
    ax2.plot(predicted['timestamp'], predicted['memory_usage'], 'r--', label='Predicted')
    ax2.set_title(f'Memory Usage for {service_name}')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Memory Usage (Normalized)')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_service_graph(df_requests, df_resources, timestamp):
    """Visualize the service dependency graph at a given timestamp."""
    G = nx.DiGraph()
    
    # Filter data for the timestamp
    requests = df_requests[df_requests['timestamp'] == timestamp]
    resources = df_resources[df_resources['timestamp'] == timestamp]
    
    # Group pods by service, handling missing Pod column gracefully
    if 'pod' in resources.columns:
        pods_by_service = resources.groupby('service')['pod'].apply(list).to_dict()
    else:
        pods_by_service = {service: [] for service in resources['service'].unique()}
    
    # Get average CPU and memory by service
    service_avg_cpu = resources.groupby('service')['cpu_usage'].mean()
    service_avg_memory = resources.groupby('service')['memory_usage'].mean()
    
    # Add service nodes
    for service in set(requests['source_service'].unique()) | set(requests['target_service'].unique()) | set(resources['service'].unique()):
        G.add_node(service, 
                  node_type='service',
                  cpu=service_avg_cpu.get(service, 0),
                  memory=service_avg_memory.get(service, 0),
                  pods=pods_by_service.get(service, []))
    
    # Add service-to-service edges
    for _, row in requests.iterrows():
        source = row['source_service']
        target = row['target_service']
        rps = row['requests_per_second']
        # Add edge with request rate as weight
        if G.has_edge(source, target):
            G[source][target]['weight'] += rps
        else:
            G.add_edge(source, target, weight=rps, edge_type='requests')
    
    # Draw the graph
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    edge_widths = [G[u][v]['weight'] / max(1, G[u][v]['weight']) * 3 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='red', arrows=True, arrowsize=15)
    for node in G.nodes():
        cpu = G.nodes[node]['cpu']
        x, y = pos[node]
        width = 0.04
        height = 0.04 * cpu
        rect = plt.Rectangle((x - width/2, y + 0.1), width, height, color='orange')
        plt.gca().add_patch(rect)
    nx.draw_networkx_labels(G, pos)
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title(f"Service Dependency Graph at {timestamp}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_predicted_graphs(df_requests, df_resources, predicted_resources, predicted_requests, timestamps):
    """Visualize the predicted service dependency graphs."""
    for timestamp in timestamps:
        # Create subplot
        plt.figure(figsize=(12, 10))
        
        # Create graph for this timestamp
        G = nx.DiGraph()
        
        # Filter data for the timestamp
        requests = predicted_requests[predicted_requests['timestamp'] == timestamp]
        resources = predicted_resources[predicted_resources['timestamp'] == timestamp]
        
        # Get average CPU and memory by service
        service_avg_cpu = resources.groupby('service')['cpu_usage'].mean()
        service_avg_memory = resources.groupby('service')['memory_usage'].mean()
        
        # Add service nodes
        for service in set(requests['source_service'].unique()) | set(requests['target_service'].unique()) | set(resources['service'].unique()):
            G.add_node(service, 
                      node_type='service',
                      cpu=service_avg_cpu.get(service, 0),
                      memory=service_avg_memory.get(service, 0))
        
        # Add service-to-service edges
        for _, row in requests.iterrows():
            source = row['source_service']
            target = row['target_service']
            rps = row['requests_per_second']
            
            # Add edge with request rate as weight
            if G.has_edge(source, target):
                G[source][target]['weight'] += rps
            else:
                G.add_edge(source, target, weight=rps, edge_type='requests')
        
        # Draw the graph
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes with color based on memory usage
        node_colors = [G.nodes[n]['memory'] for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors, cmap=plt.cm.Blues)
        
        # Draw edges with width proportional to request count
        edge_widths = [G[u][v]['weight'] / max(1, G[u][v]['weight']) * 3 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='red', arrows=True, arrowsize=15)
        
        # Draw CPU bars
        for node in G.nodes():
            cpu = G.nodes[node]['cpu']
            x, y = pos[node]
            width = 0.04
            height = 0.04 * cpu
            rect = plt.Rectangle((x - width/2, y + 0.1), width, height, color='orange')
            plt.gca().add_patch(rect)
        
        # Add labels
        nx.draw_networkx_labels(G, pos)
        
        # Add edge labels
        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title(f"Predicted Service Dependency Graph at {timestamp}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# Main execution
if __name__ == "__main__":
    try:
        # File paths
        service_requests_path = "service-to-service.csv"
        cpu_usage_path = "cpu_usage.csv"
        memory_usage_path = "memory_usage.csv"
        
        print("Loading dataset...")
        # Load data
        df_requests, df_resources = load_dataset(service_requests_path, cpu_usage_path, memory_usage_path)
        
        print(f"Loaded {len(df_requests)} service requests and {len(df_resources)} resource records")
        
        # Train model
        print("Training model...")
        sequence_length = 10  # Number of timestamps to use for prediction
        prediction_horizon = 5  # Number of timestamps to predict   (5x10 = 50 (it predicts approx 50-60 instances))
        model = train_model(df_requests, df_resources, sequence_length, prediction_horizon, 
                            hidden_channels=64, epochs=50)
        
        # Predict future behavior
        print("Predicting future behavior...")
        predicted_resources, predicted_requests = predict_future(model, df_requests, df_resources, 
                                                                last_n_timestamps=sequence_length, 
                                                                prediction_steps=prediction_horizon)
        
        print(f"Generated {len(predicted_resources)} resource predictions and {len(predicted_requests)} request predictions")

        # # Print historical service requests
        # print("\nHistorical Service Requests:")
        # print(df_requests[['timestamp', 'source_service', 'target_service', 'requests_per_second']].to_string(index=False))

        # # Print historical resource usage
        # print("\nHistorical Resource Usage:")
        # print(df_resources[['timestamp', 'service', 'cpu_usage', 'memory_usage']].to_string(index=False))

        # # Print predicted resource usage
        # print("\nPredicted Resource Usage:")
        # print(predicted_resources[['timestamp', 'service', 'cpu_usage', 'memory_usage']].to_string(index=False))

        # # Print predicted service requests
        # print("\nPredicted Service Requests:")
        # print(predicted_requests[['timestamp', 'source_service', 'target_service', 'requests_per_second']].to_string(index=False))
        

        predicted_resources[['timestamp', 'service', 'cpu_usage', 'memory_usage']].to_csv(
            "predicted_resource_usage.csv", index=False
        )
        print("\nPredicted Resource Usage saved to 'predicted_resource_usage.csv'")

        # Save predicted service requests to a CSV file
        predicted_requests[['timestamp', 'source_service', 'target_service', 'requests_per_second']].to_csv(
            "predicted_service_requests.csv", index=False
        )
        print("Predicted Service Requests saved to 'predicted_service_requests.csv'")

        # Visualize original graph
        print("Visualizing original graph...")
        visualize_service_graph(df_requests, df_resources, df_requests['timestamp'].iloc[0])
        
        # Visualize predictions for a specific service
        print("Visualizing service predictions...")
        service_name = df_resources['service'].iloc[0]  # Use the first service as an example
        visualize_predictions(df_resources, predicted_resources, service_name)
        
        # Visualize predicted graphs
        print("Visualizing predicted graphs...")
        predicted_timestamps = predicted_resources['timestamp'].unique()[:2]  # Show first two predictions
        visualize_predicted_graphs(df_requests, df_resources, predicted_resources, predicted_requests, predicted_timestamps)
        
        # Save the model
        torch.save(model.state_dict(), "microservice_tgnn_model.pth")
        print("Model saved as 'microservice_tgnn_model.pth'")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        
        # Provide error handling suggestions
        print("\nTroubleshooting suggestions:")
        print("1. Check if the CSV files exist in the correct location")
        print("2. Verify that the CSV files have the expected columns")
        print("3. Ensure there are enough timestamps in the data for the sequence length and prediction horizon")
        print("4. Check if you have PyTorch and PyTorch Geometric installed")
        print("5. Make sure you have NetworkX, matplotlib, and pandas installed")