import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np

# Define the TemporalGNN model class (same as in training)
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
        x = torch.relu(x)
        x = torch.dropout(x, p=0.2, train=self.training)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        # Reshape for LSTM layer
        x = x.unsqueeze(0)  # Add batch dimension
        # LSTM layer
        x, _ = self.lstm(x)
        # Output layer
        x = self.linear(x.squeeze(0))
        return x

# Function to load dataset
def load_dataset(service_requests_path, cpu_usage_path, memory_usage_path):
    """Load and preprocess the dataset."""
    service_requests = pd.read_csv(service_requests_path)
    cpu_usage = pd.read_csv(cpu_usage_path)
    memory_usage = pd.read_csv(memory_usage_path)

    # Normalize CPU and memory usage
    def normalize(series):
        if series.max() == series.min():
            return series * 0
        return (series - series.min()) / (series.max() - series.min())

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
    merged_usage['timestamp'] = merged_usage['Unix Timestamp']

    # Process service requests
    service_requests['timestamp'] = service_requests.get('Unix Timestamp', service_requests.get('timestamp'))

    # Group by service and timestamp
    service_cpu = merged_usage.groupby(['Service', 'timestamp'])['Cpu usage'].mean().reset_index()
    service_memory = merged_usage.groupby(['Service', 'timestamp'])['Memory Usage'].mean().reset_index()

    # Merge service metrics
    service_metrics = pd.merge(service_cpu, service_memory, on=['Service', 'timestamp'], how='outer').fillna(0)
    service_metrics.rename(columns={
        'Service': 'service',
        'Cpu usage': 'cpu_usage',
        'Memory Usage': 'memory_usage'
    }, inplace=True)

    # Rename columns for consistency with TGNN model
    service_requests_renamed = service_requests.rename(columns={
        'Source Service': 'source_service',
        'Destination Service': 'target_service',
        'Request Count': 'requests_per_second'
    })

    return service_requests_renamed, service_metrics

# Function to create a graph from dataframes
def create_graph_from_dataframe(df_requests, df_resources, timestamp):
    G = nx.DiGraph()
    requests = df_requests[df_requests['timestamp'] == timestamp]
    resources = df_resources[df_resources['timestamp'] == timestamp]

    services = set(requests['source_service'].unique()) | set(requests['target_service'].unique())
    services.update(resources['service'].unique())

    for service in services:
        service_resources = resources[resources['service'] == service]
        if not service_resources.empty:
            cpu = service_resources['cpu_usage'].mean()
            memory = service_resources['memory_usage'].mean()
            G.add_node(service, cpu=cpu, memory=memory)
        else:
            G.add_node(service, cpu=0.0, memory=0.0)

    for _, row in requests.iterrows():
        source = row['source_service']
        target = row['target_service']
        rps = row['requests_per_second']
        if G.has_edge(source, target):
            G[source][target]['weight'] += rps
        else:
            G.add_edge(source, target, weight=rps)

    return G

# Function to prepare input data
def prepare_input_data(graph):
    node_mapping = {node: idx for idx, node in enumerate(graph.nodes())}
    features = []
    for node in graph.nodes():
        features.append([graph.nodes[node]['cpu'], graph.nodes[node]['memory']])
    features = np.array(features)

    edge_index = []
    for source, target in graph.edges():
        edge_index.append([node_mapping[source], node_mapping[target]])
    edge_index = np.array(edge_index).T

    return torch.tensor(features, dtype=torch.float), torch.tensor(edge_index, dtype=torch.long)

# Load the trained model
def load_trained_model(model_path, node_features, hidden_channels, output_features):
    model = TemporalGNN(node_features, hidden_channels, output_features)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Main execution
if __name__ == "__main__":
    try:
        # File paths
        service_requests_path = "service-to-service.csv"
        cpu_usage_path = "cpu_usage.csv"
        memory_usage_path = "memory_usage.csv"
        model_path = "microservice_tgnn_model.pth"

        # Load dataset
        print("Loading dataset...")
        df_requests, df_resources = load_dataset(service_requests_path, cpu_usage_path, memory_usage_path)

        # Load the trained model
        print("Loading trained model...")
        node_features = 2  # CPU and memory usage
        hidden_channels = 64
        output_features = 10  # Example value; adjust based on your training setup
        model = load_trained_model(model_path, node_features, hidden_channels, output_features)

        # Create a graph for the latest timestamp
        latest_timestamp = df_resources['timestamp'].max()
        G = create_graph_from_dataframe(df_requests, df_resources, latest_timestamp)

        # Prepare input data
        features, edge_index = prepare_input_data(G)

        # Make predictions
        print("Making predictions...")
        with torch.no_grad():
            predictions = model(features, edge_index)

        print("Predictions:", predictions.numpy())

    except Exception as e:
        print(f"Error: {str(e)}")