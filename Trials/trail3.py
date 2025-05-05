import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
from sklearn.preprocessing import StandardScaler

class TemporalGNN(nn.Module):
    def __init__(self, node_features, hidden_channels, output_features):
        super(TemporalGNN, self).__init__()
        self.conv1 = GCNConv(node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lstm = nn.LSTM(hidden_channels, hidden_channels, batch_first=True)
        self.linear = nn.Linear(hidden_channels, output_features)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = x.unsqueeze(0)
        x, _ = self.lstm(x)
        x = self.linear(x.squeeze(0))
        return x

def create_graph_from_dataframe(df, timestamp):
    G = nx.DiGraph()
    
    # Filter data for the specific timestamp
    df_time = df[df['timestamp'] == timestamp]
    
    # Add nodes and edges
    for _, row in df_time.iterrows():
        # Add nodes if they don't exist
        if not G.has_node(row['user']):
            G.add_node(row['user'])
        if not G.has_node(row['item']):
            G.add_node(row['item'])
            
        # Add edge with features
        feature_value = float(row['features'])  # Convert to float if not already
        G.add_edge(row['user'], row['item'], 
                  weight=feature_value,
                  features=[feature_value])  # Store as a single-item list
    
    return G

def prepare_data(df, sequence_length=10):
    # Get unique timestamps
    timestamps = sorted(df['timestamp'].unique())
    
    features_list = []
    edge_indices_list = []
    
    for i in range(len(timestamps) - sequence_length + 1):
        sequence_graphs = []
        for j in range(sequence_length):
            timestamp = timestamps[i + j]
            G = create_graph_from_dataframe(df, timestamp)
            sequence_graphs.append(G)
        
        # Use last graph for structure
        last_graph = sequence_graphs[-1]
        
        # Create node mapping
        node_mapping = {node: idx for idx, node in enumerate(last_graph.nodes())}
        
        # Extract features
        features = []
        for node in last_graph.nodes():
            node_features = []
            for graph in sequence_graphs:
                # Check if node exists in the current graph
                if node in graph:
                    # Get edges where node is either source or target
                    in_edges = [e for e in graph.in_edges(node) if graph.has_edge(*e)]
                    out_edges = [e for e in graph.out_edges(node) if graph.has_edge(*e)]
                    edges = in_edges + out_edges
                    
                    if edges:
                        # Average the features for all edges connected to this node
                        edge_features = []
                        for edge in edges:
                            edge_features.append(graph.edges[edge]['features'])
                        if edge_features:
                            avg_features = np.mean(edge_features, axis=0)
                            node_features.extend(avg_features)
                        else:
                            node_features.append(0.0)
                    else:
                        node_features.append(0.0)
                else:
                    node_features.append(0.0)  # Node doesn't exist in this graph
            
            features.append(node_features)
            
        # Create edge index
        edge_index = []
        for source, target in last_graph.edges():
            edge_index.append([node_mapping[source], node_mapping[target]])
        
        if edge_index:
            edge_index = np.array(edge_index).T
            features_list.append(np.array(features))
            edge_indices_list.append(edge_index)
    
    return features_list, edge_indices_list, node_mapping

def main():
    # Load the dataset
    df = pd.read_csv('./dataset/microservices.csv')
    
    # Prepare data
    sequence_length = 10
    features_list, edge_indices_list, node_mapping = prepare_data(df, sequence_length)
    
    # Create and train model
    node_features = features_list[0].shape[1]
    hidden_channels = 64
    output_features = node_features  # Same as input for embedding
    
    model = TemporalGNN(node_features, hidden_channels, output_features)
    
    # Convert to PyTorch tensors
    features_tensor = torch.tensor(features_list[-1], dtype=torch.float)
    edge_index_tensor = torch.tensor(edge_indices_list[-1], dtype=torch.long)
    
    # Generate embeddings for next 10 timestamps
    model.eval()
    with torch.no_grad():
        embeddings = model(features_tensor, edge_index_tensor)
    
    # Create reverse mapping
    inverse_mapping = {idx: node for node, idx in node_mapping.items()}
    
    # Create DataFrame with embeddings
    embeddings_df = pd.DataFrame(
        embeddings.numpy(),
        index=[inverse_mapping[i] for i in range(len(embeddings))],
        columns=[f'embedding_{i}' for i in range(embeddings.shape[1])]
    )
    
    # Get last timestamp from data
    last_timestamp = df['timestamp'].max()
    
    # Add timestamp predictions
    all_embeddings = []
    for i in range(10):  # Next 10 timestamps
        temp_df = embeddings_df.copy()
        temp_df['timestamp'] = last_timestamp + (i + 1) * 60  # Assuming 60-second intervals
        all_embeddings.append(temp_df)
    
    # Combine all predictions
    final_embeddings = pd.concat(all_embeddings)
    final_embeddings.index.name = 'service'
    
    # Save embeddings
    final_embeddings.to_csv('node_embeddings.csv')
    print("Node embeddings saved to node_embeddings.csv")

if __name__ == "__main__":
    main()
