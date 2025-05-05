import torch
import pandas as pd
from torch_geometric.nn.models import TGNMemory, TransformerConv
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch.nn import Linear
from torch.nn.functional import binary_cross_entropy_with_logits
from tqdm import tqdm

# Create node to index mapping
def create_index_mapping(df):
    nodes = pd.concat([df['src'], df['dst']]).unique()
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    return node_to_idx, idx_to_node

# Construct TemporalData from your CSVs
def construct_temporal_data(interactions_csv, cpu_csv, mem_csv):
    df = pd.read_csv(interactions_csv)
    cpu = pd.read_csv(cpu_csv)
    mem = pd.read_csv(mem_csv)

    node_to_idx, idx_to_node = create_index_mapping(df)
    num_nodes = len(node_to_idx)

    df['src'] = df['src'].map(node_to_idx)
    df['dst'] = df['dst'].map(node_to_idx)

    # Merge CPU and Memory data
    cpu_mem = pd.merge(cpu, mem, on=['pod', 'timestamp'])
    cpu_mem['node'] = cpu_mem['pod'].map(node_to_idx)
    cpu_mem = cpu_mem.dropna(subset=['node'])

    # Create node state matrix
    cpu_mem = cpu_mem.sort_values(by='timestamp')
    state_df = cpu_mem.set_index(['node', 'timestamp'])[['cpu_usage', 'memory_usage']]

    src = torch.tensor(df['src'].values, dtype=torch.long)
    dst = torch.tensor(df['dst'].values, dtype=torch.long)
    t = torch.tensor(df['timestamp'].values, dtype=torch.long)  # ✅ FIXED: ensure long
    msg = torch.tensor(df[['req_per_s']].values, dtype=torch.float)
    y = torch.tensor(df['label'].values, dtype=torch.float)  # assuming binary labels
    data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=y)

    return data, state_df, node_to_idx, idx_to_node

# Train or load a TGN model
def train_or_load_tgn(data, num_nodes, raw_dim, mem_dim=128, time_dim=32):
    loader = TemporalDataLoader(data, batch_size=200)

    memory = TGNMemory(num_nodes=num_nodes, raw_msg_dim=raw_dim,
                       memory_dim=mem_dim, time_dim=time_dim)

    class TGNLinkPredictor(torch.nn.Module):
        def __init__(self, in_channels):
            super().__init__()
            self.lin = Linear(in_channels, 1)

        def forward(self, z_src, z_dst):
            return self.lin(z_src * z_dst).squeeze()

    tgn = TransformerConv(in_channels=mem_dim, out_channels=mem_dim, heads=2)
    link_pred = TGNLinkPredictor(mem_dim)

    opt = torch.optim.Adam(list(tgn.parameters()) + list(link_pred.parameters()), lr=0.001)

    for batch in loader:
        opt.zero_grad()
        memory.update_state(batch.src, batch.dst, batch.t.long(), batch.msg)  # ✅ FIXED
        src_emb, _ = memory(batch.src)
        dst_emb, _ = memory(batch.dst)
        neg_emb, _ = memory(batch.neg_dst)
        pos_out = link_pred(src_emb, dst_emb)
        neg_out = link_pred(src_emb, neg_emb)
        loss = binary_cross_entropy_with_logits(pos_out, batch.y) + \
               binary_cross_entropy_with_logits(neg_out, torch.zeros_like(batch.y))
        loss.backward()
        opt.step()

    return memory

# Extract future node state embeddings
def extract_with_state_events(memory, state_df, idx_to_node, future_steps=3):
    num_nodes = len(idx_to_node)
    current_max_t = max([ts for _, ts in state_df.index])

    embeddings_over_time = []

    for dt in range(1, future_steps + 1):
        t_fut = current_max_t + dt
        src = dst = torch.arange(num_nodes)
        feats = []
        for idx in range(num_nodes):
            node = idx_to_node[idx]
            try:
                feats.append(state_df.loc[(node, t_fut)].values.tolist())
            except KeyError:
                feats.append([0.0, 0.0])  # missing data default

        msg = torch.tensor(feats, dtype=torch.float)
        t_tensor = torch.full((num_nodes,), int(t_fut), dtype=torch.long)  # ✅ FIXED
        memory.update_state(src, dst, t_tensor, msg)

        emb, _ = memory(torch.arange(num_nodes))
        embeddings_over_time.append((t_fut, emb))

    return embeddings_over_time

if __name__ == "__main__":
    # Replace with actual CSV paths
    interactions_csv = "microservice_interactions.csv"
    cpu_csv = "cpu_usage.csv"
    mem_csv = "memory_usage.csv"

    data, state_df, node_to_idx, idx_to_node = construct_temporal_data(
        interactions_csv, cpu_csv, mem_csv
    )

    raw_dim = data.msg.size(1)
    mem = train_or_load_tgn(data, len(node_to_idx), raw_dim)

    embeddings = extract_with_state_events(mem, state_df, idx_to_node)

    # Optionally save or visualize embeddings
    for t, emb in embeddings:
        print(f"Timestep {t}: Embedding shape {emb.shape}")
