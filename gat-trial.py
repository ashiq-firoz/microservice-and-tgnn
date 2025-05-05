import pandas as pd                                         # for CSV I/O :contentReference[oaicite:0]{index=0}
import torch                                                # for tensor operations :contentReference[oaicite:1]{index=1}
from models.definitions.GAT import GAT, LayerType                     # the GAT class 
import numpy
# 1. Load your two CSVs
emb_df = pd.read_csv("node_embeddings.csv")
# columns: service, embedding_0…embedding_9, timestamp

edge_df = pd.read_csv("./csv_files/predicted_service_requests.csv")
# columns: timestamp, source_service, target_service, requests_per_second

# 2. Build a global service→index mapping (so node IDs align across time)
all_services = pd.concat([
    emb_df["service"],
    edge_df["source_service"],
    edge_df["target_service"]
]).unique()
service2idx = {s: i for i, s in enumerate(all_services)}
N = len(all_services)

# 3. Instantiate your GAT once
in_dim  = emb_df.filter(like="embedding_").shape[1]  # 10 
hid_dim = 8
out_dim = in_dim

# gat = GAT(
#     n_feat=in_dim,
#     n_hidden=hid_dim,
#     n_class=out_dim,
#     dropout=0.6,
#     alpha=0.2,
#     n_heads=4,
#     layer_type=None      # use default “IMP3” variant
# ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

in_dim  = 10
hid_dim = 8
out_dim = 10

gat = GAT(
    num_of_layers            = 2,                     # two GAT layers
    num_heads_per_layer      = [4, 1],                # 4 heads in layer1, 1 head in layer2
    num_features_per_layer   = [in_dim, hid_dim, out_dim],
    add_skip_connection      = True,
    bias                     = True,
    dropout                  = 0.6,
    layer_type               = LayerType.IMP3,
    log_attention_weights    = False
).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# 4. Prepare to collect dynamic outputs
dynamic_embeddings = {}   # dict: timestamp → tensor of shape [N, out_dim]

# 5. Loop over each timestamp snapshot
for t in sorted(emb_df["timestamp"].unique()):
    # a) slice embeddings at time t
    df_e = emb_df[emb_df["timestamp"] == t].set_index("service")
    # build feature matrix of shape [N, in_dim], filling missing with zeros
    X = torch.zeros((N, in_dim))
    for svc, row in df_e.iterrows():
        X[service2idx[svc]] = torch.tensor(row.filter(like="embedding_").values)

    # b) slice edges at time t
    df_edges = edge_df[edge_df["timestamp"] == t]
    src = df_edges["source_service"].map(service2idx).tolist()
    dst = df_edges["target_service"].map(service2idx).tolist()
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    # c) add self‑loops so nodes attend to themselves
    self_loops = torch.arange(N).unsqueeze(0).repeat(2,1)
    edge_index = torch.cat([edge_index, self_loops], dim=1).contiguous()

    # d) forward pass through GAT (in eval mode)
    gat.eval()
    with torch.no_grad():
        H,_ = gat((X, edge_index))   # → [N, out_dim]

    dynamic_embeddings[t] = H.cpu()  # store on CPU

# now `dynamic_embeddings` maps each timestamp to its refined embedding matrix


rows = []
for t, H in dynamic_embeddings.items():
    # H is a CPU tensor of shape [N, F]
    H = H.numpy()            # to NumPy array
    N, F = H.shape
    # invert service2idx to idx2service
    idx2service = {idx: svc for svc, idx in service2idx.items()}
    for idx in range(N):
        svc = idx2service[idx]
        row = {"service": svc, "timestamp": t}
        # add each refined dimension
        for f in range(F):
            row[f"refined_{f}"] = H[idx, f]
        rows.append(row)

# build DataFrame
out_df = pd.DataFrame(rows)

# optionally sort
out_df = out_df.sort_values(["timestamp","service"]).reset_index(drop=True)

# write to CSV
out_df.to_csv("dynamic_refined_embeddings.csv", index=False)
print("Wrote", len(out_df), "rows to dynamic_refined_embeddings.csv")