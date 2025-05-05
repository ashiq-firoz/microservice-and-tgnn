import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from models.definitions.GAT import GAT, LayerType

# ── 1) HYPERPARAMETERS ──────────────────────────────────────────────────────
EMB_CSV    = "node_embeddings.csv"   # cols: service, embedding_0…9, timestamp
EDGE_CSV   = "./csv_files/predicted_service_requests.csv"             # cols: timestamp, source_service, target_service, requests_per_second
Y_CSV      = "true_replicas.csv"         # cols: service, timestamp, replicas
IN_DIM     = 10      # embedding size
HID_DIM    = 8
GAT_OUT    = IN_DIM  # we’ll output same dim as input
NUM_LAYERS = 2
HEADS      = [4,1]
LR         = 5e-3
WD         = 5e-4
EPOCHS     = 50
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 2) LOAD DATA ─────────────────────────────────────────────────────────────
emb_df  = pd.read_csv(EMB_CSV)
edge_df = pd.read_csv(EDGE_CSV)
y_df    = pd.read_csv(Y_CSV)

# global service→idx
all_svcs = pd.concat([
    emb_df["service"],
    edge_df["source_service"], edge_df["target_service"],
    y_df["service"]
]).unique()
svc2idx = {s:i for i,s in enumerate(all_svcs)}
N = len(svc2idx)

# pivot ground‑truth to dict: t→tensor[N]
y_dict = {}
for t, grp in y_df.groupby("timestamp"):
    vec = torch.zeros(N)
    for _,r in grp.iterrows():
        vec[svc2idx[r["service"]]] = r["replicas"]
    y_dict[t] = vec

# sorted list of timestamps present in both emb & edges & y
timestamps = sorted(set(emb_df["timestamp"]) & set(edge_df["timestamp"]) & set(y_dict.keys()))

# ── 3) MODEL DEFINITION ──────────────────────────────────────────────────────
class ScalerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gat = GAT(
            num_of_layers          = NUM_LAYERS,
            num_heads_per_layer    = HEADS,
            num_features_per_layer = [IN_DIM, HID_DIM, GAT_OUT],
            add_skip_connection    = True,
            bias                   = True,
            dropout                = 0.6,
            layer_type             = LayerType.IMP3
        )
        self.fc = nn.Sequential(
            nn.Linear(GAT_OUT, 1),
            nn.ReLU()
        )
    def forward(self, x, edge_index):
        h, _ = self.gat((x, edge_index))   # → (N, GAT_OUT)
        out = self.fc(h).squeeze(1)        # → (N,)
        return out

model = ScalerModel().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
criterion = nn.MSELoss()

# ── 4) TRAINING LOOP ─────────────────────────────────────────────────────────
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0.0

    for t in timestamps:
        # a) build X_t
        df_e = emb_df[emb_df["timestamp"]==t].set_index("service")
        X = torch.zeros((N, IN_DIM), device=DEVICE)
        for svc, row in df_e.iterrows():
            idx = svc2idx[svc]
            X[idx] = torch.tensor(row.filter(like="embedding_").values, device=DEVICE)

        # b) build edge_index_t
        df_edge = edge_df[edge_df["timestamp"]==t]
        src = df_edge["source_service"].map(svc2idx).tolist()
        dst = df_edge["target_service"].map(svc2idx).tolist()
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=DEVICE)
        # add self‑loops
        self_loops = torch.arange(N, device=DEVICE).unsqueeze(0).repeat(2,1)
        edge_index = torch.cat([edge_index, self_loops], dim=1)

        # c) ground‑truth
        y_true = y_dict[t].to(DEVICE)

        # d) forward + loss + backward
        optimizer.zero_grad()
        y_pred = model(X, edge_index)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(timestamps)
    print(f"Epoch {epoch:03d}  AvgLoss: {avg_loss:.4f}")

# ── 5) SAVE FINAL EMBEDDINGS & PREDICTIONS ───────────────────────────────────
torch.save(model.state_dict(), "scaler_model.pth")
print("Training complete. Model saved to scaler_model.pth")
