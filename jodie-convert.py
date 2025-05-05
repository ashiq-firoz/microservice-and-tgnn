import pandas as pd
from collections import defaultdict

# Load CSV files
interactions = pd.read_csv("service-to-service.csv")  # CSV 1
cpu_usage = pd.read_csv("cpu_usage.csv")               # CSV 2
mem_usage = pd.read_csv("memory_usage.csv")            # CSV 3

# Rename for consistency
interactions.columns = ["Destination", "Source", "Timestamp", "RequestCount"]
cpu_usage.columns = ["Pod", "Timestamp", "CPU"]
mem_usage.columns = ["Pod", "Timestamp", "Memory"]

# Extract service name from pod (before first dash)
cpu_usage["Service"] = cpu_usage["Pod"].str.split("-").str[0]
mem_usage["Service"] = mem_usage["Pod"].str.split("-").str[0]

# Merge CPU and Memory on Pod + Timestamp
resource_df = pd.merge(cpu_usage, mem_usage, on=["Pod", "Timestamp"], how='outer')
resource_df["Service"] = resource_df["Pod"].str.split("-").str[0]

# Group by (Service, Timestamp) → average CPU, Memory, and count of pods
grouped = resource_df.groupby(["Service", "Timestamp"]).agg({
    "CPU": "mean",
    "Memory": "mean",
    "Pod": "count"
}).reset_index()

# Build lookup: {(service, timestamp): (cpu, mem, pod_count)}
resource_lookup = {
    (row.Service, row.Timestamp): (row.CPU, row.Memory, row.Pod)
    for row in grouped.itertuples()
}

# Convert interaction data to JODIE format
output = []
for row in interactions.itertuples():
    user = row.Source
    item = row.Destination
    ts = row.Timestamp
    req_count = row.RequestCount
    state_label = 0

    # Get avg CPU/mem + pod count for user/item services
    user_cpu, user_mem, user_pods = resource_lookup.get((user, ts), (0.0, 0.0, 0))
    item_cpu, item_mem, _ = resource_lookup.get((item, ts), (0.0, 0.0, 0))

    # Feature vector
    features = [req_count, user_cpu, user_mem, item_cpu, item_mem, user_pods]
    features_str = ','.join(map(str, features))

    output.append([user, item, ts, state_label, features_str])

# Write to JODIE-formatted CSV
with open("microservices.csv", "w") as f:
    f.write("user,item,timestamp,state_label,features\n")
    for line in output:
        f.write(f"{line[0]},{line[1]},{line[2]},{line[3]},{line[4]}\n")

print("✅ JODIE dataset generated with features + pod count: microservices.csv")
