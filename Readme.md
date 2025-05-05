Here's a cleaned-up and professional version of your README with improved formatting, clearer structure, fixed typos, and concise explanations:

---

# 📚 Microservice Behavior Prediction with Temporal Graph Neural Networks (TGNN) & Graph Attention Network (GAT)

This repository contains a complete pipeline for predicting future resource usage and service request patterns in a microservice-based system using **Temporal Graph Neural Network (TGNN)** & **Graph Attention Network**.

---

## 🌟 Features

**Core Capabilities:**

* 🔮 Predicts **CPU and memory usage** for each microservice.
* 🔗 Forecasts **request rates** between services.
* 📊 Visualizes historical and predicted behavior.
* 💾 Saves predictions to CSV for further analysis.

**Use Cases:**

* 🚀 Optimizing autoscaling and resource allocation.
* 🛡️ Detecting anomalies and performance issues.
* 🧠 Understanding service dependencies and patterns.

---

## 📁 Project Structure

```
.
├───csv_files/                  # 📤 Predicted outputs (CPU, memory, service requests)
├───dataset/                   # 📥 Raw data files (collected and created)
│   ├── service-to-service.csv
│   ├── cpu_usage.csv
│   ├── memory_usage.csv
│   └── microservice.csv       # JODIE-formatted dataset
├───embeddings/                # 🔮 Predicted embeddings from TGNN trials
├───models/
│   ├── binaries/
│   ├── checkpoints/
│   └── definitions/
│       └── GAT.py             # Graph Attention Network model
├───Output_Embeddings/         # ✨ Final node embeddings from TGNN + GAT
├───outputs/                   # 🔧 Temporary output files
├───runs/                      # 🧪 Training and evaluation logs
├───Trials/                    # 🚀 Main scripts for training/testing
│   ├── trial.py
│   ├── trail1.py
│   ├── trail3.py
│   └── test-model.py
├───utils/                     # ⚙️ Utility scripts for GAT + conversions
├── .gitignore
├── environment.yml
├── requirements.txt
├── Readme.md
├── figures.py
├── fully-connected.py
├── gat-trial.py
├── jodie-convert.py
└── microservice_tgnn_mode.pth
└── Readme.md
└── requirements.txt
└── tgn-embeddings.py
```

## 📦 Environment Setup

Use **Anaconda** to manage dependencies:

```bash
conda create -n devops python=3.9.21 anaconda
conda activate devops
pip install -r requirements.txt
```

✅ If using CUDA:

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
```

Alternatively, use the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate devops
```

---

## 📊 Dataset Overview

All source data is placed in the `dataset/` folder.

| File                     | Description                                            |
| ------------------------ | ------------------------------------------------------ |
| `service-to-service.csv` | Request data: source, target, timestamp, request count |
| `cpu_usage.csv`          | CPU usage: pod name, timestamp, usage value            |
| `memory_usage.csv`       | Memory usage: pod name, timestamp, usage value         |
| `microservice.csv`       | JODIE format (user, item, timestamp, features, labels) |


## 🚀 How to Run

> ⚠️ **Important:** Always cross-check and update file paths inside each script before running (e.g., `trial.py`, `test-model.py`). Most paths will be hardcoded to old structures.

### 🔧 Step 1: Train TGNN

```bash
python Trials/trial.py
```

* Preprocesses raw data
* Trains the model and saves `microservice_tgnn_model.pth`
* Generates predicted embeddings and outputs

### 🧪 Step 2: Test Model

```bash
python Trials/test-model.py
```

* Loads trained model and predicts:

  * CPU & memory usage
  * Request rates between services
* Saves predictions to `csv_files/`

---

## 📁 Output Files (in `csv_files/`)

### `predicted_resource_usage.csv`

```
timestamp,service,cpu_usage,memory_usage
1740685688,service-a,0.48,0.63
```

### `predicted_service_requests.csv`

```
timestamp,source_service,target_service,requests_per_second
1740685688,service-a,service-b,130.0
```

---

## 📈 Visualizations

Add these in `figures.py` or call from your trials:

```python
from figures import *

visualize_service_graph(...)
visualize_predictions(...)
visualize_predicted_graphs(...)
```

---

## 🔗 GAT Support

The GAT model is used for embedding refinement:

* Model definition: `models/definitions/GAT.py`
* Training: `gat-trial.py` or `fully-connected.py`
* Utilities: `utils/` directory

Run:

```bash
python gat-trial.py
```

---

## 🔄 JODIE Conversion

Convert raw service & pod data into JODIE format using:

```bash
python jodie-convert.py
```

> Make sure the input filenames inside the script point to files in `dataset/`.

---


## 🧬 Feature Encoding Overview for JODIE format

This is how each interaction is encoded internally:

| Field         | Value       | Description                         |
| ------------- | ----------- | ----------------------------------- |
| `user`        | frontend    | Source microservice                 |
| `item`        | adservice   | Target microservice                 |
| `timestamp`   | 1740681008  | Time of interaction (Unix format)   |
| `state_label` | 0           | Default state label                 |
| `features`    | 3.046586074 | Request count (or derived features) |

**Feature Index Mapping:**

| Index | Description                      |
| ----- | -------------------------------- |
| 0     | Request count                    |
| 1     | Avg CPU usage of user service    |
| 2     | Avg memory usage of user service |
| 3     | Avg CPU usage of item service    |
| 4     | Avg memory usage of item service |
| 5     | Number of pods for user service  |

---

## 🧾 File Notes

| Script/File        | Purpose                                           |
| ------------------ | ------------------------------------------------- |
| `trial.py`         | TGNN model training                               |
| `trail1.py`        | Alternate trial (TGAT or TGNN variant)            |
| `test-model.py`    | Loads model & predicts resources + request counts |
| `gat-trial.py`     | Graph Attention model for node embeddings         |
| `jodie-convert.py` | Converts CSV files to JODIE format                |
| `figures.py`       | Visualization utilities                           |

---

## 📌 Final Notes

✅ Before running any script:

* Check all **file paths** to point to your actual folders (`dataset/`, `csv_files/`, etc.)
* Ensure all dependencies are installed
* Activate your `devops` conda environment
---