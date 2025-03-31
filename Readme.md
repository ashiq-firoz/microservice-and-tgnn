<!-- pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121 -->


# 📚 Microservice Behavior Prediction with Temporal Graph Neural Networks (TGNN) 🤖

 This repository contains a complete pipeline for predicting future resource usage and service request patterns in a microservice-based system using a **Temporal Graph Neural Network (TGNN)**. Below is a detailed guide to help you understand, set up, and use this project effectively.

---

## 🌟 Features of the Project

✨ **Key Capabilities**:
- Predicts **CPU and memory usage** for each microservice.
- Forecasts **request rates** between services.
- Visualizes historical and predicted behavior for better insights.
- Saves predictions into CSV files for further analysis.

📈 **Applications**:
- Optimize resource allocation and scaling decisions.
- Detect anomalies and performance bottlenecks.
- Understand service dependencies and communication patterns.

---

## 📂 Project Structure

Here’s an overview of the project files and their purpose:

| File/Component                  | Description                                                                                   |
|---------------------------------|-----------------------------------------------------------------------------------------------|
| `service-to-service.csv`        | Contains service request data (source, target, timestamp, request count).                     |
| `cpu_usage.csv`                 | Contains CPU usage data (Pod, timestamp, CPU usage).                                          |
| `memory_usage.csv`              | Contains memory usage data (Pod, timestamp, memory usage).                                    |
| `microservice_tgnn_model.pth`   | The trained TGNN model file for making predictions.                                           |
| `test_model.py`                 | A standalone script to load the trained model and make predictions.                           |
| `visualizations/`               | Directory for saving visualizations (optional).                                               |

---

## 🛠️ Prerequisites

Before running the project, ensure you have the following installed:

### 1. **Python Libraries**
Install the required Python libraries using `pip`:
```bash
conda create -n devops python=3.9.21 anaconda
conda activate devops
pip install -r requirements.txt
```
or try
```
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
```

### 2. **Dataset Files**
Ensure the following files are present in the project directory:
- `service-to-service.csv`
- `cpu_usage.csv`
- `memory_usage.csv`

The dataset should follow the structure described below.

---

## 📊 Dataset Format

### 1. **Service Requests (`service-to-service.csv`)**
| Column Name         | Description                                      |
|---------------------|--------------------------------------------------|
| `Source Service`    | Name of the source service.                      |
| `Destination Service` | Name of the destination service.                |
| `Unix Timestamp`    | Timestamp of the request (in Unix format).       |
| `Request Count`     | Number of requests made during the timestamp.    |

### 2. **CPU Usage (`cpu_usage.csv`)**
| Column Name         | Description                                      |
|---------------------|--------------------------------------------------|
| `Pod`               | Name of the pod (e.g., `service-a-pod-1`).       |
| `Unix Timestamp`    | Timestamp of the CPU usage record.               |
| `Cpu usage`         | CPU usage value (in arbitrary units).             |

### 3. **Memory Usage (`memory_usage.csv`)**
| Column Name         | Description                                      |
|---------------------|--------------------------------------------------|
| `Pod`               | Name of the pod (e.g., `service-a-pod-1`).       |
| `Unix Timestamp`    | Timestamp of the memory usage record.            |
| `Memory Usage`      | Memory usage value (in arbitrary units).         |

---

## 🏃‍♂️ How to Run the Project

### 1. **Training the Model**
To train the TGNN model:
1. Place the dataset files (`service-to-service.csv`, `cpu_usage.csv`, `memory_usage.csv`) in the project directory.
2. Run the main script:
   ```bash
   python trial1.py
   ```
3. The script will:
   - Load and preprocess the dataset.
   - Train the TGNN model.
   - Generate predictions for future resource usage and service requests.
   - Save the trained model as `microservice_tgnn_model.pth`.

### 2. **Making Predictions**
To make predictions using the trained model:
1. Ensure the trained model file (`microservice_tgnn_model.pth`) is in the project directory.
2. Run the test script:
   ```bash
   python test-model.py
   ```
3. The script will:
   - Load the trained model.
   - Use the latest data to predict future behavior.
   - Print and save the predictions.

---

## 🎨 Visualization

The project includes several visualization functions to help you interpret the results:

### 1. **Service Dependency Graph**
Visualize the service dependency graph at a specific timestamp:
```python
visualize_service_graph(df_requests, df_resources, timestamp)
```
📊 **What it shows**:
- Nodes represent services.
- Edges represent service-to-service requests with weights indicating request rates.

### 2. **Resource Usage Predictions**
Visualize historical and predicted resource usage for a specific service:
```python
visualize_predictions(original_resources, predicted_resources, service_name)
```
📈 **What it shows**:
- Historical and predicted CPU/memory usage trends over time.

### 3. **Predicted Dependency Graphs**
Visualize the predicted service dependency graphs for future timestamps:
```python
visualize_predicted_graphs(df_requests, df_resources, predicted_resources, predicted_requests, timestamps)
```
🔮 **What it shows**:
- Predicted service interactions and resource usage for future timestamps.

---

## 📝 Example Output

### 1. **Predicted Resource Usage**
The script saves predictions into `predicted_resource_usage.csv`:
```csv
timestamp(unix),service,cpu_usage,memory_usage
1740685688,service-a,0.48,0.63
1740685748,service-a,0.49,0.64
```

### 2. **Predicted Service Requests**
The script saves predictions into `predicted_service_requests.csv`:
```csv
timestamp(unix),source_service,target_service,requests_per_second
1740685688,service-a,service-b,130.0
1740685748,service-a,service-b,135.0
```

---

## 🔧 Troubleshooting

If you encounter any issues, check the following:

1. **Error: 'Column not found'**
   - Ensure the dataset files have the expected columns (`Source Service`, `Destination Service`, `Request Count`, etc.).
   - Verify that the `Pod` column exists in `cpu_usage.csv` and `memory_usage.csv`.

2. **Error: 'Not enough timestamps'**
   - Ensure the dataset contains sufficient timestamps to cover the sequence length and prediction horizon.

3. **Error: 'ModuleNotFoundError'**
   - Install missing libraries using `pip install <library_name>`.

---



<!-- ## 🙏 Acknowledgments

Special thanks to:
- **PyTorch Geometric** for providing powerful tools for graph neural networks.
- **NetworkX** for graph visualization.
- **Pandas** and **NumPy** for data manipulation.

--- -->

