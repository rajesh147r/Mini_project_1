import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import streamlit as st
import os
from pathlib import Path

# Set random seed for reproducibility
torch.manual_seed(42)

# Optimize for CPU: Use all available CPU cores
torch.set_num_threads(torch.get_num_threads())

# Streamlit App
st.title("LSTM Time Series Prediction Dashboard")
st.write("Upload a single CSV file to generate predictions using the pre-trained LSTM model and view MSE, MAE, and result graphs.")

# File uploader (single file)
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], accept_multiple_files=False)

# Directory to save uploaded file
UPLOAD_DIR = "./fastStorage/2013-8/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=14, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_sizes[0], batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.lstm3 = nn.LSTM(hidden_sizes[1], hidden_sizes[2], batch_first=True)
        self.fc = nn.Linear(hidden_sizes[2], output_size)
    
    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out, _ = self.lstm3(out)
        out = self.fc(out[:, -1, :])
        return out

# Load pre-trained model
device = torch.device("cpu")
model = LSTMModel(input_size=14, output_size=14).to(device)
model_path = "enhanced_lstm_model_full_dataset.pth"
if not os.path.exists(model_path):
    st.error("Pre-trained model file 'enhanced_lstm_model_full_dataset.pth' not found. Please ensure the model is available.")
else:
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    st.success("Pre-trained model loaded successfully!")

    # Process uploaded file
    if uploaded_file:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully!")

        # Step 1: Load the CSV File
        try:
            df = pd.read_csv(file_path, delimiter=";", skipinitialspace=True)
            df.columns = df.columns.str.strip()
            st.write(f"Data Shape after loading: {df.shape}")
        except Exception as e:
            st.error(f"Error loading CSV file: {e}")
            st.stop()

        # Step 2: Convert Timestamp to Datetime & Sort
        try:
            df["Timestamp [ms]"] = pd.to_datetime(df["Timestamp [ms]"], unit='ms')
            df = df.sort_values("Timestamp [ms]").reset_index(drop=True)
        except Exception as e:
            st.error(f"Error processing timestamp: {e}")
            st.stop()

        # Optional: Resample to 1-second intervals
        df.set_index("Timestamp [ms]", inplace=True)
        df = df.resample("1s").mean().interpolate()
        df.reset_index(inplace=True)
        st.write(f"Data Shape after resampling: {df.shape}")

        # Step 3: Feature Engineering
        try:
            df["CPU Utilization Per Core"] = df["CPU usage [MHZ]"] / df["CPU capacity provisioned [MHZ]"]
            df["Memory Utilization [%]"] = df["Memory usage [KB]"] / df["Memory capacity provisioned [KB]"]
            df["Disk Total Throughput [KB/s]"] = df["Disk read throughput [KB/s]"] + df["Disk write throughput [KB/s]"]
            df["Network Total Throughput [KB/s]"] = df["Network received throughput [KB/s]"] + df["Network transmitted throughput [KB/s]"]
        except Exception as e:
            st.error(f"Error in feature engineering: {e}")
            st.stop()

        # Step 4: Select Features
        features = [
            "CPU cores", "CPU capacity provisioned [MHZ]", "CPU usage [MHZ]", "CPU usage [%]",
            "Memory capacity provisioned [KB]", "Memory usage [KB]", "Disk read throughput [KB/s]",
            "Disk write throughput [KB/s]", "Network received throughput [KB/s]", "Network transmitted throughput [KB/s]",
            "CPU Utilization Per Core", "Memory Utilization [%]", "Disk Total Throughput [KB/s]", "Network Total Throughput [KB/s]"
        ]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        st.write(f"Data Shape after preprocessing: {df.shape}")

        # Step 5: Normalize Data
        scaler = MinMaxScaler()
        df[features] = scaler.fit_transform(df[features])

        # Step 6: Create Sequences
        def create_sequences(data, seq_length=5):
            sequences, targets = [], []
            for i in range(len(data) - seq_length):
                sequences.append(data[i:i+seq_length])
                targets.append(data[i+seq_length])
            return np.array(sequences), np.array(targets)

        seq_length = 5
        X, y = create_sequences(df[features].values, seq_length)
        st.write(f"Sequence Shape: X={X.shape}, y={y.shape}")

        # Step 7: Define PyTorch Dataset
        class TimeSeriesDataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.tensor(X, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.float32)
                
            def __len__(self):
                return len(self.X)
                
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        # Create DataLoader for predictions
        dataset = TimeSeriesDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

        # Step 8: Generate Predictions
        y_pred = []
        with torch.no_grad():
            for X_batch, _ in data_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                y_pred.append(outputs.cpu().numpy())
        y_pred = np.concatenate(y_pred, axis=0)
        y_test = y  # Ground truth for metrics

        # Step 9: Calculate Metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # Display Metrics
        st.subheader("Model Performance Metrics")
        st.write(f"**Mean Squared Error (MSE):** {mse:.6f}")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.6f}")

        # Step 10: Visualizations for Predictions
        # --- ACTUAL vs PREDICTED VISUALIZATION ---
        st.subheader("Actual vs Predicted Values")
        fig, axes = plt.subplots(5, 3, figsize=(18, 14))
        axes = axes.flatten()
        for i in range(len(features)):
            axes[i].plot(y_test[:, i][::10], label=f"Actual {features[i]}")
            axes[i].plot(y_pred[:, i][::10], label=f"Predicted {features[i]}", linestyle="dashed")
            axes[i].set_xlabel("Time")
            axes[i].set_ylabel(features[i])
            axes[i].legend()
            axes[i].set_title(f"LSTM Prediction vs Actual - {features[i]}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # --- ERROR DISTRIBUTION ---
        st.subheader("Error Distribution")
        errors = y_test - y_pred
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.boxplot(data=errors, ax=ax)
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels(features, rotation=45, ha="right")
        ax.set_title("Prediction Error Distribution Across Features")
        ax.set_ylabel("Prediction Error")
        ax.grid()
        st.pyplot(fig)
        plt.close(fig)

        # --- CPU Usage [%] Focused Plot ---
        st.subheader("CPU Usage [%] Prediction")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(y_test[:, 3], label="Actual CPU Usage [%]")
        ax.plot(y_pred[:, 3], label="Predicted CPU Usage [%]", linestyle="dashed")
        ax.set_xlabel("Time")
        ax.set_ylabel("CPU Usage [%]")
        ax.legend()
        ax.set_title("LSTM Prediction vs Actual (CPU Usage [%])")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Please upload a single CSV file to generate predictions.")