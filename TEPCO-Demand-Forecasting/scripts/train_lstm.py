import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import joblib

class ElectricDemandDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0]) # 0 is Demand column index
    return np.array(X), np.array(y)

def train_lstm_torch():
    input_file = "data/processed/featured_dataset.csv"
    model_output = "models/lstm_model.pth"
    scaler_output = "models/scaler.pkl"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading data for LSTM (PyTorch)...")
    df = pd.read_csv(input_file)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df.sort_values('Datetime')
    
    features = ['Demand', 'Temperature', 'RollingMean_24h', 'Lag_24h']
    data = df[features].values
    
    # Scale
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create Sequences
    SEQ_LENGTH = 24
    X, y = create_sequences(data_scaled, SEQ_LENGTH)
    
    # Split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    train_dataset = ElectricDemandDataset(X_train, y_train)
    test_dataset = ElectricDemandDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Model Setup
    input_size = len(features)
    hidden_size = 64
    num_layers = 2
    output_size = 1
    
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    print("Training...")
    num_epochs = 10
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")
        
    # Evaluate
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.numpy())
            
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Inverse Transform logic
    dummy_pred = np.zeros((len(predictions), len(features)))
    dummy_pred[:, 0] = predictions.flatten()
    pred_inverse = scaler.inverse_transform(dummy_pred)[:, 0]
    
    dummy_actual = np.zeros((len(actuals), len(features)))
    dummy_actual[:, 0] = actuals.flatten()
    y_inverse = scaler.inverse_transform(dummy_actual)[:, 0]
    
    # Metrics
    mae = mean_absolute_error(y_inverse, pred_inverse)
    rmse = np.sqrt(mean_squared_error(y_inverse, pred_inverse))
    mape = np.mean(np.abs((y_inverse - pred_inverse) / y_inverse)) * 100
    
    print(f"LSTM (PyTorch) MAE: {mae:.2f}")
    print(f"LSTM (PyTorch) RMSE: {rmse:.2f}")
    print(f"LSTM (PyTorch) MAPE: {mape:.2f}%")
    
    # Save
    torch.save(model.state_dict(), model_output)
    joblib.dump(scaler, scaler_output)
    print(f"Model saved to {model_output}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_inverse[-200:], label='Actual')
    plt.plot(pred_inverse[-200:], label='Predicted')
    plt.title('LSTM Forecast (PyTorch) - Last 200 hours')
    plt.legend()
    plt.savefig("visualizations/lstm_forecast.png")
    
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig("visualizations/lstm_loss.png")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)
    train_lstm_torch()
