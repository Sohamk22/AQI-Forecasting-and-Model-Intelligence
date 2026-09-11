"""
Temporal Convolutional Network (TCN) PyTorch model for AQI forecasting.
Implements causal dilated 1D convolutions with residual connections.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, List
from src.utils.logger import get_logger

logger = get_logger("TCNModel")

class Chomp1d(nn.Module):
    """Trims future padding to ensure causal convolution (no lookahead)."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int, dilation: int, padding: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNNet(nn.Module):
    def __init__(self, input_dim: int, num_channels: List[int], kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size,
                dropout=dropout
            ))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_channels[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, seq_len, input_dim)
        # Conv1d expects (batch_size, input_dim, seq_len)
        x_trans = x.transpose(1, 2)
        y = self.network(x_trans)
        # Final sequence output
        last_step = y[:, :, -1]
        out = self.fc(last_step)
        return out.squeeze(-1)

class TCNModel:
    def __init__(self, input_dim: int, params: Dict[str, Any] = None, device: str = None):
        if params is None:
            params = {
                "num_channels": [32, 32, 64, 64],
                "kernel_size": 3,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 40,
                "patience": 8
            }
        self.params = params
        self.input_dim = input_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = TCNNet(
            input_dim=input_dim,
            num_channels=params.get("num_channels", [32, 32, 64, 64]),
            kernel_size=params.get("kernel_size", 3),
            dropout=params.get("dropout", 0.2)
        ).to(self.device)
        self.best_loss = float("inf")
        self.best_state = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            verbose: bool = False):
        
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.params["batch_size"], shuffle=False)
        
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.params["learning_rate"])
        criterion = nn.MSELoss()
        
        has_val = X_val is not None and y_val is not None
        if has_val:
            val_X = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            val_y = torch.tensor(y_val, dtype=torch.float32).to(self.device)
            
        patience_counter = 0
        best_val_loss = float("inf")
        
        for epoch in range(1, self.params["epochs"] + 1):
            self.net.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.net(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch_x)
                
            train_loss /= len(train_dataset)
            
            if has_val:
                self.net.eval()
                with torch.no_grad():
                    val_pred = self.net(val_X)
                    val_loss = criterion(val_pred, val_y).item()
                    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if verbose and epoch % 5 == 0:
                    logger.info(f"TCN Epoch {epoch:02d}: Train Loss={train_loss:.2f}, Val Loss={val_loss:.2f}")
                    
                if patience_counter >= self.params["patience"]:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                self.best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                if verbose and epoch % 5 == 0:
                    logger.info(f"TCN Epoch {epoch:02d}: Train Loss={train_loss:.2f}")
                    
        if self.best_state is not None:
            self.net.load_state_dict({k: v.to(self.device) for k, v in self.best_state.items()})
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        loader = DataLoader(torch.tensor(X, dtype=torch.float32), batch_size=64, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch_x in loader:
                batch_x = batch_x.to(self.device)
                out = self.net(batch_x)
                preds.extend(out.cpu().numpy().tolist())
        return np.array(preds, dtype=np.float32)

    def save(self, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "input_dim": self.input_dim,
            "params": self.params
        }, filepath)
        logger.info(f"Saved TCN model to {filepath}")
        
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.input_dim = checkpoint["input_dim"]
        self.params = checkpoint["params"]
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.to(self.device)
        logger.info(f"Loaded TCN model from {filepath}")
        return self
