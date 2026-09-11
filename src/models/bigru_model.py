"""
Bidirectional GRU (Bi-GRU) PyTorch model for AQI temporal forecasting.
Maintains identical unit sizes and training protocol with Bi-LSTM for controlled comparison.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any
from src.utils.logger import get_logger

logger = get_logger("BiGRUModel")

class BiGRUNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, hn = self.gru(x)
        # In bidirectional PyTorch GRU: hn[-2] is forward last step, hn[-1] is backward first step
        feat = torch.cat([hn[-2], hn[-1]], dim=1)
        out = self.fc(feat)
        return out.squeeze(-1)

class BiGRUModel:
    def __init__(self, input_dim: int, params: Dict[str, Any] = None, device: str = None):
        if params is None:
            params = {
                "hidden_dim": 64,
                "num_layers": 1,
                "dropout": 0.2,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 40,
                "patience": 8
            }
        self.params = params
        self.input_dim = input_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = BiGRUNet(
            input_dim=input_dim,
            hidden_dim=params.get("hidden_dim", 64),
            num_layers=params.get("num_layers", 1),
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
                    logger.info(f"Bi-GRU Epoch {epoch:02d}: Train Loss={train_loss:.2f}, Val Loss={val_loss:.2f}")
                    
                if patience_counter >= self.params["patience"]:
                    if verbose:
                        logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                self.best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                if verbose and epoch % 5 == 0:
                    logger.info(f"Bi-GRU Epoch {epoch:02d}: Train Loss={train_loss:.2f}")
                    
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
        logger.info(f"Saved Bi-GRU model to {filepath}")
        
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.input_dim = checkpoint["input_dim"]
        self.params = checkpoint["params"]
        self.net.load_state_dict(checkpoint["state_dict"])
        self.net.to(self.device)
        logger.info(f"Loaded Bi-GRU model from {filepath}")
        return self
