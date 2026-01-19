"""
Argus Forecasting Model
========================
Multi-task model with seizure detection and forecasting heads.

Architecture:
- Shared CNN-LSTM backbone for feature extraction
- Detection head: Binary classification (seizure NOW or background)
- Forecasting head: Binary classification (pre-ictal or inter-ictal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskSeizureModel(nn.Module):
    """
    Multi-task model for seizure detection and forecasting.
    
    Shared backbone with separate task-specific heads.
    """
    
    def __init__(self, 
                 n_channels=32,
                 sequence_length=2000,
                 hidden_size=128,
                 num_lstm_layers=2,
                 dropout=0.4,
                 use_attention=True):
        
        super(MultiTaskSeizureModel, self).__init__()
        
        self.n_channels = n_channels
        self.use_attention = use_attention
        
        # ============================================
        # SHARED FEATURE EXTRACTION (CNN)
        # ============================================
        
        # Multi-scale temporal convolutions
        self.conv1_7 = nn.Conv1d(n_channels, 64, kernel_size=7, padding=3)
        self.conv1_11 = nn.Conv1d(n_channels, 64, kernel_size=11, padding=5)
        self.conv1_15 = nn.Conv1d(n_channels, 64, kernel_size=15, padding=7)
        
        self.bn1 = nn.BatchNorm1d(192)  # 64*3 channels
        self.pool1 = nn.MaxPool1d(kernel_size=4)
        self.dropout1 = nn.Dropout(dropout)
        
        # Second conv block
        self.conv2 = nn.Conv1d(192, 128, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=4)
        self.dropout2 = nn.Dropout(dropout)
        
        # Third conv block
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=4)
        self.dropout3 = nn.Dropout(dropout)
        
        # ============================================
        # SHARED TEMPORAL MODELING (LSTM)
        # ============================================
        
        # Calculate sequence length after pooling
        seq_after_pool = sequence_length // (4 * 4 * 4)  # 2000 -> 31
        
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        lstm_output_size = hidden_size * 2  # bidirectional
        
        # ============================================
        # ATTENTION MECHANISM (SHARED)
        # ============================================
        
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_output_size, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        
        # ============================================
        # DETECTION HEAD (Seizure NOW)
        # ============================================
        
        self.detection_head = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # ============================================
        # FORECASTING HEAD (Pre-ictal prediction)
        # ============================================
        
        self.forecasting_head = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, task='detection'):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, channels, time)
            task: 'detection', 'forecasting', or 'both'
        
        Returns:
            If task='detection': detection logits
            If task='forecasting': forecasting logits  
            If task='both': (detection_logits, forecasting_logits)
        """
        batch_size = x.size(0)
        
        # ============================================
        # SHARED FEATURE EXTRACTION
        # ============================================
        
        # Multi-scale convolutions
        x1 = F.relu(self.conv1_7(x))
        x2 = F.relu(self.conv1_11(x))
        x3 = F.relu(self.conv1_15(x))
        x = torch.cat([x1, x2, x3], dim=1)
        
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second conv block
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Third conv block
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # ============================================
        # SHARED LSTM
        # ============================================
        
        # Reshape for LSTM (batch, seq, features)
        x = x.permute(0, 2, 1)
        
        lstm_out, _ = self.lstm(x)
        
        # ============================================
        # SHARED ATTENTION
        # ============================================
        
        if self.use_attention:
            # Compute attention weights
            attention_scores = self.attention(lstm_out)  # (batch, seq, 1)
            attention_weights = F.softmax(attention_scores, dim=1)
            
            # Apply attention
            context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden*2)
        else:
            # Use last hidden state
            context = lstm_out[:, -1, :]
        
        # ============================================
        # TASK-SPECIFIC HEADS
        # ============================================
        
        if task == 'detection':
            return self.detection_head(context)
        
        elif task == 'forecasting':
            return self.forecasting_head(context)
        
        elif task == 'both':
            detection_logits = self.detection_head(context)
            forecasting_logits = self.forecasting_head(context)
            return detection_logits, forecasting_logits
        
        else:
            raise ValueError(f"Unknown task: {task}")


class ForecastingOnlyModel(nn.Module):
    """
    Standalone forecasting model (same architecture as detection model).
    Use this if you want to train forecasting separately.
    """
    
    def __init__(self, 
                 n_channels=22,
                 sequence_length=2000,
                 hidden_size=128,
                 num_lstm_layers=2,
                 dropout=0.4,
                 use_attention=True):
        
        super(ForecastingOnlyModel, self).__init__()
        
        # Create multi-task model and only use forecasting head
        self.model = MultiTaskSeizureModel(
            n_channels=n_channels,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            num_lstm_layers=num_lstm_layers,
            dropout=dropout,
            use_attention=use_attention
        )
    
    def forward(self, x):
        return self.model(x, task='forecasting')


def count_parameters(model):
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    print("Testing Forecasting Model...")
    
    # Create model
    model = MultiTaskSeizureModel(
        n_channels=22,
        sequence_length=2000,
        hidden_size=128,
        num_lstm_layers=2,
        dropout=0.4,
        use_attention=True
    )
    
    # Count parameters
    n_params = count_parameters(model)
    print(f"\nTotal parameters: {n_params:,}")
    print(f"Model size: {n_params * 4 / 1024 / 1024:.2f} MB")
    
    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 22, 2000)
    
    # Test detection
    det_out = model(x, task='detection')
    print(f"\nDetection output shape: {det_out.shape}")
    
    # Test forecasting
    fore_out = model(x, task='forecasting')
    print(f"Forecasting output shape: {fore_out.shape}")
    
    # Test both
    det_out, fore_out = model(x, task='both')
    print(f"\nBoth outputs - Detection: {det_out.shape}, Forecasting: {fore_out.shape}")
    
    print("\n✓ Model test complete!")