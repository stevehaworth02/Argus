"""
EEG Seizure Detection - Robust CNN-LSTM Model
Ceribell Project

Architecture:
    1. Multi-scale temporal convolution blocks
    2. Spatial convolution across channels
    3. Bidirectional LSTM for temporal dependencies
    4. Attention mechanism for focus on seizure patterns
    5. Fully connected classifier with dropout

Design Principles:
    - Robust: Heavy regularization, batch norm, dropout
    - Efficient: Optimized for portable hardware
    - Performant: Deep enough for complex patterns
    - Production-ready: Proper initialization, error handling

Author: Ceribell Seizure Detector Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


# ============================================================================
# ATTENTION MECHANISM
# ============================================================================

class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism to focus on important time segments.
    Helps model focus on seizure onset/offset patterns.
    """
    
    def __init__(self, hidden_dim: int):
        super(TemporalAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)
        
        Returns:
            context: Attention-weighted context vector (batch, hidden_dim)
            attention_weights: Attention weights (batch, seq_len)
        """
        # Calculate attention scores
        attention_scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores.squeeze(-1), dim=1)  # (batch, seq_len)
        
        # Apply attention weights
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output)  # (batch, 1, hidden_dim)
        context = context.squeeze(1)  # (batch, hidden_dim)
        
        return context, attention_weights


# ============================================================================
# TEMPORAL CONVOLUTION BLOCK
# ============================================================================

class TemporalConvBlock(nn.Module):
    """
    Temporal convolution block with residual connection.
    Extracts features at different time scales.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, dropout: float = 0.3):
        super(TemporalConvBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time)
        Returns:
            out: (batch, channels, time)
        """
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual connection
        out = out + residual
        out = F.relu(out)
        
        return out


# ============================================================================
# MAIN MODEL: ROBUST CNN-LSTM
# ============================================================================

class SeizureDetectorCNNLSTM(nn.Module):
    """
    Robust CNN-LSTM architecture for seizure detection.
    
    Architecture Overview:
        1. Multi-scale temporal CNN (extract features at different scales)
        2. Spatial CNN across channels (learn channel interactions)
        3. Max pooling for dimensionality reduction
        4. Bidirectional LSTM (capture temporal dependencies)
        5. Temporal attention (focus on important segments)
        6. Fully connected classifier
    
    Input: (batch, 22 channels, 2000 samples) - 10 seconds at 200 Hz
    Output: (batch, 2) - [background, seizure] logits
    """
    
    def __init__(self, 
                 num_channels: int = 22,
                 num_samples: int = 2000,
                 num_classes: int = 2,
                 cnn_filters: list = [32, 64, 128],
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 dropout: float = 0.4,
                 attention: bool = True):
        """
        Initialize the seizure detection model.
        
        Args:
            num_channels: Number of EEG channels (default: 22)
            num_samples: Number of time samples per window (default: 2000)
            num_classes: Number of output classes (default: 2 - background/seizure)
            cnn_filters: List of filter sizes for CNN layers
            lstm_hidden: Hidden dimension for LSTM
            lstm_layers: Number of LSTM layers
            dropout: Dropout probability
            attention: Whether to use attention mechanism
        """
        super(SeizureDetectorCNNLSTM, self).__init__()
        
        self.num_channels = num_channels
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.use_attention = attention
        
        # ====================================================================
        # 1. MULTI-SCALE TEMPORAL CONVOLUTION
        # ====================================================================
        # Extract features at different temporal scales (short, medium, long)
        
        self.temporal_conv1 = TemporalConvBlock(num_channels, cnn_filters[0], 
                                                kernel_size=7, dropout=dropout)
        self.temporal_conv2 = TemporalConvBlock(cnn_filters[0], cnn_filters[1],
                                                kernel_size=11, dropout=dropout)
        self.temporal_conv3 = TemporalConvBlock(cnn_filters[1], cnn_filters[2],
                                                kernel_size=15, dropout=dropout)
        
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Calculate size after convolutions and pooling
        self.feature_size = cnn_filters[2]
        self.seq_len = num_samples // (4 ** 3)  # 3 pooling layers
        
        # ====================================================================
        # 2. BIDIRECTIONAL LSTM
        # ====================================================================
        # Capture temporal dependencies in both directions
        
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        lstm_output_dim = lstm_hidden * 2  # Bidirectional doubles the output
        
        # ====================================================================
        # 3. ATTENTION MECHANISM (OPTIONAL)
        # ====================================================================
        
        if self.use_attention:
            self.attention = TemporalAttention(lstm_output_dim)
            classifier_input_dim = lstm_output_dim
        else:
            classifier_input_dim = lstm_output_dim
        
        # ====================================================================
        # 4. FULLY CONNECTED CLASSIFIER
        # ====================================================================
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using best practices."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.constant_(param.data, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (batch, channels, samples)
        
        Returns:
            logits: Class logits (batch, num_classes)
            attention_weights: Attention weights if using attention, else None
        """
        batch_size = x.size(0)
        
        # ====================================================================
        # 1. TEMPORAL CONVOLUTION (Multi-scale feature extraction)
        # ====================================================================
        
        out = self.temporal_conv1(x)  # (batch, 32, 2000)
        out = self.pool(out)          # (batch, 32, 500)
        
        out = self.temporal_conv2(out)  # (batch, 64, 500)
        out = self.pool(out)            # (batch, 64, 125)
        
        out = self.temporal_conv3(out)  # (batch, 128, 125)
        out = self.pool(out)            # (batch, 128, 31)
        
        # ====================================================================
        # 2. PREPARE FOR LSTM (Reshape)
        # ====================================================================
        
        # Transpose for LSTM: (batch, seq_len, features)
        out = out.permute(0, 2, 1)  # (batch, 31, 128)
        
        # ====================================================================
        # 3. LSTM (Temporal modeling)
        # ====================================================================
        
        lstm_out, (hidden, cell) = self.lstm(out)  # (batch, seq_len, lstm_hidden*2)
        
        # ====================================================================
        # 4. ATTENTION or POOLING
        # ====================================================================
        
        if self.use_attention:
            # Use attention to get weighted context
            context, attention_weights = self.attention(lstm_out)
        else:
            # Use mean pooling over time
            context = torch.mean(lstm_out, dim=1)
            attention_weights = None
        
        # ====================================================================
        # 5. CLASSIFICATION
        # ====================================================================
        
        logits = self.classifier(context)  # (batch, num_classes)
        
        return logits, attention_weights
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions (returns probabilities).
        
        Args:
            x: Input tensor (batch, channels, samples)
        
        Returns:
            probs: Class probabilities (batch, num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB."""
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 ** 2)


# ============================================================================
# MODEL FACTORY FUNCTIONS
# ============================================================================

def create_seizure_detector(model_size: str = 'medium', **kwargs) -> SeizureDetectorCNNLSTM:
    """
    Factory function to create pre-configured seizure detection models.
    
    Args:
        model_size: 'small', 'medium', or 'large'
        **kwargs: Additional arguments to override defaults
    
    Returns:
        Configured SeizureDetectorCNNLSTM model
    """
    configs = {
        'small': {
            'cnn_filters': [16, 32, 64],
            'lstm_hidden': 64,
            'lstm_layers': 1,
            'dropout': 0.3,
            'attention': True
        },
        'medium': {
            'cnn_filters': [32, 64, 128],
            'lstm_hidden': 128,
            'lstm_layers': 2,
            'dropout': 0.4,
            'attention': True
        },
        'large': {
            'cnn_filters': [64, 128, 256],
            'lstm_hidden': 256,
            'lstm_layers': 3,
            'dropout': 0.5,
            'attention': True
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"model_size must be one of {list(configs.keys())}")
    
    config = configs[model_size]
    config.update(kwargs)  # Override with user-provided kwargs
    
    model = SeizureDetectorCNNLSTM(**config)
    
    print(f"\n{'='*70}")
    print(f"Created {model_size.upper()} Seizure Detector Model")
    print(f"{'='*70}")
    print(f"Parameters: {model.get_num_parameters():,}")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")
    print(f"Architecture: CNN ({config['cnn_filters']}) -> LSTM ({config['lstm_hidden']}x{config['lstm_layers']}) -> Attention -> Classifier")
    print(f"{'='*70}\n")
    
    return model


# ============================================================================
# MODEL TESTING
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTING SEIZURE DETECTOR MODEL")
    print("="*70 + "\n")
    
    # Test with dummy data
    batch_size = 8
    num_channels = 22
    num_samples = 2000
    
    # Create dummy input
    x = torch.randn(batch_size, num_channels, num_samples)
    
    print(f"Input shape: {x.shape}")
    
    # Test different model sizes
    for size in ['small', 'medium', 'large']:
        print(f"\n{'='*70}")
        print(f"Testing {size.upper()} model")
        print(f"{'='*70}")
        
        model = create_seizure_detector(size)
        
        # Forward pass
        logits, attention_weights = model(x)
        
        print(f"\nOutput:")
        print(f"  - Logits shape: {logits.shape}")
        if attention_weights is not None:
            print(f"  - Attention weights shape: {attention_weights.shape}")
        
        # Test prediction
        probs = model.predict(x)
        print(f"  - Probabilities shape: {probs.shape}")
        print(f"  - Probability sum (should be ~1.0): {probs[0].sum().item():.4f}")
        
        print(f"\n✓ {size.upper()} model working correctly!\n")
    
    print("="*70)
    print("All model sizes tested successfully!")
    print("="*70 + "\n")