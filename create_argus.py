"""
Create Argus Visualization - Saliency Map for Repository Header

This script:
1. Loads a seizure EEG from TUH dataset
2. Generates model predictions with attention weights
3. Computes Grad-CAM saliency maps
4. Creates artistic visualization for GitHub header

Output: High-resolution EEG + saliency overlay that Argus will "point at"
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for WSL
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import pyedflib
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import sys

# Add modules to path
sys.path.append('.')
from modules.model import SeizureDetectorCNNLSTM


# ============================================================================
# GRAD-CAM IMPLEMENTATION
# ============================================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNN visualization.
    Shows which parts of the EEG waveform the CNN focuses on.
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, class_idx=1):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: (1, channels, samples)
            class_idx: Target class (1 = seizure)
        
        Returns:
            cam: (samples,) - importance score for each timepoint
        """
        # Forward pass
        self.model.eval()
        logits, _ = self.model(input_tensor)
        
        # Backward pass for target class
        self.model.zero_grad()
        class_score = logits[0, class_idx]
        class_score.backward()
        
        # Compute weighted activation
        weights = torch.mean(self.gradients, dim=2, keepdim=True)  # Global average pooling
        cam = torch.sum(weights * self.activations, dim=1).squeeze()  # Weighted sum
        
        # ReLU + normalize
        cam = F.relu(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        # Upsample to original length
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                           size=input_tensor.shape[2], 
                           mode='linear', 
                           align_corners=False).squeeze()
        
        return cam.cpu().numpy()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_eeg_with_seizure(data_dir: str = "data/tusz/edf/dev") -> Tuple[np.ndarray, Dict]:
    """
    Load an EEG recording with a seizure event.
    
    Returns:
        data: (channels, samples) EEG data
        metadata: Dict with file info, seizure times, etc.
    """
    data_path = Path(data_dir)
    
    # Find a file with seizure annotation
    print("Searching for seizure recordings...")
    files_checked = 0
    
    for edf_file in data_path.rglob("*.edf"):
        csv_file = edf_file.with_suffix('.csv')
        files_checked += 1
        
        if files_checked % 100 == 0:
            print(f"Checked {files_checked} files...")
        
        if csv_file.exists():
            try:
                # TUH annotations can have inconsistent formatting
                # Try reading with error handling
                annotations = pd.read_csv(csv_file, on_bad_lines='skip')
                
                # Debug: print what we found
                if files_checked < 5:  # Print first few for debugging
                    print(f"CSV columns: {list(annotations.columns)}")
                    print(f"First few rows:\n{annotations.head()}")
                
                # Check if this file has a seizure (not just background)
                if 'label' in annotations.columns and any(annotations['label'] != 'bckg'):
                    print(f"\nFound seizure in: {edf_file.name}")
                    
                    # Load EEG data
                    f = pyedflib.EdfReader(str(edf_file))
                    n_channels = f.signals_in_file
                    signal_labels = f.getSignalLabels()
                    sample_rate = f.getSampleFrequency(0)
                    
                    # Read all channels
                    data = np.array([f.readSignal(i) for i in range(n_channels)])
                    f.close()
                    
                    # Get seizure timing
                    seizure_rows = annotations[annotations['label'] != 'bckg']
                    seizure_start = seizure_rows['start_time'].min()
                    seizure_stop = seizure_rows['stop_time'].max()
                    seizure_type = seizure_rows['label'].iloc[0]
                    
                    metadata = {
                        'filename': edf_file.name,
                        'channels': signal_labels,
                        'sample_rate': sample_rate,
                        'seizure_start': seizure_start,
                        'seizure_stop': seizure_stop,
                        'seizure_type': seizure_type,
                        'duration': data.shape[1] / sample_rate
                    }
                    
                    print(f"Seizure type: {seizure_type}")
                    print(f"Seizure time: {seizure_start:.1f}s - {seizure_stop:.1f}s")
                    print(f"Channels: {n_channels}, Sample rate: {sample_rate} Hz")
                    
                    return data, metadata
            except Exception as e:
                # Print error for first few files to debug
                if files_checked < 5:
                    print(f"Error reading {csv_file.name}: {e}")
                continue
    
    print(f"\nTotal files checked: {files_checked}")
    raise FileNotFoundError("No seizure recordings found in dataset!")


def extract_window(data: np.ndarray, metadata: Dict, 
                   target_length: int = 2000) -> Tuple[np.ndarray, float]:
    """
    Extract a 10-second window containing the seizure onset.
    
    Returns:
        window: (channels, 2000) preprocessed window
        window_start_time: Time in seconds where window starts
    """
    sample_rate = metadata['sample_rate']
    seizure_start_sample = int(metadata['seizure_start'] * sample_rate)
    
    # Extract 10-second window starting 2 seconds before seizure
    window_start_sample = max(0, seizure_start_sample - int(2 * sample_rate))
    window_end_sample = window_start_sample + int(10 * sample_rate)
    
    window = data[:, window_start_sample:window_end_sample]
    window_start_time = window_start_sample / sample_rate
    
    # Ensure we have exactly 22 channels (take first 22 if more)
    if window.shape[0] > 22:
        window = window[:22, :]
    elif window.shape[0] < 22:
        # Pad with zeros if fewer channels
        padding = np.zeros((22 - window.shape[0], window.shape[1]))
        window = np.vstack([window, padding])
    
    # Resample to 200 Hz if needed (your model expects 2000 samples @ 200Hz = 10s)
    if sample_rate != 200:
        from scipy import signal
        window = signal.resample(window, target_length, axis=1)
    
    # Standardize
    window = (window - window.mean(axis=1, keepdims=True)) / (window.std(axis=1, keepdims=True) + 1e-8)
    
    return window, window_start_time


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_argus_visualization(eeg_data: np.ndarray, 
                               attention_weights: np.ndarray,
                               gradcam_map: np.ndarray,
                               metadata: Dict,
                               window_start_time: float,
                               prediction_prob: float,
                               save_path: str = "argus_vision.png"):
    """
    Create the artistic visualization for Argus to "see".
    
    Shows:
    - Raw EEG traces (22 channels)
    - Attention heatmap (what the LSTM focuses on)
    - Grad-CAM heatmap (what the CNN focuses on)
    """
    fig = plt.figure(figsize=(20, 12), facecolor='#0d1117')  # GitHub dark theme
    gs = GridSpec(3, 1, height_ratios=[3, 0.3, 0.3], hspace=0.15)
    
    # Color scheme
    bg_color = '#0d1117'
    text_color = '#c9d1d9'
    seizure_color = '#ff6b6b'
    attention_color = '#4dabf7'
    
    # ========================================================================
    # PANEL 1: EEG TRACES
    # ========================================================================
    
    ax_eeg = fig.add_subplot(gs[0])
    ax_eeg.set_facecolor(bg_color)
    
    n_channels, n_samples = eeg_data.shape
    time = np.linspace(window_start_time, window_start_time + 10, n_samples)
    
    # Plot each channel with offset
    channel_spacing = 8
    for i in range(n_channels):
        offset = i * channel_spacing
        ax_eeg.plot(time, eeg_data[i] + offset, color=text_color, 
                   linewidth=0.8, alpha=0.7)
        
        # Channel labels
        ax_eeg.text(-0.5, offset, f"Ch{i+1}", 
                   fontsize=8, color=text_color, ha='right', va='center')
    
    # Highlight seizure period
    seizure_start = metadata['seizure_start']
    seizure_stop = metadata['seizure_stop']
    ax_eeg.axvspan(seizure_start, seizure_stop, 
                  alpha=0.2, color=seizure_color, label='Seizure')
    
    ax_eeg.set_xlim(window_start_time, window_start_time + 10)
    ax_eeg.set_ylim(-channel_spacing, n_channels * channel_spacing)
    ax_eeg.set_xlabel('Time (s)', fontsize=14, color=text_color)
    ax_eeg.set_ylabel('EEG Channels', fontsize=14, color=text_color)
    ax_eeg.set_title(f'ARGUS VISION: {metadata["seizure_type"].upper()} Detection\n' + 
                    f'Confidence: {prediction_prob:.1%}',
                    fontsize=18, color=text_color, fontweight='bold', pad=20)
    
    ax_eeg.tick_params(colors=text_color)
    ax_eeg.spines['bottom'].set_color(text_color)
    ax_eeg.spines['left'].set_color(text_color)
    ax_eeg.spines['top'].set_visible(False)
    ax_eeg.spines['right'].set_visible(False)
    
    # ========================================================================
    # PANEL 2: LSTM ATTENTION (what temporal segments matter)
    # ========================================================================
    
    ax_attn = fig.add_subplot(gs[1], sharex=ax_eeg)
    ax_attn.set_facecolor(bg_color)
    
    # Upsample attention to match time resolution
    attention_upsampled = np.interp(
        np.linspace(0, len(attention_weights)-1, n_samples),
        np.arange(len(attention_weights)),
        attention_weights
    )
    
    ax_attn.fill_between(time, 0, attention_upsampled, 
                        color=attention_color, alpha=0.6)
    ax_attn.set_ylabel('LSTM\nAttention', fontsize=10, color=text_color)
    ax_attn.set_ylim(0, attention_upsampled.max() * 1.1)
    ax_attn.tick_params(colors=text_color)
    ax_attn.spines['bottom'].set_color(text_color)
    ax_attn.spines['left'].set_color(text_color)
    ax_attn.spines['top'].set_visible(False)
    ax_attn.spines['right'].set_visible(False)
    
    # ========================================================================
    # PANEL 3: CNN GRAD-CAM (what waveform features matter)
    # ========================================================================
    
    ax_grad = fig.add_subplot(gs[2], sharex=ax_eeg)
    ax_grad.set_facecolor(bg_color)
    
    ax_grad.fill_between(time, 0, gradcam_map, 
                        color=seizure_color, alpha=0.6)
    ax_grad.set_ylabel('CNN\nSaliency', fontsize=10, color=text_color)
    ax_grad.set_ylim(0, 1.0)
    ax_grad.set_xlabel('Time (s)', fontsize=12, color=text_color)
    ax_grad.tick_params(colors=text_color)
    ax_grad.spines['bottom'].set_color(text_color)
    ax_grad.spines['left'].set_color(text_color)
    ax_grad.spines['top'].set_visible(False)
    ax_grad.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, facecolor=bg_color, edgecolor='none')
    print(f"\n✓ Visualization saved to: {save_path}")
    
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("ARGUS VISION - Generating Saliency Visualization")
    print("="*70 + "\n")
    
    # Load trained model
    print("Loading model...")
    model = SeizureDetectorCNNLSTM(
        num_channels=22,
        num_samples=2000,
        cnn_filters=[32, 64, 128],
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.4,
        attention=True
    )
    
    # Load weights if you have them
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
        print("✓ Loaded trained weights")
    except:
        print("⚠ Using untrained model (for visualization demo)")
    
    model.eval()
    
    # Load seizure EEG
    print("\nLoading seizure EEG...")
    eeg_data, metadata = load_eeg_with_seizure(data_dir="/mnt/c/Users/0218s/Desktop/Ceribell-SZ-DTCTR/data/tusz/edf/dev")
    window, window_start_time = extract_window(eeg_data, metadata)
    
    # Prepare input
    x = torch.FloatTensor(window).unsqueeze(0)  # (1, 22, 2000)
    
    # Get prediction + attention
    print("\nGenerating predictions...")
    with torch.no_grad():
        logits, attention_weights = model(x)
        probs = F.softmax(logits, dim=1)
        seizure_prob = probs[0, 1].item()
    
    print(f"Seizure probability: {seizure_prob:.1%}")
    attention_weights = attention_weights[0].cpu().numpy()
    
    # Generate Grad-CAM
    print("Computing CNN saliency (Grad-CAM)...")
    gradcam = GradCAM(model, model.temporal_conv3)  # Target last CNN block
    gradcam_map = gradcam.generate(x, class_idx=1)
    
    # Create visualization
    print("\nCreating visualization...")
    create_argus_visualization(
        eeg_data=window,
        attention_weights=attention_weights,
        gradcam_map=gradcam_map,
        metadata=metadata,
        window_start_time=window_start_time,
        prediction_prob=seizure_prob,
        save_path="argus_vision.png"
    )
    
    print("\n" + "="*70)
    print("COMPLETE! Now create Argus artwork and composite them together.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()