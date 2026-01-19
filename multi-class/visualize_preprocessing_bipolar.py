"""
Visualize EEG Preprocessing Pipeline - BIPOLAR MONTAGE (4 PLOTS)
Clean layout: Bipolar → Filtered → Normalized → Model Input (22ch)

Fixes:
- Channel labels placed in axes coordinates (no overlap with traces/titles)
- Short, non-wordy titles
- Only bottom subplot has x-axis label
- Better spacing + left margin reserved for labels
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

PREPROCESSED_DATA = r"C:\Users\0218s\Desktop\Ceribell-SZ-DTCTR\preprocessed\dev.npz"
OUTPUT_DIR = r"C:\Users\0218s\Desktop\Ceribell-SZ-DTCTR\multi_class\results"

# Bipolar montage (22 channels)
BIPOLAR_CHANNELS = [
    # Left temporal
    'FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1',
    # Right temporal
    'FP2-F8', 'F8-T4', 'T4-T6', 'T6-O2',
    # Left parasagittal
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    # Right parasagittal
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    # Midline
    'Fz-Cz', 'Cz-Pz',
    # Additional (to make 22)
    'T3-C3', 'C4-T4'
]

SAMPLING_RATE = 200

# ============================================================================
# SIMULATE PRE-FILTERED BIPOLAR
# ============================================================================

def add_noise_to_bipolar(clean_bipolar: np.ndarray, sampling_rate: int = 200) -> np.ndarray:
    """Add simple noise to simulate pre-filtered bipolar data (for visualization only)."""
    t = np.arange(clean_bipolar.shape[1]) / sampling_rate

    powerline = 0.2 * np.sin(2 * np.pi * 60 * t)      # 60 Hz
    drift = 0.3 * np.sin(2 * np.pi * 0.3 * t)         # slow drift
    noise = 0.15 * np.random.randn(*clean_bipolar.shape)

    noisy = clean_bipolar.copy()
    for i in range(noisy.shape[0]):
        noisy[i] = noisy[i] + powerline + drift + noise[i]

    return noisy

# ============================================================================
# FILTERING
# ============================================================================

def apply_filters(eeg: np.ndarray, sampling_rate: int = 200) -> np.ndarray:
    """Apply bandpass + notch filters."""
    # Bandpass: 0.5-50 Hz
    sos_bandpass = signal.butter(
        4, [0.5, 50], btype='bandpass', fs=sampling_rate, output='sos'
    )
    filtered = signal.sosfiltfilt(sos_bandpass, eeg, axis=1)

    # Notch: 60 Hz
    b, a = signal.iirnotch(60, 30, fs=sampling_rate)
    filtered = signal.filtfilt(b, a, filtered, axis=1)

    return filtered

# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_eeg(eeg: np.ndarray) -> np.ndarray:
    """Per-channel z-score normalization."""
    normalized = np.zeros_like(eeg)
    for i in range(eeg.shape[0]):
        mean = np.mean(eeg[i])
        std = np.std(eeg[i])
        normalized[i] = (eeg[i] - mean) / (std + 1e-8)
    return normalized

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_preprocessing_4panel(eeg_window: np.ndarray, output_path: Path) -> None:
    """Create a clean 4-panel preprocessing visualization."""
    sampling_rate = SAMPLING_RATE
    duration = eeg_window.shape[1] / sampling_rate
    time = np.linspace(0, duration, eeg_window.shape[1])

    # Processing steps
    raw_bipolar = add_noise_to_bipolar(eeg_window, sampling_rate)
    filtered = apply_filters(raw_bipolar, sampling_rate)
    normalized = normalize_eeg(filtered)

    # Figure + grid
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 1, hspace=0.28)

    num_channels = min(22, eeg_window.shape[0])
    channels_subset = list(range(min(num_channels, 12)))  # first ~12 for top panels

    def plot_stack(ax, data, ch_indices, color, lw, alpha, spacing, label_fs):
        """Stacked traces with channel labels in axes-coordinates (prevents overlap)."""
        ax.set_xlim(0, duration)
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='x', alpha=0.25)

        # Vertical extent for mapping offsets to axes fraction
        y_min = -spacing
        y_max = len(ch_indices) * spacing

        for i, ch_idx in enumerate(ch_indices):
            offset = (len(ch_indices) - 1 - i) * spacing
            ax.plot(time, data[ch_idx] + offset, linewidth=lw, color=color, alpha=alpha)

            label = BIPOLAR_CHANNELS[ch_idx] if ch_idx < len(BIPOLAR_CHANNELS) else f'Ch{ch_idx+1}'
            y_frac = (offset - y_min) / (y_max - y_min)

            ax.text(
                -0.02, y_frac, label,
                transform=ax.transAxes,
                ha='right', va='center',
                fontsize=label_fs, fontweight='bold'
            )

    # Panel 1
    ax1 = fig.add_subplot(gs[0])
    plot_stack(ax1, raw_bipolar, channels_subset, color='#2E86AB', lw=0.9, alpha=0.85, spacing=7.5, label_fs=9)
    ax1.set_title("1) Bipolar (raw)", fontweight='bold', fontsize=12, loc='left', pad=6)
    ax1.set_xticks([])

    # Panel 2
    ax2 = fig.add_subplot(gs[1])
    plot_stack(ax2, filtered, channels_subset, color='#FF8C00', lw=0.9, alpha=0.85, spacing=7.5, label_fs=9)
    ax2.set_title("2) Filtered", fontweight='bold', fontsize=12, loc='left', pad=6)
    ax2.set_xticks([])

    # Panel 3
    ax3 = fig.add_subplot(gs[2])
    plot_stack(ax3, normalized, channels_subset, color='#2D7D4D', lw=0.9, alpha=0.85, spacing=4.0, label_fs=9)
    ax3.set_title("3) Normalized", fontweight='bold', fontsize=12, loc='left', pad=6)
    ax3.set_xticks([])

    # Panel 4 (all channels)
    ax4 = fig.add_subplot(gs[3])
    plot_stack(ax4, normalized, list(range(num_channels)), color='#9B59B6', lw=0.6, alpha=0.75, spacing=2.4, label_fs=7.5)
    ax4.set_title("4) Model input (22 channels)", fontweight='bold', fontsize=12, loc='left', pad=6)
    ax4.set_xlabel("Time (s)", fontweight='bold', fontsize=11)

    # Global title + layout: reserve left margin for labels
    fig.suptitle("EEG preprocessing pipeline", fontsize=15, fontweight='bold', y=0.99)
    fig.subplots_adjust(left=0.12, right=0.99, top=0.94, bottom=0.06)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 4-panel preprocessing visualization saved: {output_path}")
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("EEG PREPROCESSING VISUALIZATION - BIPOLAR MONTAGE (4 PLOTS)")
    print("="*70)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    print("\nLoading preprocessed data...")
    data = np.load(PREPROCESSED_DATA)
    windows = data['windows']

    print(f"✅ Loaded {len(windows)} windows")
    print(f"   Shape: {windows.shape}")

    # Select window
    window_idx = 100
    eeg_window = windows[window_idx]

    print(f"\nUsing window {window_idx}")
    print(f"  Shape: {eeg_window.shape}")
    print(f"  Channels: {eeg_window.shape[0]}")
    print(f"  Samples: {eeg_window.shape[1]}")

    # Generate visualization
    output_path = output_dir / "preprocessing_pipeline_bipolar_4panel.png"
    plot_preprocessing_4panel(eeg_window, output_path)

    print("\n✅ DONE")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
