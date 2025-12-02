"""
COMPLETE TIME SERIES TO IMAGE ENCODING - ALL 6 METHODS
Based on Time-VLM Paper (ICML 2025)

Creates all 6 image encoding methods shown in your reference image:
1. Gramian Angular Field (GAF-S and GAF-D)
2. Recurrence Plot
3. Continuous Wavelet Transform (CWT)
4. Markov Transition Field (MTF)
5. Greyscale Encoding
6. Spectrogram

For Beijing PM2.5 Classification (5 classes)
"""

import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# ================= CONFIG =================
DATA_PATH = "data/beijing_pm25_5classes_paper.npz"
OUT_DIR = "data/vlm_images_all_6_methods"
IMG_SIZE = 224  # Standard for VLM models

CLASS_NAMES = [
    "L0_VeryLowRisk",
    "L1_LowRisk",
    "L2_MediumRisk",
    "L3_HighRisk",
    "L4_VeryHighRisk"
]

# Enable all 6 methods
METHODS_TO_GENERATE = {
    'gaf_summation': True,      # Method 1: GAF-S
    'gaf_difference': True,      # Method 1: GAF-D  
    'recurrence_plot': True,     # Method 2
    'cwt': True,                 # Method 3
    'mtf': True,                 # Method 4
    'greyscale': True,           # Method 5
    'spectrogram': True          # Method 6
}
# ==========================================


def normalize_to_range(x, min_val=-1, max_val=1):
    """Normalize array to specified range"""
    x_min = np.min(x)
    x_max = np.max(x)
    
    if x_max - x_min < 1e-8:
        return np.full_like(x, (min_val + max_val) / 2)
    
    x_normalized = (x - x_min) / (x_max - x_min)
    x_scaled = x_normalized * (max_val - min_val) + min_val
    x_scaled = np.clip(x_scaled, min_val, max_val)
    
    return x_scaled


def normalize_image(img_array):
    """Normalize array to [0, 255] uint8"""
    img_min = img_array.min()
    img_max = img_array.max()
    
    if img_max - img_min < 1e-8:
        return np.full(img_array.shape, 128, dtype=np.uint8)
    
    img_normalized = (img_array - img_min) / (img_max - img_min)
    img_uint8 = (img_normalized * 255).astype(np.uint8)
    return img_uint8


# ============================================
# METHOD 1: GRAMIAN ANGULAR FIELD (GAF)
# ============================================
def gramian_angular_field(ts, method='summation'):
    """
    Gramian Angular Field - Summation (GASF) or Difference (GADF)
    
    This is THE BEST method according to Time-VLM paper!
    Converts time series to polar coordinates and creates gramian matrix.
    
    Args:
        ts: (T, D) time series
        method: 'summation' or 'difference'
    Returns:
        RGB image (H, W, 3)
    """
    T, D = ts.shape
    
    # Select top 3 features by variance
    variances = np.var(ts, axis=0)
    top_3 = np.argsort(variances)[-3:] if D >= 3 else range(D)
    
    channels = []
    
    for feat_idx in top_3:
        signal_data = ts[:, feat_idx]
        
        # Normalize to [-1, 1] for arccos
        normalized = normalize_to_range(signal_data, -0.999, 0.999)
        normalized = np.clip(normalized, -1.0, 1.0)
        
        # Convert to polar coordinates
        phi = np.arccos(normalized)
        
        # Create Gramian matrix
        if method == 'summation':
            # GASF: cos(φᵢ + φⱼ)
            gaf = np.cos(phi[:, None] + phi[None, :])
        else:
            # GADF: sin(φᵢ - φⱼ)
            gaf = np.sin(phi[:, None] - phi[None, :])
        
        # Handle NaN
        if np.any(np.isnan(gaf)):
            gaf = np.nan_to_num(gaf, nan=0.0)
        
        channels.append(gaf)
    
    # Stack as RGB
    while len(channels) < 3:
        channels.append(channels[-1])
    
    rgb = np.stack(channels[:3], axis=-1)
    return normalize_image(rgb)


# ============================================
# METHOD 2: RECURRENCE PLOT
# ============================================
def recurrence_plot(ts, epsilon_percentile=10):
    """
    Recurrence Plot - Shows when states recur in time series
    
    Good for finding periodic patterns and repetitions.
    
    Args:
        ts: (T, D) time series
        epsilon_percentile: Threshold percentile for recurrence
    Returns:
        RGB image (H, W, 3)
    """
    T, D = ts.shape
    
    variances = np.var(ts, axis=0)
    top_3 = np.argsort(variances)[-3:] if D >= 3 else range(D)
    
    channels = []
    
    for feat_idx in top_3:
        signal_data = ts[:, feat_idx:feat_idx+1]
        
        # Compute pairwise distances
        distances = np.abs(signal_data - signal_data.T)
        
        # Adaptive threshold
        epsilon = np.percentile(distances, epsilon_percentile)
        
        # Binary recurrence matrix
        rp = (distances <= epsilon).astype(np.float32)
        
        channels.append(rp)
    
    while len(channels) < 3:
        channels.append(channels[-1])
    
    rgb = np.stack(channels[:3], axis=-1)
    return normalize_image(rgb)


# ============================================
# METHOD 3: CONTINUOUS WAVELET TRANSFORM
# ============================================
def continuous_wavelet_transform(ts):
    """
    Continuous Wavelet Transform (CWT) - Time-frequency representation
    
    Shows both time and frequency information simultaneously.
    
    Args:
        ts: (T, D) time series
    Returns:
        RGB image (H, W, 3)
    """
    T, D = ts.shape
    
    variances = np.var(ts, axis=0)
    top_3 = np.argsort(variances)[-3:] if D >= 3 else range(D)
    
    channels = []
    
    # Define scales for CWT
    widths = np.arange(1, min(T//4, 64))
    
    for feat_idx in top_3:
        signal_data = ts[:, feat_idx]
        
        # Manual CWT implementation with Ricker (Mexican Hat) wavelet
        cwt_matrix = np.zeros((len(widths), len(signal_data)))
        
        for i, width in enumerate(widths):
            # Create Ricker wavelet manually
            # Ricker wavelet formula: ψ(t) = (1 - t²) * exp(-t²/2)
            points = min(10 * width, len(signal_data))
            t = np.linspace(-4, 4, points)
            wavelet = (1 - t**2) * np.exp(-t**2 / 2)
            wavelet = wavelet / np.sqrt(np.sum(wavelet**2))  # Normalize
            
            # Convolve signal with wavelet
            convolved = np.convolve(signal_data, wavelet, mode='same')
            cwt_matrix[i, :] = convolved
        
        # Take absolute value (magnitude)
        cwt_abs = np.abs(cwt_matrix)
        
        channels.append(cwt_abs)
    
    while len(channels) < 3:
        channels.append(channels[-1])
    
    rgb = np.stack(channels[:3], axis=-1)
    return normalize_image(rgb)


# ============================================
# METHOD 4: MARKOV TRANSITION FIELD
# ============================================
def markov_transition_field(ts, n_bins=8):
    """
    Markov Transition Field (MTF) - State transition probabilities
    
    Captures temporal transition dynamics.
    
    Args:
        ts: (T, D) time series
        n_bins: Number of quantile bins
    Returns:
        RGB image (H, W, 3)
    """
    T, D = ts.shape
    
    variances = np.var(ts, axis=0)
    top_3 = np.argsort(variances)[-3:] if D >= 3 else range(D)
    
    channels = []
    
    for feat_idx in top_3:
        signal_data = ts[:, feat_idx]
        
        # Quantize signal into bins
        quantiles = np.percentile(signal_data, np.linspace(0, 100, n_bins + 1))
        bins = np.digitize(signal_data, quantiles[1:-1])
        
        # Clip to valid range
        bins = np.clip(bins, 0, n_bins - 1)
        
        # Create transition matrix
        trans_matrix = np.zeros((n_bins, n_bins))
        for i in range(len(bins) - 1):
            trans_matrix[bins[i], bins[i+1]] += 1
        
        # Normalize
        row_sums = trans_matrix.sum(axis=1, keepdims=True)
        trans_matrix = np.divide(
            trans_matrix, row_sums,
            where=row_sums != 0,
            out=np.zeros_like(trans_matrix)
        )
        
        # Create MTF image
        mtf = trans_matrix[bins[:, None], bins[None, :]]
        
        channels.append(mtf)
    
    while len(channels) < 3:
        channels.append(channels[-1])
    
    rgb = np.stack(channels[:3], axis=-1)
    return normalize_image(rgb)


# ============================================
# METHOD 5: GREYSCALE ENCODING
# ============================================
def greyscale_encoding(ts):
    """
    Greyscale Encoding - Direct pixel intensity mapping
    
    Simple but effective direct representation.
    Each pixel row is one time step, pixel column is one feature.
    
    Args:
        ts: (T, D) time series
    Returns:
        RGB image (H, W, 3)
    """
    T, D = ts.shape
    
    # Normalize each feature to [0, 1]
    ts_norm = np.zeros_like(ts)
    for d in range(D):
        ts_norm[:, d] = normalize_to_range(ts[:, d], 0, 1)
    
    # Create 2D representation: (T, D) -> (T, T) by repeating
    # Alternative: Use interpolation to create square image
    if T >= D:
        # Repeat columns to make square
        repeat_factor = T // D + 1
        grey_2d = np.tile(ts_norm, (1, repeat_factor))[:, :T]
    else:
        # Interpolate rows to make square
        from scipy.interpolate import interp1d
        x_old = np.linspace(0, 1, T)
        x_new = np.linspace(0, 1, T)
        grey_2d = np.zeros((T, T))
        for d in range(min(D, T)):
            f = interp1d(x_old, ts_norm[:, d], kind='linear', fill_value='extrapolate')
            grey_2d[:, d] = f(x_new)
    
    # Convert to RGB by repeating greyscale across 3 channels
    grey_2d_norm = normalize_to_range(grey_2d, 0, 255).astype(np.uint8)
    rgb = np.stack([grey_2d_norm, grey_2d_norm, grey_2d_norm], axis=-1)
    
    return rgb


# ============================================
# METHOD 6: SPECTROGRAM
# ============================================
def spectrogram_encoding(ts):
    """
    Spectrogram - Frequency content over time (STFT)
    
    Shows how frequency content evolves over time.
    
    Args:
        ts: (T, D) time series
    Returns:
        RGB image (H, W, 3)
    """
    T, D = ts.shape
    
    variances = np.var(ts, axis=0)
    top_3 = np.argsort(variances)[-3:] if D >= 3 else range(D)
    
    channels = []
    
    for feat_idx in top_3:
        signal_data = ts[:, feat_idx]
        
        # Compute STFT (Short-Time Fourier Transform)
        f, t, Sxx = signal.spectrogram(
            signal_data,
            fs=1.0,  # Sampling frequency
            nperseg=min(32, T//4),  # Segment length
            noverlap=min(16, T//8)   # Overlap
        )
        
        # Take log magnitude for better visualization
        Sxx_log = np.log10(Sxx + 1e-10)
        
        channels.append(Sxx_log)
    
    while len(channels) < 3:
        channels.append(channels[-1])
    
    rgb = np.stack(channels[:3], axis=-1)
    return normalize_image(rgb)


# ============================================
# MAIN ENCODING FUNCTION
# ============================================
def encode_to_image(ts, method='gaf_summation'):
    """
    Convert time series to image using specified method
    
    Args:
        ts: (T, D) time series
        method: One of the 6 encoding methods
    Returns:
        PIL Image (224x224x3)
    """
    # Standardize per feature
    ts_std = (ts - ts.mean(axis=0)) / (ts.std(axis=0) + 1e-8)
    
    # Apply encoding method
    if method == 'gaf_summation':
        img_array = gramian_angular_field(ts_std, 'summation')
    elif method == 'gaf_difference':
        img_array = gramian_angular_field(ts_std, 'difference')
    elif method == 'recurrence_plot':
        img_array = recurrence_plot(ts_std, epsilon_percentile=15)
    elif method == 'cwt':
        img_array = continuous_wavelet_transform(ts_std)
    elif method == 'mtf':
        img_array = markov_transition_field(ts_std, n_bins=8)
    elif method == 'greyscale':
        img_array = greyscale_encoding(ts_std)
    elif method == 'spectrogram':
        img_array = spectrogram_encoding(ts_std)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Convert to PIL Image and resize
    img = Image.fromarray(img_array)
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    
    return img


def visualize_methods(X, y, n_samples=3):
    """
    Visualize all 6 methods side by side for comparison
    """
    methods = [
        'gaf_summation', 'gaf_difference', 'recurrence_plot',
        'cwt', 'mtf', 'greyscale', 'spectrogram'
    ]
    
    method_names = [
        'GAF-Sum', 'GAF-Diff', 'Recurrence',
        'CWT', 'MTF', 'Greyscale', 'Spectrogram'
    ]
    
    print(f"\n🎨 Creating comparison visualization...")
    
    fig, axes = plt.subplots(len(CLASS_NAMES), len(methods),
                             figsize=(len(methods)*2, len(CLASS_NAMES)*2))
    
    for class_idx in range(len(CLASS_NAMES)):
        # Get samples from this class
        class_samples = X[y == class_idx]
        
        if len(class_samples) == 0:
            continue
        
        # Random sample
        sample_idx = np.random.choice(len(class_samples))
        ts = class_samples[sample_idx]
        
        for method_idx, method in enumerate(methods):
            img = encode_to_image(ts, method=method)
            
            axes[class_idx, method_idx].imshow(img)
            axes[class_idx, method_idx].axis('off')
            
            if class_idx == 0:
                axes[class_idx, method_idx].set_title(
                    method_names[method_idx], fontsize=10, fontweight='bold'
                )
            
            if method_idx == 0:
                axes[class_idx, method_idx].set_ylabel(
                    CLASS_NAMES[class_idx], 
                    rotation=0, size=9,
                    labelpad=60, ha='right'
                )
    
    plt.suptitle('All 6 Encoding Methods - Comparison Across Classes',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('all_methods_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: all_methods_comparison.png")
    plt.close()


def process_split(X, y, split, method):
    """Process train or test split"""
    base = os.path.join(OUT_DIR, method, split)
    
    for cname in CLASS_NAMES:
        os.makedirs(os.path.join(base, cname), exist_ok=True)
    
    for i in tqdm(range(len(X)), desc=f"{split}/{method}"):
        ts = X[i]
        label = int(y[i])
        
        img = encode_to_image(ts, method=method)
        
        cname = CLASS_NAMES[label]
        img.save(os.path.join(base, cname, f"{split}_{i:05}.png"))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("🎨 GENERATING ALL 6 TIME SERIES IMAGE ENCODINGS")
    print("=" * 70)
    print(f"\n📂 Loading: {DATA_PATH}")
    
    data = np.load(DATA_PATH)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]
    
    print(f"\n✅ Loaded")
    print(f"  Train: {X_train.shape}")
    print(f"  Test:  {X_test.shape}")
    
    # First, create comparison visualization
    visualize_methods(X_train, y_train, n_samples=3)
    
    # Generate images for all enabled methods
    print(f"\n🎯 Generating images for all methods...")
    print("=" * 70)
    
    for method, enabled in METHODS_TO_GENERATE.items():
        if enabled:
            print(f"\n📊 Method: {method.upper()}")
            print("-" * 70)
            
            process_split(X_train, y_train, "train", method)
            process_split(X_test, y_test, "test", method)
            
            print(f"✅ {method} images saved in:")
            print(f"   {OUT_DIR}/{method}/train/")
            print(f"   {method}/test/")
    
    print("\n" + "=" * 70)
    print("✅ ALL DONE!")
    print("=" * 70)
    print(f"\n📁 All images saved in: {OUT_DIR}/")
    print(f"📊 Comparison plot: all_methods_comparison.png")
    
    print("\n🎯 Next steps:")
    print("  1. Check 'all_methods_comparison.png' to see all 6 methods")
    print("  2. Run the VLM training script for each method")
    print("  3. Compare results to find the best encoding!")


if __name__ == "__main__":
    main()