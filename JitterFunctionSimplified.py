import numpy as np
import cv2
from scipy import signal
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import cv2
from scipy import signal
import warnings

warnings.filterwarnings('ignore')


def compute_jitter_variance(video_path, fps_override=None):
    """
    Compute Jitter Variance (Temporal) score from video, both implicit and explicit.

    Parameters:
    -----------
    video_path : str
        Path to the input video file
    fps_override : float or None
        Override the detected FPS

    Returns:
    --------
    dict : Jitter metrics including implicit, explicit, and min jitter
    """
    # ================= STEP 1: LOAD VIDEO =================
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = fps_override if fps_override else cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    max_frames = 300
    gray_frames = []

    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    cap.release()

    if len(gray_frames) < 30:
        raise ValueError(f"Video too short: only {len(gray_frames)} frames")

    gray_frames = np.array(gray_frames)
    actual_frames = len(gray_frames)

    # ================= STEP 2: EXTRACT TEMPORAL SIGNAL =================
    patch_size = 30
    center_y, center_x = height // 2, width // 2
    y_start = max(0, center_y - patch_size // 2)
    y_end = min(height, center_y + patch_size // 2)
    x_start = max(0, center_x - patch_size // 2)
    x_end = min(width, center_x + patch_size // 2)

    center_patches = gray_frames[:, y_start:y_end, x_start:x_end]
    brightness_signal = np.mean(center_patches, axis=(1, 2))

    brightness_signal = brightness_signal - np.mean(brightness_signal)
    if np.std(brightness_signal) > 0:
        brightness_signal = brightness_signal / np.std(brightness_signal)

    # ================= STEP 3: IMPLICIT JITTER (High-pass filter) =================
    cutoff_freq = 2.0  # Hz
    nyquist = fps / 2

    if cutoff_freq < nyquist:
        b, a = signal.butter(N=4, Wn=cutoff_freq / nyquist, btype='high')
        jitter_implicit = signal.filtfilt(b, a, brightness_signal)
    else:
        jitter_implicit = np.diff(brightness_signal, n=2)
        jitter_implicit = np.pad(jitter_implicit, (1, 1), 'edge')

    # ================= STEP 3B: EXPLICIT JITTER (FFT high-frequency) =================
    B = np.fft.fft(brightness_signal)
    freq = np.fft.fftfreq(len(B), d=1 / fps)
    high_freq_mask = np.abs(freq) > cutoff_freq  # only high frequencies
    B_high = B * high_freq_mask
    jitter_explicit = np.fft.ifft(B_high).real  # back to time domain

    # ================= STEP 4: COMPUTE METRICS =================
    jitter_metrics = {
        'implicit_jitter_variance': float(np.var(jitter_implicit)),
        'explicit_jitter_variance': float(np.var(jitter_explicit)),
        'jitter_min': float(np.min(jitter_implicit)),  # minimum of implicit jitter
        'jitter_max': float(np.max(jitter_implicit)),  # max for reference
        'jitter_rms': float(np.sqrt(np.mean(jitter_implicit ** 2))),
        'jitter_peak_to_peak': float(np.max(jitter_implicit) - np.min(jitter_implicit))
    }

    # Add video info
    jitter_metrics['video_info'] = {
        'path': video_path,
        'fps': float(fps),
        'frames_analyzed': int(actual_frames),
        'duration': float(actual_frames / fps),
        'resolution': (int(width), int(height))
    }

    return jitter_metrics
# ============================================================================
# SIMPLE VISUALIZATION FUNCTION (Optional)
# ============================================================================
def visualize_jitter_analysis(video_path, save_plot=False):
    """
    Quick visualization of jitter analysis.
    """
    import matplotlib.pyplot as plt

    # Compute jitter variance
    jitter_score = compute_jitter_variance(video_path)

    # Create simple visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Reload video to get frames for visualization
    cap = cv2.VideoCapture(video_path)
    frames_to_read = min(300, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    gray_frames = []
    for _ in range(frames_to_read):
        ret, frame = cap.read()
        if ret:
            gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    cap.release()
    gray_frames = np.array(gray_frames)

    # Plot 1: Sample frame
    if len(gray_frames) > 0:
        axes[0, 0].imshow(gray_frames[len(gray_frames) // 2], cmap='gray')
        axes[0, 0].set_title('Sample Frame')
        axes[0, 0].axis('off')

    # Plot 2: Jitter signal (simplified - we need to recompute)
    patch_size = 30
    height, width = gray_frames.shape[1], gray_frames.shape[2]
    center_y, center_x = height // 2, width // 2
    y_start = max(0, center_y - patch_size // 2)
    y_end = min(height, center_y + patch_size // 2)
    x_start = max(0, center_x - patch_size // 2)
    x_end = min(width, center_x + patch_size // 2)

    center_patches = gray_frames[:, y_start:y_end, x_start:x_end]
    brightness_signal = np.mean(center_patches, axis=(1, 2))
    brightness_signal = brightness_signal - np.mean(brightness_signal)

    # Simple high-pass: subtract moving average
    window_size = int(info['video_info']['fps'])  # 1 second window
    if window_size > 0:
        smoothed = np.convolve(brightness_signal, np.ones(window_size) / window_size, mode='same')
        jitter_signal = brightness_signal - smoothed
    else:
        jitter_signal = brightness_signal

    time_vector = np.arange(len(jitter_signal)) / info['video_info']['fps']

    axes[0, 1].plot(time_vector, jitter_signal, 'r-', linewidth=1)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Jitter Amplitude')
    axes[0, 1].set_title(f'Jitter Signal (Variance: {jitter_score:.6f})')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Metrics bar chart
    metrics = ['Variance', 'RMS', 'Peak-to-Peak']
    values = [
        info['jitter_variance'],
        info['jitter_rms'],
        info['jitter_peak_to_peak']
    ]

    # Normalize for display
    if max(values) > 0:
        normalized = [v / max(values) for v in values]
    else:
        normalized = values

    bars = axes[1, 0].bar(metrics, normalized, color=['blue', 'green', 'red'], alpha=0.7)
    axes[1, 0].set_ylabel('Normalized Value')
    axes[1, 0].set_title('Jitter Metrics Comparison')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Add actual values
    for bar, val in zip(bars, values):
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    # Plot 4: Summary info
    axes[1, 1].axis('off')
    summary_text = f"""
    JITTER ANALYSIS SUMMARY

    Primary Score (Variance): {jitter_score:.6f}

    Additional Metrics:
    - RMS: {info['jitter_rms']:.4f}
    - Peak-to-Peak: {info['jitter_peak_to_peak']:.4f}
    - Dominant Freq: {info['dominant_jitter_frequency']:.2f} Hz
    - Signal/Jitter Ratio: {info['signal_to_jitter_ratio_db']:.1f} dB

    Video Info:
    - Duration: {info['video_info']['duration']:.1f}s
    - FPS: {info['video_info']['fps']:.1f}
    - Frames analyzed: {info['video_info']['frames_analyzed']}
    """

    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle(f'Jitter Analysis: {video_path}\nJitter Variance Score: {jitter_score:.6f}',
                 fontsize=14, y=1.02)
    plt.tight_layout()

    if save_plot:
        import os
        plot_filename = f"jitter_score_{os.path.basename(video_path).replace('.mp4', '')}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_filename}")

    plt.show()

    return jitter_score, info

if __name__ == "__main__":
    # Basic usage
    my_video = "shaky_climbing4.mp4"

    score = compute_jitter_variance(my_video)
    print(f"Jitter Score: {score}")

    # With visualization
    visualize_jitter_analysis(my_video, save_plot=True)