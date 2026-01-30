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


def compute_jitter_variance_very_simplified(video_path, fps_override=None):
    """
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
        #'jitter_min': float(np.min(jitter_implicit)),  # minimum of implicit jitter
        #'jitter_max': float(np.max(jitter_implicit)),  # max for reference
        #'jitter_rms': float(np.sqrt(np.mean(jitter_implicit ** 2))),
        #'jitter_peak_to_peak': float(np.max(jitter_implicit) - np.min(jitter_implicit))
    }


    return jitter_metrics

if __name__ == "__main__":
    # Basic usage
    my_video = "shaky_climbing4.mp4"

    score = compute_jitter_variance_very_simplified(my_video)
    print(f"Jitter Score: {score}")