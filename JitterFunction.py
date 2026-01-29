import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import signal
import warnings

warnings.filterwarnings('ignore')


def extract_jitter_signatures(video_path, output_plots=True, fps_override=None):
    """
    Extract jitter signatures from a video using Fourier domain phase analysis.
    """

    print("=" * 70)
    print("JITTER SIGNATURE EXTRACTION - Fourier Domain Analysis")
    print("=" * 70)

    # ========================================================================
    # STEP 1: LOAD AND PREPROCESS VIDEO
    # ========================================================================
    print("\n1. Loading and preprocessing video...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = fps_override if fps_override else cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read frames (limit for performance)
    max_frames = 300
    frames = []
    gray_frames = []

    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    cap.release()

    if len(frames) < 10:
        raise ValueError(f"Video too short: only {len(frames)} frames")

    gray_frames = np.array(gray_frames)
    actual_frames = len(frames)

    # ========================================================================
    # STEP 2: EXTRACT TEMPORAL SIGNAL
    # ========================================================================
    print("\n2. Extracting temporal motion signal...")

    # Use brightness variation at image center
    patch_size = 30
    center_y, center_x = height // 2, width // 2
    y_start = max(0, center_y - patch_size // 2)
    y_end = min(height, center_y + patch_size // 2)
    x_start = max(0, center_x - patch_size // 2)
    x_end = min(width, center_x + patch_size // 2)

    center_patches = gray_frames[:, y_start:y_end, x_start:x_end]
    brightness_signal = np.mean(center_patches, axis=(1, 2))

    # Normalize signal
    brightness_signal = (brightness_signal - np.mean(brightness_signal))
    if np.std(brightness_signal) > 0:
        brightness_signal = brightness_signal / np.std(brightness_signal)

    temporal_signal = brightness_signal
    time_vector = np.arange(actual_frames) / fps

    # ========================================================================
    # STEP 3: FOURIER DOMAIN ANALYSIS
    # ========================================================================
    print("\n3. Performing Fourier domain analysis...")

    # Compute Fourier transform
    signal_fft = np.fft.fft(temporal_signal)
    freq = np.fft.fftfreq(actual_frames, 1 / fps)

    # Focus on positive frequencies
    pos_mask = freq > 0
    pos_freq = freq[pos_mask]
    signal_mag = np.abs(signal_fft[pos_mask])
    signal_phase = np.angle(signal_fft[pos_mask])

    # Unwrap phase for analysis
    signal_phase_unwrapped = np.unwrap(signal_phase)

    # Compute phase slope (key to impulse detection)
    phase_slope = np.gradient(signal_phase_unwrapped, pos_freq)

    # ========================================================================
    # STEP 4: COMPUTE THE THREE KEY METRICS
    # ========================================================================
    print("\n4. Computing jitter metrics...")

    # METRIC 1: PHASE SLOPE VARIANCE
    # Variance of the phase derivative dΦ/dω
    # Measures how consistent the phase slope is across frequencies
    phase_slope_variance = np.var(phase_slope)

    # METRIC 2: OVERALL JITTER MAGNITUDE
    # Mean absolute phase slope in low frequencies (< 10 Hz)
    # Larger values indicate stronger jitter
    low_freq_mask = pos_freq < min(10, pos_freq[-1])
    if np.any(low_freq_mask):
        overall_jitter_magnitude = np.mean(np.abs(phase_slope[low_freq_mask]))
    else:
        overall_jitter_magnitude = np.mean(np.abs(phase_slope))

    # METRIC 3: JITTER VARIANCE (in time domain)
    # This is NOT the same as phase slope variance!
    # Measures temporal variation of the jitter signal

    # First, estimate jitter signal from phase residuals
    # High phase slope variance → more jitter
    jitter_estimate = np.sqrt(np.abs(phase_slope_variance)) * np.sin(2 * np.pi * 5 * time_vector)

    # Alternative: Use high-pass filtered temporal signal
    # Jitter is high-frequency component of motion
    from scipy import signal as sp_signal
    b, a = sp_signal.butter(4, 2 / (fps / 2), btype='high')  # High-pass filter at 2Hz
    jitter_signal = sp_signal.filtfilt(b, a, temporal_signal)

    # Jitter variance (temporal variance of jitter)
    jitter_variance = np.var(jitter_signal)

    print(f"   Phase Slope Variance: {phase_slope_variance:.6f}")
    print(f"   Overall Jitter Magnitude: {overall_jitter_magnitude:.6f}")
    print(f"   Jitter Variance (temporal): {jitter_variance:.6f}")

    # ========================================================================
    # STEP 5: DETECT IMPULSE SIGNATURES
    # ========================================================================
    print("\n5. Detecting impulse signatures...")

    def detect_impulse_regions(phase_slope, freq, min_slope_threshold=0.3):
        """Detect impulse regions from phase slope."""
        slope_norm = np.abs(phase_slope) / np.max(np.abs(phase_slope)) if np.max(np.abs(phase_slope)) > 0 else np.abs(
            phase_slope)

        impulse_regions = []
        in_region = False
        region_start = 0

        for i in range(len(slope_norm)):
            is_significant = slope_norm[i] > min_slope_threshold

            if is_significant and not in_region:
                in_region = True
                region_start = i
            elif not is_significant and in_region:
                in_region = False
                region_end = i

                if (region_end - region_start) >= 3:
                    region_freq = np.mean(freq[region_start:region_end])
                    region_slope = np.mean(phase_slope[region_start:region_end])
                    estimated_t_k = -region_slope / (2 * np.pi)

                    impulse_regions.append({
                        'frequency_center': region_freq,
                        'phase_slope': region_slope,
                        'estimated_time': estimated_t_k,
                        'region_indices': (region_start, region_end)
                    })

        return impulse_regions

    impulse_regions = detect_impulse_regions(phase_slope, pos_freq, 0.4)

    # Filter valid impulse times
    valid_regions = []
    for region in impulse_regions:
        t_k = region['estimated_time']
        if 0 <= t_k <= (actual_frames / fps):
            valid_regions.append(region)

    impulse_regions = valid_regions
    impulse_times = [region['estimated_time'] for region in impulse_regions]

    print(f"   Detected {len(impulse_times)} impulse signatures")

    # ========================================================================
    # STEP 6: CREATE RESULTS DICTIONARY
    # ========================================================================
    results = {
        # Core metrics (distinct!)
        'phase_slope_variance': phase_slope_variance,
        'overall_jitter_magnitude': overall_jitter_magnitude,
        'jitter_variance': jitter_variance,

        # Derived metrics
        'jitter_snr': overall_jitter_magnitude / (np.sqrt(jitter_variance) + 1e-10),
        'phase_slope_std': np.std(phase_slope),
        'phase_slope_mean': np.mean(phase_slope),

        # Impulse detection
        'impulse_times': np.array(impulse_times),
        'num_impulses': len(impulse_times),
        'impulse_regions': impulse_regions,

        # Raw data for further analysis
        'temporal_signal': temporal_signal,
        'jitter_signal': jitter_signal,
        'time_vector': time_vector,
        'frequencies': pos_freq,
        'phase_slope': phase_slope,
        'signal_magnitude': signal_mag,
        'signal_phase': signal_phase,

        # Video info
        'video_info': {
            'path': video_path,
            'fps': fps,
            'frames': actual_frames,
            'duration': actual_frames / fps,
            'resolution': (width, height)
        }
    }

    # ========================================================================
    # STEP 7: VISUALIZE ALL THREE METRICS
    # ========================================================================
    if output_plots:
        print("\n6. Generating comprehensive visualization...")

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f'Jitter Analysis: {video_path}\n'
                     f'Phase Slope Variance: {phase_slope_variance:.4f} | '
                     f'Jitter Magnitude: {overall_jitter_magnitude:.4f} | '
                     f'Jitter Variance: {jitter_variance:.4f}',
                     fontsize=14, y=1.02)

        # Plot 1: Temporal signals comparison
        ax1 = plt.subplot(3, 4, 1)
        ax1.plot(time_vector, temporal_signal, 'b-', alpha=0.7, label='Original')
        ax1.plot(time_vector, jitter_signal, 'r-', alpha=0.7, label='Jitter component')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Temporal Signals')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Magnitude spectrum
        ax2 = plt.subplot(3, 4, 2)
        display_limit = min(20, pos_freq[-1])
        freq_mask = pos_freq < display_limit
        ax2.plot(pos_freq[freq_mask], signal_mag[freq_mask], 'g-')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Magnitude')
        ax2.set_title('Magnitude Spectrum')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Phase slope (with detected regions)
        ax3 = plt.subplot(3, 4, 3)
        ax3.plot(pos_freq[freq_mask], phase_slope[freq_mask], 'c-', linewidth=2, label='Phase slope')
        ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)

        # Mark impulse regions
        for region in impulse_regions:
            start_idx, end_idx = region['region_indices']
            if start_idx < len(pos_freq) and pos_freq[start_idx] < display_limit:
                ax3.axvspan(pos_freq[start_idx], pos_freq[min(end_idx, len(pos_freq) - 1)],
                            alpha=0.3, color='red', label='Impulse region' if region == impulse_regions[0] else "")

        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('dΦ/dω (s)')
        ax3.set_title('Phase Slope with Detected Impulses')
        ax3.grid(True, alpha=0.3)
        if impulse_regions:
            ax3.legend(fontsize=8)

        # Plot 4: Phase slope histogram
        ax4 = plt.subplot(3, 4, 4)
        ax4.hist(phase_slope[freq_mask], bins=30, edgecolor='black', alpha=0.7)
        ax4.axvline(x=np.mean(phase_slope[freq_mask]), color='r', linestyle='--',
                    label=f'Mean: {np.mean(phase_slope[freq_mask]):.3f}')
        ax4.axvline(x=np.mean(phase_slope[freq_mask]) + np.std(phase_slope[freq_mask]),
                    color='g', linestyle=':', label=f'±1 std')
        ax4.axvline(x=np.mean(phase_slope[freq_mask]) - np.std(phase_slope[freq_mask]),
                    color='g', linestyle=':')
        ax4.set_xlabel('Phase Slope Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Phase Slope Distribution')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)

        # Plot 5: METRICS COMPARISON
        ax5 = plt.subplot(3, 4, 5)
        metrics = ['Phase Slope\nVariance', 'Jitter\nMagnitude', 'Jitter\nVariance']
        values = [phase_slope_variance, overall_jitter_magnitude, jitter_variance]

        # Normalize for display
        max_val = max(values) if max(values) > 0 else 1
        normalized_values = [v / max_val for v in values]

        bars = ax5.bar(metrics, normalized_values, color=['blue', 'green', 'red'], alpha=0.7)
        ax5.set_ylabel('Normalized Value')
        ax5.set_title('Key Metrics Comparison (Normalized)')
        ax5.grid(True, alpha=0.3, axis='y')

        # Add actual values on bars
        for bar, val, norm_val in zip(bars, values, normalized_values):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{val:.4f}', ha='center', va='bottom', fontsize=8)

        # Plot 6: Impulse timeline
        ax6 = plt.subplot(3, 4, 6)
        if impulse_times:
            # Get amplitudes from phase slope at impulse frequencies
            amplitudes = []
            for region in impulse_regions:
                start_idx, end_idx = region['region_indices']
                amplitudes.append(np.mean(np.abs(phase_slope[start_idx:end_idx])))

            ax6.stem(impulse_times, amplitudes, linefmt='b-', markerfmt='bo', basefmt=' ')
            ax6.set_xlabel('Time (s)')
            ax6.set_ylabel('Phase Slope Magnitude')
            ax6.set_title(f'Detected Impulses ({len(impulse_times)} total)')
            ax6.grid(True, alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No impulses detected',
                     ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('No Impulses Detected')

        # Plot 7: Phase slope vs frequency (log scale for variance visualization)
        ax7 = plt.subplot(3, 4, 7)
        ax7.plot(pos_freq[freq_mask], np.abs(phase_slope[freq_mask]), 'm-')
        ax7.set_xlabel('Frequency (Hz)')
        ax7.set_ylabel('|dΦ/dω|')
        ax7.set_title('Absolute Phase Slope')
        ax7.set_yscale('log')
        ax7.grid(True, alpha=0.3)

        # Add variance annotation
        ax7.text(0.05, 0.95, f'Variance: {phase_slope_variance:.4f}',
                 transform=ax7.transAxes, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Plot 8: Jitter signal autocorrelation
        ax8 = plt.subplot(3, 4, 8)
        if len(jitter_signal) > 1:
            autocorr = np.correlate(jitter_signal, jitter_signal, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]
            time_lags = np.arange(len(autocorr)) / fps

            ax8.plot(time_lags[:100], autocorr[:100] / autocorr[0], 'b-')
            ax8.set_xlabel('Time Lag (s)')
            ax8.set_ylabel('Normalized Autocorrelation')
            ax8.set_title('Jitter Signal Autocorrelation')
            ax8.grid(True, alpha=0.3)

        # Plot 9: Metric relationships scatter plot
        ax9 = plt.subplot(3, 4, 9)
        # Create sliding window analysis
        window_size = len(phase_slope) // 10
        if window_size > 10:
            local_variances = []
            local_means = []

            for i in range(0, len(phase_slope) - window_size, window_size // 2):
                window = phase_slope[i:i + window_size]
                local_variances.append(np.var(window))
                local_means.append(np.mean(np.abs(window)))

            if local_variances and local_means:
                ax9.scatter(local_means, local_variances, alpha=0.6)
                ax9.set_xlabel('Local Mean |dΦ/dω|')
                ax9.set_ylabel('Local Variance')
                ax9.set_title('Local Phase Slope Statistics')
                ax9.grid(True, alpha=0.3)

        # Plot 10: Summary table
        ax10 = plt.subplot(3, 4, 10)
        summary_text = f"""
        METRICS SUMMARY:

        Phase Slope Variance: {phase_slope_variance:.6f}
        (Variance of dΦ/dω across frequencies)

        Overall Jitter Magnitude: {overall_jitter_magnitude:.6f}
        (Mean |dΦ/dω| in low frequencies)

        Jitter Variance: {jitter_variance:.6f}
        (Temporal variance of jitter signal)

        Impulse Detection:
        Found {len(impulse_times)} impulses
        """

        if impulse_times:
            summary_text += "\nImpulse Times (s):\n"
            for i, t_k in enumerate(impulse_times[:3]):
                summary_text += f"  {t_k:.3f}\n"
            if len(impulse_times) > 3:
                summary_text += f"  ... +{len(impulse_times) - 3} more\n"

        ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes,
                  fontsize=8, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax10.set_axis_off()
        ax10.set_title('Analysis Summary')

        # Plot 11: Phase slope cumulative distribution
        ax11 = plt.subplot(3, 4, 11)
        sorted_slopes = np.sort(phase_slope[freq_mask])
        cdf = np.arange(1, len(sorted_slopes) + 1) / len(sorted_slopes)
        ax11.plot(sorted_slopes, cdf, 'b-', linewidth=2)
        ax11.set_xlabel('Phase Slope Value')
        ax11.set_ylabel('Cumulative Probability')
        ax11.set_title('Phase Slope CDF')
        ax11.grid(True, alpha=0.3)

        # Plot 12: Metric evolution over frequency bands
        ax12 = plt.subplot(3, 4, 12)
        freq_bands = [(0, 2), (2, 5), (5, 10), (10, 20)]
        band_means = []
        band_variances = []
        band_labels = []

        for f_min, f_max in freq_bands:
            if f_max <= display_limit:
                band_mask = (pos_freq >= f_min) & (pos_freq < f_max)
                if np.any(band_mask):
                    band_slope = phase_slope[band_mask]
                    band_means.append(np.mean(np.abs(band_slope)))
                    band_variances.append(np.var(band_slope))
                    band_labels.append(f'{f_min}-{f_max}Hz')

        if band_means:
            x = np.arange(len(band_labels))
            width = 0.35
            ax12.bar(x - width / 2, band_means, width, label='Mean |dΦ/dω|', alpha=0.7)
            ax12.bar(x + width / 2, band_variances, width, label='Variance', alpha=0.7)
            ax12.set_xlabel('Frequency Band')
            ax12.set_ylabel('Value')
            ax12.set_title('Metrics by Frequency Band')
            ax12.set_xticks(x)
            ax12.set_xticklabels(band_labels)
            ax12.legend(fontsize=8)
            ax12.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Save the comprehensive plot
        import os
        base_name = os.path.basename(video_path)
        plot_filename = f"jitter_analysis_{os.path.splitext(base_name)[0]}.png"
        fig.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"   Analysis plot saved to: {plot_filename}")

    # ========================================================================
    # STEP 8: PRINT CLEAR SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS - THREE DISTINCT METRICS")
    print("=" * 70)

    print(f"\nVIDEO: {video_path}")
    print(f"Duration: {actual_frames / fps:.2f}s | FPS: {fps:.1f} | Frames: {actual_frames}")

    print(f"\n{'=' * 50}")
    print("CORE METRICS (Distinct and Independent):")
    print(f"{'=' * 50}")

    print(f"\n1. PHASE SLOPE VARIANCE")
    print(f"   Value: {phase_slope_variance:.6f}")
    print(f"   Meaning: Variance of dΦ/dω across all frequencies")
    print(f"   Interpretation: Higher = more inconsistent phase slope")

    print(f"\n2. OVERALL JITTER MAGNITUDE")
    print(f"   Value: {overall_jitter_magnitude:.6f}")
    print(f"   Meaning: Mean absolute phase slope in low frequencies (<10Hz)")
    print(f"   Interpretation: Higher = stronger jitter presence")

    print(f"\n3. JITTER VARIANCE (Temporal)")
    print(f"   Value: {jitter_variance:.6f}")
    print(f"   Meaning: Variance of extracted jitter signal in time domain")
    print(f"   Interpretation: Higher = more temporal variation in jitter")

    print(f"\n{'=' * 50}")
    print("DERIVED METRICS:")
    print(f"{'=' * 50}")
    print(f"Jitter SNR: {results['jitter_snr']:.4f}")
    print(f"Phase Slope Std: {results['phase_slope_std']:.4f}")
    print(f"Phase Slope Mean: {results['phase_slope_mean']:.4f}")

    print(f"\n{'=' * 50}")
    print(f"IMPULSE DETECTION: {len(impulse_times)} impulses found")
    print(f"{'=' * 50}")

    if impulse_times:
        print("\nImpulse Times (s) [From phase slope analysis]:")
        for i, t_k in enumerate(impulse_times):
            print(f"  {i + 1:2d}. t = {t_k:6.3f}s")

    print(f"\n{'=' * 70}")

    return results

# ============================================================================
# MAIN EXECUTION (RUN THIS TO TEST)
# ============================================================================
if __name__ == "__main__":
    print("Jitter Signature Extraction Tool")
    print("Based on: 'A formal modelization of shakiness using Fourier Transform'")
    print("=" * 70)

    # Test with a sample video (change this path to your video)
    test_video = "shaky_climbing5.mp4"  # Replace with your video file

    import os

    if os.path.exists(test_video):
        results = extract_jitter_signatures(test_video, output_plots=True)
        print("\n✅ Analysis complete!")