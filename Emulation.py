import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
H, W = 128, 128       # spatial resolution
fps = 30             # frames per second
T = 10 * fps         # 10 seconds video

# --- Create base reference video: moving rectangle ---
video = np.zeros((T, H, W))

# Rectangle parameters
rect_h, rect_w = 30, 40

# Motion parameters for reference trajectory (make motion clearly visible)
vx = 1.2   # pixels per frame (horizontal motion)
vy = 0.8   # pixels per frame (vertical motion)

cx0, cy0 = W // 4, H // 4

# Use floating-point center to avoid quantization freezing
cx_f, cy_f = float(cx0), float(cy0)

for t in range(T):
    # clear frame
    video[t].fill(0.0)

    # update continuous position
    cx_f += vx
    cy_f += vy

    cx = int(np.round(cx_f))
    cy = int(np.round(cy_f))

    # keep inside frame
    cx = np.clip(cx, rect_w//2, W - rect_w//2)
    cy = np.clip(cy, rect_h//2, H - rect_h//2)

    video[t,
          cy - rect_h//2 : cy + rect_h//2,
          cx - rect_w//2 : cx + rect_w//2] = 1.0

# --- Create synthetic random shakiness ---
# Random jitter per frame (pixels)
np.random.seed(0)
sigma = 2.0   # strength of shakiness (pixels)

Dx = np.random.randn(T) * sigma
Dy = np.random.randn(T) * sigma


# Random jitter per frame (pixels)
np.random.seed(0)
sigma = 2.0   # strength of shakiness (pixels)

Dx = np.random.randn(T) * sigma
Dy = np.random.randn(T) * sigma


# Random jitter per frame (pixels)
np.random.seed(0)
sigma = 2.0   # strength of shakiness (pixels)

Dx = np.random.randn(T) * sigma
Dy = np.random.randn(T) * sigma

# --- Apply shakiness ---
shaky = np.zeros_like(video)

for t in range(T):
    sx = int(np.round(Dx[t]))
    sy = int(np.round(Dy[t]))
    shaky[t] = np.roll(video[t], shift=(sy, sx), axis=(0, 1))


video = np.zeros((T, H, W))

# Rectangle parameters
rect_h, rect_w = 30, 40
cy, cx = H // 2, W // 2

for t in range(T):
    video[t,
          cy - rect_h//2 : cy + rect_h//2,
          cx - rect_w//2 : cx + rect_w//2] = 1.0

# --- Create synthetic impulsive shakiness ---
# Impulse times and amplitudes
impulse_times = [15, 30, 45]
Ax = [3, -4, 2]   # horizontal pixel shifts
Ay = [-2, 3, -3]  # vertical pixel shifts

# Build displacement signals
Dx = np.zeros(T)
Dy = np.zeros(T)
for t, a, b in zip(impulse_times, Ax, Ay):
    Dx[t] = a
    Dy[t] = b

# --- Apply shakiness ---
shaky = np.zeros_like(video)

for t in range(T):
    sx = int(Dx[t])
    sy = int(Dy[t])
    shaky[t] = np.roll(video[t], shift=(sy, sx), axis=(0, 1))

# --- Compute 3D Fourier transform ---
F_stable = np.fft.fftn(video)
F_shaky  = np.fft.fftn(shaky)

# Shift for visualization
F_stable = np.fft.fftshift(F_stable)
F_shaky  = np.fft.fftshift(F_shaky)

# --- Analyze temporal frequency slice at zero spatial frequency ---
# Center spatial frequencies
kx0, ky0 = H//2, W//2

S_stable = F_stable[:, kx0, ky0]
S_shaky  = F_shaky[:, kx0, ky0]

# Magnitude and phase
mag_stable = np.abs(S_stable)
mag_shaky  = np.abs(S_shaky)
phase_stable = np.unwrap(np.angle(S_stable))
phase_shaky  = np.unwrap(np.angle(S_shaky))

# --- Visualize one frame before and after shakiness ---
plt.figure(figsize=(8, 4))

frame_id = 30

plt.subplot(1, 2, 1)
plt.title("Stable frame")
plt.imshow(video[frame_id], cmap="gray")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Shaky frame")
plt.imshow(shaky[frame_id], cmap="gray")
plt.axis("off")

plt.tight_layout()
plt.show()

# --- Animate reference vs shaky video in parallel ---
from matplotlib import animation

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

im_ref = axes[0].imshow(video[0], cmap="gray", animated=True)
axes[0].set_title("Reference (moving rectangle)")
axes[0].axis("off")

im_shaky = axes[1].imshow(shaky[0], cmap="gray", animated=True)
axes[1].set_title("Shaky video")
axes[1].axis("off")


def update(frame):
    im_ref.set_array(video[frame])
    im_shaky.set_array(shaky[frame])
    return [im_ref, im_shaky]

ani = animation.FuncAnimation(fig, update, frames=T, interval=1000/fps, blit=True)
plt.tight_layout()
plt.show()
