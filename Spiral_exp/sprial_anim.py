import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import PolyCollection
from IPython.display import Image

from neuronal_attention_circuit import NAC
from ouwrap import OUWrap
from losses import NSACLoss
from neuronal_stochastic_attention_circuit import NSAC

# --------------------------------------------------
# Setup
# --------------------------------------------------

np.random.seed(42)
tf.random.set_seed(42)

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# --------------------------------------------------
# Data
# --------------------------------------------------

def generate_noisy_spirals(
    n_spirals=200,
    steps=1000,
    turns=3,
    noise_std=0.005,
):
    all_t = []
    all_xy = []

    for _ in range(n_spirals):

        t = np.linspace(0, 2 * np.pi * turns, steps)
        r = t / (2 * np.pi * turns)

        x = r * np.cos(t)
        y = r * np.sin(t)

        xy = np.stack([x, y], axis=-1)
        xy += np.random.normal(scale=noise_std, size=xy.shape)

        all_t.append(t[:, None])
        all_xy.append(xy)

    return (
        np.concatenate(all_t, axis=0).astype(np.float32),
        np.concatenate(all_xy, axis=0).astype(np.float32),
    )


t_data, y_data = generate_noisy_spirals(
    n_spirals=300,
    steps=1000,
    turns=3,
    noise_std=0.005,
)

t_min, t_max = float(t_data.min()), float(t_data.max())

t_plot = np.linspace(t_min, t_max, 1000).astype(np.float32)
r_plot = t_plot / (2 * np.pi * 3)

y_true = np.stack(
    [
        r_plot * np.cos(t_plot),
        r_plot * np.sin(t_plot),
    ],
    axis=-1,
)

# --------------------------------------------------
# Prediction storage
# --------------------------------------------------

predictions_all = {"train": []}

# --------------------------------------------------
# Callback
# --------------------------------------------------

class PredictionHistory(tf.keras.callbacks.Callback):

    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def on_epoch_end(self, epoch, logs=None):

        preds = self.model.predict_with_uncertainty(
            tf.convert_to_tensor(t_plot[:, None, None])
        )

        predictions_all[self.mode].append(preds)

# --------------------------------------------------
# Model
# --------------------------------------------------

def model_fn():

    inputs = tf.keras.Input(shape=(1, 1))

    mean, std = OUWrap(
        NAC(d_model=64, num_heads=16, topk=32, sparsity=0.8),
        output_dim=2,
        activation="sigmoid"
    )(inputs)

    return tf.keras.Model(inputs, [mean, std])


model = NSAC(stochastic_model_fn=model_fn(), ood_std=5.0)

model.compile(
    optimizer=tf.keras.optimizers.AdamW(5e-8),
    loss=NSACLoss(),
)

model.summary()

# --------------------------------------------------
# Train
# --------------------------------------------------

model.fit(
    t_data[:, None, :],
    y_data,
    epochs=25,
    batch_size=64,
    callbacks=[PredictionHistory("train")],
    verbose=1,
    shuffle=True,
)

# --------------------------------------------------
# Animation
# --------------------------------------------------

plt.style.use("seaborn-v0_8-deep")

fig, ax = plt.subplots(figsize=(5, 5))
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect("equal")

# Ground truth spiral
ax.plot(y_true[:, 0], y_true[:, 1], color="black", label="Spiral")

# Mean prediction line
line_pred, = ax.plot([], [], color="blue", label="Mean")

# Total uncertainty tube
total_poly = PolyCollection(
    [],
    facecolor="blue",
    alpha=0.2,
    edgecolor="none",
    label="Uncertainty"
)

ax.add_collection(total_poly)
ax.legend(loc="upper left")

# --------------------------------------------------
# Update function
# --------------------------------------------------

def update(frame):

    idx = min(frame, len(predictions_all["train"]) - 1)
    mean, aleatoric, epistemic = predictions_all["train"][idx]

    mean = np.array(mean).squeeze()
    aleatoric = np.array(aleatoric).squeeze()
    epistemic = np.array(epistemic).squeeze()

    mean = np.clip(mean, -1.1, 1.1)

    # -------- Total uncertainty --------
    total_var = np.maximum(aleatoric, 0) + np.maximum(epistemic, 0)
    total_std = np.sqrt(total_var)

    radius = np.linalg.norm(total_std, axis=1, keepdims=True)

    # -------- Compute normals --------
    dx = np.gradient(mean[:, 0])
    dy = np.gradient(mean[:, 1])

    tangent = np.stack([dx, dy], axis=-1)
    norm = np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-8
    tangent /= norm

    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)

    scale = 1.0  # adjust tube thickness

    upper = mean + scale * radius * normal
    lower = mean - scale * radius * normal

    verts = np.concatenate([upper, lower[::-1]], axis=0)

    total_poly.set_verts([verts])

    # -------- Mean --------
    line_pred.set_data(mean[:, 0], mean[:, 1])

    ax.set_title(f"Epoch {idx + 1}")

    return [line_pred, total_poly]

# --------------------------------------------------
# Create animation
# --------------------------------------------------

ani = FuncAnimation(
    fig,
    update,
    frames=len(predictions_all["train"]),
    blit=False,
    interval=100,
)

gif_path = f"{PLOT_DIR}/NSAC_Total_Uncertainty_Spiral.gif"

ani.save(gif_path, writer=PillowWriter(fps=2))
plt.close(fig)

Image(filename=gif_path)