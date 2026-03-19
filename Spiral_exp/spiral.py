import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy.random as npr
import math

from neuronal_attention_circuit import NAC
from ouwrap import OUWrap
from losses import NSACLoss, NLL
from neuronal_stochastic_attention_circuit import NSAC
from deep_ensemble import DeepEnsemble
from monte_carlo import MC_dropout
from DeepEvidential import Evidential
from evidential_deep_learning.layers import DenseNormalGamma
from evidential_deep_learning.losses import EvidentialRegression
from GMLE import GMLE


np.random.seed(42)
tf.random.set_seed(42)
npr.seed(42)

PLOT_DIR = "plots"
WEIGHTS_DIR  = 'model_weights'
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)


class Spiral:
    def __init__(self,
                 n_spirals=500,
                 steps=300,
                 turns=3,
                 noise_std=0.05,
                 ntrain=350,
                 ntest=150,
                 nsample_obs=50):
        
        self.n_spirals = n_spirals
        self.steps = steps
        self.turns = turns
        self.noise_std = noise_std
        self.ntrain = ntrain
        self.ntest = ntest
        self.nsample_obs = nsample_obs

        # Generate data
        self._generate_dataset()

        self.models = {}

    # Data generation methods
    def _generate_noisy_spirals(self):
        all_t, all_xy = [], []
        for _ in range(self.n_spirals):
            t = np.linspace(0, 2 * np.pi * self.turns, self.steps)
            r = t / (2 * np.pi * self.turns)
            x = r * np.cos(t)
            y = r * np.sin(t)
            xy = np.stack([x, y], axis=-1)
            xy += np.random.normal(scale=self.noise_std, size=xy.shape)
            all_t.append(t[:, None])
            all_xy.append(xy)
        return (np.concatenate(all_t, axis=0).astype(np.float32),
                np.concatenate(all_xy, axis=0).astype(np.float32))

    def _generate_dataset(self):
        t_data, y_data = self._generate_noisy_spirals()
        t_data = t_data.reshape(self.n_spirals, self.steps, 1)
        y_data = y_data.reshape(self.n_spirals, self.steps, 2)

        # Clean ground truth (identical for all spirals)
        t_clean = np.linspace(0, 2 * np.pi * self.turns, self.steps)
        r_clean = t_clean / (2 * np.pi * self.turns)
        x_clean = r_clean * np.cos(t_clean)
        y_clean = r_clean * np.sin(t_clean)
        clean_traj = np.stack([x_clean, y_clean], axis=-1)
        clean_data = np.tile(clean_traj, (self.n_spirals, 1, 1))

        self.train_noisy = y_data[:self.ntrain]
        self.train_clean = clean_data[:self.ntrain]
        self.test_noisy = y_data[self.ntrain:]
        self.test_clean = clean_data[self.ntrain:]
        self.t_clean = t_clean

        # Fixed observation indices for test evaluation
        self.test_idx = sorted(npr.choice(self.steps, self.nsample_obs, replace=False).tolist())

    # Model creation methods 
    def _build_nsac(self):
        def model_fn():
            inputs = tf.keras.Input(shape=(1, 1))
            mean, std = OUWrap(
                NAC(d_model=64, num_heads=16, topk=32, sparsity=0.8),
                output_dim=2,
                activation="sigmoid"
            )(inputs)
            return tf.keras.Model(inputs, [mean, std])

        model = NSAC(stochastic_model_fn=model_fn(), ood_std=5.0)
        model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss=NSACLoss())
        return model

    def _build_mle(self):
        def model_fn():
            inputs = tf.keras.Input(shape=(1, 1))
            attn = NAC(
                d_model=64, num_heads=16, topk=32, mode='exact',
                sparsity=0.8, activation='sigmoid', return_sequences=False
            )(inputs)
            mu = tf.keras.layers.Dense(2)(attn)
            log_std = tf.keras.layers.Dense(2)(attn)
            # outputs = tf.keras.layers.Concatenate()(  )
            return tf.keras.Model(inputs, [mu, log_std])

        model = GMLE(model_fn())
        model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss=NLL())
        return model

    def _build_deep_ensemble(self):
        def model_fn():
            inputs = tf.keras.Input(shape=(1, 1))
            attn = NAC(
                d_model=64, num_heads=16, topk=32, mode='exact',
                sparsity=0.8, activation='sigmoid', return_sequences=False
            )(inputs)
            outputs = tf.keras.layers.Dense(2)(attn)
            return tf.keras.Model(inputs, outputs)

        model = DeepEnsemble(model_fn=model_fn())
        model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss=tf.keras.losses.MeanSquaredError())
        return model

    def _build_mc_dropout(self):
        def model_fn(dropout_rate=0.1):
            inputs = tf.keras.Input(shape=(1, 1))
            attn = NAC(
                d_model=64, num_heads=16, topk=32, mode='exact',
                sparsity=0.8, activation='sigmoid', return_sequences=False
            )(inputs)
            drop = tf.keras.layers.Dropout(dropout_rate)(attn, training=True)
            outputs = tf.keras.layers.Dense(2)(drop)
            return tf.keras.Model(inputs, outputs)

        model = MC_dropout(model_fn=model_fn())
        model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss='mse')
        return model

    def _build_evidential(self):
        def model_fn():
            inputs = tf.keras.Input(shape=(1, 1))
            attn = NAC(
                d_model=64, num_heads=16, topk=32, mode='exact',
                delta_t=1.0, sparsity=0.8, activation='sigmoid', return_sequences=False
            )(inputs)
            outputs = DenseNormalGamma(2)(attn)
            return tf.keras.Model(inputs, outputs)

        model = Evidential(model_fn=model_fn(), lambda_reg=0.5)
        model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss=EvidentialRegression)
        return model

    def create_models(self, model_list=None):
        if model_list is None:
            model_list = ["NSAC", "Deep-Ensemble", "MC-Dropout", "Evidential"]

        builders = {
            "NSAC": self._build_nsac,
            "Gaussian-MLE": self._build_mle,
            "Deep-Ensemble": self._build_deep_ensemble,
            "MC-Dropout": self._build_mc_dropout,
            "Evidential": self._build_evidential}
        
        for name in model_list:
            if name not in builders:
                raise ValueError(f"Unknown model: {name}")
            print(f"Building {name}...")
            self.models[name] = builders[name]()

    # Weight management
    def _weight_path(self, model_name):
        return os.path.join(WEIGHTS_DIR, f"{model_name}_weights.keras")

    def has_weights(self, model_name):
        return os.path.exists(self._weight_path(model_name))

    def load_weights(self, model_name):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Call create_models() first.")
        path = self._weight_path(model_name)
        if os.path.exists(path):
            self.models[model_name].load_weights(path)
            print(f"Loaded weights for {model_name} from {path}")
            return True
        else:
            print(f"No weights found for {model_name} at {path}")
            return False

    # Training (with load‑if‑exists logic)
    def train(self, model_name, epochs=500, batch_size=64, verbose=1, force_retrain=False):

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Call create_models() first.")

        # Check for existing weights
        if not force_retrain and self.has_weights(model_name):
            print(f"Found existing weights for {model_name}. Loading and skipping training.")
            self.load_weights(model_name)
            return

        model = self.models[model_name]
        t_full = self.t_clean
        steps = self.steps
        ntrain = self.ntrain
        nsample_obs = self.nsample_obs
        train_noisy = self.train_noisy

        print(f"Training {model_name} from scratch...")
        for epoch in range(epochs):
            # Sample new observation indices each epoch
            train_idx = sorted(np.random.choice(steps, nsample_obs, replace=False).tolist())

            t_train = t_full[train_idx]                       # (nsample_obs,)
            x_train = train_noisy[:, train_idx, :]            # (ntrain, nsample_obs, 2)

            # Flatten and shuffle
            t_flat = np.tile(t_train, (ntrain, 1)).flatten()[:, None, None]  # (ntrain*nsample_obs, 1, 1)
            y_flat = x_train.reshape(-1, 2)                                   # (ntrain*nsample_obs, 2)

            perm = np.random.permutation(len(t_flat))
            t_flat = t_flat[perm]
            y_flat = y_flat[perm]

            # Train one epoch
            model.fit(t_flat, y_flat, epochs=1, batch_size=batch_size, verbose=verbose, shuffle=False)

            if (epoch + 1) % 10 == 0:
                print(f"{model_name} - Epoch {epoch+1}/{epochs} completed")

        # Save weights after training
        model.save(self._weight_path(model_name))
        print(f"Saved weights for {model_name} to {self._weight_path(model_name)}")

    def train_all(self, epochs=500, batch_size=64, verbose=0, force_retrain=False):
        for name in self.models:
            print(f"\n{'='*50}\nProcessing {name}\n{'='*50}")
            self.train(name, epochs, batch_size, verbose, force_retrain)

    # Unified evaluation using predict_with_uncertainty
    def evaluate(self, model_name):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")

        model = self.models[model_name]
        t_full = self.t_clean[:, None, None]  # (steps, 1, 1)
        ntest = self.ntest

        mean_list, ale_list, epi_list = [], [], []

        for i in range(ntest):
            # All models now support predict_with_uncertainty
            mean, ale, epi = model.predict_with_uncertainty(t_full)
            mean_list.append(mean.squeeze())
            ale_list.append(ale.squeeze())
            epi_list.append(epi.squeeze())

        mean_pred = np.stack(mean_list, axis=0)  # (ntest, steps, 2)
        ale_pred = np.stack(ale_list, axis=0)
        epi_pred = np.stack(epi_list, axis=0)

        return mean_pred, ale_pred, epi_pred

    # Core plotting routine (used by both single and multi‑model figures)
    def _plot_on_axis(self, ax, model_name, mean_pred, ale_pred, epi_pred,
                      scale=1.0, show_legend=True, title=None):
        steps = self.steps
        test_clean = self.test_clean
        test_noisy = self.test_noisy
        test_idx = self.test_idx

        idx_vis = 2  # use the third test spiral for consistency
        mean_vis = mean_pred[idx_vis]
        ale_vis = ale_pred[idx_vis]
        epi_vis = epi_pred[idx_vis]
        true_vis = test_clean[idx_vis]
        obs_points_vis = test_noisy[idx_vis, test_idx, :]

        # Interpolation / extrapolation masks (as in original)
        half = steps // 3.5
        interp_mask = np.arange(steps) < half
        extrap_mask = ~interp_mask

        # Compute normals for uncertainty tube
        dx = np.gradient(mean_vis[:, 0])
        dy = np.gradient(mean_vis[:, 1])
        tangent = np.stack([dx, dy], axis=-1)
        norm_tang = np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-8
        tangent /= norm_tang
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)

        # Total uncertainty radius
        ale_radius = np.linalg.norm(ale_vis, axis=1, keepdims=True)
        epi_radius = np.linalg.norm(epi_vis, axis=1, keepdims=True)
        total_radius = np.sqrt(ale_radius**2 + epi_radius**2)

        # Ground truth
        ax.plot(true_vis[interp_mask, 0], true_vis[interp_mask, 1],
                color='green', linewidth=2.5)
        ax.plot(true_vis[extrap_mask, 0], true_vis[extrap_mask, 1],
                color='green', linestyle=':', linewidth=2.5, label='Ground Truth' if show_legend else "")

        # Observations
        ax.scatter(obs_points_vis[:, 0], obs_points_vis[:, 1],
                   color='black', s=30, zorder=5, label='Data Points' if show_legend else "")

        # Mean prediction
        ax.plot(mean_vis[interp_mask, 0], mean_vis[interp_mask, 1],
                color='blue', linewidth=2, label='Interpolation' if show_legend else "")
        ax.plot(mean_vis[extrap_mask, 0], mean_vis[extrap_mask, 1],
                color='red', linewidth=2, label='Extrapolation' if show_legend else "")

        # Uncertainty tube (gradient shading)
        n_bands = 25
        max_r = scale * total_radius.squeeze()

        for i in range(n_bands):
            frac = (i + 1) / n_bands
            r = max_r * frac
            offset_pos = mean_vis + r[:, None] * normal
            offset_neg = mean_vis - r[:, None] * normal
            verts = np.concatenate([offset_pos, offset_neg[::-1]], axis=0)
            verts = np.vstack([verts, verts[0:1]])
            alpha = 0.05 * frac
            poly = PolyCollection([verts], facecolor='purple', alpha=alpha, edgecolor='none')
            ax.add_collection(poly)

        # Legend (only if requested)
        if show_legend:
            uncertainty_patch = Patch(facecolor='purple', alpha=0.35, label='Uncertain')
            handles, labels = ax.get_legend_handles_labels()
            handles.append(uncertainty_patch)
            labels.append('Uncertain')
            ax.legend(handles, labels, loc='upper right', fontsize=8)

        ax.set_aspect("equal")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        if title is not None:
            ax.set_title(title, fontsize=12)
        else:
            ax.set_title(model_name, fontsize=12)

    # Single‑model visualisation (with optional scale)
    def visualize(self, model_name, mean_pred, ale_pred, epi_pred, scale=10.0, save_path=None):

        fig, ax = plt.subplots(figsize=(6, 6))
        self._plot_on_axis(ax, model_name, mean_pred, ale_pred, epi_pred,
                           scale=scale, show_legend=True, title=model_name)
        plt.tight_layout()
        if save_path is None:
            save_path = os.path.join(PLOT_DIR, f"{model_name}_uncertainty.png")
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"Plot saved to {save_path}")

    # Multi‑model visualisation: all models in one figure (3 per row)
    def plot_all_models(self, models_to_plot=None, scales=None, save_path=None):

        if models_to_plot is None:
            models_to_plot = list(self.models.keys())

        n_models = len(models_to_plot)
        if n_models == 0:
            print("No models to plot.")
            return

        # --- Fixed 4 columns layout ---
        cols = 5
        rows = math.ceil(n_models / cols)

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(5 * cols, 5 * rows),
            sharey=True
        )

        axes = np.atleast_1d(axes).flatten()

        # --- Plot models ---
        for idx, name in enumerate(models_to_plot):
            mean_pred, ale_pred, epi_pred = self.evaluate(name)
            ax = axes[idx]

            scale = 1.0
            if isinstance(scales, dict):
                scale = scales.get(name, 1.0)

            self._plot_on_axis(
                ax,
                name,
                mean_pred,
                ale_pred,
                epi_pred,
                scale=scale,
                show_legend=False,
                title=name
            )

        # --- Remove unused axes ---
        for ax in axes[n_models:]:
            ax.remove()

        # --- Hide y-axis labels for non-first columns ---
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if idx >= n_models:
                    continue
                if c != 0:
                    axes[idx].set_ylabel("")
                    axes[idx].tick_params(labelleft=False)

        # --- Shared legend ---
        handles = [
            Line2D([0], [0], color='green', linewidth=2.5, label='Ground Truth'),
            Line2D([0], [0], marker='o', color='black', linestyle='None',
                markersize=5, label='Data Points'),
            Line2D([0], [0], color='blue', linewidth=2, label='Interpolation'),
            Line2D([0], [0], color='red', linewidth=2, label='Extrapolation'),
            Patch(facecolor='purple', alpha=0.35, label='Uncertain')
        ]

        fig.legend(
            handles=handles,
            loc='upper center',
            ncol=len(handles),
            fontsize=10,
            bbox_to_anchor=(0.5, 1.05)
        )

        plt.tight_layout(rect=[0, 0, 1, 0.93])

        if save_path is None:
            save_path = os.path.join(PLOT_DIR, "all_models_uncertainty.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)

        print(f"Combined plot saved to {save_path}")




if __name__ == "__main__":
    #all models
    model_list = ["NSAC", "Gaussian-MLE", "Deep-Ensemble", "MC-Dropout", "Evidential"]
    #scale for visualization
    scales = {"NSAC": 10.0, "Gaussian-MLE": 10.0, "Deep-Ensemble": 1.0, "MC-Dropout": 7.0, "Evidential": 6.0}

    spiral = Spiral()  #create spiral instance
    spiral.create_models(model_list) #create model

    for name in model_list:
        print(f"\n{'='*60}")
        print(f"Processing {name}")
        print('='*60)

        spiral.train(name, epochs=500, batch_size=64, verbose=0, force_retrain=False)

        mean_pred, ale_pred, epi_pred = spiral.evaluate(name)

        half = int(spiral.steps / 3.5)
        interp_mask = np.arange(spiral.steps) < half
        extrap_mask = ~interp_mask

        y_true = spiral.test_clean

        # MAE
        mae_interp = np.mean(np.abs(mean_pred[:, interp_mask, :] - y_true[:, interp_mask, :]))
        mae_extrap = np.mean(np.abs(mean_pred[:, extrap_mask, :] - y_true[:, extrap_mask, :]))
        print(f"{name} - MAE interp: {mae_interp:.4f}, extrap: {mae_extrap:.4f}")

        # RMSE
        rmse_interp = np.mean((mean_pred[:, interp_mask, :] - y_true[:, interp_mask, :])**2)
        rmse_extrap = np.mean((mean_pred[:, extrap_mask, :] - y_true[:, extrap_mask, :])**2)
        print(f"{name} - RMSE interp: {rmse_interp:.4f}, extrap: {rmse_extrap:.4f}")

        scale = scales[name]
        spiral.visualize(name, mean_pred, ale_pred, epi_pred, scale=scale)

        # Combined plot with all models
    spiral.plot_all_models(model_list, scales=scales)
