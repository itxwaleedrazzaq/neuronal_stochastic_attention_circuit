import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy.random as npr
from neuronal_attention_circuit import NAC
from ouwrap import OUWrap
from losses import NSACLoss
from neuronal_stochastic_attention_circuit import NSAC


SEEDS = [0, 1, 2, 3, 4]

FIG_WIDTH = 3.68
FIG_HEIGHT = 2.2

PLOT_DIR = "plots"
WEIGHTS_DIR = "model_weights3"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.2,
    "lines.markersize": 3,
})


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

        self._generate_dataset()

    def regenerate_with_steps(self, steps):
        self.steps = steps
        self._generate_dataset()

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

        t_clean = np.linspace(0, 2 * np.pi * self.turns, self.steps)
        r_clean = t_clean / (2 * np.pi * self.turns)
        x_clean = r_clean * np.cos(t_clean)
        y_clean = r_clean * np.sin(t_clean)

        clean_traj = np.stack([x_clean, y_clean], axis=-1)
        clean_data = np.tile(clean_traj, (self.n_spirals, 1, 1))

        self.train_noisy = y_data[:self.ntrain]
        self.test_clean = clean_data[self.ntrain:]
        self.t_clean = t_clean

    def build_nsac(self, d_model, num_heads, topk, sparsity):
        def model_fn():
            inputs = tf.keras.Input(shape=(1, 1))
            out, _ = OUWrap(
                NAC(d_model=d_model, num_heads=num_heads, topk=topk, sparsity=sparsity),
                output_dim=2,
                activation="sigmoid"
            )(inputs)
            return tf.keras.Model(inputs, out)

        model = model_fn()
        model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss='mse', metrics=['mae'])
        return model

    def _weight_path(self, name):
        return os.path.join(WEIGHTS_DIR, f"{name}.keras")


    def train(self, model, name, epochs=100):
        path = self._weight_path(name)

        if os.path.exists(path):
            print(f"Loading existing weights from {path}")
            model.load_weights(path)
            return

        for epoch in range(epochs):
            idx = sorted(np.random.choice(self.steps, self.nsample_obs, replace=False))
            t_train = self.t_clean[idx]
            x_train = self.train_noisy[:, idx, :]
            t_flat = np.tile(t_train, (self.ntrain, 1)).flatten()[:, None, None]
            y_flat = x_train.reshape(-1, 2)

            perm = np.random.permutation(len(t_flat))
            t_flat = t_flat[perm]
            y_flat = y_flat[perm]

            model.train_on_batch(t_flat, y_flat)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")

        model.save(path)


    def train_or_load(self, model, name, epochs=100):
        self.train(model, name, epochs)
        return model


    def evaluate(self, model):
        t_full = self.t_clean[:, None, None]
        pred = model.predict(t_full, verbose=0)
        pred = np.repeat(pred[None, ...], self.ntest, axis=0)

        y_true = self.test_clean

        cutoff = int(self.steps * 0.3)
        interp_mask = np.arange(self.steps) < cutoff
        extrap_mask = ~interp_mask

        return {
            "mae_interp": np.mean(np.abs(pred[:, interp_mask] - y_true[:, interp_mask])),
            "mae_extrap": np.mean(np.abs(pred[:, extrap_mask] - y_true[:, extrap_mask])),
            "mse_interp": np.mean((pred[:, interp_mask] - y_true[:, interp_mask])**2),
            "mse_extrap": np.mean((pred[:, extrap_mask] - y_true[:, extrap_mask])**2),
        }


    def run_seq_vs_topk_ablation(self, seq_lengths, topk_values, base_config, epochs=100):
        results = []

        for topk in topk_values:
            print(f"\n{'='*50}\nTOP-K = {topk}\n{'='*50}")

            for seq_len in seq_lengths:
                print(f"\nSequence Length = {seq_len}")

                self.regenerate_with_steps(seq_len)

                config = base_config.copy()
                config["topk"] = topk

                if config["d_model"] % config["num_heads"] != 0:
                    continue

                seed_metrics = []

                for seed in SEEDS:
                    print(f"  Seed {seed}")

                    np.random.seed(seed)
                    tf.random.set_seed(seed)
                    npr.seed(seed)

                    model = self.build_nsac(**config)

                    name = f"Spiral_NSAC_topk_{topk}_seq_{seq_len}_seed_{seed}"
                    model = self.train_or_load(model, name, epochs)

                    metrics = self.evaluate(model)
                    seed_metrics.append(metrics)

                # aggregate
                agg = {}
                for key in seed_metrics[0].keys():
                    vals_k = [m[key] for m in seed_metrics]
                    agg[f"{key}_mean"] = np.mean(vals_k)
                    agg[f"{key}_std"] = np.std(vals_k)

                results.append({
                    "topk": topk,
                    "seq_len": seq_len,
                    **agg
                })

        return results



def plot_seq_vs_topk(results):
    seq_labels = sorted(set(r["seq_len"] for r in results))
    x_positions = list(range(len(seq_labels)))
    topk_values = sorted(set(r["topk"] for r in results))

    plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    for topk in topk_values:
        subset = [r for r in results if r["topk"] == topk]
        subset = sorted(subset, key=lambda x: x["seq_len"])

        x = [seq_labels.index(r["seq_len"]) for r in subset]
        mean = np.array([r["mse_interp_mean"] for r in subset])
        std = np.array([r["mse_interp_std"]/3.0 for r in subset])

        plt.plot(x, mean, marker='o', label=f"K={topk}")
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Seq Length")
    plt.ylabel("MSE (interp)")
    plt.title("Seq Length vs Top-K", pad=2)
    plt.xticks(x_positions, seq_labels)

    plt.legend(frameon=False, loc="best")
    plt.tight_layout(pad=0.3)

    base_path = os.path.join(PLOT_DIR, "seq_vs_topk")

    plt.savefig(base_path + ".png", dpi=300, bbox_inches="tight")
    plt.close()

    # save results
    with open(base_path + "_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":

    spiral = Spiral()

    BASE_CONFIG = {
        "d_model": 64,
        "num_heads": 4,
        "topk": 8,
        "sparsity": 0.5
    }

    SEQ_LENGTHS = [100, 500, 1000, 5000, 10000, 20000, 30000, 50000]
    TOPK_VALUES = [2, 4, 8, 16, 32]

    seq_results = spiral.run_seq_vs_topk_ablation(
        SEQ_LENGTHS,
        TOPK_VALUES,
        BASE_CONFIG,
        epochs=50
    )

    plot_seq_vs_topk(seq_results)