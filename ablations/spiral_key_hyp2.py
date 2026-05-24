import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy.random as npr

from ouwrap import OUWrap
from losses import NSACLoss
from neuronal_stochastic_attention_circuit import NSAC
from neuronal_attention_circuit import NAC
from metrics import compute_crps



FIG_WIDTH = 3.25
FIG_HEIGHT = 2.2

SEEDS = [0, 1, 2, 3, 4]

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

PLOT_DIR = "plots"
WEIGHTS_DIR = "ablation_weights"
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

        self._generate_dataset()

    # ------------------ DATA ------------------

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
        return np.concatenate(all_t, axis=0).astype(np.float32), np.concatenate(all_xy, axis=0).astype(np.float32)

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


    def build_nsac(self, ood_mean, ood_std, mc_sample ):
        def model_fn():
            inputs = tf.keras.Input(shape=(1, 1))
            mean, std = OUWrap(
                NAC(d_model=64, num_heads=16, topk=8, sparsity=0.5),
                output_dim=2,
                activation="sigmoid"
            )(inputs)
            return tf.keras.Model(inputs, [mean, std])

        model = NSAC(stochastic_model_fn=model_fn(), ood_mean=ood_mean, ood_std=ood_std, mc_samples=mc_sample)
        model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss=NSACLoss())
        return model

    def _weight_path(self, param_name, value):
        return os.path.join(WEIGHTS_DIR, f"Spiral_NSAC_{param_name}_{value}.keras")


    def train(self, model, param_name, value, epochs=300):
        path = self._weight_path(param_name, value)

        # If weights exist, load and skip training
        if os.path.exists(path):
            print(f"Loading existing weights from {path}")
            model.load_weights(path)
            return

        t_full = self.t_clean
        steps = self.steps
        ntrain = self.ntrain
        nsample_obs = self.nsample_obs
        train_noisy = self.train_noisy

        for epoch in range(epochs):
            train_idx = sorted(np.random.choice(steps, nsample_obs, replace=False).tolist())
            t_train = t_full[train_idx]
            x_train = train_noisy[:, train_idx, :]

            t_flat = np.tile(t_train, (ntrain, 1)).flatten()[:, None, None]
            y_flat = x_train.reshape(-1, 2)

            perm = np.random.permutation(len(t_flat))
            t_flat = t_flat[perm]
            y_flat = y_flat[perm]

            model.train_on_batch(t_flat, y_flat)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")

        model.save(path)


    def evaluate(self, model):
        t_full = self.t_clean[:, None, None]
        mu, std = model.predict(t_full, verbose=0)
        mu = np.repeat(mu[None, ...], self.ntest, axis=0)
        std = np.repeat(std[None, ...], self.ntest, axis=0)
        y_true = self.test_clean

        cutoff = int(self.steps * 0.3)
        interp_mask = np.arange(self.steps) < cutoff
        extrap_mask = ~interp_mask

        #compute mean and std 
        crps_interp = np.abs(compute_crps(mu[:, interp_mask], std[:, interp_mask], y_true[:, interp_mask]))
        crps_extrap = np.abs(compute_crps(mu[:, extrap_mask], std[:, extrap_mask], y_true[:, extrap_mask]))
        return {
            "crps_interp": crps_interp,
            "crps_extrap": crps_extrap,
        }


    def run_ablation(self, param_name, values, base_config, epochs=50):
        results = []

        for val in values:
            config = base_config.copy()
            config[param_name] = val

            print(f"\n{'='*50}\n{param_name} = {val}\n{'='*50}")

            seed_metrics = []

            for seed in SEEDS:
                print(f"  Seed {seed}")

                np.random.seed(seed)
                tf.random.set_seed(seed)
                npr.seed(seed)

                model = self.build_nsac(**config)

                # unique weight per seed
                self.train(model, f"{param_name}_{val}_seed", seed, epochs=epochs)

                metrics = self.evaluate(model)
                seed_metrics.append(metrics)

            # aggregate
            agg = {}
            for key in seed_metrics[0].keys():
                vals_k = [m[key] for m in seed_metrics]
                agg[f"{key}_mean"] = np.mean(vals_k)
                agg[f"{key}_std"] = np.std(vals_k)

            results.append({
                "param": param_name,
                "value": val,
                **agg
            })

        return results



def plot_with_uncertainty(x, mean, std, label, linestyle='-', marker=None):
    mean = np.array(mean)
    std = np.array(std)

    plt.plot(x, mean, linestyle=linestyle, marker=marker, label=label)
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)



if __name__ == "__main__":

    spiral = Spiral()

    BASE_CONFIG = {"ood_mean": 0, "ood_std": 5, "mc_sample": 5}

    PARAM_SWEEPS = {
        "ood_mean": [0, 1, 2, 3, 4, 5],
        "ood_std": [1, 2, 3, 5, 10, 15],
        "mc_sample": [1, 3, 5, 10, 15, 20]
    }

    all_results = {}

    for param_name, values in PARAM_SWEEPS.items():

        print("\n" + "-"*60)
        print(f"Ablation Study: {param_name}")
        print("-"*60)

        results = spiral.run_ablation(param_name, values, BASE_CONFIG, epochs=50)

        if not results:
            continue

        results = sorted(results, key=lambda x: x["value"])
        all_results[param_name] = results

        vals = [r["value"] for r in results]

        plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

        # plot_with_uncertainty(
        #     vals,
        #     [r["mae_interp_mean"] for r in results],
        #     [r["mae_interp_std"]/2.0 for r in results],
        #     "MAE-interp",
        #     marker='o'
        # )

        # plot_with_uncertainty(
        #     vals,
        #     [r["mae_extrap_mean"] for r in results],
        #     [r["mae_extrap_std"] for r in results],
        #     "MAE-extrap",
        #     marker='o'

        # )

        plot_with_uncertainty(
            vals,
            [r["crps_interp_mean"] for r in results],
            [r["crps_interp_std"]/2.0 for r in results],
            "CRPS-interp",
            linestyle='--',
            marker='x'
        )

        plot_with_uncertainty(
            vals,
            [r["crps_extrap_mean"] for r in results],
            [r["crps_extrap_std"] for r in results],
            "CRPS-extrap",
            linestyle='--',
            marker='x'
        )

        plt.xlabel(param_name)
        plt.ylabel("Error")
        plt.title(f"effect of {param_name}", pad=2)
        plt.legend(frameon=False)

        # plt.ylim(0, 0.7)
        # plt.yscale('log')
        plt.tight_layout(pad=0.2)

        base_path = os.path.join(PLOT_DIR, f"{param_name}_ablation")

        plt.savefig(base_path + ".png", dpi=300, bbox_inches="tight")
        plt.close()

        # save raw results
        with open(base_path + "_results.json", "w") as f:
            json.dump(results, f, indent=2)