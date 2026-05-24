import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import psutil
from neuronal_attention_circuit import NAC
from ouwrap import OUWrap
from neuronal_stochastic_attention_circuit import NSAC

plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.2,
    "lines.markersize": 6,
})

# --- CONFIG ---
TOPK_VALUES = [2, 4, 8, 16, 32]
NUM_RUNS = 10

batch_size = 1
sequence_length = 1000


# --- MODEL ---
def build_nsac(topk):
    def model_fn():
        inputs = tf.keras.Input(shape=(None, 1))
        out, _ = OUWrap(
            NAC(d_model=64, num_heads=4, topk=topk),
            output_dim=2,
            activation="sigmoid"
        )(inputs)
        return tf.keras.Model(inputs, out)

    model = model_fn()
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss='mse')
    return model


def measure_runtime(model, num_runs=10):
    dummy_input = np.random.randn(
        batch_size,
        sequence_length,
        1,
        1
    ).astype(np.float32)

    # warm-up
    for _ in range(10):
        _ = model(dummy_input, training=False).numpy()

    runtimes = []

    for _ in range(num_runs):
        start = time.time()
        _ = model(dummy_input, training=False).numpy()
        end = time.time()
        runtimes.append(end - start)
    runtimes = np.array(runtimes)
    return runtimes.mean(), runtimes.std()

def measure_memory(model, num_runs=10):
    dummy_input = np.random.randn(
            batch_size,
            sequence_length,
            1,
            1
        ).astype(np.float32)

    # warm-up
    for _ in range(10):
        _ = model(dummy_input, training=False).numpy()

    process = psutil.Process(os.getpid())
    mem_samples = []

    for _ in range(num_runs):
        mem_before = process.memory_info().rss
        _ = model(dummy_input, training=False).numpy()
        mem_after = process.memory_info().rss
        mem_samples.append((mem_after - mem_before) / (1024 ** 2))  # MB
    mem_samples = np.array(mem_samples)
    return mem_samples.mean(), mem_samples.std()


# --- MAIN SWEEP ---
results = []

for topk in TOPK_VALUES:
    print(f"Running topk={topk}")

    model = build_nsac(topk)

    mean_rt, std_rt = measure_runtime(model, NUM_RUNS)
    mean_mem, std_mem = measure_memory(model, NUM_RUNS)

    results.append({
        "topk": topk,
        "mean_rt": mean_rt,
        "std_rt": std_rt,
        "mean_mem": mean_mem,
        "std_mem": std_mem,
    })

    del model
    tf.keras.backend.clear_session()
    time.sleep(1)


# --- EXTRACT ---
topk_vals = [r["topk"] for r in results]

mean_rt = np.array([r["mean_rt"] for r in results])
std_rt = np.array([r["std_rt"] for r in results])

mean_mem = np.array([r["mean_mem"] for r in results])
std_mem = np.array([r["std_mem"] for r in results])


# --- PLOT ---
fig, ax1 = plt.subplots(figsize=(6, 4))

# Runtime
ax1.plot(topk_vals, mean_rt, marker='o', label="Runtime (s)")
ax1.fill_between(
    topk_vals,
    mean_rt - std_rt,
    mean_rt + std_rt,
    alpha=0.2
)

ax1.set_xlabel("Top-K")
ax1.set_ylabel("Runtime (s)")
ax1.grid(True)

ax1.set_xscale("log", base=2)
ax1.set_xticks(topk_vals)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())

# Memory (secondary axis)
ax2 = ax1.twinx()
ax2.plot(topk_vals, mean_mem, marker='s', color='orange', label="Memory (MB)")
ax2.fill_between(
    topk_vals,
    mean_mem - std_mem,
    mean_mem + std_mem,
    alpha=0.2,
    color='orange'
)
ax2.set_ylabel("Memory (MB)")

# Title
plt.title("Runtime and Memory vs Top-K")

# Legend merge
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.tight_layout()
plt.savefig("topk_runtime_memory.png", dpi=300)
plt.show()