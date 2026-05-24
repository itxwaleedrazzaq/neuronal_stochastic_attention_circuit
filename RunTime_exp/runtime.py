import os
import time
import pandas as pd
import numpy as np
import tensorflow as tf

from ouwrap import OUWrap
from losses import NSACLoss, NLL
from neuronal_attention_circuit import NAC
from neuronal_stochastic_attention_circuit import NSAC

from baselines import GMLE, DeepEnsemble, SDE_Net, MC_dropout,Evidential
from evidential_deep_learning.layers import DenseNormalGamma
from evidential_deep_learning.losses import EvidentialRegression

np.random.seed(42)

STAT_DIR = "statistics"

NUM_RUNS = 10
BATCH_SIZE = 1
SEQ_LEN = 1000   

# Model Builders
def _build_nsac():
    def model_fn():
        inputs = tf.keras.Input(shape=(SEQ_LEN, 1))
        mean, std = OUWrap(
            NAC(d_model=32, num_heads=4, topk=8, sparsity=0.5),
            output_dim=1,
            activation="sigmoid"
        )(inputs)
        return tf.keras.Model(inputs, [mean, std])
    model = NSAC(stochastic_model_fn=model_fn(), mc_samples=20, ood_std=5.0)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4),loss=NSACLoss())
    return model

def _build_mle():
    def model_fn():
        inputs = tf.keras.Input(shape=(SEQ_LEN, 1))
        attn = NAC(
            d_model=32, num_heads=4, topk=8, mode='exact',
            sparsity=0.5, activation='sigmoid', return_sequences=False
        )(inputs)
        mu = tf.keras.layers.Dense(1)(attn)
        log_std = tf.keras.layers.Dense(1)(attn)
        return tf.keras.Model(inputs, [mu, log_std])
    model = GMLE(model_fn())
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss=NLL())
    return model

def _build_deep_ensemble():
    def model_fn():
        inputs = tf.keras.Input(shape=(SEQ_LEN, 1))
        attn = NAC(
            d_model=32, num_heads=4, topk=8, mode='exact',
            sparsity=0.5, activation='sigmoid', return_sequences=False
        )(inputs)
        mu = tf.keras.layers.Dense(1)(attn)
        log_std = tf.keras.layers.Dense(1)(attn)
        return tf.keras.Model(inputs, [mu, log_std])

    model = DeepEnsemble(model_fn=model_fn(), num_models=20)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss=NLL())
    return model


def _build_mc_dropout():
    def model_fn(dropout_rate=0.1):
        inputs = tf.keras.Input(shape=(SEQ_LEN, 1))
        attn = NAC(
            d_model=32, num_heads=4, topk=8, mode='exact',
            sparsity=0.5, activation='sigmoid', return_sequences=False
        )(inputs)
        drop = tf.keras.layers.Dropout(dropout_rate)(attn, training=True)
        mu = tf.keras.layers.Dense(1)(drop)
        log_std = tf.keras.layers.Dense(1)(drop)
        return tf.keras.Model(inputs, [mu, log_std])

    model = MC_dropout(model_fn=model_fn(), mc_samples=20)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss=NLL())
    return model


def _build_evidential():
    def model_fn():
        inputs = tf.keras.Input(shape=(SEQ_LEN, 1))
        attn = NAC(
            d_model=32, num_heads=4, topk=8, mode='exact',
            sparsity=0.5, activation='sigmoid', return_sequences=False
        )(inputs)
        outputs = DenseNormalGamma(2)(attn)
        return tf.keras.Model(inputs, outputs)

    model = Evidential(model_fn=model_fn(), lambda_reg=0.5)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss=EvidentialRegression)
    return model

def _build_sde_net():
    model = SDE_Net(input_dim=SEQ_LEN, output_dim=1, hidden_dim=256, layer_depth=20)
    model.compile(
        optimizer_drift = tf.keras.optimizers.AdamW(0.0001),
        optimizer_fc = tf.keras.optimizers.AdamW(0.0001),
        optimizer_dsl = tf.keras.optimizers.AdamW(0.0001),
        optimizer_diffusion = tf.keras.optimizers.AdamW(0.001),
        loss = NLL()
    )
    return model




# Dummy input 
dummy_input = np.random.randn(BATCH_SIZE, SEQ_LEN, 1).astype(np.float32)


# RUNTIME FUNCTION
def measure_runtime(model, num_runs=10):
    runtimes = []
    _ = model(dummy_input)
    gpu_available = len(tf.config.list_physical_devices("GPU")) > 0
    if gpu_available:
        tf.config.experimental.reset_memory_stats("GPU:0")
    for _ in range(num_runs):
        start = time.time()
        _ = model(dummy_input, training=False)
        end = time.time()
        runtimes.append(end - start)
    runtimes = np.array(runtimes)

    mean_rt = runtimes.mean()
    std_rt = runtimes.std()
    throughput = 1.0 / mean_rt

    if gpu_available:
        print("Using GPU")
        mem_info = tf.config.experimental.get_memory_info("GPU:0")
        mem_usage = mem_info["peak"] / (1024 ** 2)

    else:
        print("Using CPU")
        import psutil
        process = psutil.Process(os.getpid())
        mem_usage = process.memory_info().rss / (1024 ** 2)

    del model
    time.sleep(2)
    return mean_rt, std_rt, throughput, mem_usage


# MODEL LIST
model_list = [
    "NSAC",
    "Gaussian-MLE",
    "Deep-Ensemble",
    "MC-Dropout",
    "Evidential",
    "SDE-Net"
]


# MAIN BENCHMARK LOOP
results = []
for name in model_list:
    print(f"\nBenchmarking {name}...")
    if name == "NSAC":
        model = _build_nsac()
    elif name == "Gaussian-MLE":
        model = _build_mle()
    elif name == "Deep-Ensemble":
        model = _build_deep_ensemble()
    elif name == "MC-Dropout":
        model = _build_mc_dropout()
    elif name == "Evidential":
        model = _build_evidential()
    elif name == "SDE-Net":
        model = _build_sde_net()

    mean_rt, std_rt, throughput, mem_usage = measure_runtime(model, NUM_RUNS)

    results.append({
        "Model": name,
        "Mean Runtime (s)": round(mean_rt, 6),
        "Std Dev (s)": round(std_rt, 6),
        "Throughput (seq/s)": round(throughput, 2),
        "Peak Memory (MB)": round(mem_usage, 2),
    })


# SAVE RESULTS
df = pd.DataFrame(results)

print("\n=== Runtime Benchmark Results ===")
print(df.to_string(index=False))

df.to_csv(f"{STAT_DIR}/run-time.csv", index=False)