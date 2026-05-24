import openml
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from metrics import compute_crps, compute_ece, compute_auroc

from ouwrap import OUWrap
from losses import NSACLoss, NLL
from neuronal_attention_circuit import NAC
from neuronal_stochastic_attention_circuit import NSAC

from baselines import GMLE, DeepEnsemble, MC_dropout, Evidential, SDE_Net
from evidential_deep_learning.layers import DenseNormalGamma
from evidential_deep_learning.losses import EvidentialRegression


MODEL_NAME = 'kin8nm'
WEIGHTS_DIR = 'model_weights'
STAT_DIR = 'statistics'


dataset = openml.datasets.get_dataset("kin8nm")
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

X = X.values.astype(np.float32)
y = y.values.astype(np.float32)
y = y / np.max(y)


#models
def _build_nsac():
    def model_fn():
        inputs = tf.keras.layers.Input(shape=(1, X_train.shape[2]))
        mean, std = OUWrap(
            NAC(d_model=16, num_heads=4, topk=8, sparsity=0.8),
            output_dim=1,
            activation="sigmoid"
        )(inputs)
        return tf.keras.Model(inputs, [mean, std])

    model = NSAC(stochastic_model_fn=model_fn(), mc_samples=5, ood_std=5.0)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss=NSACLoss())
    return model


def _build_mle():
    def model_fn():
        inputs = tf.keras.layers.Input(shape=(1, X_train.shape[2]))
        attn = NAC(
            d_model=16, num_heads=4, topk=8,
            mode='exact', sparsity=0.5,
            activation='sigmoid', return_sequences=False
        )(inputs)
        mu = tf.keras.layers.Dense(1)(attn)
        log_std = tf.keras.layers.Dense(1)(attn)
        return tf.keras.Model(inputs, [mu, log_std])

    model = GMLE(model_fn())
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss=NLL())
    return model


def _build_deep_ensemble():
    def model_fn():
        inputs = tf.keras.layers.Input(shape=(1, X_train.shape[2]))
        attn = NAC(
            d_model=16, num_heads=4, topk=8,
            mode='exact', sparsity=0.5,
            activation='sigmoid', return_sequences=False
        )(inputs)
        mu = tf.keras.layers.Dense(1)(attn)
        log_std = tf.keras.layers.Dense(1)(attn)
        return tf.keras.Model(inputs, [mu, log_std])

    model = DeepEnsemble(model_fn=model_fn(), num_models=3)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss=NLL())
    return model


def _build_mc_dropout():
    def model_fn(dropout_rate=0.1):
        inputs = tf.keras.layers.Input(shape=(1, X_train.shape[2]))
        attn = NAC(
            d_model=16, num_heads=4, topk=8,
            mode='exact', sparsity=0.5,
            activation='sigmoid', return_sequences=False
        )(inputs)
        drop = tf.keras.layers.Dropout(dropout_rate)(attn, training=True)
        mu = tf.keras.layers.Dense(1)(attn)
        log_std = tf.keras.layers.Dense(1)(attn)
        return tf.keras.Model(inputs, [mu, log_std])

    model = MC_dropout(model_fn=model_fn(),mc_samples=5)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss=NLL())
    return model


def _build_evidential():
    def model_fn():
        inputs = tf.keras.layers.Input(shape=(1, X_train.shape[2]))
        attn = NAC(
            d_model=16, num_heads=4, topk=8,
            mode='exact', sparsity=0.5,
            activation='sigmoid', return_sequences=False
        )(inputs)
        outputs = DenseNormalGamma(1)(attn)
        return tf.keras.Model(inputs, outputs)

    model = Evidential(model_fn=model_fn(), lambda_reg=0.5)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss=EvidentialRegression)
    return model


def _build_sde_net():
    model = SDE_Net(input_dim=X_train.shape[2], output_dim=1, hidden_dim=64, layer_depth=6)
    model.compile(
        optimizer_drift=tf.keras.optimizers.AdamW(0.0001),
        optimizer_fc=tf.keras.optimizers.AdamW(0.0001),
        optimizer_dsl=tf.keras.optimizers.AdamW(0.0001),
        optimizer_diffusion=tf.keras.optimizers.AdamW(0.001),
        loss=NLL()
    )
    return model


model_list = [
    "NSAC",
    "Gaussian-MLE",
    "Deep-Ensemble",
    "MC-Dropout",
    "Evidential",
    "SDE-Net"
]


NUM_RUNS = 5

results_rows = []
model_counter = 1
TEST_NAME = "kin8nm"

for name in model_list:

    print(f"\n================ {name} ================")

    mse_list, nll_list, crps_list, ece_list, auroc_list = [], [], [], [], []

    for run in range(NUM_RUNS):

        seed = 42 + run
        np.random.seed(seed)
        tf.random.set_seed(seed)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = X_train.reshape(-1, 1, X_train.shape[1])
        X_test = X_test.reshape(-1, 1, X_test.shape[1])

        print(f"\nRun {run+1}/{NUM_RUNS}")

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

        # model.fit(
        #     X_train,
        #     y_train,
        #     epochs=100,
        #     batch_size=16,
        #     validation_split=0.1,
        #     verbose=1
        # )
        # model.save(f'{WEIGHTS_DIR}/{name}_{MODEL_NAME}_weights_{run}.keras')

        model.load_weights(f'{WEIGHTS_DIR}/{name}_{MODEL_NAME}_weights_{run}.keras')

        mu, std, nll, mse = model.comprehensive_evaluate(X_test, y_test)

        mu, std = np.array(mu), np.array(std)

        crps = compute_crps(y_test, mu, std)
        ece = compute_ece(y_test, mu, std)
        auroc = compute_auroc(y_test, mu, std)

        mse_list.append(mse)
        nll_list.append(nll)
        crps_list.append(crps)
        ece_list.append(ece)
        auroc_list.append(auroc)

    print("\nFINAL RESULTS (mean ± std)")

    mse_mean, mse_std = np.mean(mse_list), np.std(mse_list)
    nll_mean, nll_std = np.mean(nll_list), np.std(nll_list)
    crps_mean, crps_std = np.mean(crps_list), np.std(crps_list)
    ece_mean, ece_std = np.mean(ece_list), np.std(ece_list)
    auroc_mean, auroc_std = np.mean(auroc_list), np.std(auroc_list)

    print("MSE:", mse_mean, "±", mse_std)
    print("NLL:", nll_mean, "±", nll_std)
    print("CRPS:", crps_mean, "±", crps_std)
    print("ECE:", ece_mean, "±", ece_std)
    print("AUROC:", auroc_mean, "±", auroc_std)

    # ===== Save results =====
    model_id = f"Model_{model_counter}_{name}"

    metrics = [
        ("MSE", mse_mean, mse_std),
        ("NLL", nll_mean, nll_std),
        ("CRPS", crps_mean, crps_std),
        ("ECE", ece_mean, ece_std),
        ("AUROC", auroc_mean, auroc_std),
    ]

    for metric_name, mean_val, std_val in metrics:
        results_rows.append({
            "Test_name": TEST_NAME,
            "Model": model_id,
            "Metric": metric_name,
            "Mean ± Std": f"{mean_val:.6f} ± {std_val:.6f}"
        })

    model_counter += 1

#csv export
df_results = pd.DataFrame(results_rows)
metric_order = ["MSE", "NLL", "CRPS", "ECE", "AUROC"]
df_results["Metric"] = pd.Categorical(df_results["Metric"], categories=metric_order, ordered=True)
df_results = df_results.sort_values(by=["Metric", "Model"])
df_results.to_csv(f"{STAT_DIR}/{MODEL_NAME}_results.csv", index=False)
print("\nCSV saved as kin8nm_results.csv")