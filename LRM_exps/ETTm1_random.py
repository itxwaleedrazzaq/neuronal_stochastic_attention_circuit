import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from neuronal_attention_circuit import NAC
from ouwrap import OUWrap
from losses import NSACLoss, NLL
from neuronal_stochastic_attention_circuit import NSAC
from baselines import GMLE, DeepEnsemble, Evidential, MC_dropout, SDE_Net
from evidential_deep_learning.layers import DenseNormalGamma
from evidential_deep_learning.losses import EvidentialRegression
from metrics import compute_crps, compute_ece, compute_auroc

MODEL_NAME = 'ETTm1'
WEIGHTS_DIR = 'model_weights'
STAT_DIR = 'statistics'


# ETTm1 is sampled every 15 mins. 
# 48 steps = 12 hours. We predict 24 steps (6 hours) ahead.
LOOKBACK = 48      
HORIZON = 24
TARGET_COL = 'OT'  # Oil Temperature


def prepare_ettm1_data(file_path):
    df = pd.read_csv(file_path)
    df = df.iloc[:15000]
    features = df.drop(columns=['date'])
    n_features = features.shape[1]
    n = len(features)
    train_df = features[:int(n*0.9)]
    test_df = features[int(n*0.9):]

    scaler = MinMaxScaler()
    scaler.fit(train_df)
    train_s = scaler.transform(train_df)
    test_s = scaler.transform(test_df)

    def create_windows(data):
        X, y = [], []
        for i in range(len(data) - LOOKBACK - HORIZON):
            X.append(data[i:i+LOOKBACK, :])
            y.append(data[i+LOOKBACK:i+LOOKBACK+HORIZON, -1])
        return np.array(X), np.array(y)

    X_train, y_train = create_windows(train_s)
    X_test, y_test = create_windows(test_s)
    return X_train, y_train, X_test, y_test, scaler, n_features

X, y, X_test, y_test, scaler, n_feats = prepare_ettm1_data('ETTm1.csv')

def _build_nsac():
    def model_fn():
        inputs = tf.keras.layers.Input(shape=(LOOKBACK, n_feats))
        mean, std = OUWrap(
            NAC(d_model=64, num_heads=4, topk=32, sparsity=0.5),
            output_dim=HORIZON,
            activation="relu",
        )(inputs)
        return tf.keras.Model(inputs, [mean, std])

    model = NSAC(stochastic_model_fn=model_fn(), mc_samples=5, ood_std=5.0)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss=NSACLoss())
    return model


def _build_mle():
    def model_fn():
        inputs = tf.keras.layers.Input(shape=(LOOKBACK, n_feats))
        attn = NAC(
            d_model=64, num_heads=4, topk=8,
            mode='exact', sparsity=0.5,
            activation='sigmoid', return_sequences=False
        )(inputs)
        mu = tf.keras.layers.Dense(HORIZON)(attn)
        log_std = tf.keras.layers.Dense(HORIZON)(attn)
        return tf.keras.Model(inputs, [mu, log_std])

    model = GMLE(model_fn())
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss=NLL())
    return model


def _build_deep_ensemble():
    def model_fn():
        inputs = tf.keras.layers.Input(shape=(LOOKBACK, n_feats))
        attn = NAC(
            d_model=64, num_heads=4, topk=32, sparsity=0.5,
            activation='relu', return_sequences=False
        )(inputs)
        mu = tf.keras.layers.Dense(HORIZON)(attn)
        log_std = tf.keras.layers.Dense(HORIZON)(attn)
        return tf.keras.Model(inputs, [mu, log_std])

    model = DeepEnsemble(model_fn=model_fn(), num_models=3)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss=NLL())
    return model


def _build_mc_dropout():
    def model_fn(dropout_rate=0.1):
        inputs = tf.keras.layers.Input(shape=(LOOKBACK, n_feats))
        attn = NAC(
            d_model=64, num_heads=4, topk=32, sparsity=0.5,
            activation='relu', return_sequences=False
        )(inputs)
        drop = tf.keras.layers.Dropout(dropout_rate)(attn, training=True)
        mu = tf.keras.layers.Dense(HORIZON)(drop)
        log_std = tf.keras.layers.Dense(HORIZON)(drop)
        return tf.keras.Model(inputs, [mu, log_std])

    model = MC_dropout(model_fn=model_fn(), mc_samples=5)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss=NLL())
    return model


def _build_evidential():
    def model_fn():
        inputs = tf.keras.layers.Input(shape=(LOOKBACK, n_feats))
        attn = NAC(
            d_model=64, num_heads=4, topk=32, sparsity=0.5,
            activation='sigmoid', return_sequences=False
        )(inputs)
        outputs = DenseNormalGamma(HORIZON)(attn)
        return tf.keras.Model(inputs, outputs)

    model = Evidential(model_fn=model_fn(), lambda_reg=0.5)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss=EvidentialRegression)
    return model


def _build_sde_net():
    model = SDE_Net(input_dim=336, output_dim=HORIZON, hidden_dim=64, layer_depth=6)
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
TEST_NAME = "ETTm1"

for name in model_list:

    print(f"\n================ {name} ================")

    mse_list, nll_list, crps_list, ece_list, auroc_list = [], [], [], [], []

    for run in range(NUM_RUNS):

        seed = 42 + run
        np.random.seed(seed)
        tf.random.set_seed(seed)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=seed, shuffle=False)

        print(f"\nRun {run + 1}/{NUM_RUNS}")

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
        #     epochs=50,
        #     batch_size=64,
        #     validation_data=(X_val,y_val),
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

        # print("MSE:", mse)
        # print("NLL:", nll)
        # print("CRPS:", crps)
        # print("ECE:", ece)
        # print("AUROC:", auroc)

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


df_results = pd.DataFrame(results_rows)
metric_order = ["MSE", "NLL", "CRPS", "ECE", "AUROC"]
df_results["Metric"] = pd.Categorical(df_results["Metric"], categories=metric_order, ordered=True)
df_results = df_results.sort_values(by=["Metric", "Model"])
df_results.to_csv(f"{STAT_DIR}/{MODEL_NAME}_results.csv", index=False)
print("\nCSV saved as ETTm1_results.csv")