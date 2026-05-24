import numpy as np
import pandas as pd
import tensorflow as tf


from sklearn.preprocessing import StandardScaler
from metrics import compute_crps, compute_ece, compute_auroc, score

from ouwrap import OUWrap
from losses import NSACLoss, NLL
from neuronal_attention_circuit import NAC
from neuronal_stochastic_attention_circuit import NSAC

from baselines import DeepEnsemble, MC_dropout, Evidential, GMLE, SDE_Net
from evidential_deep_learning.layers import DenseNormalGamma
from evidential_deep_learning.losses import EvidentialRegression

from utils.preprocess import process_features


MODEL_NAME = 'rul'
WEIGHTS_DIR = 'model_weights'
STAT_DIR = 'statistics'

# Model Builders
def _build_nsac():
    def model_fn(input_shape=(16,)):
        inputs = tf.keras.layers.Input(input_shape)
        x = tf.keras.layers.Reshape((1, input_shape[0]))(inputs)
        mean, log_std = OUWrap(
            NAC(d_model=100, num_heads=20, topk=8, sparsity=0.7),
            output_dim=1,
            activation='relu',
            return_sequences=False
        )(x)
        return tf.keras.Model(inputs=inputs, outputs=[mean, log_std])
    model = NSAC(stochastic_model_fn=model_fn(), ood_std=5.0)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4),loss=NSACLoss())
    return model

def _build_mle():
    def model_fn(input_shape=(16,)):
        inputs = tf.keras.layers.Input(input_shape)
        x = tf.keras.layers.Reshape((1, input_shape[0]))(inputs)
        x = NAC(
            d_model=100, num_heads=20, topk=8, mode='exact',
            sparsity=0.8, activation='relu', return_sequences=False)(x)
        mean = tf.keras.layers.Dense(1, activation='linear')(x)
        log_std = tf.keras.layers.Dense(1, activation='linear')(x)
        return tf.keras.Model(inputs=inputs, outputs=[mean, log_std])
    model = GMLE(model_fn())
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss=NLL())
    return model

def _build_deep_ensemble():
    def model_fn(input_shape=(16,)):
        inp = tf.keras.layers.Input(input_shape)
        x = tf.keras.layers.Reshape((1, input_shape[0]))(inp)
        x = NAC(
            d_model=100, num_heads=20, topk=8, mode='exact',
            sparsity=0.8, activation='relu', return_sequences=False)(x)
        mu = tf.keras.layers.Dense(1)(x)
        log_std = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model(inp, [mu, log_std])
    model = DeepEnsemble(model_fn=model_fn())
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss=NLL())
    return model

def _build_mc_dropout():
    def model_fn(input_shape=(16,)):
        inp = tf.keras.layers.Input(input_shape)
        x = tf.keras.layers.Reshape((1, input_shape[0]))(inp)
        x = NAC(
            d_model=100, num_heads=20, topk=8, mode='exact',
            sparsity=0.8, activation='relu', return_sequences=False)(x)
        mu = tf.keras.layers.Dense(1)(x)
        log_std = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model(inp, [mu, log_std])
    model = MC_dropout(model_fn=model_fn())
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss=NLL())
    return model


def _build_evidential():
    def model_fn(input_shape=(16,)):
        inputs = tf.keras.layers.Input(input_shape)
        x = tf.keras.layers.Reshape((1, input_shape[0]))(inputs)
        x = NAC(
            d_model=100, num_heads=20, topk=8, mode='exact',
            sparsity=0.8, activation='relu', return_sequences=False)(x)
        outputs = DenseNormalGamma(1)(x)
        return tf.keras.Model(inputs, outputs)
    model = Evidential(model_fn=model_fn(), lambda_reg=0.5)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss=EvidentialRegression)
    return model

def _build_sde_net():
    model = SDE_Net(input_dim=X.shape[1], output_dim=1, hidden_dim=64, layer_depth=6)
    model.compile(
        optimizer_drift = tf.keras.optimizers.AdamW(0.0001),
        optimizer_fc = tf.keras.optimizers.AdamW(0.0001),
        optimizer_dsl = tf.keras.optimizers.AdamW(0.0001),
        optimizer_diffusion = tf.keras.optimizers.AdamW(0.001),
        loss = NLL()
    )
    return model


# Model List
model_list = [
    "NSAC",
    "Gaussian-MLE",
    "Deep-Ensemble",
    "MC-Dropout",
    "Evidential",
    "SDE-Net",
]

datasets = {
    "xjtu": {
        "label": "XJTU",
        "dir": "tf_features_xjtu",
        "bearings": ["Bearing1_1"],
        "suffix": "_features_with_labels.csv"
    },
    "pronostia": {
        "label": "PRONOSTIA",
        "dir": "tf_features_pronostia",
        "bearings": ["Bearing2_1"],
        "suffix": "_features.csv"
    },
    "hust": {
        "label": "HUST",
        "dir": "tf_features_hust",
        "bearings": ["Bearing1"],
        "suffix": "_features.csv"
    }
}

NUM_RUNS = 5
results_rows = []
model_counter = 1
TEST_NAME = "RUL"

for dataset_name, info in datasets.items():
    dataset_label = info["label"]
    print(f"\n================ DATASET: {dataset_label} ================")
    FEATURE_DIR = info["dir"]

    for bearing in info["bearings"]:
        print(f"\n----------- Bearing: {bearing} -----------")
        for name in model_list:
            print(f"\nModel: {name}")

            mse_list, nll_list, crps_list, ece_list, auroc_list, score_list = [], [], [], [], [], []

            for run in range(NUM_RUNS):

                seed = 42 + run
                np.random.seed(seed)
                tf.random.set_seed(seed)

                print(f"Run {run+1}/{NUM_RUNS}")

                file_path = f"{FEATURE_DIR}/{bearing}{info['suffix']}"
                df = pd.read_csv(file_path)
                horizontal = np.array(df['Horizontal'].apply(eval).tolist())
                X_h = process_features(horizontal)
                vertical = np.array(df['Vertical'].apply(eval).tolist())
                X_v = process_features(vertical)
                vibration_features = np.concatenate((X_h, X_v), axis=-1)
                t_data = df['Time'].values.reshape(-1,1)
                try:
                    T_data = df['Temperature'].values.reshape(-1,1)
                except:
                    T_data = np.full((df.shape[0],1), 25 + 273.15)
                y = df['Degradation'].values.reshape(-1,1)
                X = np.concatenate([vibration_features, t_data, T_data], axis=1)

                scaler = StandardScaler()
                X = scaler.fit_transform(X)

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

                model.load_weights(f'{WEIGHTS_DIR}/{name}_{MODEL_NAME}_weights_{run}.keras')

                mu, std, nll, mse = model.comprehensive_evaluate(X, y)

                mu = np.array(mu)
                std = np.array(std)

                crps = compute_crps(y, mu, std)
                ece = compute_ece(y, mu, std)
                auroc = compute_auroc(y, mu, std)
                score_val = score(y, mu)

                mse_list.append(mse)
                nll_list.append(nll)
                crps_list.append(crps)
                ece_list.append(ece)
                auroc_list.append(auroc)
                # score_list.append(score_val)

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
            # score_mean, score_std = np.mean(score_list), np.std(score_list)

            print("MSE:", mse_mean, "±", mse_std)
            print("NLL:", nll_mean, "±", nll_std)
            print("CRPS:", crps_mean, "±", crps_std)
            print("ECE:", ece_mean, "±", ece_std)
            print("AUROC:", auroc_mean, "±", auroc_std)
            # print("Score:", score_mean, "±", score_std)

            model_id = f"Model_{model_counter}_{name}"

            metrics = [
                ("MSE", mse_mean, mse_std),
                ("NLL", nll_mean, nll_std),
                ("CRPS", crps_mean, crps_std),
                ("ECE", ece_mean, ece_std),
                ("AUROC", auroc_mean, auroc_std),
                # ("Score", score_mean, score_std),
            ]

            for metric_name, mean_val, std_val in metrics:
                results_rows.append({
                    "Test_name": TEST_NAME,
                    "Model": model_id,
                    "Dataset": dataset_label,
                    "Metric": metric_name,
                    "Mean ± Std": f"{mean_val:.6f} ± {std_val:.6f}"
                })

            model_counter += 1


# EXPORT CSV
df_results = pd.DataFrame(results_rows)
metric_order = ["MSE", "NLL", "CRPS", "ECE", "AUROC", "Score"]
df_results["Metric"] = pd.Categorical(
    df_results["Metric"],
    categories=metric_order,
    ordered=True
)
df_results = df_results.sort_values(by=["Metric", "Model"])
df_results.to_csv(f"{STAT_DIR}/{MODEL_NAME}_results.csv", index=False)

print("\nCSV saved as rul_results_all_datasets.csv")