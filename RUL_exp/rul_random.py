import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from metrics import compute_crps, compute_ece

from ouwrap import OUWrap
from losses import NSACLoss, NLL
from neuronal_attention_circuit import NAC
from neuronal_stochastic_attention_circuit import NSAC

from baselines import GMLE, DeepEnsemble, MC_dropout, Evidential, SDE_Net
from evidential_deep_learning.layers import DenseNormalGamma
from evidential_deep_learning.losses import EvidentialRegression

from utils.preprocess import process_features


# Load Dataset
MODEL_NAME = 'rul'
WEIGHTS_DIR = 'model_weights'
STAT_DIR = 'statistics'
FEATURE_DIR = 'tf_features_xjtu'
bearings = ['Bearing1_1', 'Bearing1_2', 'Bearing1_3']


# Load and preprocess data
dfs = [pd.read_csv(f'{FEATURE_DIR}/{bearing}_features_with_labels.csv') for bearing in bearings]

# Process features
horizontal_data = [np.array(df['Horizontal'].apply(eval).tolist()) for df in dfs]
X_h = np.vstack([process_features(data) for data in horizontal_data])
vertical_data = [np.array(df['Vertical'].apply(eval).tolist()) for df in dfs]
X_v = np.vstack([process_features(data) for data in vertical_data])
vibration_features = np.concatenate((X_h, X_v), axis=-1)
t_data = np.concatenate([df['Time'].values.reshape(-1, 1) for df in dfs], axis=0)
T_data = np.concatenate([np.full((df.shape[0], 1), 25 + 273.15) for df in dfs], axis=0)
# T_data = np.concatenate([df['Temperature'].values.reshape(-1, 1) for df in dfs], axis=0)
y = np.concatenate([df['Degradation'].values.reshape(-1, 1) for df in dfs], axis=0)
RPM = np.concatenate([df['RPM'].values.reshape(-1, 1) for df in dfs], axis=0)
Load = np.concatenate([df['Load'].values.reshape(-1, 1) for df in dfs], axis=0)

# Combine features and normalize
X = np.concatenate([vibration_features, t_data, T_data], axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

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

# Experiment Loop
NUM_RUNS = 5
for name in model_list:

    print(f"\n================ {name} ================")

    mse_list = []
    nll_list = []
    crps_list = []
    ece_list = []

    for run in range(NUM_RUNS):

        seed = 42 + run
        np.random.seed(seed)
        tf.random.set_seed(seed)

        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=seed)

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

        model.fit(
            X_train,
            y_train,
            epochs=20,
            batch_size=64,
            validation_split=0.1,
            verbose=0
        )
        model.save(f'{WEIGHTS_DIR}/{name}_{MODEL_NAME}_weights_{run}.keras')

        model.load_weights(f'{WEIGHTS_DIR}/{name}_{MODEL_NAME}_weights_{run}.keras')

        mu, std, nll, mse = model.comprehensive_evaluate(X_test, y_test)

        mu = np.array(mu)
        std = np.array(std)

        crps = compute_crps(y_test, mu, std)
        ece = compute_ece(y_test, mu, std)

        mse_list.append(mse)
        nll_list.append(nll)
        crps_list.append(crps)
        ece_list.append(ece)

        # print("MSE:", mse)
        # print("NLL:", nll)
        # print("CRPS:", crps)
        # print("ECE:", ece)

    print("\nFINAL RESULTS (mean ± std)")

    print("MSE:", np.mean(mse_list), "±", np.std(mse_list))
    print("NLL:", np.mean(nll_list), "±", np.std(nll_list))
    print("CRPS:", np.mean(crps_list), "±", np.std(crps_list))
    print("ECE:", np.mean(ece_list), "±", np.std(ece_list))