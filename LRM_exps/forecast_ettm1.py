import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from baselines import GMLE, DeepEnsemble, Evidential, MC_dropout, SDE_Net
from evidential_deep_learning.layers import DenseNormalGamma
from evidential_deep_learning.losses import EvidentialRegression

from neuronal_attention_circuit import NAC
from ouwrap import OUWrap
from losses import NSACLoss, NLL
from neuronal_stochastic_attention_circuit import NSAC


# config
LOOKBACK = 48      
HORIZON = 24
BATCH_SIZE = 64
EPOCHS = 50
TARGET_COL = 'OT'  # Oil Temperature

WEIGHTS_DIR = 'forecast_weights'
BASE_MODEL_NAME = 'ETTm1_forecast'
os.makedirs(WEIGHTS_DIR, exist_ok=True)


# data preprocess
def prepare_ettm1_data(file_path):
    df = pd.read_csv(file_path)
    df = df.iloc[:25000]
    features = df.drop(columns=['date'])
    n_features = features.shape[1]
    n = len(features)
    train_df = features[:int(n*0.8)]
    val_df = features[int(n*0.8):int(n*0.9)]
    test_df = features[int(n*0.9):]

    scaler = MinMaxScaler()
    scaler.fit(train_df)

    train_s = scaler.transform(train_df)
    val_s = scaler.transform(val_df)
    test_s = scaler.transform(test_df)

    def create_windows(data):
        X, y = [], []
        for i in range(len(data) - LOOKBACK - HORIZON):
            X.append(data[i:i+LOOKBACK, :])
            y.append(data[i+LOOKBACK:i+LOOKBACK+HORIZON, -1])
        return np.array(X), np.array(y)

    X_train, y_train = create_windows(train_s)
    X_val, y_val = create_windows(val_s)
    X_test, y_test = create_windows(test_s)
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler, n_features

X_train, y_train, X_val, y_val, X_test, y_test, scaler, n_feats= prepare_ettm1_data('ETTm1.csv')


# all models
def _build_nsac():
    def model_fn():
        inputs = tf.keras.Input(shape=(LOOKBACK, n_feats))
        mean, std = OUWrap(
            NAC(d_model=64, num_heads=4, topk=32),
            output_dim=HORIZON,
            activation="relu"
        )(inputs)
        return tf.keras.Model(inputs, [mean, std])
    model = NSAC(stochastic_model_fn=model_fn(), ood_std=5.0, mc_samples=5)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4),loss=NSACLoss())
    return model

def _build_mle():
    def model_fn():
        inputs = tf.keras.Input(shape=(LOOKBACK, n_feats))
        attn = NAC(d_model=64, num_heads=4, topk=32, activation='relu')(inputs)
        mu = tf.keras.layers.Dense(HORIZON)(attn)
        log_std = tf.keras.layers.Dense(HORIZON)(attn)
        return tf.keras.Model(inputs, [mu, log_std])
    model = GMLE(model_fn())
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss=NLL())
    return model

def _build_deep_ensemble():
    def model_fn():
        inputs = tf.keras.Input(shape=(LOOKBACK, n_feats))
        attn = NAC(d_model=64, num_heads=4, topk=32, activation='relu')(inputs)
        mu = tf.keras.layers.Dense(HORIZON)(attn)
        log_std = tf.keras.layers.Dense(HORIZON)(attn)
        return tf.keras.Model(inputs, [mu,log_std])
    model = DeepEnsemble(model_fn=model_fn(),num_models=3)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss=NLL())
    return model


def _build_mc_dropout():
    def model_fn(dropout_rate=0.1):
        inputs = tf.keras.Input(shape=(LOOKBACK, n_feats))
        attn = NAC(d_model=64, num_heads=4, topk=32, activation='relu')(inputs)
        drop = tf.keras.layers.Dropout(dropout_rate)(attn, training=True)
        mu = tf.keras.layers.Dense(HORIZON)(drop)
        log_std = tf.keras.layers.Dense(HORIZON)(drop)
        return tf.keras.Model(inputs, [mu, log_std])
    model = MC_dropout(model_fn=model_fn(),mc_samples=5)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss=NLL())
    return model

def _build_evidential():
    def model_fn():
        inputs = tf.keras.Input(shape=(LOOKBACK, n_feats))
        attn = NAC(d_model=64, num_heads=4, topk=32, activation='relu')(inputs)
        outputs = DenseNormalGamma(HORIZON)(attn)
        return tf.keras.Model(inputs, outputs)

    model = Evidential(model_fn=model_fn(), lambda_reg=0.5)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-4), loss=EvidentialRegression)
    return model

def _build_sde_net():
    model = SDE_Net(input_dim=336, output_dim=HORIZON, hidden_dim=64, layer_depth=6)
    model.compile(
        optimizer_drift = tf.keras.optimizers.AdamW(0.0001),
        optimizer_fc = tf.keras.optimizers.AdamW(0.0001),
        optimizer_dsl = tf.keras.optimizers.AdamW(0.0001),
        optimizer_diffusion = tf.keras.optimizers.AdamW(0.001),
        loss = NLL()
    )
    return model


MODEL_LIST = [ 
              "NSAC",
            #   "Gaussian-MLE",
            #   "Deep-Ensemble",
            #   "MC-Dropout",
            #   "Evidential",
            #   "SDE-Net",
              ]
results = {}

for name in MODEL_LIST:
    print(f"\n===== Training {name} =====")

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

    # callbacks = [
    #     # EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    #     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    #     # ModelCheckpoint(f"{WEIGHTS_DIR}/{BASE_MODEL_NAME}_{name}.keras",monitor='val_loss',save_best_only=True)
    # ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        # callbacks=callbacks,
        verbose=1
    )

    model.save(f"{WEIGHTS_DIR}/{BASE_MODEL_NAME}_{name}.keras")

    model.load_weights(f"{WEIGHTS_DIR}/{BASE_MODEL_NAME}_{name}.keras")
    nll, _ = model.evaluate(X_test, y_test)

    print(f"{name} -> NLL: {nll:.4f}")

#     results[name] = {"NLL": nll}


# print("\n===== FINAL RESULTS =====")
# for k, v in results.items():
#     print(f"{k}: MSE={v['MSE']:.4f}, MAE={v['MAE']:.4f}")