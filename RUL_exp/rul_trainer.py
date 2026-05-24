# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
import pandas as pd
from utils.preprocess import process_features
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamW
from neuronal_stochastic_attention_circuit import NSAC
from neuronal_attention_circuit import NAC
from ouwrap import OUWrap
from losses import NSACLoss

MODEL_NAME = 'NSAC_RUL'
\
FEATURE_DIR = 'tf_features_xjtu'
WEIGHTS_DIR = 'model_weights'
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


#build BO tuned model
def model_fn(input_shape=(16,)):
    inp = layers.Input(input_shape)
    x = layers.Reshape((1, input_shape[0]))(inp)
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    x = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    
    mean, std = OUWrap(
        NAC(d_model=100, num_heads=20, topk=8, sparsity=0.7),
        output_dim=1,
        activation='relu',
        return_attention=False,
        return_sequences=False
    )(x)
    x = layers.Dense(64, activation='relu')(mean)
    mean = layers.Dense(1, activation='linear')(x)
    return Model(inputs=inp, outputs=[mean, std])

model = NSAC(model_fn(), ood_std=5.0)
model.summary()
model.compile(AdamW(learning_rate=1e-3),loss=NSACLoss(lambda_reg=0.5))


callbacks = [ 
    # tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-15),
    tf.keras.callbacks.ModelCheckpoint(f"{WEIGHTS_DIR}/{MODEL_NAME}.keras", save_best_only=True, monitor='loss')
]


history = model.fit(X,y,epochs=100, validation_split=0.1, batch_size=64, callbacks=callbacks, verbose=1)
model.evaluate(X,y)
model.save(f"{WEIGHTS_DIR}/{MODEL_NAME}.keras")