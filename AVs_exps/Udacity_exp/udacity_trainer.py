import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.layers import Input, Dense, Reshape, Conv2D, Lambda
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau
from keras._tf_keras.keras.regularizers import l2

from ouwrap import OUWrap
from neuronal_stochastic_attention_circuit import NSAC
from neuronal_attention_circuit import NAC
from losses import NSACLoss
from utils_udacity import INPUT_SHAPE, batch_generator



# Set random seed for reproducibility
np.random.seed(42)

# Configuration
MODEL_NAME = "Udacity_NSAC"
WEIGHTS_DIR = "model_weights"
DATA_DIR = "data/"
TEST_SIZE = 0.1
BATCH_SIZE = 40
EPOCHS = 15
SAMPLES_PER_EPOCH = 20000
LEARNING_RATE = 1e-4

# Callbacks
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=2,
    verbose=1,
    mode="min",
    min_lr=1e-10
)


# Data loading
data_df = pd.read_csv(os.path.join(DATA_DIR, "driving_log.csv"))
X = data_df[["center", "left", "right"]].values
y = data_df["steering"].values

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=42
)


def model_fn(input_shape):
        '''
    Model architecture based on NVIDIA's model modified with LAN.'
    '''
        inp = Input(shape=input_shape)

        x = Lambda(lambda x: x / 127.5 - 1.0)(inp)
        x = Conv2D(24, (5, 5), strides=(2, 2), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Conv2D(36, (5, 5), strides=(2, 2), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Conv2D(48, (5, 5), strides=(2, 2), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Conv2D(64, (3, 3), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Conv2D(64, (3, 3), activation="elu", kernel_regularizer=l2(0.001))(x)
        
        x = Reshape((-1, x.shape[1] * x.shape[2]))(x)
        mu, log_std = OUWrap(
                NAC(d_model=100, num_heads=20, sparsity=0.7, topk=16),
                output_dim=1,
                activation="elu", 
                )(x)
        return tf.keras.Model(inp, [mu, log_std])

model = NSAC(stochastic_model_fn=model_fn(INPUT_SHAPE),ood_std=10.0)
model.summary()

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=NSACLoss(),
)


# Training
model.fit(
    batch_generator(DATA_DIR, X_train, y_train, BATCH_SIZE, True),
    steps_per_epoch=SAMPLES_PER_EPOCH // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=batch_generator(DATA_DIR, X_valid, y_valid, BATCH_SIZE, False),
    validation_steps=len(X_valid) // BATCH_SIZE,
    callbacks=[reduce_lr],
    verbose=1,
)


# Save final model
model.save(f"{WEIGHTS_DIR}/{MODEL_NAME}.keras")

# Optional: Evaluate model
loss, mae = model.evaluate(
    batch_generator(DATA_DIR, X_valid, y_valid, BATCH_SIZE, False),
    steps=len(X_valid) // BATCH_SIZE,
    verbose=1
)
print(f"Validation Loss: {loss}, MAE: {mae}")
