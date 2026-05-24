import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pickle
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

from keras._tf_keras.keras.layers import (
    Input, Dense, Reshape, Conv1D, Flatten, MaxPool1D,
    BatchNormalization, GlobalAveragePooling1D,
    GlobalMaxPooling1D, Dropout, TimeDistributed,
    Conv2D, MaxPool2D
)
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam, AdamW
from keras._tf_keras.keras.callbacks import (
    ReduceLROnPlateau, ModelCheckpoint
)

from neuronal_attention_circuit import NAC
from neuronal_stochastic_attention_circuit import NSAC
from ouwrap import OUWrap
from losses import NSACLoss

from ouwrap import OUWrap
from losses import NSACLoss, NLL
from neuronal_attention_circuit import NAC
from neuronal_stochastic_attention_circuit import NSAC
from metrics import compute_crps, compute_ece, compute_auroc



from baselines import DeepEnsemble, MC_dropout, Evidential, GMLE, SDE_Net
from evidential_deep_learning.layers import DenseNormalGamma
from evidential_deep_learning.losses import EvidentialRegression



MODEL_NAME = 'CarRacing'
WEIGHTS_DIR = 'model_weights2'
os.makedirs(WEIGHTS_DIR, exist_ok=True)


with open('/home/ustc15/research/idea10/CarRacing/data.pkl','rb') as f:
    data = pickle.load(f)

X = np.array(data["observations"])
y = np.array(data["actions"])
y = np.squeeze(y, axis=1)
num_outputs = y.shape[-1]
X = np.transpose(X, (0, 2, 3, 4, 1))         # (N, 84, 84, 4)
X = X.astype("float32") / 255.0
print("Fixed X shape:", X.shape)
num_classes = len(np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(10000)
    .batch(64)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .batch(64)
)



# Model building functions
def _build_nsac():
    def model_fn(input_shape=(None, 84, 84, 1)):
        inp = Input(shape=input_shape)

        x = TimeDistributed(Conv2D(32,(5,5),activation='relu',strides=3))(inp)
        x = TimeDistributed(MaxPool2D(2,2))(x)
        x = TimeDistributed(Conv2D(48,(3,3),activation='relu',strides=2))(x)
        x = TimeDistributed(MaxPool2D(2,2))(x)
        x = TimeDistributed(Conv2D(64,(2,2),activation='relu'))(x)
        x = TimeDistributed(MaxPool2D(2,2))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Flatten())(x)
        mu,log_std = OUWrap(
            NAC(d_model=64,num_heads=16, sparsity=0.7, topk=8),
            bn_mean=0.0,
            bn_std=0.0001,
            output_dim=3,
            activation='tanh',
            return_sequences=False
            )(x)
        return Model(inp, [mu,log_std])
    model = NSAC(stochastic_model_fn=model_fn(),mc_samples=3, ood_mean=0.0, ood_std=0.0001)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss=NSACLoss())
    return model

def _build_mle():
    def model_fn(input_shape=(None, 84, 84, 1)):
        inp = Input(shape=input_shape)
        x = TimeDistributed(Conv2D(32,(5,5),activation='relu',strides=3))(inp)
        x = TimeDistributed(MaxPool2D(2,2))(x)
        x = TimeDistributed(Conv2D(48,(3,3),activation='relu',strides=2))(x)
        x = TimeDistributed(MaxPool2D(2,2))(x)
        x = TimeDistributed(Conv2D(64,(2,2),activation='relu'))(x)
        x = TimeDistributed(MaxPool2D(2,2))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Flatten())(x)
        x = NAC(
            d_model=64,
            num_heads=16,
            mode='exact',
            sparsity=0.5,
            topk=16,
            activation='relu',
            return_sequences=False
        )(x)
        mu = tf.keras.layers.Dense(3)(x)
        log_std = tf.keras.layers.Dense(3)(x)
        return tf.keras.Model(inp, [mu,log_std])
    model = GMLE(model_fn())
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss=NLL())
    return model

def _build_deep_ensemble():
    def model_fn(input_shape=(None, 84, 84, 1)):
        inp = Input(shape=input_shape)
        x = TimeDistributed(Conv2D(32,(5,5),activation='relu',strides=3))(inp)
        x = TimeDistributed(MaxPool2D(2,2))(x)
        x = TimeDistributed(Conv2D(48,(3,3),activation='relu',strides=2))(x)
        x = TimeDistributed(MaxPool2D(2,2))(x)
        x = TimeDistributed(Conv2D(64,(2,2),activation='relu'))(x)
        x = TimeDistributed(MaxPool2D(2,2))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Flatten())(x)
        x = NAC(
            d_model=64,
            num_heads=16,
            mode='exact',
            sparsity=0.5,
            topk=16,
            activation='relu',
            return_sequences=False
        )(x)
        mu = tf.keras.layers.Dense(3)(x)
        log_std = tf.keras.layers.Dense(3)(x)
        return tf.keras.Model(inp, [mu,log_std])
    model = DeepEnsemble(model_fn=model_fn(), num_models=3)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss=NLL())
    return model

def _build_mc_dropout():
    def model_fn(input_shape=(None, 84, 84, 1),dropout_rate=0.1):
        inp = Input(shape=input_shape)
        x = TimeDistributed(Conv2D(32,(5,5),activation='relu',strides=3))(inp)
        x = TimeDistributed(MaxPool2D(2,2))(x)
        x = TimeDistributed(Conv2D(48,(3,3),activation='relu',strides=2))(x)
        x = TimeDistributed(MaxPool2D(2,2))(x)
        x = TimeDistributed(Conv2D(64,(2,2),activation='relu'))(x)
        x = TimeDistributed(MaxPool2D(2,2))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Flatten())(x)
        x = NAC(
            d_model=64,
            num_heads=16,
            mode='exact',
            sparsity=0.5,
            topk=16,
            activation='relu',
            return_sequences=False
        )(x)
        mu = tf.keras.layers.Dense(3)(x)
        log_std = tf.keras.layers.Dense(3)(x)
        return tf.keras.Model(inp, [mu,log_std])
    model = MC_dropout(model_fn=model_fn(),mc_samples=5)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss=NLL())
    return model

def _build_evidential():
    def model_fn(input_shape=(None, 84, 84, 1)):
        inp = Input(shape=input_shape)
        x = TimeDistributed(Conv2D(32,(5,5),activation='relu',strides=3))(inp)
        x = TimeDistributed(MaxPool2D(2,2))(x)
        x = TimeDistributed(Conv2D(48,(3,3),activation='relu',strides=2))(x)
        x = TimeDistributed(MaxPool2D(2,2))(x)
        x = TimeDistributed(Conv2D(64,(2,2),activation='relu'))(x)
        x = TimeDistributed(MaxPool2D(2,2))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Flatten())(x)
        x = NAC(
            d_model=64,
            num_heads=16,
            mode='exact',
            sparsity=0.5,
            topk=16,
            activation='relu',
            return_sequences=False
        )(x)
        output = DenseNormalGamma(3)(x)
        return tf.keras.Model(inp, output)
    model = Evidential(model_fn(),lambda_reg=0.5)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss=EvidentialRegression)
    return model

class CNN_SDE_Wrapper(tf.keras.Model):
    def __init__(self, sde_net: SDE_Net):
        super().__init__()
        self.cnn = self._build_cnn()
        self.sde_net = sde_net
        self.optimizer_cnn = tf.keras.optimizers.AdamW(1e-3)
        self.mc_samples = 10

    def _build_cnn(self):
        return tf.keras.Sequential([
            TimeDistributed(Conv2D(32, (5,5), activation='relu', strides=3)),
            TimeDistributed(MaxPool2D(2, 2)),
            TimeDistributed(Conv2D(48, (3,3), activation='relu', strides=2)),
            TimeDistributed(MaxPool2D(2, 2)),
            TimeDistributed(Conv2D(64, (2,2), activation='relu')),
            TimeDistributed(MaxPool2D(2, 2)),
            TimeDistributed(BatchNormalization()),
            TimeDistributed(Flatten()),
        ])

    def _cnn_to_sde(self, x, training_diffusion=False):
        features = self.cnn(x)                      
        shape    = tf.shape(features)
        flat     = tf.reshape(features, (shape[0] * shape[1], features.shape[-1]))
        return self.sde_net(flat, training_diffusion=training_diffusion)

    def call(self, inputs, training_diffusion=False):
        return self._cnn_to_sde(inputs, training_diffusion)
    
    def compile(self, optimizer_drift, optimizer_fc, optimizer_dsl, optimizer_diffusion, loss, **kwargs):
        super().compile(**kwargs)  
        self.sde_net.compile(
            optimizer_drift=optimizer_drift,
            optimizer_fc=optimizer_fc,
            optimizer_dsl=optimizer_dsl,
            optimizer_diffusion=optimizer_diffusion,
            loss=loss,
        )

    def train_step(self, data):
        x, y = data
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        with tf.GradientTape() as tape:
            features  = self.cnn(x, training=True)   
            last_feat = features[:, -1, :]
            mean, sigma = self.sde_net(last_feat, training_diffusion=False)
            nll = self.sde_net.loss_fn(y, [mean, sigma])

        cnn_grad = [tf.clip_by_norm(g, 100)
                    for g in tape.gradient(nll, self.cnn.trainable_variables)]
        self.optimizer_cnn.apply_gradients(zip(cnn_grad, self.cnn.trainable_variables))
        last_feat_detached = tf.stop_gradient(features[:, -1, :])
        sde_metrics = self.sde_net.train_step((last_feat_detached, y))
        return sde_metrics

    def test_step(self, data):
        x, y      = data
        features  = self.cnn(tf.cast(x, tf.float32), training=False)
        last_feat = features[:, -1, :]
        return self.sde_net.test_step((last_feat, tf.cast(y, tf.float32)))
    
    def calculate_uncertainty(self, X):
        X = tf.cast(X, tf.float32)
        means, variances = [], []

        for _ in range(self.mc_samples):
            mean, sigma = self.call(X, training_diffusion=False)
            means.append(mean.numpy())
            variances.append((sigma ** 2).numpy())

        means = np.stack(means, axis=0)         # [T, N, 1]
        variances = np.stack(variances, axis=0)         # [T, N, 1]

        mean_pred = np.mean(means, axis=0)
        epistemic_unc = np.var(means, axis=0)
        aleatoric_unc = np.mean(variances, axis=0)
        total_uncertainty = epistemic_unc + aleatoric_unc

        return mean_pred, aleatoric_unc, epistemic_unc, total_uncertainty

    def predict_with_uncertainty(self, X):
        mean_pred, aleatoric, epistemic, _ = self.calculate_uncertainty(X)
        return mean_pred, aleatoric, epistemic

    def comprehensive_evaluate(self, X_test, y_test):
        X_test = tf.cast(X_test, tf.float32)
        y_test = tf.cast(y_test, tf.float32)

        mean_pred, aleatoric, epistemic, total_uncertainty = self.calculate_uncertainty(X_test)
        seq_len = X_test.shape[1]
        mean_pred_last = mean_pred[seq_len-1::seq_len]  # shape [batch, output_dim]

        variance = tf.clip_by_value(total_uncertainty, 1e-6, 1e6)
        variance = variance[seq_len-1::seq_len]  # shape [batch, output_dim]

        results = super().evaluate(
            X_test,
            y_test,
            return_dict=True,
            verbose=0
        )
        nll = float(results["loss"])
        mse = tf.reduce_mean(tf.square(y_test - mean_pred_last))
        predictive_std = tf.sqrt(variance)

        return (mean_pred_last, predictive_std.numpy(), nll, mse.numpy())

    def get_config(self):
        config = super().get_config()
        config.update({
            "sde_net": self.sde_net.get_config(),
        })
        return config
    def from_config(cls, config):
        model = tf.keras.utils.deserialize_keras_object(config.pop("model"))
        return cls(model_fn=model, **config)
    

def _build_sde_net():
    sde = SDE_Net(input_dim=64, output_dim=3, hidden_dim=32, layer_depth=5)
    model = CNN_SDE_Wrapper(sde)

    model.compile(
        optimizer_drift=tf.keras.optimizers.AdamW(1e-3),
        optimizer_fc=tf.keras.optimizers.AdamW(1e-3),
        optimizer_dsl=tf.keras.optimizers.AdamW(1e-3),
        optimizer_diffusion=tf.keras.optimizers.AdamW(1e-3),
        loss=NLL(),
    )
    dummy_input = tf.zeros((1, 1, 84, 84, 1))  
    _ = model(dummy_input, training_diffusion=False)
    return model


rl = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-10,mode='min')

model_list = [
    # "NSAC",
    # "Gaussian-MLE",
    # "Deep-Ensemble",
    # "MC-Dropout",
    # "Evidential",
    "SDE-Net",
]

for name in model_list:
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

    model.fit(train_ds, validation_data=test_ds, epochs=30, callbacks=[rl])

    model.save(f"{WEIGHTS_DIR}/{MODEL_NAME}_{name}.keras")

    # =========================
    # Evaluation
    # =========================

    model.load_weights(f"{WEIGHTS_DIR}/{MODEL_NAME}_{name}.keras")

    loss = model.evaluate(test_ds)
