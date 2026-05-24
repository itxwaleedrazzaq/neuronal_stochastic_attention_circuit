import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from neuronal_attention_circuit import NAC
from ouwrap import OUWrap
from losses import NSACLoss, NLL
from evidential_deep_learning.layers import DenseNormalGamma
from evidential_deep_learning.losses import EvidentialRegression
import pandas as pd
from keras._tf_keras.keras.layers import Input, Reshape, Conv2D, Lambda
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau
from keras._tf_keras.keras.regularizers import l2
from neuronal_stochastic_attention_circuit import NSAC
from utils_udacity import INPUT_SHAPE, batch_generator

from baselines import GMLE, DeepEnsemble, MC_dropout, Evidential, SDE_Net
from metrics import compute_crps, compute_ece, compute_auroc 


MODEL_NAME = "udacity"
WEIGHTS_DIR = "model_weights"
STAT_DIR = "statistics"

# Configuration
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


# Model creation methods 
def _build_nsac():
    def model_fn(input_shape):
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
    model = NSAC(stochastic_model_fn=model_fn(INPUT_SHAPE), mc_samples=5, ood_std=5.0)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss=NSACLoss())
    return model

def _build_mle():
    def model_fn(input_shape):
        inp = Input(shape=input_shape)
        x = Lambda(lambda x: x / 127.5 - 1.0)(inp)
        x = Conv2D(24, (5, 5), strides=(2, 2), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Conv2D(36, (5, 5), strides=(2, 2), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Conv2D(48, (5, 5), strides=(2, 2), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Conv2D(64, (3, 3), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Conv2D(64, (3, 3), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Reshape((-1, x.shape[1] * x.shape[2]))(x)
        x = NAC(d_model=100, 
            num_heads=20, 
            mode = "exact",
            activation="elu", 
            sparsity=0.7, 
            topk=15,
            return_sequences=False)(x)
        mu = tf.keras.layers.Dense(1)(x)
        log_std = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model(inp, [mu, log_std])
    model = GMLE(model_fn(INPUT_SHAPE))
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss=NLL())
    return model

def _build_deep_ensemble():
    def model_fn(input_shape):
        inp = Input(shape=input_shape)
        x = Lambda(lambda x: x / 127.5 - 1.0)(inp)
        x = Conv2D(24, (5, 5), strides=(2, 2), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Conv2D(36, (5, 5), strides=(2, 2), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Conv2D(48, (5, 5), strides=(2, 2), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Conv2D(64, (3, 3), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Conv2D(64, (3, 3), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Reshape((-1, x.shape[1] * x.shape[2]))(x)
        x = NAC(d_model=100, 
            num_heads=20, 
            mode = "exact",
            activation="elu", 
            sparsity=0.7, 
            topk=15,
            return_sequences=False)(x)
        mu = tf.keras.layers.Dense(1)(x)
        log_std = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model(inp, [mu, log_std])
    model = DeepEnsemble(model_fn=model_fn(INPUT_SHAPE), num_models=3)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss=NLL())
    return model

def _build_mc_dropout():
    def model_fn(input_shape,dropout_rate=0.01):
        inp = Input(shape=input_shape)
        x = Lambda(lambda x: x / 127.5 - 1.0)(inp)
        x = Conv2D(24, (5, 5), strides=(2, 2), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Conv2D(36, (5, 5), strides=(2, 2), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Conv2D(48, (5, 5), strides=(2, 2), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Conv2D(64, (3, 3), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Conv2D(64, (3, 3), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Reshape((-1, x.shape[1] * x.shape[2]))(x)
        x = NAC(d_model=100, 
            num_heads=20, 
            mode = "exact",
            activation="elu", 
            sparsity=0.7, 
            topk=15,
            return_sequences=False)(x)
        drop = tf.keras.layers.Dropout(dropout_rate)(x, training=True)
        mu = tf.keras.layers.Dense(1)(drop)
        log_std = tf.keras.layers.Dense(1)(drop)
        return tf.keras.Model(inp, [mu, log_std])
    model = MC_dropout(model_fn=model_fn(INPUT_SHAPE), mc_samples=5)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss=NLL())
    return model

def _build_evidential():
    def model_fn(input_shape):
        inp = Input(shape=input_shape)
        x = Lambda(lambda x: x / 127.5 - 1.0)(inp)
        x = Conv2D(24, (5, 5), strides=(2, 2), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Conv2D(36, (5, 5), strides=(2, 2), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Conv2D(48, (5, 5), strides=(2, 2), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Conv2D(64, (3, 3), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Conv2D(64, (3, 3), activation="elu", kernel_regularizer=l2(0.001))(x)
        x = Reshape((-1, x.shape[1] * x.shape[2]))(x)
        x = NAC(d_model=100, 
            num_heads=20, 
            mode = "exact",
            activation="elu", 
            sparsity=0.7, 
            topk=15,
            return_sequences=False)(x)
        outputs = DenseNormalGamma(1)(x)
        return tf.keras.Model(inp, outputs)
    model = Evidential(model_fn=model_fn(INPUT_SHAPE), lambda_reg=0.5)
    model.compile(optimizer=tf.keras.optimizers.AdamW(1e-3), loss=EvidentialRegression)
    return model

class CNN_SDE_Wrapper(tf.keras.Model):
    def __init__(self, sde_net, input_shape):
        super().__init__()

        self.cnn = self._build_cnn(input_shape)
        self.sde_net = sde_net
        self.optimizer_cnn = tf.keras.optimizers.AdamW(1e-3)
        self.mc_samples = 10

    def _build_cnn(self, input_shape):
        inp = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Lambda(lambda x: x / 127.5 - 1.0)(inp)
        x = tf.keras.layers.Conv2D(24, (5, 5), strides=(2, 2),activation="elu",kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.Conv2D(36, (5, 5), strides=(2, 2),activation="elu",kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.Conv2D(48, (5, 5), strides=(2, 2),activation="elu",kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3),activation="elu",kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3),activation="elu",kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        shape = x.shape
        h, w, c = shape[1], shape[2], shape[3]
        x = tf.keras.layers.Reshape((h * w, c))(x)
        return tf.keras.Model(inp, x)
        
    def _cnn_to_sde(self, x, training_diffusion=False):
        features = self.cnn(x)  # (B, T, F)
        shape = tf.shape(features)
        flat = tf.reshape(features, (shape[0] * shape[1], shape[2]))
        return self.sde_net(flat, training_diffusion=training_diffusion)

    def call(self, inputs, training_diffusion=False):
        return self._cnn_to_sde(inputs, training_diffusion)

    def compile(
        self,
        optimizer_drift,
        optimizer_fc,
        optimizer_dsl,
        optimizer_diffusion,
        loss,
        **kwargs
    ):
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
            features = self.cnn(x, training=True)   # (B, T, F)
            last_feat = features[:, -1, :]          # use last token

            mean, sigma = self.sde_net(last_feat, training_diffusion=False)
            nll = self.sde_net.loss_fn(y, [mean, sigma])

        grads = tape.gradient(nll, self.cnn.trainable_variables)
        grads = [
            tf.clip_by_norm(g, 100) if g is not None else None
            for g in grads
        ]

        self.optimizer_cnn.apply_gradients(zip(grads, self.cnn.trainable_variables))
        last_feat_detached = tf.stop_gradient(last_feat)
        sde_metrics = self.sde_net.train_step((last_feat_detached, y))
        return sde_metrics

    def test_step(self, data):
        x, y = data
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)

        features = self.cnn(x, training=False)
        last_feat = features[:, -1, :]

        return self.sde_net.test_step((last_feat, y))

    def calculate_uncertainty(self, X):
        X = tf.cast(X, tf.float32)

        means = []
        variances = []

        for _ in range(self.mc_samples):
            features = self.cnn(X, training=False)
            last_feat = features[:, -1, :]
            mean, sigma = self.sde_net(last_feat, training_diffusion=False)
            means.append(mean.numpy())
            variances.append((sigma ** 2).numpy())

        means = np.stack(means, axis=0)
        variances = np.stack(variances, axis=0)

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
        return (
            mean_pred_last,
            predictive_std.numpy(),
            nll,
            mse.numpy()
        )


def _build_sde_net():
    sde = SDE_Net(input_dim=64,output_dim=1,hidden_dim=64,layer_depth=6)
    model = CNN_SDE_Wrapper(sde_net=sde,input_shape=(66,200, 3))

    model.compile(
        optimizer_drift=tf.keras.optimizers.AdamW(1e-3),
        optimizer_fc=tf.keras.optimizers.AdamW(1e-3),
        optimizer_dsl=tf.keras.optimizers.AdamW(1e-3),
        optimizer_diffusion=tf.keras.optimizers.AdamW(1e-4),
        loss=NLL(),
    )

    dummy_input = tf.zeros((1,66, 200, 3), dtype=tf.float32)
    _ = model(dummy_input, training_diffusion=False)

    return model


# Model List
model_list = [
    "NSAC",
    "Gaussian-MLE",
    "Deep-Ensemble",
    "MC-Dropout",
    "Evidential",
    "SDE-Net"
]

# Experiment Loop
NUM_RUNS = 5

results_rows = []
model_counter = 1
TEST_NAME = "Udacity"

for name in model_list:

    print(f"\n================ {name} ================")

    mse_list, nll_list, crps_list, ece_list, auroc_list = [], [], [], [], []

    for run in range(NUM_RUNS):

        seed = 42 + run
        np.random.seed(seed)
        tf.random.set_seed(seed)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)

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
        #     batch_generator(DATA_DIR, X_train, y_train, BATCH_SIZE, True),
        #     steps_per_epoch=SAMPLES_PER_EPOCH // BATCH_SIZE,
        #     epochs=EPOCHS,
        #     validation_data=batch_generator(DATA_DIR, X_test, y_test, BATCH_SIZE, False),
        #     validation_steps=len(y_train) // BATCH_SIZE,
        #     callbacks=[reduce_lr],
        #     verbose=1,
        # )
        # model.save(f'{WEIGHTS_DIR}/{name}_{MODEL_NAME}_weights_{run}.keras')

        model.load_weights(f'{WEIGHTS_DIR}/{name}_{MODEL_NAME}_weights_{run}.keras')


        x_t, y_t = [], []
        test_generator = batch_generator(DATA_DIR, X_test, y_test, BATCH_SIZE, False)
        steps = len(X_test) // BATCH_SIZE + int(len(X_test) % BATCH_SIZE != 0)

        for _ in range(int(steps/10)):
            batch_x, batch_y = next(test_generator)
            x_t.append(np.array(batch_x))
            y_t.append(np.array(batch_y))

        x_t = np.vstack(x_t)
        y_t = np.hstack(y_t)

        mu, std, nll, mse = model.comprehensive_evaluate(x_t, y_t)
        mu, std = np.array(mu), np.array(std)

        crps = compute_crps(y_t, mu, std)
        ece = compute_ece(y_t, mu, std)
        auroc = compute_auroc(y_t, mu, std)

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
print("\nCSV saved as Udacity_results.csv")