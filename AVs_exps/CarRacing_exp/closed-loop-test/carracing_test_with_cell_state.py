import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import gym
import numpy as np
import pandas as pd
import tensorflow as tf

from keras._tf_keras.keras.layers import Input, Dense, Flatten, Dropout, TimeDistributed, Conv2D, MaxPool2D, BatchNormalization
from keras._tf_keras.keras.models import Model
from neuronal_attention_circuit import NAC
from ouwrap import OUWrap

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import WarpFrame
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage, VecMonitor


tf.config.run_functions_eagerly(True)
tf.random.set_seed(100)

STEPS = 850
MODEL_NAME = "CarRacing_NSAC"
WEIGHTS_DIR = "model_weights2"
CSV_FILE = "nsac_with_track.csv"



def preprocess(obs):
    obs = np.transpose(obs, (0, 2, 3, 1))
    obs = np.expand_dims(obs,axis=0)
    obs = obs.astype("float32") / 255.0
    return obs


TRACK_WIDTH = 7


def build_model(input_shape=(None, 84, 84, 1)):
    inp = Input(shape=input_shape)

    x = TimeDistributed(Conv2D(32,(5,5),activation='relu',strides=3))(inp)
    x = TimeDistributed(MaxPool2D(2,2))(x)
    x = TimeDistributed(Conv2D(48,(3,3),activation='relu',strides=2))(x)
    x = TimeDistributed(MaxPool2D(2,2))(x)
    x = TimeDistributed(Conv2D(64,(2,2),activation='relu'))(x)
    x = TimeDistributed(MaxPool2D(2,2))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Flatten())(x)

    nsac_layer = OUWrap(
            NAC(d_model=64,num_heads=16, sparsity=0.7, topk=8),
            bn_mean=0.0,
            bn_std=0.0001,
            output_dim=3,
            activation='tanh',
            return_sequences=False,
            return_cell_state=True,
            return_attention=False
            )
    
    mean, std , cell_states = nsac_layer(x)
    return Model(inp, [mean, std, cell_states]), nsac_layer




def get_neuron_groups(nac_layer):
    return {
        "sensory": nac_layer.cell.sensory_indices,
        "inter": nac_layer.cell.inter_indices,
        "command": nac_layer.cell.command_indices,
        "motor": nac_layer.cell.motor_indices
    }


# =========================
# MAIN
# =========================
def main():
    obs = env.reset()
    model, nac_layer = build_model()
    model.load_weights(f"{WEIGHTS_DIR}/{MODEL_NAME}.keras")

    # ===== BUILD INDEX =====
    row_index = []

    # NSAC rows
    for gate_name in ["q_proj", "k_proj", "v_proj", "ncp_out"]:
        groups = get_neuron_groups(getattr(nac_layer, gate_name))
        for group_name, indices in groups.items():
            for idx in indices:
                row_index.append((gate_name, group_name, idx))

    # Track rows
    track_rows = [
        ("track", "center", "x"),
        ("track", "center", "y"),
        ("track", "left", "x"),
        ("track", "left", "y"),
        ("track", "right", "x"),
        ("track", "right", "y"),
    ]

    row_index += track_rows

    multi_index = pd.MultiIndex.from_tuples(row_index, names=["gate", "group", "neuron_idx"])
    df = pd.DataFrame(index=multi_index, columns=[f"step_{i}" for i in range(STEPS)])

    for step in range(STEPS):
        obs_tensor = preprocess(obs)

        mean, std, cell_states = model(obs_tensor)
        action = mean.numpy().squeeze()
        action[0] = np.clip(action[0], -1, 1)
        action[1] = np.clip(action[1], 0, 1)
        action[2] = np.clip(action[2], 0, 1)
        action = action.reshape(1, -1)
        # ---- NAC STATES ----
        for gate_name, voltages in cell_states.items():
            groups = get_neuron_groups(getattr(nac_layer, gate_name))
            for group_name, indices in groups.items():
                for idx in indices:
                    voltage = voltages[0, 0, idx].numpy()
                    df.loc[(gate_name, group_name, idx), f"step_{step}"] = voltage

        obs, _, done, _ = env.step(action)

        if done:
            break

    # ===== TRACK EXTRACTION =====
    track = env.get_attr("track")[0]
    center_x = np.array([tile[2] for tile in track])
    center_y = np.array([tile[3] for tile in track])

    center_x = np.append(center_x, center_x[0])
    center_y = np.append(center_y, center_y[0])

    dx = np.gradient(center_x)
    dy = np.gradient(center_y)

    length = np.sqrt(dx**2 + dy**2)
    length = np.maximum(length, 1e-8)

    dx /= length
    dy /= length

    nx = -dy
    ny = dx

    left_x = center_x + nx * TRACK_WIDTH / 2
    left_y = center_y + ny * TRACK_WIDTH / 2

    right_x = center_x - nx * TRACK_WIDTH / 2
    right_y = center_y - ny * TRACK_WIDTH / 2

    # ===== STORE TRACK INTO DF =====
    max_len = min(len(center_x), STEPS)

    for i in range(max_len):
        df.loc[("track", "center", "x"), f"step_{i}"] = center_x[i]
        df.loc[("track", "center", "y"), f"step_{i}"] = center_y[i]

        df.loc[("track", "left", "x"), f"step_{i}"] = left_x[i]
        df.loc[("track", "left", "y"), f"step_{i}"] = left_y[i]

        df.loc[("track", "right", "x"), f"step_{i}"] = right_x[i]
        df.loc[("track", "right", "y"), f"step_{i}"] = right_y[i]

    env.close()

    # ===== SAVE =====
    df.to_csv(CSV_FILE)
    print(f"Saved NAC + TRACK to {CSV_FILE}")


if __name__ == "__main__":
    env = make_vec_env("CarRacing-v3", n_envs=1, wrapper_class=WarpFrame, env_kwargs={"continuous": True})
    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    env.seed(42)
    main()