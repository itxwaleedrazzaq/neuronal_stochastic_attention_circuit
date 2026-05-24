import pickle
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (
    VecFrameStack,
    VecTransposeImage,
    VecMonitor
)
from stable_baselines3.common.atari_wrappers import WarpFrame


def main():
    # Load trained PPO model
    model = PPO.load("/home/ustc15/research/idea10/CarRacing/best_model.zip")

    # Create environment (NO human render needed for data collection)
    env = make_vec_env(
        "CarRacing-v3",
        n_envs=1,
        wrapper_class=WarpFrame,
        env_kwargs={"continuous": True}
    )

    env = VecMonitor(env)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    observations = []
    actions = []

    num_episodes = 50
    episode_count = 0

    obs = env.reset()

    while episode_count < num_episodes:
        action, _ = model.predict(obs, deterministic=True)

        # Store observation and action
        observations.append(obs.copy())
        actions.append(action.copy())

        obs, rewards, dones, infos = env.step(action)

        if dones:
            episode_count += 1
            print(f"Finished episode {episode_count}")
            obs = env.reset()

    env.close()

    # Convert to numpy arrays
    observations = np.array(observations)
    actions = np.array(actions)

    # Save to pickle
    with open("./data.pkl", "wb") as f:
        pickle.dump({
            "observations": observations,
            "actions": actions
        }, f)

    print("Dataset saved as car_racing_dataset.pkl")
    print("Observations shape:", observations.shape)
    print("Actions shape:", actions.shape)


if __name__ == "__main__":
    main()