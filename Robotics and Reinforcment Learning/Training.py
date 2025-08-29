import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from wandb.integration.sb3 import WandbCallback
import wandb
from ot2_gym_wrapper import OT2Env  # Custom environment wrapper
from clearml import Task  # Import ClearML's Task
import typing_extensions
import numpy as np

env = OT2Env(render=False)

# Initialize ClearML Task
task = Task.init(
    project_name="Mentor Group S/Group 3",  # Replace with your project name
    task_name="RL_train_group_3",
)
task.set_base_docker("deanis/2023y2b-rl:latest")  # Set the base docker image
task.execute_remotely(queue_name="default")  # Execute remotely on the default queue

os.environ['WANDB_API_KEY'] = 'ad7961aa16de00343e0cd159062e4c79502f7184'

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)

args = parser.parse_args()

# initialize wandb project
run = wandb.init(project="Task11-RL", sync_tensorboard=True)

# create wandb callback
wandb_callback = WandbCallback(model_save_freq=1000,
                                model_save_path=f"models/{run.id}",
                                verbose=2,
                                )

model = PPO('MlpPolicy', env, verbose=1, 
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs, 
            tensorboard_log=f"runs/{run.id}",)

# Total timesteps for training
time_steps = 500000

save_interval = 5

for i in range(10):
    # add the reset_num_timesteps=False argument to the learn function to prevent the model from resetting the timestep counter
    # add the tb_log_name argument to the learn function to log the tensorboard data to the correct folder
    model.learn(total_timesteps=time_steps, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")
    # save the model to the models folder with the run id and the current timestep
    if (i + 1) % save_interval == 0 or (i + 1) == 10:
        model.save(f"models/{run.id}/{time_steps * (i + 1)}")

    #model.save(f"models/{run.id}/{time_steps*(i+1)}")
