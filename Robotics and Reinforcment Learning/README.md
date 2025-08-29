# OT2 Robot Simulation with PID control

This repository contains Python scripts for controlling an OT2 pipette robot in a simulation environment using PyBullet. The pipette movement is controlled either using a PID controller or a Reinforcement Learning (RL) model. The environment is implemented using OpenAI Gymnasium.

## Installation & Dependencies

Ensure you have Python installed (recommended version: 3.8 or later). Install the required dependencies by running:

### Required Python Packages

- `numpy`
- `opencv-python`
- `matplotlib`
- `gymnasium`
- `stable-baselines3`
- `simple-pid`
- `pybullet`

If using RL-based control, ensure you have Stable-Baselines3 installed:

```bash
pip install stable-baselines3
```

## Files Overview

### 1. `ot2_gym_wrapper.py`

Defines the custom OpenAI Gymnasium environment (`OT2Env`) for the simulation. This includes action and observation spaces, step logic, reset functions, and reward computation.

### 2. `sim_class.py`

Handles the PyBullet simulation setup, including robot creation, camera setup, movement, and physics interactions.

### 3. `pid_control.py`

Implements pipette movement using a PID controller. The script generates performance graphs for the final run.

### 4. `rl_control.py`

Uses a trained RL model (`Astra_V3.0.zip`) to control the pipette's movement. The model predicts actions based on observations.

## Usage

### Running the Simulation with PID Control

To run the simulation with a PID controller:

```bash
python Task_13.py
```

This script will:

1. Initialize the PyBullet simulation.
2. Control the pipette using a PID controller.
3. Move to predefined root locations.
4. Generate performance plots for position tracking.

### Running the Simulation with RL Control

To run the simulation with a RL controller:

```bash
python Task_13_RL_Controller.py
```

## Robot Environment Details

- **Action Space:**
  - `[-1, -1, -1]` (min) to `[1, 1, 1]` (max) for XYZ movement.
- **Observation Space:**
  - `(6,)` containing pipette position and goal position.
- **Goal Space:**
  - 3D Cartesian space within `[-0.1904, -0.1712, -0.1205]` to `[0.255, 0.2203, 0.2906]`.

## Hyperparameters used for final model

- **Batch Size:** 128
- **Learning Rate:** Linear schedule (0.1) this means that it decays overtime 
- **Time Steps:** 1.000.000