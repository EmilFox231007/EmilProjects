import numpy as np
from stable_baselines3 import PPO
from ot2_gym_wrapper import OT2Env

# Reproducibility
np.random.seed(42)

log_file_path = "evaluation_output.txt"

# Load model
model = PPO.load("Astra_V3.0.zip")

# Create environment
env = OT2Env(render=False)

# Define petri dish bounds
PLATE_MIN_X = 0.10775
PLATE_MAX_X = PLATE_MIN_X + 0.150
PLATE_MIN_Y = 0.062
PLATE_MAX_Y = 0.088
PLATE_Z = 0.1695

# Evaluation config
n_episodes = 10
distance_threshold = 0.001
max_steps = 300

# Generate random targets
eval_targets = [
    np.array([
        np.random.uniform(PLATE_MIN_X, PLATE_MAX_X),
        np.random.uniform(PLATE_MIN_Y, PLATE_MAX_Y),
        PLATE_Z
    ]) for _ in range(n_episodes)
]

# Metrics
success_count = 0
total_steps = []
final_errors = []

# Logging function
def log(text, file):
    print(text)
    file.write(text + "\n")

with open(log_file_path, "w") as f:
    for idx, goal in enumerate(eval_targets):
        log(f"\nEpisode {idx + 1}: Target = {goal}", f)
        state, _ = env.reset()
        pipette_position = state[:3]

        for step in range(max_steps):
            obs = np.concatenate([pipette_position, goal])
            action, _ = model.predict(obs, deterministic=True)
            action = np.clip(action, -1.0, 1.0)
            full_action = np.append(action, 0.0) if action.shape[0] == 3 else action

            state, _, _, _, _ = env.step(full_action)
            pipette_position = state[:3]
            error = np.linalg.norm(goal - pipette_position)

            if error < distance_threshold:
                log(f"  Reached goal in {step + 1} steps. Final error: {error:.6f}", f)
                success_count += 1
                total_steps.append(step + 1)
                final_errors.append(error)
                break
        else:
            log(f"  Failed to reach goal. Final error: {error:.6f}", f)
            total_steps.append(max_steps)
            final_errors.append(error)

    env.close()

    # Final summary
    log("\nEvaluation Summary:", f)
    log(f"Success rate: {success_count}/{n_episodes} = {success_count / n_episodes * 100:.1f}%", f)
    log(f"Average steps: {np.mean(total_steps):.2f}", f)
    log(f"Average final error: {np.mean(final_errors):.6f}", f)

