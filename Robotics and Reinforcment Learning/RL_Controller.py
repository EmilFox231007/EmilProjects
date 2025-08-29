import numpy as np
import cv2
import matplotlib.pyplot as plt
from ot2_gym_wrapper import OT2Env
from stable_baselines3 import PPO

def crop_petri_dish(image):
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    blurred = cv2.GaussianBlur(gray_image, (9, 9), 0)
    _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) < image.shape[0] * image.shape[1] * 0.95]
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    margin = 150
    x += margin
    y += margin
    w -= 2 * margin
    h -= 2 * margin
    x = max(x, 0)
    y = max(y, 0)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    cropped_image = image[y:y+h, x:x+w]
    return cropped_image, (x, y, w, h)

# Initialize the environment
env = OT2Env(render=True)

# Load and crop petri dish image
image_path = env.sim.get_plate_image()
plate_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
plate_image, bbox = crop_petri_dish(plate_image)
cv2.imwrite("cropped_petri_dish.png", plate_image)

# Plate calibration
plate_size_pixels = plate_image.shape[0]
plate_position_robot = np.array([0.10775, 0.088 - 0.026, 0.057])
plate_size_mm = 150
conversion_factor = plate_size_mm / plate_size_pixels

# Movement bounds
PLATE_MIN_X = plate_position_robot[0]
PLATE_MAX_X = plate_position_robot[0] + (plate_size_mm / 1000)
PLATE_MIN_Y = plate_position_robot[1] - (plate_size_mm / 1000)
PLATE_MAX_Y = plate_position_robot[1]
PLATE_Z = plate_position_robot[2]

def convert_pixel_to_robot_coordinates(pixel_coords):
    x_pixels, y_pixels = pixel_coords
    root_tip_mm_X = x_pixels * conversion_factor / 1000 + plate_position_robot[0]
    root_tip_mm_Y = y_pixels * conversion_factor / 1000 + plate_position_robot[1]
    root_tip_mm_X = np.clip(root_tip_mm_X, PLATE_MIN_X, PLATE_MAX_X)
    root_tip_mm_Y = np.clip(root_tip_mm_Y, PLATE_MIN_Y, PLATE_MAX_Y)
    return np.array([root_tip_mm_X, root_tip_mm_Y, PLATE_Z])

# Loading the RL model
model = PPO.load("Astra_V3.0.zip")

def move_to_goal(env, goal_position, run_index, pipette_position, drop=False, last=False):
    """Moves the pipette to the target position using RL control. Optionally performs a drop at the goal."""

    print(f"\nRun {run_index + 1}: Moving from {pipette_position} to goal {goal_position}")

    # Define stopping threshold
    error_threshold = 0.005

    # Main control loop
    for step in range(300):
        obs = np.concatenate([pipette_position, goal_position])
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action, -1.0, 1.0)

        # Extracting control values for XYZ
        if action.shape[0] >= 3:
            x_action, y_action, z_action = action[:3]
        else:
            x_action = y_action = z_action = 0.0  # fallback for safety

        drop_command = 0  # No drop during movement

        # Apply the action
        full_action = np.array([x_action, y_action, z_action, drop_command])
        state, _, _, _, _ = env.step(full_action)
        pipette_position = state[:3]

        # Compute distance to goal
        distance = np.linalg.norm(goal_position - pipette_position)

        print(f"Step {step}: X={pipette_position[0]:.5f}, Y={pipette_position[1]:.5f}, Z={pipette_position[2]:.5f}, Drop: {drop_command}, Error={distance:.6f}")

        # Stop when goal is reached
        if distance < error_threshold:
            print(f"Goal {goal_position} reached in {step} steps.")

            # Perform drop if required
            if drop:
                print("Dropping liquid at the root tip.")

                drop_duration = 1 if not last else 1
                for _ in range(drop_duration):
                    drop_action = np.array([0, 0, 0, 1])
                    env.step(drop_action)

                wait_steps = 10 if not last else 50
                for _ in range(wait_steps):
                    env.step(np.array([0, 0, 0, 0]))

            return pipette_position

    print(f"Run {run_index + 1} complete.")
    return pipette_position


# Reset the environment
state, _ = env.reset()
pipette_position = state[:3]

# Target root positions in pixel coordinates
coordinates_nodes = [(1318, 372), (1072, 818), (1064, 1322), (1039, 1799), (1090, 2360)]

# Convert to robot coordinates
coordinates_plants = [
    [i[0] * conversion_factor / 1000 + plate_position_robot[0],
     i[1] * conversion_factor / 1000 + plate_position_robot[1],
     0.1695] for i in coordinates_nodes
]

# Move to each root and drop
for index, robot_goal in enumerate(coordinates_plants):
    is_last = index == len(coordinates_plants) - 1
    print(f"Moving to root at: {robot_goal}")
    pipette_position = move_to_goal(env, robot_goal, index, pipette_position, drop=True, last=is_last)

env.close()
