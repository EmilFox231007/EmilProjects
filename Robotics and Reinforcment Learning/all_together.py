import numpy as np
import time
import matplotlib.pyplot as plt
from ot2_gym_wrapper import OT2Env
from simple_pid import PID  

# ✅ Initialize the environment
env = OT2Env(render=True)
state, info = env.reset()

# ✅ Define plate position in the robot's coordinate space (meters)
plate_position_robot = np.array([0.10775, 0.062, 0.057])  # (x, y, z)

# ✅ Calculate the conversion factor from pixel-space to mm-space
plate_size_mm = 150
plate_size_pixels = 1099
conversion_factor = plate_size_mm / plate_size_pixels  # mm/pixel

# ✅ Given pixel-space coordinates of root tip
root_tip_pixel = np.array([1500, 1000])  # (x, y) in pixels
root_tip_mm = np.array([
    root_tip_pixel[0] * conversion_factor,  # Convert x from pixels to mm
    root_tip_pixel[1] * conversion_factor,  # Convert y from pixels to mm
    plate_position_robot[2]  # Use fixed Z position from plate
])

# ✅ Convert root tip position to robot's coordinate space (meters)
goal_position = (root_tip_mm + plate_position_robot) / 1000  # Convert mm to meters
print(f"Moving to calculated root tip goal: {goal_position}")

# ✅ Improved PID Controllers with stronger control
pid_x = PID(100.0, 0., 0, setpoint=goal_position[0])  # Increased Kp, Ki, Kd
pid_y = PID(100.0, 0, 0, setpoint=goal_position[1])
pid_z = PID(50.0, 0, 0, setpoint=goal_position[2])

# ✅ Increase output limits for better movement speed
pid_x.output_limits = (-0.05, 0.05)  # Increased movement range
pid_y.output_limits = (-0.05, 0.05)
pid_z.output_limits = (-0.05, 0.05)

# ✅ Scaling factor for PID output (if still too slow, increase this)
pid_scale = 10  # Adjust if needed

# ✅ Time step and error threshold
dt = 0.1
error_threshold = 0.0005  # Precise stopping condition

# ✅ Tracking lists for visualization
time_steps = []
x_positions, y_positions, z_positions = [], [], []
x_controls, y_controls, z_controls = [], [], []
x_errors, y_errors, z_errors = [], [], []
setpoint_x, setpoint_y, setpoint_z = [], [], []

# ✅ Move the pipette towards the goal
for step in range(500):
    current_x, current_y, current_z = state[:3]  # Extract pipette position

    # ✅ Compute PID control signals
    control_x = pid_x(current_x) * pid_scale
    control_y = pid_y(current_y) * pid_scale
    control_z = pid_z(current_z) * pid_scale

    action = np.array([control_x, control_y, control_z])

    # ✅ Apply action and update state
    state, reward, terminated, truncated, info = env.step(action)

    # ✅ Compute error (distance to goal)
    pipette_position = np.array([current_x, current_y, current_z])
    distance = np.linalg.norm(goal_position - pipette_position)

    # ✅ Store data for visualization
    time_steps.append(step)
    x_positions.append(current_x)
    y_positions.append(current_y)
    z_positions.append(current_z)
    x_controls.append(control_x)
    y_controls.append(control_y)
    z_controls.append(control_z)
    x_errors.append(goal_position[0] - current_x)
    y_errors.append(goal_position[1] - current_y)
    z_errors.append(goal_position[2] - current_z)
    setpoint_x.append(goal_position[0])
    setpoint_y.append(goal_position[1])
    setpoint_z.append(goal_position[2])

    # ✅ Debugging output
    print(f"Step {step}: X={current_x:.5f}, Y={current_y:.5f}, Z={current_z:.5f}, "
          f"Control X={control_x:.5f}, Y={control_y:.5f}, Z={control_z:.5f}, "
          f"Goal: {goal_position}, Error={distance:.5f}")

    # ✅ Check if pipette reached the goal
    if distance < error_threshold:
        print(f"✅ Goal reached at {goal_position} in {step} steps!")
        
        # ✅ Drop the inoculum
        action = np.array([0, 0, 0, 1])  # Last value triggers inoculation
        state, reward, terminated, truncated, info = env.step(action)
        break  # ✅ Exit loop

    # ✅ Render the environment
    if hasattr(env, "render") and callable(env.render):
        env.render()
    
    time.sleep(dt)

env.close()

# --- VISUALIZATION ---

def plot_position_vs_setpoint(time_steps, positions, setpoints, errors, axis_name):
    """Plots position, setpoint, and error for each axis"""
    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel(f"{axis_name} Position", color="tab:red")
    ax1.plot(time_steps, positions, label=f"{axis_name} Position", color="red")
    ax1.plot(time_steps, setpoints, label=f"{axis_name} Setpoint", linestyle="dashed", color="blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel(f"{axis_name} Error", color="tab:green")
    ax2.plot(time_steps, errors, label=f"{axis_name} Error", color="green", linestyle="dotted")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.title(f"{axis_name}-Axis Performance")
    plt.grid(True)
    plt.show()

# ✅ Plot position tracking, setpoint, and error
plot_position_vs_setpoint(time_steps, x_positions, setpoint_x, x_errors, "X")
plot_position_vs_setpoint(time_steps, y_positions, setpoint_y, y_errors, "Y")
plot_position_vs_setpoint(time_steps, z_positions, setpoint_z, z_errors, "Z")

print("✅ Evaluation complete. Plots generated.")
