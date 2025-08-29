import numpy as np
import matplotlib.pyplot as plt
from ot2_gym_wrapper import OT2Env
from simple_pid import PID  

# initializing the environment with visualization enabled
env = OT2Env(render=True)

# defining min and max coordinate boundaries for goal space
GOAL_SPACE_LOW = np.array([-0.1904, -0.1712, -0.1205])
GOAL_SPACE_HIGH = np.array([0.255, 0.2203, 0.2906])

# function for controlling the pipette to reach a predefined goal
def move_to_goal(env, goal_position, run_index, pipette_position):
    """ Move the pipette to a goal using PID controllers. """

    print(f"{run_index + 1}: Moving from {pipette_position} to goal {goal_position}")

    # setting up the PID controllers for each axis with optimized gains
    pid_x = PID(200.0, 0.15, 10.0, setpoint=goal_position[0], output_limits=(-1, 1))
    pid_y = PID(200.0, 0.15, 10.0, setpoint=goal_position[1], output_limits=(-1, 1))
    pid_z = PID(150.0, 0.1, 8.0, setpoint=goal_position[2], output_limits=(-1, 1))

    # definingthe error threshold for termination
    error_threshold = 0.0003  

    # tracking pipette movement data over time
    time_steps, x_positions, y_positions, z_positions = [], [], [], []
    setpoint_x, setpoint_y, setpoint_z = [], [], []

    # executing movement within the step limit
    for step in range(200):  # limiting the maximum steps per movement
        current_x, current_y, current_z = pipette_position

        # computing control outputs based on the PID response
        control_x = pid_x(current_x) * 0.6
        control_y = pid_y(current_y) * 0.6
        control_z = pid_z(current_z) * 0.4  

        drop_command = 0  # disabling liquid dispensing

        # preparing and executing action in the environment
        action = np.array([control_x, control_y, control_z, drop_command])
        state, _, _, _, _ = env.step(action)

        # updating the pipette position after executing the action
        pipette_position = state[:3]

        # computing the current distance to the target
        distance = np.linalg.norm(goal_position - pipette_position)

        # storing movement data for visualization in the final run
        if run_index == 4:
            time_steps.append(step)
            x_positions.append(current_x)
            y_positions.append(current_y)
            z_positions.append(current_z)
            setpoint_x.append(goal_position[0])
            setpoint_y.append(goal_position[1])
            setpoint_z.append(goal_position[2])

        # logging current movement status
        print(f"Step {step}: X={current_x:.4f}, Y={current_y:.4f}, Z={current_z:.4f}, "
              f"Control X={control_x:.4f}, Y={control_y:.4f}, Z={control_z:.4f}, "
              f"Goal: {goal_position}, Error={distance:.6f}")

        # terminating early if within acceptable error range
        if distance < error_threshold:
            print(f"{goal_position} reached in {step} steps")
            return pipette_position, time_steps, x_positions, y_positions, z_positions, setpoint_x, setpoint_y, setpoint_z

    print(f"{run_index + 1} complete.")
    return pipette_position, time_steps, x_positions, y_positions, z_positions, setpoint_x, setpoint_y, setpoint_z


# resetting environment to get the initial pipette position
state, _ = env.reset()
pipette_position = state[:3]  

# sequentially moving the pipette to five randomly selected goals
for i in range(5):
    random_goal = np.random.uniform(GOAL_SPACE_LOW, GOAL_SPACE_HIGH)  
    pipette_position, time_steps, x_positions, y_positions, z_positions, setpoint_x, setpoint_y, setpoint_z = move_to_goal(env, random_goal, i, pipette_position)

# closing the simulation after execution
env.close()

plt.ioff()  # ensuring non-interactive plotting mode

# function to visualize movement tracking per axis
def plot_position_vs_setpoint(time_steps, positions, setpoints, axis_name):
    """ Plot position vs setpoint for a given axis to assess tracking accuracy. """
    plt.figure(figsize=(8, 4))
    plt.xlabel("Time Steps")
    plt.ylabel(f"{axis_name} Position")
    plt.plot(time_steps, positions, label=f"{axis_name} Position", color="red")
    plt.plot(time_steps, setpoints, label=f"{axis_name} Setpoint", linestyle="dashed", color="blue")
    plt.title(f"{axis_name}-Axis Position Tracking (Final Run)")
    plt.legend()
    plt.grid(True)
    plt.show(block=True)

# generating tracking plots for the final run
plot_position_vs_setpoint(time_steps, x_positions, setpoint_x, "X")
plot_position_vs_setpoint(time_steps, y_positions, setpoint_y, "Y")
plot_position_vs_setpoint(time_steps, z_positions, setpoint_z, "Z")
