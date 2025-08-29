from sim_class import Simulation
import numpy as np

# initializing the simulation with a specified number of agents
sim = Simulation(num_agents=1) 

# defining min and max coordinates
min_x = float('inf')
max_x = float('-inf')
min_y = float('inf')
max_y = float('-inf')
min_z = float('inf')
max_z = float('-inf')

# defining the actions to reach predefined corners and explore dynamically
corners = [
    (1, 1, 1), (-1, 1, 1), (1, -1, 1), (-1, -1, 1),
    (1, 1, -1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1)
]

# defining the total number of steps for each movement
num_steps = 700

# have it explore each corner and dynamically around it
for corner in corners:
    # setting the veloicty 
    velocities = [
        (np.sign(corner[0]) * np.random.uniform(0.1, 1),
         np.sign(corner[1]) * np.random.uniform(0.1, 1),
         np.sign(corner[2]) * np.random.uniform(0.1, 1))
    ]

    for _ in range(num_steps):
        actions = [list(velocities[0]) + [0]]  # turn off drop commans
        state = sim.run(actions)

        # Update the robots position
        position = state['robotId_1']['pipette_position']

        # update the working envelope coordinates
        max_x = max(max_x, position[0])
        min_x = min(min_x, position[0])
        max_y = max(max_y, position[1])
        min_y = min(min_y, position[1])
        max_z = max(max_z, position[2])
        min_z = min(min_z, position[2])

print(min_x, max_x)
print(min_y, max_y)
print(min_z, max_z)

sim.close()
