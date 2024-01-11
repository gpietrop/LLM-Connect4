import gym
import pybullet

import numpy as np
import pybulletgym

# Create the environment
environment = gym.make("Walker2DPyBulletEnv-v0")

# Reset the environment to apply modifications
obs = environment.reset()

# Access the PyBullet client via the Gym environment
pybullet_client = environment.robot._p

# Modify initial conditions
initial_position = [0, 0, 1.25]  # Example position
initial_orientation = pybullet_client.getQuaternionFromEuler([0, 0, 0])  # Example orientation
environment.robot.robot_body.reset_position(initial_position)
environment.robot.robot_body.reset_orientation(initial_orientation)

# Modify physics parameters
pybullet_client.setGravity(0, 0, -10)  # Custom gravity

import numpy as np
import pybullet as p

# Parameters for the terrain
num_rows = 100
num_columns = 100
terrain_scale = [0.05, 0.05, 2.5]  # Adjust the scale to fit your simulation needs

# Generating the heightfield data
# Ensure the size matches num_rows * num_columns
heightfield_data = np.random.rand(num_rows, num_columns) * 2 - 1  # Random terrain
heightfield_data = heightfield_data.flatten()  # Flatten the array to match expected dimensions

# Create the collision shape for the terrain
terrain_shape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD,
                                       meshScale=terrain_scale,
                                       heightfieldData=heightfield_data,
                                       numHeightfieldRows=num_rows,
                                       numHeightfieldColumns=num_columns)

# Create the terrain body
terrain_body = p.createMultiBody(0, terrain_shape)

