# import urdfpy
# robot = urdfpy.URDF.load('./arm.urdf')
# robot.show()


import time
import numpy as np
import pybullet as p
import pybullet_data as p_data



client_id = p.connect(p.GUI) # p.DIRECT for faster, non-visual connection
p.setGravity(0, 0, -9.8, physicsClientId=client_id)
p.setAdditionalSearchPath(p_data.getDataPath())
planeId = p.loadURDF("plane.urdf")
carId = p.loadURDF("./arm.urdf", basePosition=[0,0,0.])


time.sleep(10)
