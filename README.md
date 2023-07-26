# RRT-for-Path-Planning
Project for Computational Aspects of Robotics Course (COMSW4733) from Columbia University's School of Engineering and Applied Science, April 2023

Manipulation and path planning is an important aspect of robotics. One important application is autonomous bin picking and order fulfillment solutions where a robot picks an item from one bin and places it into another. More often than not, the robot has to move in a cluttered environment while avoiding obstacles. In this project, I manipulated a simulated UR5 robot to perform grasping in simulation in the PyBullet physics simulation engine. Once the object is grasped, I implemented the RRT (Rapidly-exploring random tree) path
planning algorithm to find a collision-free path.

Project Parts
1. Basic Robot movement (PyBulletSim.move_tool method inside sim.py)
2. Grasping (functions get_grasp_position_angle and PyBullet-Sim.execute_grasp in file sim.py)
3. Path planning using RRT (sim.py, main.py)
