# RL-youBot

This project is dedicated to solve the problem of reaching and grabing the "object" by KUKA youBot, guided by the distance and the angle info, using the depth camera and reinforcement learning approach, applied to the algorithm of platform control.
![fin](https://user-images.githubusercontent.com/49807173/110941450-72fd8280-8349-11eb-9f95-55e78c5cc823.gif)
### Launch requirements
Vrep_env - a superclass for V-REP gym environments

https://github.com/ycps/vrep-env#vrep-env

V-REP, pytorch, opencv,

Edit yobot_gym.py scene path:`24| p='YOUR_PATH'`


Locate yobot_gym.py in  `...VREP_FOLDER/programming/remoteApiBindings/python/python/`

Put youbot_gym.ttt in `...VREP_FOLDER/scenes/`

