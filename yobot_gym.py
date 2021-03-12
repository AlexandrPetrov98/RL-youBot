import vrep_env
import cv2
import time
import math
import ctypes
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import os

try:
    import vrep
except:
    print ('--------------------------------------------------------------')
    print ('"vrep.py" could not be imported. This means very probably that')
    print ('either "vrep.py" or the remoteApi library could not be found.')
    print ('Make sure both are in the same folder as this file,')
    print ('or appropriately adjust the file "vrep.py"')
    print ('--------------------------------------------------------------')
    print ('')
import os
p='/home/alexandr/vrep_3.5/scenes'#path to the scene folder
vrep_scenes_path =p

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

print('Libraries imported')



def decrease_probab(state):
	ten = torch.rand(6)
	ten[state] = -1
	return ten*0.2

def cyl2dec (ro,h,phi):
	x = ro*math.cos(phi)
	y = ro*math.sin(phi)
	z = h
	return x,y,z



def state_machine(action):
	for a in range(len(action)):
		if action[a] == max(action):
			cnt = a
	move = {
		0: [1, 0, 0],
		1: [-1, 0, 0],
		2: [0, 1, 0],
		3: [0, -1, 0],
		4: [0, 0, 1],
		5: [0, 0, -1],

	}
	return move[cnt], cnt

_, re = state_machine([1,2,2,5,])
Train_mode = False
Load_mode = True

# the env class name
class Youbotgym_Env(vrep_env.VrepEnv):
	metadata = {'render.modes': [],
	}
	def __init__(
		self,
		server_addr='127.0.0.1',
		server_port=19997,
		# the filename of your v-rep scene

		scene_path=vrep_scenes_path+'/youbot_gym.ttt',
		scene_path1=vrep_scenes_path + '/youbot_gym1.ttt',
	):
		if Train_mode:
			vrep_env.VrepEnv.__init__(self,server_addr,server_port,scene_path1)
		else:
			vrep_env.VrepEnv.__init__(self, server_addr, server_port, scene_path)
		# #modify: the name of the joints to be used in action space
		print("Connection to V-REP established")

		
		# Getting object handles

		self.Agent= self.get_object_handle('youBot')
		if Load_mode:
			self.camera = self.get_object_handle('kinect_depth')
			print("youBot connected")
			self.camera1 = self.get_object_handle('kinect_rgb')
			self.kinect = self.get_object_handle('kinect_body')
			self.Joint = self.get_object_handle('youBotArmJoint3')
			self.Check_angle = self.get_object_handle('youBotArmJoint1')
			print("Camera connected")
		self.Goal=self.get_object_handle('Rectangle13')
		self.Goal1 = self.get_object_handle('youBot_positionTarget')


		print("Goal_cube connected")
		self.leftwheel=self.get_object_handle('rollingJoint_fl')
		self.rightwheel = self.get_object_handle('rollingJoint_fr')
		print("Wheels handling")
		self.Manangle = self.get_object_handle('youBotArmJoint0')
		
		
		# #modify: if size of action space is different than number of joints
		# Example: One action per joint
		num_act = 3
		
		# #modify: if size of observation space is different than number of joints
		# Example: 3 dimensions of linear and angular (2) velocities + 6 additional dimension
		num_obs = 2
		
		# #modify: action_space and observation_space to suit your needs
		self.joints_max_velocity = 3.0
		act =np.array( [np.inf]  * num_act ) #np.array( [self.joints_max_velocity] * num_act )
		obs = np.array(          [np.inf]          * num_obs )
		self.action_space = spaces.Box(low=-1, high=1, shape=(1,))
		#self.action_space      = spaces.Box(-act,act)
		self.observation_space = spaces.Box(-obs,obs)

		# #modify: optional message
		print('Youbotgym_Env: initialized')
	
	def _make_observation(self):
		"""Query V-rep to make observation.
		   The observation is stored in self.observation
		"""
		# start with empty list
		lst_o = [];
		
		# #modify: optionally include positions or velocities
		cube_pos, agent_pos = self.obj_get_position(self.Goal), self.obj_get_position(self.Agent)

		lin_vel , ang_vel = self.obj_get_velocity(self.Agent)

		wheel_1=self.obj_get_position(self.leftwheel)
		wheel_2=self.obj_get_position(self.rightwheel)
		mid_wheel=[(wheel_1[0]+wheel_2[0])/2, (wheel_1[1]+wheel_2[1])/2,(wheel_1[2]+wheel_2[2])/2]
		distance = math.sqrt((cube_pos[0] - mid_wheel[0]) ** 2 + (cube_pos[1] - mid_wheel[1]) ** 2 + (cube_pos[2] - mid_wheel[2]) ** 2)
		robvec = [wheel_2[0] - mid_wheel[0], wheel_2[1] - mid_wheel[1]]
		goalvec= [cube_pos[0] - mid_wheel[0],cube_pos[1] - mid_wheel[1]]

		rad = math.acos((robvec[0]*goalvec[0] + robvec[1]*goalvec[1])/((math.sqrt(robvec[0]**2 + robvec[1]**2))*(math.sqrt(goalvec[0]**2 + goalvec[1]**2))))
		angle = (self.obj_get_joint_angle(self.Manangle)*180/math.pi )/30

		print('Angle to the object',angle)

		lst_o += cube_pos
		lst_o += lin_vel
		
		# #modify
		# example: include position, linear and angular velocities of all shapes

		
		self.observation = [distance, angle]

	def _make_action(self, a, cube_diff,joint_x,joint_y,joint_z, FK_mode,close_gripp, raise_cube):
		"""Query V-rep to make action.
		   no return value
		"""
		emptyBuff = bytearray()

		#640X480
		print('Joint_Y',self.obj_get_joint_angle(self.Joint) * 180 / math.pi)
		print('Joint_X',self.obj_get_joint_angle(self.Manangle)* 180 / math.pi)

		if cube_diff!=[] and FK_mode == 1:
			if	cube_diff[1]<300:
				joint_x+=500
			if cube_diff[1] > 340:
				joint_x -= 500

			if cube_diff[0] > 260:
				joint_y += 500
			if cube_diff[0] < 220:
				joint_y -= 500

		print(joint_x, joint_y)

		ints = [1, close_gripp]
		if a.shape !=(1,):
			a=a.tolist()
			print('Prediction made:',np.float32(a))
			a, _ = state_machine(a)
		else:
			a = [0, 0, 0] #No movement
		if FK_mode ==0:
			a = [0, 0, 0]
		if FK_mode == 0:
			joint_y =750

		if cube_diff != [] and FK_mode == 0:
			joint_y -=200



		if FK_mode ==1:
			a.append(joint_x),a.append(300),a.append(1500),a.append(joint_y),a.append(500),a.append(joint_z)
		if FK_mode == 0 and raise_cube==0:
			a.append(joint_x), a.append(1500), a.append(700), a.append(joint_y), a.append(500), a.append(joint_z)
		if FK_mode == 0 and raise_cube == 1:
			a.append(joint_x), a.append(500), a.append(500), a.append(500), a.append(500), a.append(joint_z)


		res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(self.cID, 'youBot',vrep.sim_scripttype_childscript,'youBot_f', ints, a, [], emptyBuff,vrep.simx_opmode_blocking)
		if res == vrep.simx_return_ok:

			print('Action made:',a)

	def step(self, action,prev_dst, cube_diff,joint_x,joint_y,joint_z, FK_mode,close_gripp, raise_cube):
		"""Gym environment 'step'
		"""
		# #modify Either clip the actions outside the space or assert the space contains them

		
		# Actuate
		self._make_action(action, cube_diff,joint_x,joint_y,joint_z, FK_mode,close_gripp,raise_cube)
		# Step
		self.step_simulation()
		# Observe
		self._make_observation()
		
		# Reward
		# #modify the reward computation
		# example: possible variables used in reward
		dst = self.observation[0]
		ang = self.observation[1]
		self.timepenalty+= 0.1
		# example: different weights in reward
		print("Distance", dst)
		reward = (-dst)*200 -math.fabs(ang*30) - 30


		prev_dst=dst
		# Early stop
		# #modify if the episode should end earlier
		tolerable_threshold = 1
		done = (dst > 4 or dst<tolerable_threshold *0.2 )

		
		return self.observation, reward, done, prev_dst, {}
	
	def reset(self):
		"""Gym environment 'reset'
		"""
		if self.sim_running:
			self.stop_simulation()
		self.start_simulation()
		self.randang = np.random.rand()

		self._make_observation()
		dist, _ = self.observation
		self.timepenalty = 0
		return self.observation
	
	def render(self, mode='human', close=False):
		"""Gym environment 'render'
		"""
		pass
	
	def seed(self, seed=None):
		"""Gym environment 'seed'
		"""
		return []





def main(args):
	NN = nn.Sequential(
		nn.Linear(2, 512),
		nn.Tanhshrink(),
		nn.Linear(512, 512),
		nn.PReLU(),
		nn.Linear(512, 512),
		nn.Tanh(),
		nn.Linear(512, 256),
		nn.ReLU(),
		nn.Linear(256, 6),
		nn.Sigmoid()
	)


	if Load_mode:
		NN.load_state_dict(torch.load('Trained_Model'))

	optimizer = optim.Adam(NN.parameters(), lr=0.001)#0.01)

	dis_fac = 0.5
	lr = 0.0001
	previous_distance=0

	env = Youbotgym_Env()
	#observation = env.reset()

	for i_episode in range(2000):


		Initial_State = False
		Search_Left = False
		Search_Right = False
		observation = env.reset()
		time1 = time.clock()
		total_reward = 0
		act_rew = torch.tensor(([0, 0, 0]), dtype=torch.float, requires_grad=False)

		Mem_stack = [[]]
		FK_mode = int(1)
		joint_z = 500
		ok = False
		close_gripp = 0
		grip_time = 0
		raise_cube = 0
		for t in range(700):#150
			done=False

			img = env.obj_get_vision_image(env.camera)
			img1 = env.obj_get_vision_image(env.camera1)
			joint_2 = env.obj_get_joint_angle(env.Joint) * 180 / math.pi
			joint_1 = env.obj_get_joint_angle(env.Manangle) * 180 / math.pi
			joint_3 = env.obj_get_joint_angle(env.Check_angle) * 180 / math.pi

			lower_cube = np.array([150, 180, 200])
			upper_cube =  np.array([190, 220, 230])
			mask = cv2.inRange(img1, lower_cube, upper_cube)
			# Bitwise-AND mask and original image
			res = cv2.bitwise_and(img1, img1, mask=mask)
			bgr= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			indices = np.where(res != [0])

			cube_diff =[]
			action = torch.tensor(([0]), dtype=torch.float, requires_grad=True)
			if not Train_mode:
				if t>0:
					coord = np.asarray([np.asarray(indices[0]), np.asarray(indices[1])])
					shape = coord.shape
					if shape[1] != 0:
						cube_cord_x = coord[0][len(coord[0])//2]
						cube_cord_y = coord[1][len(coord[1])//2]
						dep = img[cube_cord_x][cube_cord_y]
						cube_diff = [cube_cord_x,cube_cord_y]
						depth = float(dep[2]) - float(dep[1])

						k=0.0719

						camdist = -k + (depth * math.sin(90 - env.obj_get_joint_angle(env.Joint) ))
						moom = ((camdist+500)/100)*0.7-1.9

						print('Distance(cam)',moom)
						X = torch.tensor(([moom, observation[1]]), dtype=torch.float, requires_grad=True)
						action = NN.forward(X)

						if not joint_1>-100 :
							action = torch.tensor(([0, 0, 0, 0, 1, 0]), dtype=torch.float, requires_grad=False)
						if not joint_1 < 100:
							action =torch.tensor(([0, 0, 0, 0, 0, 1]), dtype=torch.float, requires_grad=False)
						if moom < 0.45:
							FK_mode = int(0)


			#FK_MODE
			if FK_mode == 1:
				if  Initial_State == False:
					joint_x = 500
					joint_y = 650
				if joint_2 <= 31 and Initial_State == False:
					Initial_State = True
				if not Search_Right and Initial_State==True:
					joint_x = 0
					joint_y = 650
				if joint_1<-167 and not Search_Right:
					Search_Right = True
				if not Search_Left and  Search_Right:
					joint_x = 1000
					joint_y = 650
				if cube_diff != []:
					joint_x = 500
					joint_y = 650

			if FK_mode ==0 and not ok:
				joint_x = (joint_1+169)*2.96
				ok = True

			if FK_mode == 0 and joint_3>75 and cube_diff==[]:
				grip_time = t
				close_gripp = 1
			if grip_time< t- 5 and close_gripp == 1:
				raise_cube = 1




			observation, reward, done, previous_distance, _ = env.step(action, previous_distance, cube_diff,joint_x,joint_y,joint_z,FK_mode,close_gripp,raise_cube)



	env.close()
	return 0


if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))

