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
p='/home/alexandr/vrep_3.5/scenes'
vrep_scenes_path =p #os.environ['VREP_SCENES_PATH']

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

print('Libraries imported')



def decrease_probab(state):
	ten = torch.rand(6)
	ten[state] = -1
	return ten*0.2#######################################

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
Load_mode = False
while (True):
	print('Train - "1"...... Load - "2" ......Load & Train - "3"')
	input1 = int(input())
	if input1==1:
		print("Train chosen")
		Train_mode = True
		Load_mode = False
		break
	if input1==2:
		print("Load chosen")
		Load_mode = True
		Train_mode = False
		break
	if input1==3:
		print("Load chosen")
		Load_mode = True
		Train_mode = True
		break
	#print(Train_mode, Load_mode )

#print(re)
#time.sleep(6)
# #modify: the env class name
class Youbotgym_Env(vrep_env.VrepEnv):
	metadata = {'render.modes': [],
	}
	def __init__(
		self,
		server_addr='127.0.0.1',
		server_port=19997,
		# #modify: the filename of your v-rep scene

		scene_path=vrep_scenes_path+'/youbot_gym.ttt',
		scene_path1=vrep_scenes_path + '/youbot_gym1.ttt',
	):
		if Train_mode:
			vrep_env.VrepEnv.__init__(self,server_addr,server_port,scene_path1)
		else:
			vrep_env.VrepEnv.__init__(self, server_addr, server_port, scene_path)
		# #modify: the name of the joints to be used in action space
		print("Connection to V-REP established")
		'''joint_names = [
			'example_joint_0',
			'example_left_joint_0','example_right_joint_0',
			'example_joint_1',
			'example_left_joint_1','example_right_joint_1',
		]'''
		# #modify: the name of the shapes to be used in observation space
		shape_names = [
			'example_head',
			'example_left_arm','example_right_arm',
			'example_torso',
			'example_left_leg','example_right_leg',
		]
		
		# Getting object handles
		
		# we will store V-rep object handles (oh = object handle)

		# Meta
		# #modify: if you want additional object handles
		self.Agent= self.get_object_handle('youBot')
		self.camera = self.get_object_handle('kinect_depth')
		print("youBot connected")
		self.camera1 = self.get_object_handle('kinect_rgb')
		self.kinect = self.get_object_handle('kinect_body')
		print("Camera connected")
		self.Goal=self.get_object_handle('Rectangle13')
		self.Goal1 = self.get_object_handle('youBot_positionTarget')#'Rectangle13')#)
		self.Joint = self.get_object_handle('youBotArmJoint3')
		print("Goal_cube connected")
		self.leftwheel=self.get_object_handle('rollingJoint_fl')
		self.rightwheel = self.get_object_handle('rollingJoint_fr')
		print("Wheels handling")
		self.Manangle = self.get_object_handle('youBotArmJoint0')
		self.Check_angle = self.get_object_handle('youBotArmJoint1')
		# Actuators
		#self.oh_joint = list(map(self.get_object_handle, joint_names))
		# Shapes
		#self.oh_shape = list(map(self.get_object_handle, shape_names))
		
		
		# #modify: if size of action space is different than number of joints
		# Example: One action per joint
		num_act = 3#len(self.oh_joint)
		
		# #modify: if size of observation space is different than number of joints
		# Example: 3 dimensions of linear and angular (2) velocities + 6 additional dimension
		num_obs = 2#(len(self.oh_shape)*3*2) + 3*2
		
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
		#difp = self.obj_get_position(self.Goal1)
		lin_vel , ang_vel = self.obj_get_velocity(self.Agent)
		#distance = math.sqrt((cube_pos[0] - agent_pos[0])**2 + (cube_pos[1] - agent_pos[1])**2 + (cube_pos[2] - agent_pos[2])**2)
		wheel_1=self.obj_get_position(self.leftwheel)
		wheel_2=self.obj_get_position(self.rightwheel)
		mid_wheel=[(wheel_1[0]+wheel_2[0])/2, (wheel_1[1]+wheel_2[1])/2,(wheel_1[2]+wheel_2[2])/2]
		distance = math.sqrt((cube_pos[0] - mid_wheel[0]) ** 2 + (cube_pos[1] - mid_wheel[1]) ** 2 + (cube_pos[2] - mid_wheel[2]) ** 2)
		robvec = [wheel_2[0] - mid_wheel[0], wheel_2[1] - mid_wheel[1]]
		goalvec= [cube_pos[0] - mid_wheel[0],cube_pos[1] - mid_wheel[1]]
		#angle =math.acos((robvec[0]*goalvec[0] + robvec[1]*goalvec[1])/((math.sqrt(robvec[0]**2 + robvec[1]**2))*(math.sqrt(goalvec[0]**2 + goalvec[1]**2)))) *180/math.pi -90
		rad = math.acos((robvec[0]*goalvec[0] + robvec[1]*goalvec[1])/((math.sqrt(robvec[0]**2 + robvec[1]**2))*(math.sqrt(goalvec[0]**2 + goalvec[1]**2))))
		angle = (self.obj_get_joint_angle(self.Manangle)*180/math.pi )/30

		#, Jointang
		print('ANGLE',angle)
		#print('VOT TUTAAAAA',math.sqrt((cube_pos[0]-mid_wheel[0])**2 + (cube_pos[1]-mid_wheel[1])**2 ))


		#print("MID WHEEL",mid_wheel,"ANGLE", angle)#,rad)
		#print("position of cube:", cube_pos)
		#print("Mean distance",distance)
		#print("position of agent:", agent_pos)
		lst_o += cube_pos
		lst_o += lin_vel
		
		# #modify
		# example: include position, linear and angular velocities of all shapes
		'''for i_oh in self.oh_shape:
			rel_pos = self.obj_get_position(i_oh, relative_to=vrep.sim_handle_parent)
			lst_o += rel_pos ;
			lin_vel , ang_vel = self.obj_get_velocity(i_oh)
			lst_o += ang_vel ;
			lst_o += lin_vel ;'''
		
		self.observation = [distance, angle] #np.array(lst_o).astype('float32');

	def _make_action(self, a, cube_diff,joint_x,joint_y,joint_z, FK_mode,close_gripp, raise_cube):
		"""Query V-rep to make action.
		   no return value
		"""
		# #modify
		# example: set a velocity for each joint
		'''VSTAVIT' VIHOD IS NN SETI'''
		#np.random.rand() * 2 - 1
		#a=tuple([],[np.random.rand() * 2 - 1,np.random.rand() * 2 - 1,np.random.rand() * 2 - 1],[],[])
		emptyBuff = bytearray()
		#n = [np.random.rand() * 2 - 1, np.random.rand() * 2 - 1, np.random.rand() * 2 - 1]#np.random.rand() * 2 - 1,np.random.rand() * 2 - 1]
		#a=([1,2],n ,['',''],bytearray())
		#self.call_childscript_function('youBot','youBot_f',n)
		'''for i_oh, i_a in zip(self.oh_joint, a):
			self.obj_set_velocity(i_oh, i_a)'''
		#print n
		#640X480

		print('Joint_Y',self.obj_get_joint_angle(self.Joint) * 180 / math.pi)
		print('Joint_X',self.obj_get_joint_angle(self.Manangle)* 180 / math.pi)
		#time.sleep(2)
		#joint_x = 500
		#joint_y = 500
		if cube_diff!=[] and FK_mode == 1:
			if	cube_diff[1]<300:
				joint_x+=500
			if cube_diff[1] > 340:
				joint_x -= 500

			if cube_diff[0] > 260:
				joint_y += 500
			if cube_diff[0] < 220:
				joint_y -= 500
		#else:
		#	joint_x = 500
		#	joint_y = 650

		print(joint_x, joint_y)
		#FK_mode = int(1)
		#FK_mode = int(0)
		ints = [1, close_gripp]
		if a.shape !=(1,):
			a=a.tolist()
			print('Prediction made:',np.float32(a))

			#a = np.float32(a)
			a, _ = state_machine(a)
		else:
			a = [0, 0, 0] #NET DVIJENIYA
		if FK_mode ==0:
			a = [0, 0, 0]
		if FK_mode == 0:
			joint_y =750

		if cube_diff != [] and FK_mode == 0:
			joint_y -=200


		print('Fk MODE', FK_mode,'a', a)
		if FK_mode ==1:
			a.append(joint_x),a.append(300),a.append(1500),a.append(joint_y),a.append(500),a.append(joint_z)# upravlenie v FK_mode
		if FK_mode == 0 and raise_cube==0:												#600
			a.append(joint_x), a.append(1500), a.append(700), a.append(joint_y), a.append(500), a.append(joint_z)
		if FK_mode == 0 and raise_cube == 1:  # 600
			a.append(joint_x), a.append(500), a.append(500), a.append(500), a.append(500), a.append(joint_z)
		#a.append(500), a.append(300), a.append(1500), a.append(500), a.append(500), a.append(joint_z)
		#inarray =self.obj_get_position(self.Goal)
		#inarr = [a[0],a[1],a[2], inarray[0],inarray[1],inarray[2]]

		res, retInts, retFloats, retStrings, retBuffer = vrep.simxCallScriptFunction(self.cID, 'youBot',vrep.sim_scripttype_childscript,'youBot_f', ints, a, [], emptyBuff,vrep.simx_opmode_blocking)  # sysCall_init youBot_f
		if res == vrep.simx_return_ok:
			#print(n)
			print('Action made:',a)#, retFloats)

	def step(self, action,prev_dst, cube_diff,joint_x,joint_y,joint_z, FK_mode,close_gripp, raise_cube):
		"""Gym environment 'step'
		"""
		# #modify Either clip the actions outside the space or assert the space contains them
		#actions = np.clip(actions,-self.joints_max_velocity, self.joints_max_velocity)
		#assert self.action_space.contains(action), "Action {} ({}) is invalid".format(action, type(action))
		
		# Actuate
		self._make_action(action, cube_diff,joint_x,joint_y,joint_z, FK_mode,close_gripp,raise_cube)
		# Step
		self.step_simulation()
		# Observe
		self._make_observation()
		
		# Reward
		# #modify the reward computation
		# example: possible variables used in reward
		dst = self.observation[0]     #[0] # front/back YA TU POMENYAL !!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
		ang = self.observation[1]# ugol
	#	print("abs",math.fabs(ang))
		#head_pos_y = self.observation[1] # left/right
		#head_pos_z = self.observation[2] # up/down
		#nrm_action  = np.linalg.norm(action)
		#r_regul     = -(nrm_action**2)
		self.timepenalty+= 0.1
		# example: different weights in reward
		reward = (-dst)*200 -math.fabs(ang*30) - 30#*self.timepenalty #(8.0)*(r_alive) +(4.0)*(head_pos_x) +(1.0)*(head_pos_z)
		#print("Mean distance", dst, "Prevdist", prev_dst)
		#print("Mean distance",dst,"Reward",reward) #"Previous step dst",prev_dst,"Diff",prev_dst-dst,
		print("Reward", reward)
		prev_dst=dst
		# Early stop
		# #modify if the episode should end earlier
		tolerable_threshold = 1 # bilo 2
		done = (dst > 4 or dst<tolerable_threshold *0.2 )
		#done = False
		
		return self.observation, reward, done, prev_dst, {}
	
	def reset(self):
		"""Gym environment 'reset'
		"""
		if self.sim_running:
			self.stop_simulation()
		self.start_simulation()


			#self.obj_set_position(self.Goal, [math.cos(self.randang * math.pi ),math.sin(self.randang  * math.pi), 0.5])
			#self.obj_set_position(self.Goal, [0, 0, 0.5])
			#self.obj_set_position(self.Goal,[self.Agent+self.randang+2,self.Agent+self.randang +2, 0.5 ])
			#self.obj_set_position(self.Goal, [np.random.rand() * 2 - 1, np.random.rand() * 2 - 1, 0.5])

		self.randang = np.random.rand()
		#self.obj_set_position(self.Goal, [np.random.rand() * 2 - 1, np.random.rand() * 2 - 1, 0.5])
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
	"""main function used as test and example.
	   Agent does random actions with 'action_space.sample()'
	"""
	# #modify: the env class name
	NN = nn.Sequential(  #MOYA SETOCHKA
		nn.Linear(2, 512),
		nn.Tanhshrink(),#nn.PReLU(),#nn.Sigmoid(),
		nn.Linear(512, 512),
		nn.PReLU(),#nn.Sigmoid(),
		#nn.Linear(512, 512),
		#nn.PReLU(),#nn.Sigmoid(),
		nn.Linear(512, 512),
		nn.Tanh(),#nn.PReLU(),#nn.Sigmoid(),
		nn.Linear(512, 256),
		nn.ReLU(),
		#nn.Tanh(),
		nn.Linear(256, 6),
		#nn.Tanh()
		nn.Sigmoid()
		#nn.ReLU()
	)

	#print(Train_mode, Load_mode )
	if Load_mode:
		NN.load_state_dict(torch.load('Trained_Model'))
	#loss_function = nn.CrossEntropyLoss()
	optimizer = optim.Adam(NN.parameters(), lr=0.001)#0.01)
	#optimizer = optim.SGD(NN.parameters(), lr=0.1, momentum=0.5)
	#optimizer = optim.Adagrad(NN.parameters(), lr=0.1, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
	dis_fac = 0.5
	lr = 0.0001
	previous_distance=0

	env = Youbotgym_Env()
	#observation = env.reset()

	for i_episode in range(2000):
		#NN.zero_grad()
		#loss = F.smooth_l1_loss(NN.forward(X), )

		Initial_State = False
		Search_Left = False
		Search_Right = False
		observation = env.reset()
		time1 = time.clock()
		total_reward = 0
		act_rew = torch.tensor(([0, 0, 0]), dtype=torch.float, requires_grad=False)
		#time.sleep(1)
		#if i_episode == 0:
			#Mem_stack =[[]]
		Mem_stack = [[]]
		FK_mode = int(1)
		joint_z = 500
		ok = False
		close_gripp = 0
		grip_time = 0
		raise_cube = 0
		for t in range(700):#150
			done=False
			"""VMESTO n BUDET SET' """
			#n = [np.random.rand() * 2 - 1, np.random.rand() * 2 - 1, np.random.rand() * 2 - 1]
			#X = torch.tensor(([observation[0], observation[1]]), dtype=torch.float, requires_grad=True)
			#action =NN.forward(X)
			#print(observation[0])
			#cnt = t
			#img = env.obj_get_vision_image(env.camera)
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
			if t>0:
				coord = np.asarray([np.asarray(indices[0]), np.asarray(indices[1])])
				#cordX = np.array_split(coord[0], len(coord[0]))
				#coordY = np.array_split(coord[1], len(coord[1]))
				#dep = np.mean(img[coord[0]][coord[1]], axis = 0)
				shape = coord.shape
				#print("TBA", sh)
				if shape[1] != 0:
					cube_cord_x = coord[0][len(coord[0])//2]
					cube_cord_y = coord[1][len(coord[1])//2]
					dep = img[cube_cord_x][cube_cord_y]
					cube_diff = [cube_cord_x,cube_cord_y]
					#print(dep[2] , dep[1])
					depth = float(dep[2]) - float(dep[1])
					#print(depth)
					#print(coord[0][len(coord[0])//2], coord[1][len(coord[1])//2])
					#print(len(coord[0]),len(coord[1]))
					#0.318
					k=0.0719
					#print(env.obj_get_joint_angle(env.Joint)*180/math.pi)
					camdist = -k + (depth * math.sin(90 - env.obj_get_joint_angle(env.Joint) ))
					moom = ((camdist+500)/100)*0.7-1.9

					print('DISTANCIYA S KAMERY',moom)
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
			if grip_time< t- 10 and close_gripp == 1:
				raise_cube = 1

			#FK_mode = 0
			'''if FK_mode == 0 and not ok:
				#ro =  ((depth+500)/100)*0.7-1.9*math.cos(env.obj_get_joint_angle(env.Joint))
				#h =  ((depth+500)/100)*math.sin(env.obj_get_joint_angle(env.Joint))
				#phi = joint_1
				#joint_x, joint_y,joint_z = cyl2dec(ro,h,phi)
				joint_x = 500
				joint_y = 500
				joint_z = 500
				ok = True
				#100 = 0.09'''

			'''_, _, prev_z = env.obj_get_position(env.Goal1)
				print('HERE AYAYAYYAYAYYAYAYYAYYAYAYAYAYYAYAYAYAYAY',prev_z)
			if t>30:
				joint_z=500
				_, _, now_z = env.obj_get_position(env.Goal1)
				print('HERE WOWOWOWOWOWOOWOWOWOWOWOWOWOOWOWOWOWOWOWOW', prev_z - now_z)'''

			#0.3222595453262329   0.2772518992424011

			print('Fk MODE',FK_mode)
			print('Smotri suda',joint_3)
			#env.obj_get_joint_angle(env.Joint) * 180 / math.pi
			#env.obj_get_joint_angle(env.Manangle)* 180 / math.pi
			#cam = env.obj_get_position(env.kinect)
			#gol = env.obj_get_position(env.Goal1)
			#f = open('text.txt', 'w')
			#f.write( str(cam)+','+str(gol))
			#f.close()
			#cv2.imshow('image', img1)
			#cv2.imshow('depth', bgr),cv2.imshow('image2', res)
			#cv2.waitKey(0)
			#time.sleep(1.5)
			#cv2.destroyAllWindows()
			#action = env.action_space.sample()
			observation, reward, done, previous_distance, _ = env.step(action, previous_distance, cube_diff,joint_x,joint_y,joint_z,FK_mode,close_gripp,raise_cube)
			if not Load_mode:

				Mem_stack.append([X, action, reward])

			#step_learn
			#print(act_mem)
			#time.sleep(0.05)
				total_reward += reward
				if done:
				#reward +=
					break
			avg_rew = total_reward/(t+0.0000001)
			print("Episode", i_episode, " finished after {} timesteps.\tTotal reward: {}".format(t+1,total_reward)+"Avg reward =",avg_rew)
			print("Action time:",time.clock()-time1)
		#print('BAMBABMAB', Mem_stack[1])
		#act_mem+=action
		#obs_mem   torch.tensor((NN.forward()), dtype=torch.float, requires_grad=True)
		#print('BAMBABMAB', Mem_stack[t][0])
		#NN.zero_grad()
		if Train_mode:

			for n in range(len(Mem_stack)-2):
				if n%2==0 :
					NN.zero_grad()

				if done :
					#Mem_stack[n + 1][2] + 200
					pass
				else:
					max_Q_val = 200
					_, num_act = state_machine(Mem_stack[n+1][1])
					Q_delta = Mem_stack[n+2][2] + dis_fac * max_Q_val - NN.forward(Mem_stack[n+1][0]) # Ocenka + discf* maks znach - vihod sety
					Q_loss = F.smooth_l1_loss(NN.forward(Mem_stack[n+1][0]),NN.forward(Mem_stack[n+1][0])+decrease_probab(num_act)*lr* Q_delta )#(Mem_stack[n+1][1], Mem_stack[n+1][1]+0.05)#([reward*10,reward*10,reward*10]), dtype=torch.float, requires_grad=True),torch.tensor((NN.forward()), dtype=torch.float, requires_grad=True) )#NN.forward(X), Q_delta)
					print( 'Loss =',Q_loss,"Q DELTA:",Q_delta)#                                           decrease_probab(num_act)*
					optimizer.step()
					Q_loss.backward()
			#torch.tensor((Mem_stack[n+1][2],Mem_stack[n+1][2],Mem_stack[n+1][2]), dtype=torch.float, requires_grad=True), torch.tensor((Mem_stack[n+1][2],Mem_stack[n+1][2],Mem_stack[n+1][2]), dtype=torch.float, requires_grad=True)
			torch.save(NN.state_dict(),'/home/alexandr/python_scripts/NewTrainmodel')
		#time.sleep(5)
	env.close()
	return 0


if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))

	'''
	
	(a,a,a,a,a,a,a,a,a,a)	
	
	'''