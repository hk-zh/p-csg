import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque

import torch
import numpy as np
from PIL import Image

from leaderboard.autoagents import autonomous_agent
from team_code.models.p_csg import P_CSG
from team_code.conf.config import GlobalConfig
from team_code.dataset.utils import scale_and_crop_image, lidar_to_histogram_features, transform_2d_points, img_pre_processing_org
from planner import RoutePlanner

import math
from matplotlib import cm

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

SAVE_PATH = os.environ.get('SAVE_PATH', None)


def get_entry_point():
	return 'PCSGAgent'


class PCSGAgent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file):
		self.lidar_processed = list()
		self.track = autonomous_agent.Track.SENSORS
		self.config_path = path_to_conf_file
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False
		self.unified = None

		self.input_buffer = {'rgb': deque(), 'rgb_left': deque(), 'rgb_right': deque(), 
							'rgb_rear': deque(), 'lidar': deque(), 'gps': deque(), 'thetas': deque()}

		self.config = GlobalConfig()
		self.net = P_CSG(self.config)
		self.net.to("cuda")
		self.net.load_state_dict(torch.load(os.path.join(path_to_conf_file, 'best_model.pth')))
		self.net.cuda()
		self.net.eval()
		self.net.inference = True
		self.global_gt_v = 0.0
		self.old_light = 0
		self.accumulation = 0.0
		self.time_gap = 0.05

		self.save_path = None
		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			string = 'Town05_long_' #pathlib.Path(os.environ['ROUTES']).stem + '_'
			string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

			print (string)

			self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
			self.save_path.mkdir(parents=True, exist_ok=False)

			(self.save_path / 'rgb').mkdir(parents=True, exist_ok=False)
			(self.save_path / 'lidar_0').mkdir(parents=True, exist_ok=False)
			(self.save_path / 'lidar_1').mkdir(parents=True, exist_ok=False)
			(self.save_path / 'meta').mkdir(parents=True, exist_ok=False)

	def _init(self):
		self._route_planner = RoutePlanner(4.0, 50.0)
		self._route_planner.set_route(self._global_plan, True)

		self.initialized = True

	def _get_position(self, tick_data):
		gps = tick_data['gps']
		gps = (gps - self._route_planner.mean) * self._route_planner.scale

		return gps

	def sensors(self):
		return [
				{
					'type': 'sensor.camera.rgb',
					'x': 1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 400, 'height': 300, 'fov': 100,
					'id': 'rgb'
					},
				{
					'type': 'sensor.camera.rgb',
					'x': 1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
					'width': 400, 'height': 300, 'fov': 100,
					'id': 'rgb_left'
					},
				{
					'type': 'sensor.camera.rgb',
					'x': 1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
					'width': 400, 'height': 300, 'fov': 100,
					'id': 'rgb_right'
					},
				{
					'type': 'sensor.camera.rgb',
					'x': -1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': -180.0,
					'width': 400, 'height': 300, 'fov': 100,
					'id': 'rgb_rear'
					},
                {   
                    'type': 'sensor.lidar.ray_cast',
                    'x': 1.3, 'y': 0.0, 'z': 2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                    'id': 'lidar'
                    },
				{
					'type': 'sensor.other.imu',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.05,
					'id': 'imu'
					},
				{
					'type': 'sensor.other.gnss',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.01,
					'id': 'gps'
					},
				{
					'type': 'sensor.speedometer',
					'reading_frequency': 20,
					'id': 'speed'
					}
				]

	def tick(self, input_data):
		self.step += 1

		rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		rgb_rear = cv2.cvtColor(input_data['rgb_rear'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		gps = input_data['gps'][1][:2]
		speed = input_data['speed'][1]['speed']
		compass = input_data['imu'][1][-1]
		if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
			compass = 0.0
		lidar = input_data['lidar'][1][:, :3]

		result = {
				'rgb': rgb,
				'rgb_left': rgb_left,
				'rgb_right': rgb_right,
				'rgb_rear': rgb_rear,
				'lidar': lidar,
				'gps': gps,
				'speed': speed,
				'compass': compass,
				}
		
		pos = self._get_position(result)
		result['gps'] = pos
		next_wp, next_cmd = self._route_planner.run_step(pos)
		result['next_command'] = next_cmd.value

		theta = compass + np.pi/2
		R = np.array([
			[np.cos(theta), -np.sin(theta)],
			[np.sin(theta), np.cos(theta)]
			])

		local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
		local_command_point = R.T.dot(local_command_point)
		result['target_point'] = tuple(local_command_point)

		return result

	@torch.no_grad()
	def run_step(self, input_data, timestamp):
		if not self.initialized:
			self._init()

		tick_data = self.tick(input_data)

		if self.step < self.config.seq_len:
			rgb = img_pre_processing_org(tick_data['rgb']).unsqueeze(0)
			self.input_buffer['rgb'].append(rgb.to('cuda', dtype=torch.float32))
			self.input_buffer['lidar'].append(tick_data['lidar'])
			self.input_buffer['gps'].append(tick_data['gps'])
			self.input_buffer['thetas'].append(tick_data['compass'])

			control = carla.VehicleControl()
			control.steer = 0.0
			control.throttle = 0.0
			control.brake = 0.0

			self.global_throttle = 0.0
			self.global_steer = 0.0
			self.global_brake = 0.0

			return control

		gt_velocity = tick_data['speed']
		self.global_gt_v = gt_velocity.squeeze(0).item()

		if self.old_light == 0 and self.accumulation >= 20.0:
			gt_velocity = 2.0
		elif self.old_light == 1 and self.accumulation >= 10.0:
			gt_velocity = 2.0
		if self.old_light == 2 and self.accumulation >= 30.0:
			gt_velocity = 2.5

		gt_velocity = torch.FloatTensor([gt_velocity]).to('cuda', dtype=torch.float32)

		command = torch.FloatTensor([tick_data['next_command']]).to('cuda', dtype=torch.float32)
		gt_throttle = torch.FloatTensor([self.global_throttle]).to('cuda', dtype=torch.float32)
		gt_steer = torch.FloatTensor([self.global_steer]).to('cuda', dtype=torch.float32)
		gt_brake = torch.FloatTensor([self.global_brake]).to('cuda', dtype=torch.float32)
		gt_theta = torch.FloatTensor([tick_data['compass']]).to('cuda', dtype=torch.float32)
		# gt_measurements = torch.cat([gt_velocity, gt_steer, gt_throttle, gt_brake]).view(4, -1)
		gt_measurements = torch.stack([gt_velocity,gt_steer,gt_throttle,gt_brake], 1)

		tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
											torch.FloatTensor([tick_data['target_point'][1]])]
		target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)

		encoding = []
		rgb = img_pre_processing_org(tick_data['rgb']).unsqueeze(0)
		self.input_buffer['rgb'].popleft()
		self.input_buffer['rgb'].append(rgb.to('cuda', dtype=torch.float32))
		self.input_buffer['lidar'].popleft()
		self.input_buffer['lidar'].append(tick_data['lidar'])
		self.input_buffer['gps'].popleft()
		self.input_buffer['gps'].append(tick_data['gps'])
		self.input_buffer['thetas'].popleft()
		self.input_buffer['thetas'].append(tick_data['compass'])
		# print(tick_data['compass'])

		# transform the lidar point clouds to local coordinate frame
		ego_theta = self.input_buffer['thetas'][-1]
		ego_x, ego_y = self.input_buffer['gps'][-1]

		#Only predict every second step because we only get a LiDAR every second frame.
		if(self.step  % 2 == 0 or self.step <= 4):
			for i, lidar_point_cloud in enumerate(self.input_buffer['lidar']):
				curr_theta = self.input_buffer['thetas'][i]
				curr_x, curr_y = self.input_buffer['gps'][i]
				lidar_point_cloud[:,1] *= -1 # inverts x, y
				lidar_transformed = transform_2d_points(lidar_point_cloud,
						np.pi/2-curr_theta, -curr_x, -curr_y, np.pi/2-ego_theta, -ego_x, -ego_y)
				lidar_transformed = torch.from_numpy(lidar_to_histogram_features(lidar_transformed, crop=self.config.input_resolution)).unsqueeze(0)
				self.lidar_processed = list()
				self.lidar_processed.append(lidar_transformed.to('cuda', dtype=torch.float32))

			unified = self.input_buffer['rgb']

			self.unified = unified
			self.ret = self.net(unified[0], self.lidar_processed[0], target_point, gt_measurements)

		steer, throttle, brake, metadata = self.net.control_pid(self.ret[0], torch.FloatTensor([self.global_gt_v]))
		self.pid_metadata = metadata
		self.global_steer = steer
		self.global_throttle = throttle
		self.global_brake = brake

		light_prob = torch.nn.functional.softmax(self.ret[1].cpu().squeeze(0), dim=0)
		recent_light = torch.argmax(light_prob, dim=0)
		if self.global_gt_v < 0.1:
			self.accumulation += self.time_gap
		else:
			self.accumulation = 0.0
		if recent_light == self.old_light:
			pass
		else:
			self.old_light = recent_light

		if brake < 0.05: brake = 0.0
		if throttle > brake: brake = 0.0

		# if self.old_light == 1 and self.accumulation > 7.0:
		# 	steer += 1.
		# 	brake = 0.0
		
		# if self.old_light == 2 and self.accumulation > 20.0:
		# 	steer -=0.03
		# 	brake = 0.0
		# if self.old_light == 0 and self.accumulation > 30.0:
		# 	steer -=0.03
		# 	brake = 0.0

		control = carla.VehicleControl()
		control.steer = float(steer)
		control.throttle = float(throttle)
		control.brake = float(brake)
		# print('steer: ', steer, 'throttle: ', throttle, 'brake: ', brake)

		# import pdb; pdb.set_trace()
		if SAVE_PATH is not None and self.unified is not None and self.step % 100 == 0:
			self.save(tick_data)

		return control

	def save(self, tick_data):
		frame = self.step // 100

		output = self.unified[-1].squeeze(0).permute(1, 2, 0).cpu().numpy()
		Image.fromarray(np.uint8(output)).save(self.save_path / 'rgb' / ('%04d.png' % frame))

		Image.fromarray(cm.gist_earth(self.lidar_processed[0].cpu().numpy()[0, 0], bytes=True)).save(self.save_path / 'lidar_0' / ('%04d.png' % frame))
		Image.fromarray(cm.gist_earth(self.lidar_processed[0].cpu().numpy()[0, 1], bytes=True)).save(self.save_path / 'lidar_1' / ('%04d.png' % frame))


		outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
		json.dump(self.pid_metadata, outfile, indent=4)
		outfile.close()

	def destroy(self):
		del self.net

