
import carla
import numpy as np
import weakref
from carla import ColorConverter as cc
from copy import deepcopy

assigned_location_dict = {'E1': (13.700, 2.600),
						'E2': (13.700, 6.000),
						'E3': (47.200, 1.800),
						'E4': (47.200, 5.100),
						'A1': (31.600, 18.100),
						'A2': (35.100, 18.100),
						'A3': (31.400, -18.100),
						'A4': (34.900, -18.100),
						'B1': (27.900, -18.100),
						'B2': (24.400, -18.100),
						'B3': (28.200, 18.100),
						'B4': (24.600, 18.100),
						'C1': (47.200, -1.700),
						'C2': (47.200, -5.200),
						'C3': (10.700, -0.900),
						'C4': (10.700, -4.400),
						'center': (29.550, 0.85)
						}

# ==============================================================================
# -- CameraSensor -----------------------------------------------------------------
# ==============================================================================
class SensorManager(object):
	def __init__(self, parent_actor):
		self._parent = parent_actor
		self.id = parent_actor.id
		self.world = parent_actor.get_world()
		self.bp_library = self.world.get_blueprint_library()
		self.Attachment = carla.AttachmentType

		self.sensors_dict = {}
		self.data = {}
		self.use_lidar = False

	def setup(self, sensor_spec_list):		
		for sensor_spec in sensor_spec_list:
			sensor_transform = (carla.Transform(carla.Location(x=sensor_spec['x'], y=sensor_spec['y'], z=sensor_spec['z']), carla.Rotation(roll=sensor_spec['roll'], pitch=sensor_spec['pitch'], yaw=sensor_spec['yaw'])), self.Attachment.Rigid)	
			sensor_id = sensor_spec['id']
			if sensor_spec['type'].startswith('sensor.camera'):
				sensor_rgb_bp = self.bp_library.find(str(sensor_spec['type']))
				sensor_rgb_bp.set_attribute('image_size_x', str(sensor_spec['width']))
				sensor_rgb_bp.set_attribute('image_size_y', str(sensor_spec['height']))
				sensor_rgb_bp.set_attribute('fov', str(sensor_spec['fov']))
				sensor_rgb_bp.set_attribute('lens_circle_multiplier', str(3.0))
				sensor_rgb_bp.set_attribute('lens_circle_falloff', str(3.0))
				sensor_rgb_bp.set_attribute('chromatic_aberration_intensity', str(0.5))
				sensor_rgb_bp.set_attribute('chromatic_aberration_offset', str(0))
				if sensor_id.startswith("rgb_center"):
					self.sensor_instance_rgb = self.world.spawn_actor(
					sensor_rgb_bp,
					sensor_transform[0],
				)
				else:
					self.sensor_instance_rgb = self.world.spawn_actor(
					sensor_rgb_bp,
					sensor_transform[0],
					attach_to=self._parent,
					attachment_type=sensor_transform[1]
				)
				self.data['image'] = None
				# print("create rgb sensor")
				# self.sensors_dict[sensor_id] = sensor_instance_rgb
    
			if sensor_spec['type'].startswith('sensor.lidar'):
				self.use_lidar = True
				sensor_lidar_bp = self.bp_library.find(str(sensor_spec['type']))
				sensor_lidar_bp.set_attribute('range', str(85))
				sensor_lidar_bp.set_attribute('rotation_frequency', str(10))
				sensor_lidar_bp.set_attribute('channels', str(64))
				sensor_lidar_bp.set_attribute('upper_fov', str(10))
				sensor_lidar_bp.set_attribute('lower_fov', str(-30))
				sensor_lidar_bp.set_attribute('points_per_second', str(600000))
				sensor_lidar_bp.set_attribute('atmosphere_attenuation_rate', str(0.004))
				sensor_lidar_bp.set_attribute('dropoff_general_rate', str(0.45))
				sensor_lidar_bp.set_attribute('dropoff_intensity_limit', str(0.8))
				sensor_lidar_bp.set_attribute('dropoff_zero_intensity', str(0.4))

				self.sensor_instance_lidar = self.world.spawn_actor(
					sensor_lidar_bp,
					sensor_transform[0],
					attach_to=self._parent,
					attachment_type=sensor_transform[1]
				)
				self.data['lidar'] = None
			
    
	  # We need to pass the lambda a weak reference to self to avoid
		# circular reference.
		weak_self = weakref.ref(self)

		self.sensor_instance_rgb.listen(lambda data: SensorManager._parse_data(weak_self, data, "image"))
	
		if self.use_lidar:
			self.sensor_instance_lidar.listen(lambda data: SensorManager._parse_data(weak_self, data, 'lidar'))
		# print(self.sensors_dict)
		# for sensor_id, sensor in self.sensors_dict.items():
		# 		sensor.listen(lambda data: SensorManager._parse_data(weak_self, data, sensor_id))
  
  
	def destroy(self):
		self.sensor_instance_rgb.destroy()
		if self.use_lidar:
			self.sensor_instance_lidar.destroy()	
  
	def get_data(self, frame, sensor_id=None):
		while True:
			if not self.data[sensor_id]:
				print(f'wait for {self.id} {sensor_id} sensor at frame {frame}')
				continue
			if self.data[sensor_id].frame == frame:
				break

		if sensor_id == "image":
			self.data[sensor_id].convert(cc.Raw)
			array = np.frombuffer(self.data[sensor_id].raw_data, dtype=np.dtype("uint8"))
			array = deepcopy(array)
			array = np.reshape(array, (self.data[sensor_id].height, self.data[sensor_id].width, 4))		# array = array[:, :, ::-1]
			return array
		else:
			points = np.frombuffer(self.data[sensor_id].raw_data, dtype=np.dtype('f4'))
			points = deepcopy(points)
			points = np.reshape(points, (int(points.shape[0] / 4), 4))
			return points
						

	@staticmethod
	def _parse_data(weak_self, data, sensor_id):
		self = weak_self()
		if not self:
			return
		
		if sensor_id == 'image':
			self.data['image'] = data	
		if sensor_id == 'lidar':
			self.data['lidar'] = data
  	# image.convert(self.sensors[1])
		# array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
		# array = deepcopy(array)
		# array = np.reshape(array, (image.height, image.width, 4))		# array = array[:, :, ::-1]
