
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
		self.world = parent_actor.get_world()
		self.bp_library = self.world.get_blueprint_library()
		self.Attachment = carla.AttachmentType

		self.sensors_dict = {}
		self.data = {}

	def setup(self, parent_actor_list):		
		for parent_actor, sensor_spec_list in parent_actor_list.items():
			if sensor_spec_list:
				actor_id = parent_actor.id
				actor_sensor_dict = {}
				self.data[actor_id] = {}
				for sensor_spec in sensor_spec_list:
					sensor_transform = (carla.Transform(carla.Location(x=sensor_spec['x'], y=sensor_spec['y'], z=sensor_spec['z']), carla.Rotation(roll=sensor_spec['roll'], pitch=sensor_spec['pitch'], yaw=sensor_spec['yaw'])), self.Attachment.Rigid)	
					sensor_bp = self.bp_library.find(str(sensor_spec['type']))
					sensor_id = sensor_spec['id']
					if sensor_spec['type'].startwith('sensor.camera'):
						sensor_bp.set_attribute('image_size_x', str(sensor_spec['width']))
						sensor_bp.set_attribute('image_size_y', str(sensor_spec['height']))
						sensor_bp.set_attribute('fov', str(sensor_spec['fov']))
						sensor_bp.set_attribute('lens_circle_multiplier', str(3.0))
						sensor_bp.set_attribute('lens_circle_falloff', str(3.0))
						sensor_bp.set_attribute('chromatic_aberration_intensity', str(0.5))
						sensor_bp.set_attribute('chromatic_aberration_offset', str(0))
					
					if sensor_spec['type'].startwith('sensor.lidar'):
						sensor_bp.set_attribute('range', str(85))
						sensor_bp.set_attribute('rotation_frequency', str(10))
						sensor_bp.set_attribute('channels', str(64))
						sensor_bp.set_attribute('upper_fov', str(10))
						sensor_bp.set_attribute('lower_fov', str(-30))
						sensor_bp.set_attribute('points_per_second', str(600000))
						sensor_bp.set_attribute('atmosphere_attenuation_rate', str(0.004))
						sensor_bp.set_attribute('dropoff_general_rate', str(0.45))
						sensor_bp.set_attribute('dropoff_intensity_limit', str(0.8))
						sensor_bp.set_attribute('dropoff_zero_intensity', str(0.4))
				
					sensor_instance = self.world.spawn_actor(
						sensor_bp,
						sensor_transform[0],
						attach_to=parent_actor,
						attachment_type=sensor_transform[1]
					)
					actor_sensor_dict[sensor_id] = sensor_instance
					self.data[actor_id][sensor_id] = None
				self.sensors_dict[actor_id] = actor_sensor_dict
				

			else:
				sensor_transform = (carla.Transform(carla.Location(x=assigned_location_dict['center'][0], y=assigned_location_dict['center'][1], z=50), carla.Rotation(pitch=-90, yaw=0)), self.Attachment.Rigid)
		
				sensor_bp = self.bp_library.find('sensor.camera.rgb')
				sensor_bp.set_attribute('image_size_x', str(512))
				sensor_bp.set_attribute('image_size_y', str(512))
				sensor_bp.set_attribute('fov', str(60.0))

				actor_sensor = self.world.spawn_actor(
					sensor_bp,
					sensor_transform[0],
				)
				self.sensors_dict['bev'] = actor_sensor
				self.data['bev'] = None
    
	  # We need to pass the lambda a weak reference to self to avoid
		# circular reference.
		weak_self = weakref.ref(self)

		for actor_id, actor_sensor in self.sensors_dict.items():
			if actor_id == 'bev':
				actor_sensor.listen(lambda data: SensorManager._parse_data(weak_self, data, actor_id))
			else:
				for sensor_id, sensor in actor_sensor.items():
					sensor.listen(lambda data: SensorManager._parse_data(weak_self, data, actor_id, sensor_id))
      
	def get_data(self, frame, actor_id, sensor_id=None):
		if not sensor_id:
			while True:
				if self.data[actor_id].frame == frame:
					self.data[actor_id].convert(self.sensors[1])
					array = np.frombuffer(self.data[actor_id].raw_data, dtype=np.dtype("uint8"))
					array = deepcopy(array)
					array = np.reshape(array, (self.data[actor_id].height, self.data[actor_id].width, 4))		# array = array[:, :, ::-1]
					return array
		else:
			if sensor_id.startwith('rgb'):
				while True:
					if self.data[actor_id][sensor_id].frame == frame:
						self.data[actor_id][sensor_id].convert(self.sensors[1])
						array = np.frombuffer(self.data[actor_id][sensor_id].raw_data, dtype=np.dtype("uint8"))
						array = deepcopy(array)
						array = np.reshape(array, (self.data[actor_id][sensor_id].height, self.data[actor_id][sensor_id].width, 4))		# array = array[:, :, ::-1]
						return array
			else:
				while True:
					points = np.frombuffer(self.data[actor_id][sensor_id].raw_data, dtype=np.dtype('f4'))
					points = deepcopy(points)
					points = np.reshape(points, (int(points.shape[0] / 4), 4))
					return points
						

	@staticmethod
	def _parse_data(weak_self, data, id, type=None):
		self = weak_self()
		if not self:
			return
		
		if not type:
			self.data[id] = data	
		else:
			self.data[id][type] = data

  	# image.convert(self.sensors[1])
		# array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
		# array = deepcopy(array)
		# array = np.reshape(array, (image.height, image.width, 4))		# array = array[:, :, ::-1]
