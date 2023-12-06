#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.


from __future__ import print_function

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import argparse
import carla
import cv2
import datetime
import json
import logging
import numpy as np
import math
import pygame
import random
import re
import time
import weakref

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from autopilot import AutoPilot
from bird_eye_view.Mask import Loc
from carla import ColorConverter as cc
# from navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from random_actors import spawn_actor_assigned
from roach_agent import BEV_MAP
from threading import Thread

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================
class World(object):
    def __init__(self, carla_world, hud, args):
        self.world = carla_world
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True  # Enables synchronous mode
        self.world.apply_settings(settings)
        self.actor_role_name = args.rolename
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
            
            
        # layer map remove ...    
        
        self.world.unload_map_layer(carla.MapLayer.Buildings)     
        self.world.unload_map_layer(carla.MapLayer.Decals)     
        self.world.unload_map_layer(carla.MapLayer.Foliage)     
        self.world.unload_map_layer(carla.MapLayer.Ground)     
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)         
        self.world.unload_map_layer(carla.MapLayer.Particles)     
        self.world.unload_map_layer(carla.MapLayer.Props)     
        self.world.unload_map_layer(carla.MapLayer.StreetLights)     
        self.world.unload_map_layer(carla.MapLayer.Walls)     

        self.hud = hud
        self.player = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.restart()
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.constant_velocity_enabled = False
        self.current_map_layer = 0
        self.map_layer_names = [
            carla.MapLayer.NONE,
            carla.MapLayer.Buildings,
            carla.MapLayer.Decals,
            carla.MapLayer.Foliage,
            carla.MapLayer.Ground,
            carla.MapLayer.ParkedVehicles,
            carla.MapLayer.Particles,
            carla.MapLayer.Props,
            carla.MapLayer.StreetLights,
            carla.MapLayer.Walls,
            carla.MapLayer.All
        ]

    def restart(self):
        self.player_max_speed = 1.3 #1.589
        self.player_max_speed_fast = 3.713
        # Keep same camera config if the camera manager exists.
        # cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        # cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint = self.world.get_blueprint_library().find('vehicle.lincoln.mkz_2017')
        blueprint.set_attribute('role_name', self.actor_role_name)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        if blueprint.has_attribute('is_invincible'):
            blueprint.set_attribute('is_invincible', 'true')
        # set the max speed
        if blueprint.has_attribute('speed'):
            self.player_max_speed = float(blueprint.get_attribute('speed').recommended_values[1])
            self.player_max_speed_fast = float(blueprint.get_attribute('speed').recommended_values[2])
        else:
            print("No recommended values for 'speed' attribute")


       # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            
            # spawn_point = carla.Transform()
            
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        # Set up the sensors.
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.set_sensor(notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):

        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        # print(preset[1])
        while ('Night' in preset[1]):
            self._weather_index += -1 if reverse else 1
            self._weather_index %= len(self._weather_presets)
            preset = self._weather_presets[self._weather_index]
            # print(preset[1])
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def next_map_layer(self, reverse=False):
        self.current_map_layer += -1 if reverse else 1
        self.current_map_layer %= len(self.map_layer_names)
        selected = self.map_layer_names[self.current_map_layer]
        self.hud.notification('LayerMap selected: %s' % selected)

    def load_map_layer(self, unload=False):
        selected = self.map_layer_names[self.current_map_layer]
        if unload:
            self.hud.notification('Unloading map layer: %s' % selected)
            self.world.unload_map_layer(selected)
        else:
            self.hud.notification('Loading map layer: %s' % selected)
            self.world.load_map_layer(selected)

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock, frame, display):
        end = self.hud.tick(self, clock, self.camera_manager, frame, display)
        return end
    def render(self, display):
        self.camera_manager.render(display)
        # self.hud.render(display)

    def destroy_sensors(self):
        pass

    def destroy(self):        
        sensors = [
            self.camera_manager.sensor_rgb_bev,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()
        


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================
class HUD(object):
    def __init__(self, width, height, distance=25.0, town='Town05', v_id=1):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        # self._notifications = FadingText(font, (width, 40), (0, height - 40))
        # self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self.recording = False
        self.recording_frame = 0
        self.v_id = int(v_id)
        self.ego_data = {}

        self.d_2_intersection = distance
        self.d_last = distance
        self.jam = 0
        
        self.ss_front = []
        self.ss_left = []
        self.ss_right = []
        self.ss_rear = []
        self.ss_rear_left = []
        self.ss_rear_right = []
        
        self.depth_front = []
        self.depth_left = []
        self.depth_right = []
        self.depth_rear = []
        self.depth_rear_left = []
        self.depth_rear_right = []
        self.counter = 0


        self.data_list = []
        self.frame_list = []
        self.sensor_data_list = []
        self.id_list = []
        self.ego_list = []
        
        self.record_flag = False

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def save_ego_data(self, path):
        with open(os.path.join(path, 'ego_data.json'), 'w') as f:
            json.dump(self.ego_data, f, indent=4)
        self.ego_data = {}

    def record_speed_control_transform(self, world, frame):
        v = world.player.get_velocity()
        c = world.player.get_control()
        t = world.player.get_transform()
        if frame not in self.ego_data:
            self.ego_data[frame] = {}
        self.ego_data[frame]['speed'] = {'constant': math.sqrt(v.x**2 + v.y**2 + v.z**2),
                                         'x': v.x, 'y': v.y, 'z': v.z}
        self.ego_data[frame]['control'] = {'throttle': c.throttle, 'steer': c.steer,
                                           'brake': c.brake, 'hand_brake': c.hand_brake,
                                           'manual_gear_shift': c.manual_gear_shift,
                                           'gear': c.gear}
        self.ego_data[frame]['transform'] = {'x': t.location.x, 'y': t.location.y, 'z': t.location.z,
                                             'pitch': t.rotation.pitch, 'yaw': t.rotation.yaw, 'roll': t.rotation.roll}

    
                                           
    def tick(self, world, clock, camera, frame, display):
        # self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        # print("vehicle height", t)

        compass = world.imu_sensor.compass
        heading = 'N' if compass > 270.5 or compass < 89.5 else ''
        heading += 'S' if 90.5 < compass < 269.5 else ''
        heading += 'E' if 0.5 < compass < 179.5 else ''
        heading += 'W' if 180.5 < compass < 359.5 else ''
        vehicles = world.world.get_actors().filter('vehicle.*')
        peds = world.world.get_actors().filter('walker.pedestrian.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Compass:% 17.0f\N{DEGREE SIGN} % 2s' % (compass, heading),
            'Accelero: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.accelerometer),
            'Gyroscop: (%5.1f,%5.1f,%5.1f)' % (world.imu_sensor.gyroscope),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]

        moving = False
        acc = world.player.get_acceleration().length()

        if c.throttle == 0:
            self.jam += 1
            # print(acc)
            if self.jam > 100:
                return 0
        else:
            self.jam = 0 

        return 1

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        pass
        # self._notifications.set_text(text, seconds=seconds)
        
    def error(self, text):
        pass
        # self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        # self._notifications.render(display)
        # self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================
class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================
class HelpText(object):
    """Helper class to handle text output using pygame"""

    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.line_space = 18
        self.dim = (780, len(lines) * self.line_space + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * self.line_space))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- GnssSensor ----------------------------------------------------------------
# ==============================================================================
class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)))
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- IMUSensor -----------------------------------------------------------------
# ==============================================================================
class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================
class CameraManager(object):
    def __init__(self, parent_actor, hud, gamma_correction, save_mode=True):
        # self.sensor_front = None
        self.sensor_rgb_bev = None

        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self.save_mode = save_mode

        self.rgb_front = None
        self.ss_front = None


        Attachment = carla.AttachmentType

        self.camera_transform = (carla.Transform(carla.Location(x=assigned_location_dict['center'][0], y=assigned_location_dict['center'][1], z=50), carla.Rotation(pitch=-90, yaw=-90)), Attachment.SpringArmGhost)
        self.sensor = ['sensor.camera.rgb', cc.Raw, 'Camera RGB']
        
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        
        self.sensor_rgb_bp = bp_library.find('sensor.camera.rgb')
        self.sensor_rgb_bp.set_attribute('image_size_x', str(512))
        self.sensor_rgb_bp.set_attribute('image_size_y', str(512))
        self.sensor_rgb_bp.set_attribute('fov', str(60.0))
        
       
        bp = bp_library.find(self.sensor[0])
        if self.sensor[0].startswith('sensor.camera'):
            bp.set_attribute('image_size_x', str(hud.dim[0]))
            bp.set_attribute('image_size_y', str(hud.dim[1]))
            if bp.has_attribute('gamma'):
                bp.set_attribute('gamma', str(gamma_correction))
            # for attr_name, attr_value in self.sensor[3].items():
            #     bp.set_attribute(attr_name, attr_value)


        self.sensor.append(bp)
        self.index = None

    def toggle_camera(self):
        self.set_sensor(notify=False, force_respawn=True)

    def set_sensor(self, notify=True, force_respawn=False):
        if self.sensor_rgb_bev is not None:
            self.sensor_rgb_bev.destroy()
            self.surface = None

        # rgb sensor
            ## setup sensors [ tf sensors  ( total 6 * 3 sensors ) ] 
            # rgb 
        self.sensor_rgb_bev = self._parent.get_world().spawn_actor(
            self.sensor_rgb_bp,
            self.camera_transform[0]
        )
        

        # We need to pass the lambda a weak reference to self to avoid
        # circular reference.
        weak_self = weakref.ref(self)

        self.sensor_rgb_bev.listen(lambda image: CameraManager._parse_image(weak_self, image, 'rgb_bev'))

        # if notify:
        #     self.hud.notification(self.sensors[index][2])
        # self.index = index

        
    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image, view='rgb_bev'):
        self = weak_self()
        if not self:
            return
        
        elif view == 'rgb_bev':
            image.convert(self.sensor[1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (512, 512, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            # render the view shown in monitor
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            
            
def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []
        

def check_close(ev_loc, loc0, distance = 3):
    # ev_location = self.vehicle.get_location()
    # closest_idx = 0

    # for i in range(len(self._global_route)-1):
    #     if i > windows_size:
    #         break

    #     loc0 = self._global_route[i][0].transform.location
    #     loc1 = self._global_route[i+1][0].transform.location
    if ev_loc.distance(loc0) < distance:
        return True

def init_multi_agent(world, planner, agent_list, start_list, dest_list, roach_policy=None):
    map = world.get_map()
    # planner = GlobalRoutePlanner(map, sampling_resolution=1.0)

    ego_agent_list = []
    interactive_agent_list = []
    for i, agent in enumerate(agent_list):
        agent_dict = {}
        spawn_trans = carla.Transform(carla.Location(assigned_location_dict[start_list[i]][0], assigned_location_dict[start_list[i]][1]))
        spawn_trans.location.z += 2.0
        spawn_trans.rotation.roll = 0.0
        spawn_trans.rotation.pitch = 0.0
        
        #####################  determine yaws by comparing the relative positions of spawn points and the center point ################
        x_diff = assigned_location_dict['center'][0] - assigned_location_dict[start_list[i]][0]
        y_diff = assigned_location_dict['center'][1] - assigned_location_dict[start_list[i]][1]
        if x_diff < 0 and y_diff < 0:
            spawn_trans.rotation.yaw = 270
        elif x_diff > 0 and y_diff < 0:
            spawn_trans.rotation.yaw = 0
        elif x_diff < 0 and y_diff > 0:
            spawn_trans.rotation.yaw = 180
        else:
            spawn_trans.rotation.yaw = 90

        ####################  set up the locations of destinations, the locations will later be used to calculate A* routes by planner ################
        dest_trans = carla.Location(assigned_location_dict[dest_list[i]][0], assigned_location_dict[dest_list[i]][1])
        
        # get blueprint from world
        blueprint = world.get_blueprint_library().find('vehicle.lincoln.mkz_2017')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)

        try:
            carla_agent = world.spawn_actor(blueprint, spawn_trans)
            agent_dict['id'] = carla_agent.id
            agent_dict["agent"] = carla_agent
            agent_dict['dest'] = dest_trans
            route = planner.trace_route(carla_agent.get_location(), dest_trans)
            agent_dict['route'] = route                    
            agent_dict['done'] = 0

        except:
            print("Spawn failed because of collision at spawn position")

        if agent == "ego":
            # TODO
            agent_dict['name'] = 'e2e'
        else:
            if agent == 'roach':            
                # Initialize roach agent
                roach_agent = BEV_MAP(map.name.split('/')[-1])
                roach_agent.init_vehicle_bbox(carla_agent.id)
                roach_agent.set_policy(roach_policy)
                agent_dict['model'] = roach_agent
                agent_dict['name'] = agent
            if agent == "auto":
                agent_dict['imu'] = IMUSensor(carla_agent)
                auto_agent = AutoPilot(carla_agent, world, route)
                agent_dict['model'] = auto_agent   
                agent_dict['name'] = 'auto' 
            
            # set route
            interactive_agent_list.append(agent_dict)                

    
    return ego_agent_list, interactive_agent_list


assigned_location_dict = {'E1': (-188.1, 18.5),
                    'E2': (-184.1, 18.5),
                    'E3': (-188.2, -15.0),
                    'E4': (-184.5, -15.0),
                    'A1': (-174.5, -0.4),
                    'A2': (-174.5, -4.1),
                    'A3': (-204.8, -0.5),
                    'A4': (-204.8, -3.9),
                    'B1': (-204.8, 3.5),
                    'B2': (-204.8, 6.6),
                    'B3': (-174.5, 3.1),
                    'B4': (-174.5, 6.78),
                    'C1': (-191.8, -15.0),
                    'C2': (-195.3, -15.0),
                    'C3': (-191.5, 18.5),
                    'C4': (-195.2, 18.5),
                    'center': (-190.0, 2.0)
                    }


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================
def game_loop(args):
    f = open(args.eval_config)
    eval_config = json.load(f)
    scenario_index = 0
    for town, scenarios in eval_config["available_scenarios"].items():
        for scene in scenarios:
            #try:
            client = carla.Client(args.host, args.port)
            client.set_timeout(10.0)

            # Initialize pygame
            pygame.init()
            pygame.font.init()
            display = pygame.display.set_mode(
                (512, 512),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
            display.fill((0,0,0))
            pygame.display.flip()
            
            scenario_index += 1
            logging.info(f'Running scenario {scenario_index} at {town}')
            hud = HUD(args.width, args.height, args.distance, town)
            world = World(client.load_world(town), hud, args)
            avg_FPS = 0
            clock = pygame.time.Clock()

            settings = world.world.get_settings()
            settings.fixed_delta_seconds = 0.05
            settings.synchronous_mode = True  # Enables synchronous mode
            world.world.apply_settings(settings)

            # spawn other agent 
            map = world.world.get_map()
            spawn_points = map.get_spawn_points()
            planner = GlobalRoutePlanner(map, sampling_resolution=1.0)                

            # Initialize a global roach for all roach agent to avoid generating multipile HD maps
            global_roach = None
            global_roach_policy = None
            if 'roach' in scene['agent']:
                global_roach = BEV_MAP(town)
                global_roach.init_vehicle_bbox(world.player.id)
                global_roach_policy = global_roach.init_policy()
                
            ego_agent_list, interactive_agent_list = init_multi_agent(world.world, planner, scene['agent'], scene['start'], scene['dest'], global_roach_policy)

            while True:
                clock.tick_busy_loop(30)
                frame = world.world.tick()

                view = pygame.surfarray.array3d(display)
                view = view.transpose([1, 0, 2]) 
                image = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)        

                world.tick(clock, frame, image)
                avg_FPS = 0.98 * avg_FPS + 0.02 * clock.get_fps()

                # collect data for all roach if needed
                if global_roach:
                    processed_data = global_roach.collect_actor_data(world)
                
                ###################player control#####################
                for agent_dict in interactive_agent_list:                        
                    # regenerate a route when the agent deviates from the current route
                    if not check_close(agent_dict["agent"].get_location(), agent_dict['route'][0][0].transform.location, 6):
                        print("route deviation")
                        destination = agent_dict['route'][-1][0].transform.location
                        agent_dict['route'] = planner.trace_route(agent_dict["agent"].get_location(), destination)
                    
                    # Delete reached points from current route
                    while check_close(agent_dict["agent"].get_location(), agent_dict['route'][0][0].transform.location):
                        agent_dict['route'].pop(0)
                        if len(agent_dict['route']) == 0:
                            print("route len == 0")
                            agent_dict['done'] = 1
                            new_destination = random.choice(spawn_points).location
                            agent_dict['route'] = planner.trace_route(agent_dict["agent"].get_location(), new_destination)
                    # Generate new destination if the current one is close 
                    if len(agent_dict['route']) < 5:
                        print("route len < 5")
                        agent_dict['done'] = 1
                        new_destination = random.choice(spawn_points).location
                        new_route = planner.trace_route(agent_dict['route'][-1][0].transform.location, new_destination)
                        temp_route = agent_dict['route'] + new_route
                        agent_dict['route'] = temp_route       
                
                for agent_dict in ego_agent_list:                            
                    # regenerate a route when the agent deviates from the current route
                    if not check_close(agent_dict['agent'].get_location(), agent_dict['route'][0][0].transform.location, 6):
                        print("deviates")
                        destination = agent_dict['route'][-1][0].transform.location
                        agent_dict['route'] = planner.trace_route(agent_dict['agent'].get_location(), destination)
                    
                    # Delete reached points from current route
                    while check_close(agent_dict['agent'].get_location(), agent_dict['route'][0][0].transform.location):
                        agent_dict['route'].pop(0)
                        if len(agent_dict['route']) == 0:
                            agent_dict['done'] = 1
                            new_destination = random.choice(spawn_points).location
                            agent_dict['route'] = planner.trace_route(agent_dict['agent'].get_location(), new_destination)

                    # Generate new destination if the current one is close 
                    if len(agent_dict['route']) < 10:
                        agent_dict['done'] = 1
                        new_destination = random.choice(spawn_points).location
                        new_route = planner.trace_route(agent_dict['route'][-1][0].transform.location, new_destination)
                        temp_route = agent_dict['route'] + new_route
                        agent_dict['route'] = temp_route

                #############################################################
                # TODO: Need to prepare input for e2e driving model -> tick()
                #                                                           #
                #                                                           #
                #                                                           #
                #############################################################
                t_list = []
                all_agent_list = interactive_agent_list + ego_agent_list
                for agent_dict in all_agent_list:
                    if agent_dict['name'] == 'roach':
                        route_list = [wp[0].transform.location for wp in agent_dict['route'][0:60]]
                        start_time = time.time()
                        inputs = [route_list, agent_dict]
                        agent_dict['model'].set_data(processed_data)
                    
                    if agent_dict['name'] == 'e2e':
                        # TODO: Prepare sensor data -> tick()
                        pass
                    
                    if agent_dict['name'] == "auto":
                        route_list = [wp for wp in agent_dict['route'][0:60]]
                        agent_dict['model'].tick(agent_dict)
                        inputs = [route_list, agent_dict]


                    t = Thread(target=agent_dict['model'].run_step, args=tuple(inputs))
                    t_list.append(t)
                    t.start()
                
                for t in t_list:
                    t.join()

                start_time = time.time()
                for agent_dict in all_agent_list:
                    control_elements_list = agent_dict["control"] 
                    control_elements = control_elements_list[0]
                    control = carla.VehicleControl(throttle=control_elements['throttle'], steer=control_elements['steer'], brake=control_elements['brake'])
                    agent_dict["agent"].apply_control(control)

                world.render(display)
                
                pygame.display.flip()
                
                scene_done = 1
                for agent_dict in all_agent_list:
                    scene_done = scene_done and agent_dict['done']
                if scene_done:
                    break
                
            
            
            print("destroy env")
            # Destroy all agent, world and HUD
            for agent_dict in all_agent_list:
                agent_dict['agent'].destroy()
            del all_agent_list
            del ego_agent_list
            del interactive_agent_list                
            world.destroy()
            del client
            del world
            del hud
            del global_roach
            del global_roach_policy
            
            pygame.quit()
                


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================
def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='512x512',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='model3',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--distance',
        default=25.0,
        type=float,
        help='distance to intersection for toggling camera)')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--eval_config',
        default='eval_config.json',
        type=str,
        help='path to evaluation config')
    argparser.add_argument(
        '--ego_agent',
        default='roach',
        type=str,
        help='select ego agent type')
    # TODO: change to agent entry_point later.
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    # print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

if __name__ == '__main__':

    main()