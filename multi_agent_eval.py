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
from multiprocessing import Process
import json
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

from collections import deque

from bird_eye_view.BirdViewProducer import BirdViewProducer, BirdView
from bird_eye_view.Mask import PixelDimensions, Loc

################


from threading import Thread
import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import time
import logging
import math
import random
import re
import weakref
from auto_random_actors import spawn_actor_nearby
import cv2
import pygame
# from auto_light import *

import datetime

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from carla_gym.core.obs_manager.obs_manager_handler import ObsManagerHandler
from carla_gym.core.task_actor.ego_vehicle.ego_vehicle_handler import EgoVehicleHandler
from carla_gym.utils import config_utils
from carla_gym.utils.traffic_light import TrafficLightHandler
from navigation.global_route_planner import GlobalRoutePlanner
from multiprocessing import Process, Pool
import copy
from random_actors import spawn_actor_assigned

import torch.multiprocessing as mp
# import torch.threading as threading
import threading
import gym
from omegaconf import DictConfig, OmegaConf
from importlib import import_module
import wandb
def load_entry_point(name):
    mod_name, attr_name = name.split(":") #agents.rl_birdview.rl_birdview_agent:RlBirdviewAgent
    mod = import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


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
        print(blueprint)
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





            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.show_vehicle_telemetry = False
            self.modify_vehicle_physics(self.player)
        # Set up the sensors.
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        
        
        # self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        # self.camera_manager.transform_index = cam_pos_index
        # self.camera_manager.set_sensor(cam_index, notify=False)
        
        
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

    # def toggle_radar(self):
    #     if self.radar_sensor is None:
    #         self.radar_sensor = RadarSensor(self.player)
    #     elif self.radar_sensor.sensor is not None:
    #         self.radar_sensor.sensor.destroy()
    #         self.radar_sensor = None

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock, frame, display, stored_path):
        end = self.hud.tick(self, clock, self.camera_manager, frame, display, stored_path)
        return end
    def render(self, display):
        # self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        # self.camera_manager.sensor.destroy()
        # self.camera_manager.sensor = None
        # self.camera_manager.index = None
        pass

    def destroy(self):
        # if self.radar_sensor is not None:
        #     self.toggle_radar()

        
        
        sensors = [
            # self.camera_manager.sensor_rgb_front,
            # self.camera_manager.sensor_ss_front,


            self.gnss_sensor.sensor,
            self.imu_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    """Class that handles keyboard input."""
    def __init__(self, world, start_in_autopilot):
        #self._autopilot_enabled = start_in_autopilot
        self._autopilot_enabled = False
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)



# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height, distance=25.0, town='Town05', stored_path='', v_id=1):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 16), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self.recording = False
        self.recording_frame = 0
        self.stored_path = stored_path
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

    
                                           
    def tick(self, world, clock, camera, frame, display, root_path):
        self._notifications.tick(world, clock)
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
        self._info_text += [
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

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
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

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
        self._notifications.render(display)
        self.help.render(display)


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
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
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
        self.sensor_rgb_front = None

        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self.save_mode = save_mode

        self.rgb_front = None
        self.ss_front = None



        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        Attachment = carla.AttachmentType

        if not self._parent.type_id.startswith("walker.pedestrian"):
            self._camera_transforms = [
                # front view
                (carla.Transform(carla.Location(x=+0.8*bound_x,
                 y=+0.0*bound_y, z=1.3*bound_z)), Attachment.Rigid),
                # front-left view
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y,
                 z=1.3*bound_z), carla.Rotation(yaw=-55)), Attachment.Rigid),
                # front-right view
                (carla.Transform(carla.Location(x=+0.8*bound_x, y=+0.0*bound_y,
                 z=1.3*bound_z), carla.Rotation(yaw=55)), Attachment.Rigid),
                # back view
                (carla.Transform(carla.Location(x=-0.8*bound_x, y=+0.0*bound_y,
                 z=1.3*bound_z), carla.Rotation(yaw=180)), Attachment.Rigid),
                # back-left view
                (carla.Transform(carla.Location(x=-0.8*bound_x, y=+0.0*bound_y,
                 z=1.3*bound_z), carla.Rotation(yaw=235)), Attachment.Rigid),
                # back-right view
                (carla.Transform(carla.Location(x=-0.8*bound_x, y=+0.0*bound_y,
                 z=1.3*bound_z), carla.Rotation(yaw=-235)), Attachment.Rigid),
                # top view
                (carla.Transform(carla.Location(x=-0.8*bound_x, y=+0.0*bound_y,
                 z=23*bound_z), carla.Rotation(pitch=18.0)), Attachment.SpringArm),
                # LBC top view
                # (carla.Transform(carla.Location(x=0, y=0,
                #  z=25.0), carla.Rotation(pitch=-90.0)), Attachment.SpringArm),
                (carla.Transform(carla.Location(x=0, y=0,
                 z=20.0), carla.Rotation(pitch=-90.0)), Attachment.SpringArm),
                
                # sensor config for transfuser camera settings 
                #  front view 8
                (carla.Transform(carla.Location(x=1.3, y=0,
                 z=2.3), carla.Rotation(roll=0.0, pitch=0.0, yaw=0.0)), Attachment.Rigid),
                # left view  9 
                (carla.Transform(carla.Location(x=1.3, y=0,
                 z=2.3), carla.Rotation(roll=0.0, pitch=0.0, yaw=-60.0)), Attachment.Rigid),
                # right view 10
                (carla.Transform(carla.Location(x=1.3, y=0,
                 z=2.3), carla.Rotation(roll=0.0, pitch=0.0, yaw=60.0)), Attachment.Rigid),
                # rear 11 
                (carla.Transform(carla.Location(x=-1.3, y=0,
                 z=2.3), carla.Rotation(roll=0.0, pitch=0.0, yaw=180.0)), Attachment.Rigid),
                # rear left 12 
                (carla.Transform(carla.Location(x=-1.3, y=0,
                 z=2.3), carla.Rotation(roll=0.0, pitch=0.0, yaw=-120.0)), Attachment.Rigid),
                # rear right 13
                (carla.Transform(carla.Location(x=-1.3, y=0,
                 z=2.3), carla.Rotation(roll=0.0, pitch=0.0, yaw=120.0)), Attachment.Rigid)
            ]
        else:
            pass
            # self._camera_transforms = [
            #     (carla.Transform(carla.Location(x=-5.5, z=2.5),
            #      carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            #     (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            #     (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)),
            #      Attachment.SpringArm),
            #     (carla.Transform(carla.Location(x=-8.0, z=6.0),
            #      carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            #     (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]

        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth,
                'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.Raw,
                'Camera Semantic Segmentation (Raw)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)', {}],
            ['sensor.lidar.ray_cast', None,
                'Lidar (Ray-Cast)', {'range': '85', 'rotation_frequency': '25'}],
            ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
            ['sensor.camera.optical_flow', None, 'Optical Flow', {}],
            ['sensor.other.lane_invasion', None, 'Lane lane_invasion', {}],
            ['sensor.camera.instance_segmentation', cc.CityScapesPalette,
                'Camera Instance Segmentation (CityScapes Palette)', {}],
        ]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()

        # self.bev_bp = bp_library.find('sensor.camera.rgb')
        # self.bev_bp.set_attribute('image_size_x', str(300))
        # self.bev_bp.set_attribute('image_size_y', str(300))
        # self.bev_bp.set_attribute('fov', str(50.0))
        # if self.bev_bp.has_attribute('gamma'):
        #     self.bev_bp.set_attribute('gamma', str(gamma_correction))
        
        
        # self.sensor_rgb_bp = bp_library.find('sensor.camera.rgb')
        # self.sensor_rgb_bp.set_attribute('image_size_x', str(400))
        # self.sensor_rgb_bp.set_attribute('image_size_y', str(300))
        # self.sensor_rgb_bp.set_attribute('fov', str(100.0))
        
        self.sensor_rgb_bp = bp_library.find('sensor.camera.rgb')
        self.sensor_rgb_bp.set_attribute('image_size_x', str(512))
        self.sensor_rgb_bp.set_attribute('image_size_y', str(512))
        self.sensor_rgb_bp.set_attribute('fov', str(60.0))
        
        # self.sensor_ss_bp = bp_library.find('sensor.camera.instance_segmentation')
        self.sensor_ss_bp = bp_library.find('sensor.camera.semantic_segmentation')
        # 
        self.sensor_ss_bp.set_attribute('image_size_x', str(512))
        self.sensor_ss_bp.set_attribute('image_size_y', str(512))
        self.sensor_ss_bp.set_attribute('fov', str(60.0))
        
        self.sensor_depth_bp = bp_library.find('sensor.camera.depth')
        self.sensor_depth_bp.set_attribute('image_size_x', str(1280))
        self.sensor_depth_bp.set_attribute('image_size_y', str(720))
        self.sensor_depth_bp.set_attribute('fov', str(60.0))
        

        self.bev_seg_bp = bp_library.find('sensor.camera.instance_segmentation')
        self.bev_seg_bp.set_attribute('image_size_x', str(400))
        self.bev_seg_bp.set_attribute('image_size_y', str(400))
        self.bev_seg_bp.set_attribute('fov', str(50.0))

        self.front_cam_bp = bp_library.find('sensor.camera.rgb')
        self.front_cam_bp.set_attribute('image_size_x', str(768))
        self.front_cam_bp.set_attribute('image_size_y', str(256))
        self.front_cam_bp.set_attribute('fov', str(120.0))
        self.front_cam_bp.set_attribute('lens_circle_multiplier', '0.0')
        self.front_cam_bp.set_attribute('lens_circle_falloff', '0.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_intensity', '3.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_offset', '500')
        # self.front_cam_bp.set_attribute('focal_distance', str(500))
        if self.front_cam_bp.has_attribute('gamma'):
            self.front_cam_bp.set_attribute('gamma', str(gamma_correction))

        self.front_seg_bp = bp_library.find('sensor.camera.instance_segmentation')
        self.front_seg_bp.set_attribute('image_size_x', str(768))
        self.front_seg_bp.set_attribute('image_size_y', str(256))
        self.front_seg_bp.set_attribute('fov', str(120.0))
        self.front_cam_bp.set_attribute('lens_circle_multiplier', '0.0')
        self.front_cam_bp.set_attribute('lens_circle_falloff', '0.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_intensity', '3.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_offset', '500')
        # self.front_seg_bp.set_attribute('focal_distance', str(500))
        if self.front_seg_bp.has_attribute('gamma'):
            self.front_seg_bp.set_attribute('gamma', str(gamma_correction))

        self.depth_bp = bp_library.find('sensor.camera.depth')
        self.depth_bp.set_attribute('image_size_x', str(768))
        self.depth_bp.set_attribute('image_size_y', str(256))
        self.depth_bp.set_attribute('fov', str(120.0))
        self.front_cam_bp.set_attribute('lens_circle_multiplier', '0.0')
        self.front_cam_bp.set_attribute('lens_circle_falloff', '0.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_intensity', '3.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_offset', '500')
        # self.depth_bp.set_attribute('focal_distance', str(500))
        if self.depth_bp.has_attribute('gamma'):
            self.depth_bp.set_attribute('gamma', str(gamma_correction))

        # self.flow_bp = bp_library.find('sensor.camera.optical_flow')
        # self.flow_bp.set_attribute('image_size_x', str(1236))
        # self.flow_bp.set_attribute('image_size_y', str(256))
        # self.flow_bp.set_attribute('fov', str(120.0))
        self.front_cam_bp.set_attribute('lens_circle_multiplier', '0.0')
        self.front_cam_bp.set_attribute('lens_circle_falloff', '0.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_intensity', '3.0')
        self.front_cam_bp.set_attribute('chromatic_aberration_offset', '500')
        # self.flow_bp.set_attribute('focal_distance', str(500))
        # if self.flow_bp.has_attribute('gamma'):
        #     self.flow_bp.set_attribute('gamma', str(gamma_correction))
        for item in self.sensors:
            
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))
                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
            elif item[0].startswith('sensor.lidar'):
                self.lidar_range = 50

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)
                    if attr_name == 'range':
                        self.lidar_range = float(attr_value)

            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index +
                                1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index]
             [2] != self.sensors[self.index][2]))
        if needs_respawn:
            if self.sensor_rgb_front is not None:
                self.sensor_rgb_front.destroy()
                self.surface = None

            # rgb sensor
            if self.save_mode:
                ## setup sensors [ tf sensors  ( total 6 * 3 sensors ) ] 
                # rgb 
                self.sensor_rgb_front = self._parent.get_world().spawn_actor(
                    self.sensor_rgb_bp,
                    self._camera_transforms[8][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])
                
                # ss
                self.sensor_ss_front = self._parent.get_world().spawn_actor(
                    self.sensor_ss_bp,
                    self._camera_transforms[7][0],
                    attach_to=self._parent,
                    attachment_type=self._camera_transforms[0][1])

            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)

            if self.save_mode:
                self.sensor_rgb_front.listen(lambda image: CameraManager._parse_image(weak_self, image, 'rgb_front'))
                self.sensor_ss_front.listen(lambda image: CameraManager._parse_image(weak_self, image, 'ss_front'))
                
                           

                
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)
        
    def render(self, display):
        if self.surface is not None:
            
            # print(self.surface)
            
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image, view='top'):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / (2.0 * self.lidar_range)
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)

        elif view == 'rgb_front':
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (512, 512, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            # render the view shown in monitor
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        #if self.recording and image.frame % 1 == 0:
        if True:
            # if view == 'front':
            #     self.front_img = image

            ## tf sensors  ( total 6 * 3 sensors )
            if view == 'rgb_front':
                self.rgb_front = image
            elif view == 'rgb_left':
                self.rgb_left = image
            elif view == 'rgb_right':
                self.rgb_right = image
                
            elif view == 'rgb_rear':
                self.rgb_rear = image
            elif view == 'rgb_rear_left':
                self.rgb_rear_left = image                
            elif view == 'rgb_rear_right':
                self.rgb_rear_right = image
            elif view == 'depth_front':
                self.depth_front = image
            elif view == 'depth_left':
                self.depth_left = image
            elif view == 'depth_right':
                self.depth_right = image
            elif view == 'depth_rear':
                self.depth_rear = image
            elif view == 'depth_rear_left':
                self.depth_rear_left = image
            elif view == 'depth_rear_right':
                self.depth_rear_right = image
            elif view == 'ss_front':
                self.ss_front = image
            elif view == 'ss_left':
                self.ss_left = image
            elif view == 'ss_right':
                self.ss_right = image
            elif view == 'ss_rear':
                self.ss_rear = image
            elif view == 'ss_rear_left':
                self.ss_rear_left = image
            elif view == 'ss_rear_right':
                self.ss_rear_right = image
            
            
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
    
def _get_traffic_light_waypoints(traffic_light, carla_map):
    """
    get area of a given traffic light
    adapted from "carla-simulator/scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_criteria.py"
    """
    base_transform = traffic_light.get_transform()
    tv_loc = traffic_light.trigger_volume.location
    tv_ext = traffic_light.trigger_volume.extent

    # Discretize the trigger box into points
    x_values = np.arange(-0.9 * tv_ext.x, 0.9 * tv_ext.x, 1.0)  # 0.9 to avoid crossing to adjacent lanes
    area = []
    for x in x_values:
        point_location = base_transform.transform(tv_loc + carla.Location(x=x))
        area.append(point_location)

    # Get the waypoints of these points, removing duplicates
    ini_wps = []
    for pt in area:
        wpx = carla_map.get_waypoint(pt)
        # As x_values are arranged in order, only the last one has to be checked
        if not ini_wps or ini_wps[-1].road_id != wpx.road_id or ini_wps[-1].lane_id != wpx.lane_id:
            ini_wps.append(wpx)

    # Leaderboard: Advance them until the intersection
    stopline_wps = []
    stopline_vertices = []
    junction_wps = []
    for wpx in ini_wps:
        # Below: just use trigger volume, otherwise it's on the zebra lines.
        # stopline_wps.append(wpx)
        # vec_forward = wpx.transform.get_forward_vector()
        # vec_right = carla.Vector3D(x=-vec_forward.y, y=vec_forward.x, z=0)

        # loc_left = wpx.transform.location - 0.4 * wpx.lane_width * vec_right
        # loc_right = wpx.transform.location + 0.4 * wpx.lane_width * vec_right
        # stopline_vertices.append([loc_left, loc_right])

        while not wpx.is_intersection:
            next_wp = wpx.next(0.5)[0]
            if next_wp and not next_wp.is_intersection:
                wpx = next_wp
            else:
                break
        junction_wps.append(wpx)

        stopline_wps.append(wpx)
        vec_forward = wpx.transform.get_forward_vector()
        vec_right = carla.Vector3D(x=-vec_forward.y, y=vec_forward.x, z=0)

        loc_left = wpx.transform.location - 0.4 * wpx.lane_width * vec_right
        loc_right = wpx.transform.location + 0.4 * wpx.lane_width * vec_right
        stopline_vertices.append([loc_left, loc_right])

    # all paths at junction for this traffic light
    junction_paths = []
    path_wps = []
    wp_queue = deque(junction_wps)
    while len(wp_queue) > 0:
        current_wp = wp_queue.pop()
        path_wps.append(current_wp)
        next_wps = current_wp.next(1.0)
        for next_wp in next_wps:
            if next_wp.is_junction:
                wp_queue.append(next_wp)
            else:
                junction_paths.append(path_wps)
                path_wps = []

    return carla.Location(base_transform.transform(tv_loc)), stopline_wps, stopline_vertices, junction_paths

    

class BEV_MAP():
    def __init__(self, args) -> None:
        
        self.args = args
        
        
        self.data = None
        self.birdview_producer = BirdViewProducer(
                args.town, 
                # PixelDimensions(width=512, height=512), 
                PixelDimensions(width=304, height=304), 
                pixels_per_meter=5)
        
        
       
        # init Roach-model 
        self.model = None # load roach model 

        self.vehicle_bbox_1_16 = {}
        self.pedestrain_bbox_1_16 = {}

        self.Y_bbox_1_16 = {}
        self.G_bbox_1_16 = {}
        self.R_bbox_1_16 = {}

    def init_policy(self, policy = None):    # prepare policy
        # inti roach model 
        wandb_path = "sonicokuo/cata_train_rl_FixSeed/o6gx5mka"
        if wandb_path[-1] == '\n':
               wandb_path = wandb_path[:-1]
        api = wandb.Api()
        run = api.run(wandb_path)
        all_ckpts = [f for f in run.files() if 'ckpt' in f.name] #load a check points
        f = max(all_ckpts, key=lambda x: int(x.name.split('_')[1].split('.')[0]))
        print(f'Resume checkpoint latest {f.name}')
        f.download(replace=True)
        run.file('config_agent.yaml').download(replace=True)
        cfg = OmegaConf.load('config_agent.yaml')
        cfg = OmegaConf.to_container(cfg)
        self._ckpt = f.name

        ckpt = f.name

        train_cfg = cfg['training'] 

        policy_class = load_entry_point(cfg['policy']['entry_point'])
        policy_kwargs = cfg['policy']['kwargs']
        print(f'Loading wandb checkpoint: {ckpt}')
        policy, train_cfg['kwargs'] = policy_class.load(ckpt)
        policy = policy.eval()

        self._wrapper_class = load_entry_point(cfg['env_wrapper']['entry_point'])
        self._wrapper_kwargs = cfg['env_wrapper']['kwargs']
        self.policy = policy

        return policy
    def set_policy(self, processed_policy):
        policy = copy.deepcopy(processed_policy)
        self.policy = policy

    def init_vehicle_bbox(self, ego_id):
        self.vehicle_bbox_1_16[ego_id] = []
        self.pedestrain_bbox_1_16[ego_id] = []
        self.Y_bbox_1_16[ego_id] = []
        self.G_bbox_1_16[ego_id] = []
        self.R_bbox_1_16[ego_id] = []

    def collect_actor_data(self, world):
        vehicles_id_list = []
        bike_blueprint = ["vehicle.bh.crossbike","vehicle.diamondback.century","vehicle.gazelle.omafiets"]
        motor_blueprint = ["vehicle.harley-davidson.low_rider","vehicle.kawasaki.ninja","vehicle.yamaha.yzf","vehicle.vespa.zx125"]
        
        def get_xyz(method, rotation=False):

            if rotation:
                roll = method.roll
                pitch = method.pitch
                yaw = method.yaw
                return {"pitch": pitch, "yaw": yaw, "roll": roll}

            else:
                x = method.x
                y = method.y
                z = method.z

                # return x, y, z
                return {"x": x, "y": y, "z": z}

        ego_loc = world.player.get_location()
        data = {}

        vehicles = world.world.get_actors().filter("*vehicle*")
        for actor in vehicles:

            _id = actor.id
            actor_loc = actor.get_location()
            location = get_xyz(actor_loc)
            rotation = get_xyz(actor.get_transform().rotation, True)

            cord_bounding_box = {}
            bbox = actor.bounding_box
            if actor.type_id in motor_blueprint:
                bbox.extent.x = 1.177870
                bbox.extent.y = 0.381839
                bbox.extent.z = 0.75
                bbox.location = carla.Location(0, 0, bbox.extent.z)
            elif actor.type_id in bike_blueprint:
                bbox.extent.x = 0.821422
                bbox.extent.y = 0.186258
                bbox.extent.z = 0.9
                bbox.location = carla.Location(0, 0, bbox.extent.z)
                
            verts = [v for v in bbox.get_world_vertices(
                actor.get_transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1

            distance = ego_loc.distance(actor_loc)


            # if distance < 50:
            vehicles_id_list.append(_id)

            acceleration = get_xyz(actor.get_acceleration())
            velocity = get_xyz(actor.get_velocity())
            angular_velocity = get_xyz(actor.get_angular_velocity())

            v = actor.get_velocity()

            speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)

            vehicle_control = actor.get_control()
            control = {
                "throttle": vehicle_control.throttle,
                "steer": vehicle_control.steer,
                "brake": vehicle_control.brake,
                "hand_brake": vehicle_control.hand_brake,
                "reverse": vehicle_control.reverse,
                "manual_gear_shift": vehicle_control.manual_gear_shift,
                "gear": vehicle_control.gear
            }


            data[_id] = {}
            data[_id]["location"] = location
            data[_id]["rotation"] = rotation
            data[_id]["distance"] = distance
            data[_id]["acceleration"] = acceleration
            data[_id]["velocity"] = velocity
            data[_id]["vel_x"] = v.x
            data[_id]["vel_y"] = v.y
            data[_id]["speed"] = speed
            data[_id]["angular_velocity"] = angular_velocity
            data[_id]["control"] = control
            data[_id]["cord_bounding_box"] = cord_bounding_box
            data[_id]["type"] = "vehicle"

        pedestrian_id_list = []

        walkers = world.world.get_actors().filter("*pedestrian*")
        for actor in walkers:

            _id = actor.id

            actor_loc = actor.get_location()
            location = get_xyz(actor_loc)
            rotation = get_xyz(actor.get_transform().rotation, True)

            cord_bounding_box = {}
            bbox = actor.bounding_box
            verts = [v for v in bbox.get_world_vertices(
                actor.get_transform())]
            counter = 0
            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1

            distance = ego_loc.distance(actor_loc)

            # if distance < 50:
            pedestrian_id_list.append(_id)

            acceleration = get_xyz(actor.get_acceleration())
            velocity = get_xyz(actor.get_velocity())
            angular_velocity = get_xyz(actor.get_angular_velocity())

            walker_control = actor.get_control()
            control = {"direction": get_xyz(walker_control.direction),
                       "speed": walker_control.speed, "jump": walker_control.jump}

            data[_id] = {}
            data[_id]["location"] = location
            # data[_id]["rotation"] = rotation
            data[_id]["distance"] = distance
            data[_id]["acceleration"] = acceleration
            data[_id]["velocity"] = velocity
            data[_id]["angular_velocity"] = angular_velocity
            data[_id]["control"] = control

            data[_id]["cord_bounding_box"] = cord_bounding_box
            data[_id]["type"] = 'pedestrian'


        
        
        traffic_id_list = []
        lights = world.world.get_actors().filter("*traffic_light*")
        for actor in lights:
            tv_loc, stopline_wps, stopline_vtx, junction_paths = _get_traffic_light_waypoints( actor, world.world.get_map())
            # tv_loc
            # stopline_vtx
            # print(stopline_vtx)
            if len(stopline_vtx) < 2:
                continue
            _id = actor.id
            traffic_id_list.append(_id)
            traffic_light_state = int(actor.state)  # traffic light state
            cord_bounding_box = {}
            cord_bounding_box["cord_0"] = [stopline_vtx[0][0].x, stopline_vtx[0][0].y, stopline_vtx[0][0].z]
            cord_bounding_box["cord_1"] = [stopline_vtx[1][0].x, stopline_vtx[1][0].y, stopline_vtx[1][0].z]
            cord_bounding_box["cord_2"] = [stopline_vtx[0][1].x, stopline_vtx[0][1].y, stopline_vtx[0][1].z]
            cord_bounding_box["cord_3"] = [stopline_vtx[1][1].x, stopline_vtx[1][1].y, stopline_vtx[1][1].z]


            data[_id] = {}
            data[_id]["state"] = traffic_light_state
            data[_id]["cord_bounding_box"] = cord_bounding_box


        

        obstacle_id_list = []

        obstacle = world.world.get_actors().filter("*static.prop*")
        for actor in obstacle:

            _id = actor.id

            actor_loc = actor.get_location()
            distance = ego_loc.distance(actor_loc)

            for loc in verts:
                cord_bounding_box["cord_"+str(counter)] = [loc.x, loc.y, loc.z]
                counter += 1

            # if distance < 50:
            obstacle_id_list.append(_id)

            data[_id] = {}
            data[_id]["distance"] = distance
            data[_id]["type"] = "obstacle"
            data[_id]["cord_bounding_box"] = cord_bounding_box


        data["traffic_light_ids"] = traffic_id_list

        data["obstacle_ids"] = obstacle_id_list
        data["vehicles_ids"] = vehicles_id_list
        data["pedestrian_ids"] = pedestrian_id_list

        self.data = data
        return data

    def set_data(self, proccessed_data):
        self.data = proccessed_data

    def run_step(self, result, frame, ego_id, route_list = None):
        start_time = time.time()
        actor_dict =  copy.deepcopy(self.data)

        ego_loc = carla.Location(x=actor_dict[ego_id]["location"]["x"], y=actor_dict[ego_id]["location"]["y"])
        ego_pos = Loc(x=actor_dict[ego_id]["location"]["x"], y=actor_dict[ego_id]["location"]["y"])
        ego_yaw = actor_dict[ego_id]["rotation"]["yaw"]


        obstacle_bbox_list = []
        pedestrian_bbox_list = []
        vehicle_bbox_list = []
        agent_bbox_list = []

        r_traffic_light_list = []
        g_traffic_light_list = []
        y_traffic_light_list = []

        # interactive id 
        vehicle_id_list = list(actor_dict["vehicles_ids"])
        # pedestrian id list 
        pedestrian_id_list = list(actor_dict["pedestrian_ids"])
        # obstacle id list 
        obstacle_id_list = list(actor_dict["obstacle_ids"])

        # traffic light id list 

        traffic_light_id_list = list(actor_dict["traffic_light_ids"])

        for id in traffic_light_id_list:
            traffic_light_loc = carla.Location(x=actor_dict[id]["cord_bounding_box"]["cord_0"][0], y=actor_dict[id]["cord_bounding_box"]["cord_0"][1])
            if not check_close(ego_loc, traffic_light_loc, 45): #35.95m
                continue
            pos_0 = actor_dict[id]["cord_bounding_box"]["cord_0"]
            pos_1 = actor_dict[id]["cord_bounding_box"]["cord_1"]
            pos_2 = actor_dict[id]["cord_bounding_box"]["cord_2"]
            pos_3 = actor_dict[id]["cord_bounding_box"]["cord_3"]
            # obstacle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
            #                             Loc(x=pos_1[0], y=pos_1[1]), 
            #                             Loc(x=pos_2[0], y=pos_2[1]), 
            #                             Loc(x=pos_3[0], y=pos_3[1]), 
            #                             ])
            # print(
            # # 0 - R
            # # 1 - Y
            # # 2 - G
            
            if actor_dict[id]["state"] == 0 :
                g_traffic_light_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ])
            elif actor_dict[id]["state"] == 2:
                g_traffic_light_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ])  
            elif actor_dict[id]["state"] == 1:
                g_traffic_light_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ])  
            



        for id in obstacle_id_list:
            pos_0 = actor_dict[id]["cord_bounding_box"]["cord_0"]
            pos_1 = actor_dict[id]["cord_bounding_box"]["cord_4"]
            pos_2 = actor_dict[id]["cord_bounding_box"]["cord_6"]
            pos_3 = actor_dict[id]["cord_bounding_box"]["cord_2"]
            obstacle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ])
            
        for id in vehicle_id_list:
            traffic_light_loc = carla.Location(x=actor_dict[id]['location']['x'], y=actor_dict[id]['location']['y'])
            # distance filter
            # if not check_close(ego_loc, traffic_light_loc, 36): #35.95m
            #     continue
            pos_0 = actor_dict[id]["cord_bounding_box"]["cord_0"]
            pos_1 = actor_dict[id]["cord_bounding_box"]["cord_4"]
            pos_2 = actor_dict[id]["cord_bounding_box"]["cord_6"]
            pos_3 = actor_dict[id]["cord_bounding_box"]["cord_2"]
            

            if int(id) == int(ego_id):
                agent_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ])
            else:
                vehicle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ])
                    
        for id in pedestrian_id_list:
            pos_0 = actor_dict[id]["cord_bounding_box"]["cord_0"]
            pos_1 = actor_dict[id]["cord_bounding_box"]["cord_4"]
            pos_2 = actor_dict[id]["cord_bounding_box"]["cord_6"]
            pos_3 = actor_dict[id]["cord_bounding_box"]["cord_2"]

            pedestrian_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ])
            


        if len(self.pedestrain_bbox_1_16[ego_id]) < 16:
            self.pedestrain_bbox_1_16[ego_id].append(pedestrian_bbox_list)
        else:
            self.pedestrain_bbox_1_16[ego_id].pop(0)
            self.pedestrain_bbox_1_16[ego_id].append(pedestrian_bbox_list)

        if len(self.vehicle_bbox_1_16[ego_id]) < 16:
            self.vehicle_bbox_1_16[ego_id].append(vehicle_bbox_list)
        else:
            self.vehicle_bbox_1_16[ego_id].pop(0)
            self.vehicle_bbox_1_16[ego_id].append(vehicle_bbox_list)


        if len(self.Y_bbox_1_16[ego_id]) < 16:
            self.Y_bbox_1_16[ego_id].append(y_traffic_light_list)
        else:
            self.Y_bbox_1_16[ego_id].pop(0)
            self.Y_bbox_1_16[ego_id].append(y_traffic_light_list)


        if len(self.G_bbox_1_16[ego_id]) < 16:
            self.G_bbox_1_16[ego_id].append(g_traffic_light_list)
        else:
            self.G_bbox_1_16[ego_id].pop(0)
            self.G_bbox_1_16[ego_id].append(g_traffic_light_list)

        if len(self.R_bbox_1_16[ego_id]) < 16:
            self.R_bbox_1_16[ego_id].append(r_traffic_light_list)
        else:
            self.R_bbox_1_16[ego_id].pop(0)
            self.R_bbox_1_16[ego_id].append(r_traffic_light_list)


        # if self.vehicle_bbox_1_16:

        birdview: BirdView = self.birdview_producer.produce(ego_pos, ego_yaw,
                                                       agent_bbox_list, 
                                                       self.vehicle_bbox_1_16[ego_id],
                                                       self.pedestrain_bbox_1_16[ego_id],
                                                       self.R_bbox_1_16[ego_id],
                                                       self.G_bbox_1_16[ego_id],
                                                       self.Y_bbox_1_16[ego_id],
                                                       obstacle_bbox_list,
                                                       route_list)
    
        # print(birdview.shape)



        camera_pos = Loc(x=assigned_location_dict['center'][0], y=assigned_location_dict['center'][1]+10)
        camera_yaw = -90
        camera: BirdView = self.birdview_producer.produce(camera_pos, camera_yaw,
                                                       agent_bbox_list, 
                                                       self.vehicle_bbox_1_16[ego_id],
                                                       self.pedestrain_bbox_1_16[ego_id],
                                                       self.R_bbox_1_16[ego_id],
                                                       self.G_bbox_1_16[ego_id],
                                                       self.Y_bbox_1_16[ego_id],
                                                       obstacle_bbox_list,
                                                       route_list)        
        # topdown = BirdViewProducer.as_rgb(birdview)
        topdown = BirdViewProducer.as_rgb(camera)

        # input BEV representation
        roach_obs, new_array =  BirdViewProducer.as_roach_input(birdview)

        # input state   
        throttle =  actor_dict[ego_id]["control"]["throttle"]
        steer =  actor_dict[ego_id]["control"]["steer"]
        brake =  actor_dict[ego_id]["control"]["brake"]
        gear =  actor_dict[ego_id]["control"]["gear"]
        vel_x =  actor_dict[ego_id]["vel_x"]
        vel_y =  actor_dict[ego_id]["vel_y"]
        # throttle, steer, brake, gear/5.0 , vel_x, vel_y

        # combine input


        # birdview = np.expand_dims(birdview, 0)
        # state = np.expand_dims(state, 0)
        
        # obs_dict = {
        #     'state': state.astype(np.float32),
        #     'birdview': birdview
        # }

        policy_input = {}
        policy_input['state'] = np.expand_dims([throttle, steer, brake, gear/5, vel_x, vel_y], 0).astype(np.float32)
        # print(policy_input['state'])
        policy_input['birdview'] = np.expand_dims(roach_obs, 0)

       
        # return roach_obs
        result["bev_map_rgb"] = topdown[0:192, 56:248]
        # result["bev_map_rgb"] = new_array
        result["policy_input"] = policy_input
        # return topdown[0:192, 56:248], policy_input
        # return new_array, policy_input
        self.policy_forward(result, [policy_input])
        # print("finish one both", ego_id ,start_time - time.time(), frame - time.time())
        
    def policy_forward(self, result, policy_inputs_list):    
        control_elements_list = []
        for policy_input in policy_inputs_list:
            actions, values, log_probs, mu, sigma, features = self.policy.forward(
                policy_input, deterministic=True, clip_action=True)
                     
            # print(actions)
            for action in actions:
                acc, steer = action.astype(np.float64)
                if acc >= 0.0:
                    throttle = acc
                    brake = 0.0
                else:
                    throttle = 0.0
                    brake = np.abs(acc)

                throttle = np.clip(throttle, 0, 0.5)
                steer = np.clip(steer, -1, 1)
                brake = np.clip(brake, 0, 1)
                control_elements = {}
                control_elements['throttle'] = throttle
                control_elements['steer'] = steer
                control_elements['brake'] = brake
                control_elements_list.append(control_elements)
        result["control_elements_list"] = control_elements_list
        # return control_elements_list
        # return contorl 
        
        
# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================

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

def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None

    #try:
    if True:
    
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        # display = pygame.display.set_mode(
        #     (args.width, args.height),
        #     pygame.HWSURFACE | pygame.DOUBLEBUF)
        
        
        display = pygame.display.set_mode(
            (512, 900),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        display.fill((0,0,0))
        pygame.display.flip()


        stored_path = os.path.join('data_collection', args.town)
        # if not os.path.exists(stored_path) :
        #     os.makedirs(stored_path)


        hud = HUD(args.width, args.height, args.distance, args.town, stored_path)
        world = World(client.load_world(args.town), hud, args)

        settings = world.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True  # Enables synchronous mode
        world.world.apply_settings(settings)

        controller = KeyboardControl(world, args.autopilot)
        
            
        
        e2e_vehicles_list = []
        e2e_agent_dict = {}
        # spawn other agent 
        map = world.world.get_map()
        spawn_points = map.get_spawn_points()
        
        # waypoint_list = []
        
        # for waypoint in spawn_points:            
        #     waypoint_list.append(waypoint)
        # random.shuffle(waypoint_list)
        # blueprints = world.world.get_blueprint_library().filter('vehicle.*')

        # actor_numbers = args.n
        # actor_numbers = 4

        ############## set up agent spawnpoints and destinations by parsing the location.txt ###############
        with open('locations.txt') as f:
            assigned_spawns_dests = f.readlines()
        for i in range(len(assigned_spawns_dests)):
            assigned_spawns_dests[i] = assigned_spawns_dests[i].strip()
            # print(assigned_spawns_dests[i])

        # remove empty string
        assigned_spawns_dests[:] = [loc for loc in assigned_spawns_dests if loc]

        # print(assigned_spawns_dests)

        parsed_spawns_dests = []

        for loc in assigned_spawns_dests:
            parsed_location = loc.split(", ")
            parsed_spawns_dests.append(parsed_location)
        # print(parsed_spawns_dests)


        ####################  setup transforms and destinations according to the locations given #####################
        assigned_transforms = []
        assigned_destinations = []
        for pair in parsed_spawns_dests:
            spawn_trans = carla.Transform(carla.Location(assigned_location_dict[pair[0]][0], assigned_location_dict[pair[0]][1]))
            spawn_trans.location.z += 2.0
            spawn_trans.rotation.roll = 0.0
            spawn_trans.rotation.pitch = 0.0
            #####################  determine yaws by comparing the relative positions of spawn points and the center point ################
            x_diff = assigned_location_dict['center'][0] - assigned_location_dict[pair[0]][0]
            y_diff = assigned_location_dict['center'][1] - assigned_location_dict[pair[0]][1]
            if x_diff < 0 and y_diff < 0:
                spawn_trans.rotation.yaw = 270
            elif x_diff > 0 and y_diff < 0:
                spawn_trans.rotation.yaw = 0
            elif x_diff < 0 and y_diff > 0:
                spawn_trans.rotation.yaw = 180
            else:
                spawn_trans.rotation.yaw = 90
            ####################  set up the locations of destinations, the locations will later be used to calculate A* routes by planner ################
            dest_trans = carla.Location(assigned_location_dict[pair[1]][0], assigned_location_dict[pair[1]][1])

            # print(spawn_trans)
            # print(dest_trans)

            assigned_transforms.append(spawn_trans)
            assigned_destinations.append(dest_trans)

        ############# spawn e2e actors according to assigned transforms ###############
        if args.mode != 'auto':
            actor_numbers = len(assigned_transforms)
        else:
            actor_numbers = 1
            
        for num_of_vehicles, transform in enumerate(assigned_transforms):
            if num_of_vehicles == actor_numbers:
                break
            blueprint = world.world.get_blueprint_library().find('vehicle.lincoln.mkz_2017')

            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
                
            # blueprint, transform
            
            try:
        
                other_agent = client.get_world().spawn_actor(blueprint, transform)
                id = other_agent.id
                e2e_vehicles_list.append(id)
                
                e2e_agent_dict[id] = {}
                e2e_agent_dict[id]["agent"] = other_agent
                
            except:
                print("Spawn failed because of collision at spawn position")
      
        ################################## implement traffic manager below #########################################            
        auto_agent_list, all_auto_actors = spawn_actor_assigned(world.world, client, assigned_transforms[actor_numbers:])
        
        agent = BehaviorAgent(world.player, behavior='aggressive')
        destination = random.choice(spawn_points).location
        agent.set_destination(destination)


        planner = GlobalRoutePlanner(map, resolution=1.0)

        clock = pygame.time.Clock()

        player_bev_map = BEV_MAP(args)
        processed_policy = player_bev_map.init_policy()
        player_bev_map.init_vehicle_bbox(world.player.id)
        destination = random.choice(spawn_points).location
        current_route = planner.trace_route(world.player.get_location(), destination)

        agent_bev_maps = {}
        current_routes = {}
        print("E2E agent:")
        for id in e2e_vehicles_list:
            print(id)
            agent_bev_map = BEV_MAP(args)
            agent_bev_map.init_vehicle_bbox(id)
            agent_bev_map.set_policy(processed_policy)
            destination = random.choice(spawn_points).location

            agent_bev_maps[id] = agent_bev_map
            current_routes[id] = []

        auto_current_routes = {}
        print('Auto agent:')
        for id in auto_agent_list:
            print(id)
            destination = random.choice(spawn_points).location
            auto_current_routes[id] = []            
            
        avg_FPS = 0

        while True:

            clock.tick_busy_loop(30)
            frame = world.world.tick()
            # print("++++start++++")
            # start_time = time.time()
            if controller.parse_events(client, world, clock):
                return

            view = pygame.surfarray.array3d(display)
            view = view.transpose([1, 0, 2]) 
            image = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)        

            world.tick(clock, frame, image, stored_path)
            avg_FPS = 0.98 * avg_FPS + 0.02 * clock.get_fps()
            # print("Multithreading avg FPS:", avg_FPS)
            # print(start_time - time.time())
            # start_time = time.time()
            # print("player_route")
            # regenerate a destination when the agent deviates from the current route
            if not check_close(world.player.get_location(), current_route[0][0].transform.location, 6):
                destination = current_route[-1][0].transform.location
                current_route = planner.trace_route(world.player.get_location(), destination)
            
            # Delete reached points from current route
            while check_close(world.player.get_location(), current_route[0][0].transform.location):
                current_route.pop(0)
                if len(current_route) == 0:
                    new_destination = random.choice(spawn_points).location
                    current_route = planner.trace_route(world.player.get_location(), new_destination)

            
            # Generate new destination if the current one is close 
            if len(current_route) < 10:
                new_destination = random.choice(spawn_points).location
                new_route = planner.trace_route(current_route[-1][0].transform.location, new_destination)
                temp_route = current_route + new_route
                current_route = temp_route

            route_trace = current_route[0:80]    
            # collect data for all agents
            processed_data = player_bev_map.collect_actor_data(world)
            route_list = []
            for wp in route_trace:
                wp = wp[0]
                route_list.append(Loc(x=wp.transform.location.x, y=wp.transform.location.y))
            # print(start_time - time.time())

            ###################player control#####################
            # start_time = time.time()
            # print("player_runstep")
            result = {}
            player_bev_map.run_step(result, frame, world.player.id, route_list)
            bev_map_rgb = result["bev_map_rgb"]
            # policy_input = result["policy_input"]
            # print(start_time - time.time())
            # start_time = time.time()
            # print("player_forward")            
            # player_bev_map.policy_forward(result, [policy_input])
            control_elements_list = result["control_elements_list"]
            # print("player_all", start_time - time.time())
            # start_time = time.time()
            # print("player_render")

            # for render
            surface = pygame.surfarray.make_surface(bev_map_rgb)
            surface = pygame.transform.flip(surface, True, False)
            surface = pygame.transform.rotate(surface, 90)
            display.blit(surface, (256, 0))
            # print(start_time - time.time())
            # start_time = time.time()
            # print("player_control")
            control_elements = control_elements_list[0]
            control = carla.VehicleControl(throttle=control_elements['throttle'], steer=control_elements['steer'], brake=control_elements['brake'])
        
            world.player.apply_control( control )
            ###################player control#####################

            # print(start_time - time.time())
            # print("+++++++actors++++++++++")
            route_trace = {}
            count = 1
            inputs = []
            start_time_r = time.time()

            for num_of_vehicles, id in enumerate(e2e_vehicles_list):
                if len(current_routes[id]) == 0:
                    # new_destination = random.choice(spawn_points).location
                    new_destination = assigned_destinations[num_of_vehicles]
                    print(f'{id}: {new_destination}')
                    # print(new_destination)
                    current_routes[id] = planner.trace_route(e2e_agent_dict[id]['agent'].get_location(), new_destination)
                
                # regenerate a route when the agent deviates from the current route
                if not check_close(e2e_agent_dict[id]["agent"].get_location(), current_routes[id][0][0].transform.location, 6):
                    destination = current_routes[id][-1][0].transform.location
                    current_routes[id] = planner.trace_route(e2e_agent_dict[id]["agent"].get_location(), destination)
                
                # Delete reached points from current route
                while check_close(e2e_agent_dict[id]["agent"].get_location(), current_routes[id][0][0].transform.location):
                    current_routes[id].pop(0)
                    if len(current_routes[id]) == 0:
                        new_destination = random.choice(spawn_points).location
                        current_routes[id] = planner.trace_route(e2e_agent_dict[id]["agent"].get_location(), new_destination)
                # Generate new destination if the current one is close 
                if len(current_routes[id]) < 10:
                    new_destination = random.choice(spawn_points).location
                    new_route = planner.trace_route(current_routes[id][-1][0].transform.location, new_destination)
                    temp_route = current_routes[id] + new_route
                    current_routes[id] = temp_route       

                route_trace[id] = current_routes[id][0:60]
                # bev_map_rgb, control = bev_map.run_step(frame, id, route_trace[id]) 
                # e2e_agent_dict[id]["agent"].apply_control(control)

                route_list = []
                for wp in route_trace[id]:
                    wp = wp[0]
                    route_list.append(Loc(x=wp.transform.location.x, y=wp.transform.location.y))
                # #original

                # bev_map_rgb, control_elements = bev_map.run_step(frame, id, route_list) 
                # control = carla.VehicleControl(throttle=control_elements['throttle'], steer=control_elements['steer'], brake=control_elements['brake'])
                # e2e_agent_dict[id]["agent"].apply_control(control)
                start_time = time.time()
                inputs.append([start_time, id, route_list])
            
            traffic_manager = client.get_trafficmanager()
            for num_of_vehicles, id in enumerate(auto_agent_list):
                # action = traffic_manager.get_all_actions(all_auto_actors[num_of_vehicles])
                # print(f'{id}: {action}')
                if len(auto_current_routes[id]) == 0:
                    # new_destination = random.choice(spawn_points).location
                    new_destination = assigned_destinations[actor_numbers + num_of_vehicles]
                    # print(f'{id}: {new_destination}')
                    auto_current_routes[id] = planner.trace_route(all_auto_actors[num_of_vehicles].get_location(), new_destination)
                    
                # regenerate a route when the agent deviates from the current route
                if not check_close(all_auto_actors[num_of_vehicles].get_location(), auto_current_routes[id][0][0].transform.location, 6):
                    print("deviates")
                    destination = auto_current_routes[id][-1][0].transform.location
                    auto_current_routes[id] = planner.trace_route(all_auto_actors[num_of_vehicles].get_location(), destination)
                
                # Delete reached points from current route
                while check_close(all_auto_actors[num_of_vehicles].get_location(), auto_current_routes[id][0][0].transform.location):
                    auto_current_routes[id].pop(0)
                    if len(auto_current_routes[id]) == 0:
                        print('empty route')
                        new_destination = random.choice(spawn_points).location
                        auto_current_routes[id] = planner.trace_route(all_auto_actors[num_of_vehicles].get_location(), new_destination)
                # Generate new destination if the current one is close 
                if len(auto_current_routes[id]) < 5:
                    new_destination = random.choice(spawn_points).location
                    new_route = planner.trace_route(auto_current_routes[id][-1][0].transform.location, new_destination)
                    temp_route = auto_current_routes[id] + new_route
                    auto_current_routes[id] = temp_route
                
                route_trace[id] = auto_current_routes[id][0:60]
                # bev_map_rgb, control = bev_map.run_step(frame, id, route_trace[id]) 
                # e2e_agent_dict[id]["agent"].apply_control(control)

                route_list = []
                for wp in route_trace[id]:
                    route_list.append(wp[0].transform.location)

                traffic_manager.set_path(all_auto_actors[num_of_vehicles], route_list)
                
            start_time_run = time.time()
            t_list = []
            results = [{} for i in range(actor_numbers)]
            for i in range(actor_numbers):
                input = [results[i]]
                input = input + inputs[i]
                agent_bev_maps[e2e_vehicles_list[i]].set_data(processed_data)
                t = Thread(target=agent_bev_maps[e2e_vehicles_list[i]].run_step, args=tuple(input))
                t_list.append(t)
                t.start()
            for t in t_list:
                t.join()

            # print("finish both",start_time_run - time.time())
            # print("finish runstep",start_time_run - time.time())
            
            
            # start_time = time.time()
            # t2_list = []
            # for i in range(actor_numbers):
            #     policy_input = [results[i]]
            #     policy_input = policy_input + [[results[i]["policy_input"]]]
            #     t = Thread(target=agent_bev_maps[vehicles_list[i]].policy_forward, args=tuple(policy_input))
            #     t2_list.append(t)
            #     t.start()
            # for t in t2_list:
            #     t.join()
            # print("finish infer",start_time - time.time())

            for i in range((min(4, actor_numbers))):
                bev_map_rgb = results[i]["bev_map_rgb"]
                surface = pygame.surfarray.make_surface(bev_map_rgb)
                surface = pygame.transform.flip(surface, True, False)
                surface = pygame.transform.rotate(surface, 90)
                display.blit(surface, (256, 210*i))  

            start_time = time.time()
            for i in range(actor_numbers):
                control_elements_list = results[i]["control_elements_list"] 
                control_elements = control_elements_list[0]
                control = carla.VehicleControl(throttle=control_elements['throttle'], steer=control_elements['steer'], brake=control_elements['brake'])
                e2e_agent_dict[e2e_vehicles_list[i]]["agent"].apply_control(control)
                

            # print("finish all",start_time_r - time.time())





            # policy_inputs_list = []
            # for bev_map_rgb, policy_input in image_inputs:

            #     policy_inputs_list.append(policy_input)

            #     # Visualize other agents
            #     if count < 4:
            #         surface = pygame.surfarray.make_surface(bev_map_rgb)
            #         surface = pygame.transform.flip(surface, True, False)
            #         surface = pygame.transform.rotate(surface, 90)
            #         display.blit(surface, (256, 210*count))                
            #     count += 1
                
            # # print("finish control and render",start_time - time.time())
            # # start_time = time.time()
            #     # print("finish one",start_time - time.time())
            #     # start_time = time.time()
            #     # agent_dict[id]["agent"].apply_control()
            
            # # print(policy_inputs_list)
            # start_time = time.time()
            # control_elements_list = bev_map.policy_forward(policy_inputs_list)
            # print("finish all forward",start_time - time.time())
            # start_time = time.time()
            # i = 0
            # for control_elements in control_elements_list:
            #     control = carla.VehicleControl(throttle=control_elements['throttle'], steer=control_elements['steer'], brake=control_elements['brake'])
            #     agent_dict[vehicles_list[i]]["agent"].apply_control(control)
            #     i+=1
            # print("finish all control",start_time - time.time())
            # start_time = time.time()
            world.hud.render(display)
            
            pygame.display.flip()

            # print("++++end++++")

            
            
            
            # world.render(display)
            #print("location: ",world.player.get_location())
            
            # if agent.done():
            #     agent.set_destination(random.choice(spawn_points).location)
            # control = agent.run_step()
            # control.manual_gear_shift = False
            # world.player.apply_control(control)
            
    # except Exception as e:
    #       print(e)
    # finally:
        settings = world.world.get_settings()
        settings.synchronous_mode = False 
        world.world.apply_settings(settings)

        print('destroying vehicles')
        
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
        



        time.sleep(0.5)


        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()

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
        '--town',
        default='Town05',
        type=str,
        help='map)')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    # argparser.add_argument(
    #     '--n',
    #     default='4',
    #     type=int,
    #     help='number of Roach agents')
    argparser.add_argument(
        '--mode',
        default='auto',
        type=str,
        help='Selecet behavior mode of other agent')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()


