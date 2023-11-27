from bird_eye_view.BirdViewProducer import BirdViewProducer, BirdView
from bird_eye_view.Mask import PixelDimensions, Loc

import wandb
import copy

from omegaconf import DictConfig, OmegaConf
from importlib import import_module


def load_entry_point(name):
    mod_name, attr_name = name.split(":") #agents.rl_birdview.rl_birdview_agent:RlBirdviewAgent
    mod = import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


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