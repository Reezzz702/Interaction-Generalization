import numpy as np 
from typing import NamedTuple, List, Tuple, Optional
import cv2 
import os
import ujson

from bird_eye_view.Mask import PixelDimensions, square_fitting_rect_at_any_rotation, MapMaskGenerator, RenderingWindow, BirdViewMasks, Coord, Loc, COLOR_OFF, COLOR_ON

RgbCanvas = np.ndarray  # [np.uint8] with shape (y, x, 3)
BirdView = np.ndarray  # [np.uint8] with shape (level, y, x)


class RGB:
    VIOLET = (173, 127, 168)
    ORANGE = (252, 175, 62)
    CHOCOLATE = (233, 185, 110)
    CHAMELEON = (138, 226, 52)
    SKY_BLUE = (114, 159, 207)
    DIM_GRAY = (105, 105, 105)
    DARK_GRAY = (50, 50, 50)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)
    WHITE = (255, 255, 255)

    ROAD = (46, 52, 54)
    ROUTE = (136, 138, 133)
    ROAD_LINE = (255, 140, 255)
    PEDESTRIANS_16 = (51, 255, 255)
    PEDESTRIANS_11 = (102, 255, 255)
    PEDESTRIANS_6 = (153, 255, 255)
    PEDESTRIANS_1 = (204, 255, 255)
    VEHICLES_16 = (51, 51, 255)
    VEHICLES_11 = (102, 102, 255)
    VEHICLES_6 = (153, 153, 255)
    VEHICLES_1 = (204, 204, 255)

    R_LIGHT_STOP_16 = (255, 51, 51)
    R_LIGHT_STOP_11 = (255, 102, 102)
    R_LIGHT_STOP_6 = (255, 153, 153)
    R_LIGHT_STOP_1 = (255, 204, 204)

    Y_LIGHT_STOP_16 = (255, 255, 51)
    Y_LIGHT_STOP_11 = (255, 255, 102)
    Y_LIGHT_STOP_6 = (255, 255, 153)
    Y_LIGHT_STOP_1 = (255, 255, 204)

    G_LIGHT_STOP_16 = (51, 255, 51)
    G_LIGHT_STOP_11 = (102, 255, 102)
    G_LIGHT_STOP_6 = (153, 255, 153)
    G_LIGHT_STOP_1 = (204, 255, 204)

    # 
    # UNLABELES = (0, 0, 0)

# Drivable areas
# Desired route
# Lane boundaries
# Vehicles t-16
# Vehicles t-11
# Vehicles t-6
# Vehicles t-1
# Pedestrians t-16
# Pedestrians t-11
# Pedestrians t-6
# Pedestrians t-1
# Lights and stops t-16
# Lights and stops t-11
# Lights and stops t-6
# Lights and stops t-1

#if the frame of t-16, t-11 or t-6 does not exist, replace it with the oldest frame


# 9 channel 
RGB_BY_MASK = {

    BirdViewMasks.AGENT: RGB.CHAMELEON,

    BirdViewMasks.Y_LIGHT_STOP_1: RGB.Y_LIGHT_STOP_1,    
    BirdViewMasks.Y_LIGHT_STOP_6: RGB.Y_LIGHT_STOP_6,    
    BirdViewMasks.Y_LIGHT_STOP_11: RGB.Y_LIGHT_STOP_11,    
    BirdViewMasks.Y_LIGHT_STOP_16: RGB.Y_LIGHT_STOP_16,   

    BirdViewMasks.G_LIGHT_STOP_1: RGB.G_LIGHT_STOP_1,    
    BirdViewMasks.G_LIGHT_STOP_6: RGB.G_LIGHT_STOP_6,    
    BirdViewMasks.G_LIGHT_STOP_11: RGB.G_LIGHT_STOP_11,    
    BirdViewMasks.G_LIGHT_STOP_16: RGB.G_LIGHT_STOP_16,   

    BirdViewMasks.R_LIGHT_STOP_1: RGB.R_LIGHT_STOP_1,    
    BirdViewMasks.R_LIGHT_STOP_6: RGB.R_LIGHT_STOP_6,    
    BirdViewMasks.R_LIGHT_STOP_11: RGB.R_LIGHT_STOP_11,    
    BirdViewMasks.R_LIGHT_STOP_16: RGB.R_LIGHT_STOP_16,    

    BirdViewMasks.PEDESTRIANS_1: RGB.PEDESTRIANS_1,
    BirdViewMasks.PEDESTRIANS_6: RGB.PEDESTRIANS_6,
    BirdViewMasks.PEDESTRIANS_11: RGB.PEDESTRIANS_11,
    BirdViewMasks.PEDESTRIANS_16: RGB.PEDESTRIANS_16,
    BirdViewMasks.VEHICLES_1: RGB.VEHICLES_1,
    BirdViewMasks.VEHICLES_6: RGB.VEHICLES_6,
    BirdViewMasks.VEHICLES_11: RGB.VEHICLES_11,
    BirdViewMasks.VEHICLES_16: RGB.VEHICLES_16,
    BirdViewMasks.ROAD_LINE: RGB.ROAD_LINE,
    BirdViewMasks.ROUTE: RGB.ROUTE,
    BirdViewMasks.ROAD: RGB.ROAD,
    # BirdViewMasks.UNLABELES: RGB.UNLABELES,
    
}

class CroppingRect(NamedTuple):
    x: int
    y: int
    width: int
    height: int

    @property
    def vslice(self) -> slice:
        return slice(self.y, self.y + self.height)

    @property
    def hslice(self) -> slice:
        return slice(self.x, self.x + self.width)


def rotate(image, angle, center=None, scale=1.0):
    assert image.dtype == np.uint8

    """Copy paste of imutils method but with INTER_NEAREST and BORDER_CONSTANT flags"""
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(
        image,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # return the rotated image
    return rotated



class BirdViewProducer:
    def __init__(self,
                 Town_Name,
                 target_size: PixelDimensions,
                 pixels_per_meter: int = 4
                 ) -> None:
        
        
        self.target_size = target_size
        self._pixels_per_meter = pixels_per_meter
        
        
        rendering_square_size = round(square_fitting_rect_at_any_rotation(self.target_size))
        self.rendering_area = PixelDimensions(
            width=rendering_square_size, height=rendering_square_size
        )

            
    
        self.masks_generator = MapMaskGenerator(town=Town_Name, pixels_per_meter=self._pixels_per_meter)
        
        
        self.full_road_cache = self.masks_generator.road_mask()
        self.full_lanes_cache = self.masks_generator.road_line_mask()
        
        # self.road_mask()
        # self.road_line_mask()
        # self.crosswalk_mask()
        
        
    # draw 
    def produce(
        self, vehicle_loc, yaw, agent_bbox_list, vehicle_bbox_1_16, pedestrain_bbox_1_16, r_bbox_1_16, g_bbox_1_16, y_bbox_1_16, obstacle_bbox_list, 
    ) -> BirdView:
        
        # Reusing already generated static masks for whole map
        self.masks_generator.disable_local_rendering_mode()
        agent_global_px_pos = self.masks_generator.location_to_pixel(vehicle_loc)
        # agent_global_px_pos:  Coord(x=461, y=760) 

        cropping_rect = CroppingRect(
            x=int(agent_global_px_pos.x - self.rendering_area.width / 2),
            y=int(agent_global_px_pos.y - self.rendering_area.height / 2),
            width=self.rendering_area.width,
            height=self.rendering_area.height,
        )

        
        # CroppingRect(x=333, y=632, width=255, height=255)
        masks = np.zeros(
            shape=(
                len(BirdViewMasks),
                self.rendering_area.height,
                self.rendering_area.width,
            ),
            dtype=np.uint8,
        )
        
        masks[BirdViewMasks.ROAD.value] = self.full_road_cache[
            cropping_rect.vslice, cropping_rect.hslice
        ]
        masks[BirdViewMasks.ROAD_LINE.value] = self.full_lanes_cache[
            cropping_rect.vslice, cropping_rect.hslice
        ]
        
        rendering_window = RenderingWindow(
            origin=vehicle_loc, area=self.rendering_area
        )
        self.masks_generator.enable_local_rendering_mode(rendering_window)
        
        
        # draw actor 
        masks = self._render_actors_masks(agent_bbox_list, 
                                          vehicle_bbox_1_16, 
                                          pedestrain_bbox_1_16,
                                          r_bbox_1_16, g_bbox_1_16, y_bbox_1_16, 
                                          obstacle_bbox_list, 
                                          masks)
        
        # agent_bbox_list,
        # vehicle_bbox_list,
        # pedestrians_bbox_list,
        # obstacle_bbox_list,
        
        
        # -------------------------------------------------------------------- # 
        
        cropped_masks = self.apply_agent_following_transformation_to_masks(
           yaw, masks,
        )
        ordered_indices = [mask.value for mask in BirdViewMasks.bottom_to_top()]

        
        return cropped_masks[ordered_indices]
    
    

    def _render_actors_masks(
        self,
        agent_bbox_list,
        vehicle_bbox_1_16,
        pedestrain_bbox_1_16,
        r_bbox_1_16, g_bbox_1_16, y_bbox_1_16,
        obstacle_bbox_list,
        masks: np.ndarray,
    ) -> np.ndarray:
        """Fill masks with ones and zeros (more precisely called as "bitmask").
        Although numpy dtype is still the same, additional semantic meaning is being added.
        """
        # lights_masks = self.masks_generator.traffic_lights_masks(
        #     segregated_actors.traffic_lights
        # )
        
        # red_lights_mask, yellow_lights_mask, green_lights_mask = lights_masks
        # masks[BirdViewMasks.RED_LIGHTS.value] = red_lights_mask
        # masks[BirdViewMasks.YELLOW_LIGHTS.value] = yellow_lights_mask
        # masks[BirdViewMasks.GREEN_LIGHTS.value] = green_lights_mask
        

        
        masks[BirdViewMasks.AGENT.value] = self.masks_generator.draw_bbox_mask(
            agent_bbox_list
        )

        if len(vehicle_bbox_1_16) < 16:
            vehicle_bbox_list = vehicle_bbox_1_16[0]
        else:
            vehicle_bbox_list = vehicle_bbox_1_16[-16]
        masks[BirdViewMasks.VEHICLES_16.value] = self.masks_generator.draw_bbox_mask(
            vehicle_bbox_list
        )
        if len(vehicle_bbox_1_16) < 11:
            vehicle_bbox_list = vehicle_bbox_1_16[0]
        else:
            vehicle_bbox_list = vehicle_bbox_1_16[-11]
        masks[BirdViewMasks.VEHICLES_11.value] = self.masks_generator.draw_bbox_mask(
            vehicle_bbox_list
        )
        if len(vehicle_bbox_1_16) < 6:
            vehicle_bbox_list = vehicle_bbox_1_16[0]
        else:
            vehicle_bbox_list = vehicle_bbox_1_16[-6]
        masks[BirdViewMasks.VEHICLES_6.value] = self.masks_generator.draw_bbox_mask(
            vehicle_bbox_list
        )
        vehicle_bbox_list = vehicle_bbox_1_16[-1]
        masks[BirdViewMasks.VEHICLES_1.value] = self.masks_generator.draw_bbox_mask(
            vehicle_bbox_list
        )


        # ped 
        if len(pedestrain_bbox_1_16) < 16:
            pedestrians_bbox_list = pedestrain_bbox_1_16[0]
        else:
            pedestrians_bbox_list = pedestrain_bbox_1_16[-16]
        masks[BirdViewMasks.PEDESTRIANS_16.value] = self.masks_generator.draw_bbox_mask(
            pedestrians_bbox_list
        )
        if len(pedestrain_bbox_1_16) < 11:
            pedestrians_bbox_list = pedestrain_bbox_1_16[0]
        else:
            pedestrians_bbox_list = pedestrain_bbox_1_16[-11]
        masks[BirdViewMasks.PEDESTRIANS_11.value] = self.masks_generator.draw_bbox_mask(
            pedestrians_bbox_list
        )
        if len(pedestrain_bbox_1_16) < 6:
            pedestrians_bbox_list = pedestrain_bbox_1_16[0]
        else:
            pedestrians_bbox_list = pedestrain_bbox_1_16[-6]
        masks[BirdViewMasks.PEDESTRIANS_6.value] = self.masks_generator.draw_bbox_mask(
            pedestrians_bbox_list
        )
        pedestrians_bbox_list = pedestrain_bbox_1_16[-1]
        masks[BirdViewMasks.PEDESTRIANS_1.value] = self.masks_generator.draw_bbox_mask(
            pedestrians_bbox_list
        )

        # ToDos: Add stop sign 

        # r_bbox_1_16, g_bbox_1_16, y_bbox_1_16,
        # ped 
        if len(r_bbox_1_16) < 16:
            r_bbox = r_bbox_1_16[0]
        else:
            r_bbox = r_bbox_1_16[-16]
        masks[BirdViewMasks.R_LIGHT_STOP_16.value] = self.masks_generator.draw_line_mask(
            r_bbox
        )
        if len(r_bbox_1_16) < 11:
            r_bbox = r_bbox_1_16[0]
        else:
            r_bbox = r_bbox_1_16[-11]
        masks[BirdViewMasks.R_LIGHT_STOP_11.value] = self.masks_generator.draw_line_mask(
            r_bbox
        )
        if len(r_bbox_1_16) < 6:
            r_bbox = r_bbox_1_16[0]
        else:
            r_bbox = r_bbox_1_16[-6]
        masks[BirdViewMasks.R_LIGHT_STOP_6.value] = self.masks_generator.draw_line_mask(
            r_bbox
        )
        r_bbox = r_bbox_1_16[-1]
        masks[BirdViewMasks.R_LIGHT_STOP_1.value] = self.masks_generator.draw_line_mask(
            r_bbox
        )

        if len(g_bbox_1_16) < 16:
            g_bbox = g_bbox_1_16[0]
        else:
            g_bbox = g_bbox_1_16[-16]
        masks[BirdViewMasks.G_LIGHT_STOP_16.value] = self.masks_generator.draw_line_mask(
            g_bbox
        )
        if len(g_bbox_1_16) < 11:
            g_bbox = g_bbox_1_16[0]
        else:
            g_bbox = g_bbox_1_16[-11]
        masks[BirdViewMasks.G_LIGHT_STOP_11.value] = self.masks_generator.draw_line_mask(
            g_bbox
        )
        if len(g_bbox_1_16) < 6:
            g_bbox = g_bbox_1_16[0]
        else:
            g_bbox = g_bbox_1_16[-6]
        masks[BirdViewMasks.G_LIGHT_STOP_6.value] = self.masks_generator.draw_line_mask(
            g_bbox
        )
        g_bbox = g_bbox_1_16[-1]
        masks[BirdViewMasks.G_LIGHT_STOP_1.value] = self.masks_generator.draw_line_mask(
            g_bbox
        )

        if len(y_bbox_1_16) < 16:
            y_bbox = y_bbox_1_16[0]
        else:
            y_bbox = y_bbox_1_16[-16]
        masks[BirdViewMasks.Y_LIGHT_STOP_16.value] = self.masks_generator.draw_line_mask(
            y_bbox
        )
        if len(y_bbox_1_16) < 11:
            y_bbox = y_bbox_1_16[0]
        else:
            y_bbox = y_bbox_1_16[-11]
        masks[BirdViewMasks.Y_LIGHT_STOP_11.value] = self.masks_generator.draw_line_mask(
            y_bbox
        )
        if len(y_bbox_1_16) < 6:
            y_bbox = y_bbox_1_16[0]
        else:
            y_bbox = y_bbox_1_16[-6]
        masks[BirdViewMasks.Y_LIGHT_STOP_6.value] = self.masks_generator.draw_line_mask(
            y_bbox
        )
        y_bbox = y_bbox_1_16[-1]
        masks[BirdViewMasks.Y_LIGHT_STOP_1.value] = self.masks_generator.draw_line_mask(
            y_bbox
        )
        # masks[BirdViewMasks.PEDESTRIANS.value] = self.masks_generator.draw_bbox_mask(
        #     pedestrians_bbox_list
        # )
        # masks[BirdViewMasks.OBSTACLES.value] = self.masks_generator.draw_bbox_mask(
        #     obstacle_bbox_list
        # )
        
        
        return masks
      
    @staticmethod
    def as_rgb(birdview: BirdView) -> RgbCanvas:
        _, h, w = birdview.shape
        rgb_canvas = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        nonzero_indices = lambda arr: arr == COLOR_ON

        for mask_type in BirdViewMasks.bottom_to_top():
            rgb_color = RGB_BY_MASK[mask_type]
            # print(rgb_color)
            mask = birdview[mask_type]
            # If mask above contains 0, don't overwrite content of canvas (0 indicates transparency)
            rgb_canvas[nonzero_indices(mask)] = rgb_color
        return rgb_canvas
    

    @staticmethod
    def as_roach_input(birdview: BirdView):
        h, w = 192, 192

        road_mask = birdview[BirdViewMasks.ROAD][0:192, 56:248] * 255
        route_mask = birdview[BirdViewMasks.ROUTE][0:192, 56:248] * 255
        road_line_mask = birdview[BirdViewMasks.ROAD_LINE][0:192, 56:248] * 255

        vehicle_history = []
        # crop for roach input 
        # target -> 192 x 192 
        # 304 --> [56, 248]
        #y [0, 192]

        print(birdview[BirdViewMasks.VEHICLES_16][0:192, 56:248].shape)


        for i in range(4):
            vehicle_history.append(birdview[BirdViewMasks.VEHICLES_16+i][0:192, 56:248] * 255)

        walker_history = []
        for i in range(4):
            walker_history.append(birdview[BirdViewMasks.PEDESTRIANS_16+i][0:192, 56:248] * 255)

        trafficlight_stop_history = []
        for i in range(4):
            single_time_frame = np.zeros([h, w], dtype=np.uint8)
            single_time_frame[birdview[BirdViewMasks.G_LIGHT_STOP_16+i][0:192, 56:248]] = 80
            single_time_frame[birdview[BirdViewMasks.Y_LIGHT_STOP_16+i][0:192, 56:248]] = 170
            single_time_frame[birdview[BirdViewMasks.R_LIGHT_STOP_16+i][0:192, 56:248]] = 255
            trafficlight_stop_history.append(single_time_frame)

        observation = np.stack((road_mask, route_mask, road_line_mask, *vehicle_history, *walker_history, *trafficlight_stop_history), axis=2)
        observation = np.transpose(observation, [2, 0, 1])
        # print(observation)
        return observation


    
    @staticmethod
    def as_ss(birdview: BirdView):
        _, h, w = birdview.shape
        canvas = np.zeros(shape=(h, w), dtype=np.uint8)
        nonzero_indices = lambda arr: arr == COLOR_ON

        for mask_type in BirdViewMasks.bottom_to_top():
            mask = birdview[mask_type]
            # If mask above contains 0, don't overwrite content of canvas (0 indicates transparency)
            canvas[nonzero_indices(mask)] = mask_type.value
        return canvas
    
    def apply_agent_following_transformation_to_masks(
        self, yaw,  masks: np.ndarray,
    ) -> np.ndarray:
        
        # agent_transform = agent_vehicle.get_transform()
        angle = ( yaw + 90)  # vehicle's front will point to the top

        # Rotating around the center
        crop_with_car_in_the_center = masks
        masks_n, h, w = crop_with_car_in_the_center.shape
        rotation_center = Coord(x=w // 2, y=h // 2)

        # warpAffine from OpenCV requires the first two dimensions to be in order: height, width, channels
        crop_with_centered_car = np.transpose(
            crop_with_car_in_the_center, axes=(1, 2, 0)
        )
        rotated = rotate(crop_with_centered_car, angle, center=rotation_center)
        rotated = np.transpose(rotated, axes=(2, 0, 1))

        half_width = self.target_size.width // 2
        hslice = slice(rotation_center.x - half_width, rotation_center.x + half_width)

        # if self._crop_type is BirdViewCropType.FRONT_AREA_ONLY:
        #     vslice = slice(rotation_center.y - self.target_size.height, rotation_center.y)
        # elif self._crop_type is BirdViewCropType.FRONT_AND_REAR_AREA:
            
        half_height = self.target_size.height // 2
        vslice = slice(
            rotation_center.y - half_height, rotation_center.y + half_height
        )
        
            
        # else:
        #     raise NotImplementedError
        assert (
            vslice.start > 0 and hslice.start > 0
        ), "Trying to access negative indexes is not allowed, check for calculation errors!"
        car_on_the_bottom = rotated[:, vslice, hslice]
        return car_on_the_bottom

        
        

if __name__ == "__main__":
    
    folder_path = "./sample_data/obstacle"
    town = "Town10HD"
    
    measurement_list = sorted(os.listdir(f"{folder_path}/ego_data"))
    
    birdview_producer = BirdViewProducer(
                                        town, 
                                        PixelDimensions(width=200, height=200), 
                                        pixels_per_meter=5)

    for name in measurement_list:
        # print(name)
        # 0000.json.gz
        frame = name.split(".")[0]
        # print(frame)
        
        with open(f"{folder_path}/ego_data/{name}", 'rt') as f1:
            data = ujson.load(f1)
    
        pos = Loc(x=data["location"]["x"], y=data["location"]["y"]) # data["pos_global"]
        yaw = data["rotation"]["yaw"]
            
        # read bbox from actors data 

        obstacle_bbox_list = []
        pedestrian_bbox_list = []
        vehicle_bbox_list = []
        agent_bbox_list = []
              
        with open(f"{folder_path}/actor_attribute.json", 'rt') as f1:
            data = ujson.load(f1)

        ego_id = data["ego_id"]
        # interactive id 
        vehicle_id_list = list(data["vehicle"].keys())
        
        # pedestrian id list 
        
        pedestrian_id_list = list(data["pedestrian"].keys())
                
        # obstacle id list 
        obstacle_id_list = list(data["obstacle"].keys())
        
        # obstacle bbox store in actor_attribute.json
        for id in obstacle_id_list:
            pos_0 = data["obstacle"][str(id)]["cord_bounding_box"]["cord_0"]
            pos_1 = data["obstacle"][str(id)]["cord_bounding_box"]["cord_4"]
            pos_2 = data["obstacle"][str(id)]["cord_bounding_box"]["cord_6"]
            pos_3 = data["obstacle"][str(id)]["cord_bounding_box"]["cord_2"]
            
            obstacle_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ])
        with open(f"{folder_path}/actors_data/{frame}.json", 'rt') as f1:
            data = ujson.load(f1)
        
        for id in vehicle_id_list:
            pos_0 = data[str(id)]["cord_bounding_box"]["cord_0"]
            pos_1 = data[str(id)]["cord_bounding_box"]["cord_4"]
            pos_2 = data[str(id)]["cord_bounding_box"]["cord_6"]
            pos_3 = data[str(id)]["cord_bounding_box"]["cord_2"]

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
            pos_0 = data[str(id)]["cord_bounding_box"]["cord_0"]
            pos_1 = data[str(id)]["cord_bounding_box"]["cord_4"]
            pos_2 = data[str(id)]["cord_bounding_box"]["cord_6"]
            pos_3 = data[str(id)]["cord_bounding_box"]["cord_2"]
            
            pedestrian_bbox_list.append([Loc(x=pos_0[0], y=pos_0[1]), 
                                        Loc(x=pos_1[0], y=pos_1[1]), 
                                        Loc(x=pos_2[0], y=pos_2[1]), 
                                        Loc(x=pos_3[0], y=pos_3[1]), 
                                        ])
            

            
        birdview: BirdView = birdview_producer.produce(pos, yaw=yaw,
                                                       agent_bbox_list=agent_bbox_list, 
                                                       vehicle_bbox_list=vehicle_bbox_list,
                                                       pedestrians_bbox_list=pedestrian_bbox_list,
                                                       obstacle_bbox_list=obstacle_bbox_list)
    
        bgr = cv2.cvtColor(BirdViewProducer.as_rgb(birdview), cv2.COLOR_BGR2RGB)
        cv2.imshow("BirdView RGB", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
        