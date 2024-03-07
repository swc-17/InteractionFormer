import json
import os
import numpy as np
import torch
from tqdm import tqdm
import multiprocessing
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
from collections import defaultdict
import copy
import pandas as pd
object_type_convert = {0: 'TYPE_UNSET', 1: 'TYPE_VEHICLE', 2: 'TYPE_PEDESTRIAN', 3: 'TYPE_CYCLIST'}
from enum import Enum
from shapely.geometry import Polygon
import math

def minimum_rectangle(polygon):
    poly = Polygon(polygon)
    convex_hull = poly.convex_hull
    min_bd = convex_hull.minimum_rotated_rectangle
    vex = list(min_bd.exterior.coords)[::-1]
    length = math.sqrt((vex[1][1] - vex[0][1]) ** 2 + (vex[1][0] - vex[0][0]) ** 2)
    width = math.sqrt((vex[-1][1] - vex[0][1]) ** 2 + (vex[-1][0] - vex[-1][0]) ** 2)
    if length > width:
        edge1 = [vex[0], vex[1]]
        edge2 = [vex[3], vex[2]]
    else:
        edge1 = [vex[0], vex[-1]]
        edge2 = [vex[1].vex[2]]
    return edge1, edge2


def rotate(x, y, angle):
    if isinstance(x, torch.Tensor):
        other_x_trans = torch.cos(angle) * x - torch.sin(angle) * y
        other_y_trans = torch.cos(angle) * y + torch.sin(angle) * x
        output_coords = torch.stack((other_x_trans, other_y_trans), dim=-1)

    else:
        other_x_trans = np.cos(angle) * x - np.sin(angle) * y
        other_y_trans = np.cos(angle) * y + np.sin(angle) * x
        output_coords = np.stack((other_x_trans, other_y_trans), axis=-1)
    return output_coords


def get_mode(action_list):
    res = np.unique(action_list)
    return ",".join(res)


class LaneMarkType(str, Enum):
    """Color and pattern of a painted lane marking, located on either the left or ride side of a lane segment.

    The `NONE` type indicates that lane boundary is not marked by any paint; its extent should be implicitly inferred.
    """

    DASH_SOLID_YELLOW: str = "DASH_SOLID_YELLOW"
    DASH_SOLID_WHITE: str = "DASH_SOLID_WHITE"
    DASHED_WHITE: str = "DASHED_WHITE"
    DASHED_YELLOW: str = "DASHED_YELLOW"
    DOUBLE_SOLID_YELLOW: str = "DOUBLE_SOLID_YELLOW"
    DOUBLE_SOLID_WHITE: str = "DOUBLE_SOLID_WHITE"
    DOUBLE_DASH_YELLOW: str = "DOUBLE_DASH_YELLOW"
    DOUBLE_DASH_WHITE: str = "DOUBLE_DASH_WHITE"
    SOLID_YELLOW: str = "SOLID_YELLOW"
    SOLID_WHITE: str = "SOLID_WHITE"
    SOLID_DASH_WHITE: str = "SOLID_DASH_WHITE"
    SOLID_DASH_YELLOW: str = "SOLID_DASH_YELLOW"
    SOLID_BLUE: str = "SOLID_BLUE"
    NONE: str = "NONE"
    UNKNOWN: str = "UNKNOWN"


LaneMarkTypes = set([member.value for member in LaneMarkType])


def process_sublane(polyline, ego, offset, lane_range):
    center_yaw = ego[-2]
    center_pos = ego[:2]
    new_axis = polyline[..., :2] - center_pos
    rot_corn = rotate(copy.deepcopy(new_axis[:, 0]), copy.deepcopy(new_axis[:, 1]), -center_yaw)
    lane_point_mask = (abs(rot_corn[..., 0] - offset) < lane_range) * (abs(rot_corn[..., 1]) < lane_range - 40)
    sub_poly = polyline[lane_point_mask]
    return sub_poly


def process_subref(ref, ego, offset, lane_range):
    new_ref = {}
    sub_ref = process_sublane(ref[:, :3], ego, offset, lane_range)
    if len(sub_ref) >= 2:
        new_ref["refline"] = [{'x': sub_ref[row, 0],
                               'y': sub_ref[row, 1],
                               'z': sub_ref[row, 2]} for row in
                                range(sub_ref.shape[0])]
    return new_ref

def process_subbounds(bounds, ego, offset, lane_range):
    new_bounds = copy.deepcopy(bounds)
    for bound_id, bound in bounds.items():
        bound = np.stack(np.array([[point["x"], point["y"], point["z"]] for point in bound['area_boundary']]),
                         axis=0)
        sub_bound = process_sublane(bound, ego, offset, lane_range)
        if len(sub_bound) >= 2:
            new_bounds[bound_id]["area_boundary"] = [{'x': sub_bound[row, 0],
                                                      'y': sub_bound[row, 1],
                                                      'z': sub_bound[row, 2]} for row in
                                                     range(sub_bound.shape[0])]
        else:
            new_bounds.pop(bound_id)
            continue
    return new_bounds


def process_connections(bounds, ego, offset, lane_range):
    new_bounds = copy.deepcopy(bounds)
    for bound_id, bound in bounds.items():
        bound = np.stack(np.array([[point["x"], point["y"], point["z"]] for point in bound['refline']]),
                         axis=0)
        sub_bound = process_sublane(bound, ego, offset, lane_range)
        if len(sub_bound) >= 2:
            new_bounds[bound_id]["refline"] = [{'x': sub_bound[row, 0],
                                                'y': sub_bound[row, 1],
                                                'z': sub_bound[row, 2]} for row in
                                               range(sub_bound.shape[0])]
        else:
            new_bounds.pop(bound_id)
            continue
    return new_bounds

def process_sub_crosswalks(crosswalks, ego, offset, lane_range):
    new_bounds = copy.deepcopy(crosswalks)
    for bound_id, bound in crosswalks.items():
        bound = np.stack(np.array([[point["x"], point["y"], point["z"]] for point in bound['edge1']]),
                         axis=0)
        sub_bound = process_sublane(bound, ego, offset, lane_range)
        if len(sub_bound) != 2:
            new_bounds.pop(bound_id)
    return new_bounds


def process_sublanes(all_polylines, ego, offset, lane_range):
    new_all_polylines = copy.deepcopy(all_polylines)
    for polyline_id, polyline in all_polylines.items():
        centerline = np.stack(np.array([[point["x"], point["y"], point["z"]] for point in polyline['centerline']]),
                              axis=0)
        centerline_type = np.zeros([centerline.shape[0], 1])
        centerline = np.concatenate([centerline, centerline_type], axis=-1)
        if len(polyline['left_lane_boundary']) != 0:
            left_polyline = np.stack(
                np.array([[point["x"], point["y"], point["z"]] for point in polyline['left_lane_boundary']]),
                axis=0)
            left_centerline_type = np.ones([left_polyline.shape[0], 1])
            left_polyline = np.concatenate([left_polyline, left_centerline_type], axis=-1)
        else:
            left_polyline = np.zeros([0, centerline.shape[1]])
        if len(polyline['right_lane_boundary']) != 0:
            right_polyline = np.stack(
                np.array([[point["x"], point["y"], point["z"]] for point in polyline['right_lane_boundary']]),
                axis=0)
            right_centerline_type = np.ones([right_polyline.shape[0], 1])
            right_centerline_type[:, 0] = 2
            right_polyline = np.concatenate([right_polyline, right_centerline_type], axis=-1)
        else:
            right_polyline = np.zeros([0, centerline.shape[1]])
        all_polyline = np.concatenate([centerline, left_polyline, right_polyline], axis=0)
        sub_poly = process_sublane(all_polyline, ego, offset, lane_range)
        sub_centerline = sub_poly[sub_poly[:, -1] == 0, :3]
        sub_left = sub_poly[sub_poly[:, -1] == 1, :3]
        sub_right = sub_poly[sub_poly[:, -1] == 2, :3]
        if len(sub_centerline) >= 2:
            new_all_polylines[polyline_id]["centerline"] = [{'x': sub_centerline[row, 0],
                                                             'y': sub_centerline[row, 1],
                                                             'z': sub_centerline[row, 2]} for row in
                                                            range(sub_centerline.shape[0])]
        else:
            new_all_polylines.pop(polyline_id)
            continue
        if len(sub_left) >= 2:
            new_all_polylines[polyline_id]["left_lane_boundary"] = [{'x': sub_left[row, 0],
                                                                     'y': sub_left[row, 1],
                                                                     'z': sub_left[row, 2]} for row in
                                                                    range(sub_left.shape[0])]
        else:
            new_all_polylines.pop(polyline_id)
            continue
        if len(sub_right) >= 2:
            new_all_polylines[polyline_id]["right_lane_boundary"] = [{'x': sub_right[row, 0],
                                                                      'y': sub_right[row, 1],
                                                                      'z': sub_right[row, 2]} for row in
                                                                     range(sub_right.shape[0])]
        else:
            new_all_polylines.pop(polyline_id)
            continue
    return new_all_polylines


def process_sub_track(tracks, lane_range):
    new_tracks = tracks.transpose(1, 0, 2)
    ego = new_tracks[0, 50, :]
    center_pos = ego[:2]
    trajs = new_tracks
    new_line = trajs[..., :2] - center_pos
    ref_mask = (abs(new_line[..., 0]) < lane_range) * (abs(new_line[..., 1]) < lane_range)
    in_range_track = np.any(ref_mask, axis=1)
    ref_mask = ref_mask[in_range_track]
    trajs = trajs[in_range_track]
    trajs[~ref_mask] = np.zeros(12)
    return trajs.transpose(1, 0, 2)


def get_polyline_dir(polyline):
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
    return polyline_dir


def line_type_trans(line_type):
    bina = bin(line_type)
    bina_str = bina[2:].zfill(5)
    color = None
    if bina_str[-1] == "1":
        color = "WHITE"
    elif bina_str[-2] == "1":
        color = "YELLOW"
    solid = "SOLID" if bina_str[-3] == "1" else "DASHED"
    virtual = True if bina_str[0] == "1" else False
    roadside = True if bina_str[0] == "2" else False
    argo_line_type = "UNKNOWN"
    if virtual:
        argo_line_type = "NONE"
    elif roadside:
        argo_line_type = "UNKNOWN"
    elif color is not None:
        argo_line_type = solid + '_' + color
    assert argo_line_type in LaneMarkTypes
    return argo_line_type


lane_type_hash = {
    "driving": "VEHICLE",
    "bus": "BUS",
    "biking": "BIKE",
    "bicycle": "BIKE",
    "emergency": "BUS"
}


def traj_dict2np(polyline):
    polyline = np.stack(
        np.array([[point["x"], point["y"], point["z"]] for point in polyline]),
        axis=0)
    return polyline


def traj_np2dict(polyline):
    return [{'x': polyline[row, 0], 'y': polyline[row, 1], 'z': 0} for row in range(polyline.shape[0])]


class DataProcessor:
    file_name_list = [
        "agent_property",
        "agent_state",
        "lane_meta",
        "lane_property",
        "traffic_light",
        "target_point",
        "refline"
    ]

    def __init__(self, path, package, mode="pickle", task="", admap=None):
        self.name = package
        self.mode = mode
        self.ad_map = admap
        self.ceph = None
        self.task = task
        if task != "safer":
            DataProcessor.file_name_list.append("interval_info")
        if "s3" in path:
            from utils.cluster_reader import LoadScenarioFromCeph
            self.ceph = LoadScenarioFromCeph()
            for name in DataProcessor.file_name_list:
                setattr(self, name, self.ceph.read_csv(os.path.join(path, name + ".csv")))
            if task == "safer":
                self.take_over = self.ceph.read_json(os.path.join(path, "takeover" + ".json"))["takeover_interval"]
        elif mode == "simulation":
            for name in DataProcessor.file_name_list:
                setattr(self, name, pd.read_csv(os.path.join(path, name + ".csv")))
        else:
            for name in DataProcessor.file_name_list:
                setattr(self, name, pd.read_pickle(os.path.join(path, name + ".csv")))
            if task == "safer":
                with open(os.path.join(path, "takeover" + ".json"), 'r') as f:
                    self.take_over = json.load(f)["takeover_interval"]

    def interplating_polyline(self, polylines, distance):
        # Calculate the cumulative distance along the path, up-sample the polyline to 0.5 meter
        dist_along_path = np.zeros(polylines.shape[0])
        for i in range(1, polylines.shape[0]):
            dist_along_path[i] = dist_along_path[i - 1] + euclidean(polylines[i, :2], polylines[i - 1, :2])

        # Create interpolation functions for x and y coordinates
        fx = interp1d(dist_along_path, polylines[:, 0])
        fy = interp1d(dist_along_path, polylines[:, 1])

        # Create an array of distances at which to interpolate
        new_dist_along_path = np.arange(0, dist_along_path[-1], distance)

        # Use the interpolation functions to generate new x and y coordinates
        new_x = fx(new_dist_along_path)
        new_y = fy(new_dist_along_path)

        # Combine the new x and y coordinates into a single array
        new_polylines = np.vstack((new_x, new_y)).T
        return new_polylines

    def build_lane_boundary(self, line, width, cur_polyline_dir, direction="left"):
        # compute the new_yaw
        # pick the next_boundary along the line
        assert cur_polyline_dir.shape[0] == line.shape[0]
        rotate_dir = np.zeros_like(cur_polyline_dir)
        if direction == "left":
            rotate_dir[:, 0] = -cur_polyline_dir[:, 1]
            rotate_dir[:, 1] = cur_polyline_dir[:, 0]
        elif direction == "right":
            rotate_dir[:, 0] = cur_polyline_dir[:, 1]
            rotate_dir[:, 1] = -cur_polyline_dir[:, 0]
        left_polyline = line[:, :2] + rotate_dir[:, :2] * (width / 2)
        return left_polyline

    def process_bound(self, bound):
        down_sample_bounds = self.interplating_polyline(bound, 1)
        property = np.zeros([down_sample_bounds.shape[0], 2])
        property[:, 0] = 15
        property[:, 1] = 10000
        new_poly_line = np.concatenate([down_sample_bounds, torch.zeros(down_sample_bounds.shape[0], 1)], axis=1)
        poly_dir = get_polyline_dir(new_poly_line[:, :3])
        new_poly_line = np.concatenate([new_poly_line, poly_dir, property], axis=1)
        return new_poly_line

    def get_hdmap_data(self, lane_meta):
        min_x = lane_meta["x"].min()
        min_y = lane_meta["y"].min()
        max_x = lane_meta["x"].max()
        max_y = lane_meta["y"].max()
        self.ad_map.select(selected_boundary=[min_x, max_x, min_y, max_y],
                           include_ids=[],
                           exclude_ids=[])
        new_ad_map = self.ad_map.to_dict()
        hd_map = new_ad_map['value0']
        lines = hd_map['semantic_map_data']['lines']
        lane_links = hd_map['routing_map_data']['lane_links']
        section_links = hd_map['routing_map_data']['section_links']
        lanes = hd_map['routing_map_data']['lanes']
        line_hash = {line['id']: line for line in lines}
        lane_link_hash = {line['id']: line for line in lane_links}
        section_link_hash = {line['id']: line for line in section_links}
        lane_hash = {line['id']: line for line in lanes}
        routing_map_data = hd_map['routing_map_data']
        sections = routing_map_data["sections"]
        section_hash = {line['id']: line for line in sections}
        junctions = routing_map_data["junctions"]
        crosswalks = hd_map['semantic_map_data']['crosswalks']
        drivable_areas = {}
        connections = {}
        cross_walks = {}
        for crosswalk in crosswalks:
            geo = crosswalk["geometry"]
            geo = [(point['x'], point['y']) for point in geo]
            edge1, edge2 = minimum_rectangle(geo)
            cross_walks[crosswalk["id"]] = {"edge1": [{"x": point[0], "y": point[1], "z": 0} for point in edge1],
                                            "edge2": [{"x": point[0], "y": point[1], "z": 0} for point in edge2],
                                            "id": crosswalk["id"]}
        processed_section = set()
        for junction in junctions:
            overlap_ids = junction["overlap_ids"]
            for overlap in overlap_ids:
                if "lane" not in overlap or overlap not in lane_hash:
                    continue
                else:
                    lane = lane_hash[overlap]
                    successor_ids = lane['successor_ids']
                    predecessor_ids = lane['predecessor_ids']
                    for successor_id in successor_ids:
                        successor_lane_link = lane_link_hash[successor_id]
                        if len(successor_lane_link['ref_line_id']) != 0:
                            ref_line = line_hash[successor_lane_link['ref_line_id']]
                            connection = {'refline': traj_np2dict(
                                self.interplating_polyline(traj_dict2np(ref_line['geometry']), 1)), "id": successor_id}
                            connections[successor_id] = connection
                    for predecessor_id in predecessor_ids:
                        predecessor_lane_link = lane_link_hash[predecessor_id]
                        if len(predecessor_lane_link['ref_line_id']) != 0:
                            ref_line = line_hash[predecessor_lane_link['ref_line_id']]
                            connection = {'refline': traj_np2dict(
                                self.interplating_polyline(traj_dict2np(ref_line['geometry']), 1)),
                                "id": predecessor_id}
                            connections[predecessor_id] = connection
            boundary = {}
            junction_id = junction["id"]
            boundary["id"] = junction["id"]
            boundaries = traj_np2dict(
                self.interplating_polyline(traj_dict2np(junction["boundaries"][0]["geometry"]), 5))
            boundary["area_boundary"] = boundaries
            drivable_areas[junction_id] = boundary

        argo_lanes = {}
        for lane in lanes:
            argo_lane = {}
            argo_lane['id'] = lane["id"]
            if len(lane["overlap_ids"]) != 0 and "junction" in "".join(lane["overlap_ids"]):
                is_junction = True
            else:
                is_junction = False
            argo_lane['is_intersection'] = is_junction
            argo_lane['lane_type'] = lane_type_hash[lane["lane_type"]]
            centerline = line_hash[lane['center_line_id']]
            argo_lane['centerline'] = traj_np2dict(self.interplating_polyline(traj_dict2np(centerline['geometry']), 1))
            left_lane_boundary_id = lane["left_line_id"]
            left_lane_bound = line_hash[left_lane_boundary_id]
            argo_lane['left_lane_boundary'] = traj_np2dict(
                self.interplating_polyline(traj_dict2np(left_lane_bound['geometry']), 1))
            argo_lane['left_lane_mark_type'] = line_type_trans(left_lane_bound["line_type"])
            argo_lane['left_neighbor_id'] = lane['left_neighbor_lane']['lane_id']
            argo_lane['predecessors'] = []
            for link_id in lane["predecessor_ids"]:
                link = lane_link_hash[link_id]
                if link['ref_line_id'] == "":
                    argo_lane['predecessors'].append(lane_link_hash[link_id]['from_lane_id'])
                else:
                    if link_id in connections:
                        argo_lane['predecessors'].append(link_id)
            right_lane_boundary_id = lane["right_line_id"]
            right_lane_bound = line_hash[right_lane_boundary_id]
            argo_lane['right_lane_boundary'] = traj_np2dict(
                self.interplating_polyline(traj_dict2np(right_lane_bound['geometry']), 1))
            argo_lane['right_lane_mark_type'] = line_type_trans(right_lane_bound["line_type"])
            argo_lane['right_neighbor_id'] = lane['right_neighbor_lane']['lane_id']
            argo_lane['successors'] = []
            for link_id in lane["successor_ids"]:
                link = lane_link_hash[link_id]
                if link['ref_line_id'] == "":
                    argo_lane['successors'].append(lane_link_hash[link_id]['to_lane_id'])
                else:
                    if link_id in connections:
                        argo_lane['successors'].append(link_id)
            argo_lanes[lane["id"]] = argo_lane
        return argo_lanes, cross_walks, drivable_areas, connections

    def process_data(self, path):
        if self.task == "safer":
            scenarios = [0]
        else:
            scenarios = list(self.interval_info["scenario_index"].unique()[1:])
        for scenario in scenarios:
            try:
                scenario_cache = {}
                scenario_id = self.name + "_{}".format(scenario)
                agent_p = self.agent_property.loc[self.agent_property["scenario_index"] == scenario]
                refline = self.refline.loc[self.refline["scenario_index"] == scenario]
                agents = self.agent_state.loc[self.agent_state["scenario_index"] == scenario]
                lane_meta = self.lane_meta.loc[self.lane_meta["scenario_index"] == scenario]
                argo_lanes, crosswalks, drivable_areas, connections = self.get_hdmap_data(lane_meta)
                start_interval = agents["interval"].min()
                end_interval = agents["interval"].max()
                agents_array = self.process_agent(agents, agent_p, start_interval, end_interval)
                # traffic_light = self.traffic_light.loc[self.traffic_light["scenario_index"] == scenario]
                # lane_ids = lane_meta["laneid"].unique()
                # lane_ids_hash = {lane_id: idx for idx, lane_id in enumerate(lane_ids)}
                # lanes_topo = self.topo_info(l2l_relation, lane_ids_hash)
                # lanes = self.process_lane(lane_meta, lane_ids_hash, lanes_topo, lane_property)
                # if traffic_light.shape[0] != 0:
                #     traffic_light = traffic_light.copy()
                #     traffic_light["laneid"] = traffic_light["laneid"].apply(lambda x: lane_ids_hash[x])
                #     all_lights_info, dynamic_map_infos = self.light_info(traffic_light, start_interval, end_interval)
                # else:
                #     all_lights_info, dynamic_map_infos = None, None
                # scenario_cache["trafficlight"] = all_lights_info
                # scenario_cache["dynamic_map_infos"] = dynamic_map_infos
                # refline = refline.copy()
                # refline["laneid"] = refline["laneid"].apply(lambda x: lane_ids_hash[x])
                scenario_cache["agent"] = agents_array
                scenario_cache["refline"] = np.array(refline)
                scenario_cache["all_polylines"] = argo_lanes
                scenario_cache["boundaries"] = drivable_areas
                scenario_cache["crosswalks"] = crosswalks
                scenario_cache["connections"] = connections
                if self.task == "safer":
                    self.transfer_scenario_safer(scenario_id, scenario_cache, path, start_interval, end_interval)
                else:
                    self.transfer_scenario_cache2argo(scenario_id, scenario_cache, path)
            except Exception as e:
                print(e)
                continue
        return []

    def process_xiaoxin_agent(self, agents_array, scenario_id, start_timestamp, end_timestamp):
        def type_hash(x):
            type_re_hash = {
                0: "vehicle",
                1: "pedestrian",
                2: "cyclist",
                3: "background",
                4: "unknown"
            }
            return type_re_hash[x]

        columns = ['observed', 'track_id', 'object_type', 'object_category', 'timestep',
                   'position_x', 'position_y', 'heading', 'velocity_x', 'velocity_y',
                   'scenario_id', 'start_timestamp', 'end_timestamp', 'num_timestamps',
                   'focal_track_id', 'city']
        new_columns = np.ones((agents_array.shape[0], agents_array.shape[1], 9))
        for index in range(new_columns.shape[0]):
            new_columns[index, :, 0] = int(index)
        scored = agents_array[..., -1].all(axis=0).astype(bool)
        unscored = agents_array[..., -1].sum(axis=0) > 80
        shit = ~ (scored | unscored)
        new_columns[..., 1] = 1
        new_columns[:, unscored, 2] = 1
        new_columns[:, scored, 2] = 2
        new_columns[:, shit, 2] = 0
        new_columns[..., 3] = int(start_timestamp)
        new_columns[..., 4] = int(end_timestamp)
        new_columns[..., 5] = int(110)
        new_columns[..., 6] = agents_array[..., -2]
        new_columns[..., 7] = 10086
        new_columns[:50, :, 8] = True
        new_columns[50:, :, 8] = False
        new_columns = new_columns
        new_agents_array = np.concatenate([new_columns, agents_array], axis=-1)
        new_agents_array = new_agents_array[new_agents_array[..., -1] == 1.0].reshape(-1, new_agents_array.shape[-1])
        new_agents_array = new_agents_array[..., [8, -2, -3, 2, 0, 9, 10, 14, 12, 13, 1, 3, 4, 5, 6, 7]]
        new_agents_array = pd.DataFrame(data=new_agents_array, columns=columns)
        new_agents_array["object_type"] = new_agents_array["object_type"].apply(func=type_hash)
        new_agents_array["start_timestamp"] = new_agents_array["start_timestamp"].astype(int)
        new_agents_array["end_timestamp"] = new_agents_array["end_timestamp"].astype(int)
        new_agents_array["num_timestamps"] = new_agents_array["num_timestamps"].astype(int)
        new_agents_array["scenario_id"] = scenario_id
        new_agents_array.loc[new_agents_array["track_id"] == 0.0, "track_id"] = "AV"
        new_agents_array.loc[new_agents_array["track_id"] == "AV", "object_category"] = 3.0
        return new_agents_array

    def transfer_scenario_safer(self, data_file, raw_data, output_path, start_interval, end_interval):
        whole_range = 110
        his_horizon = 30
        end_interval = min(self.take_over + 250, end_interval)
        start_time = np.random.randint(self.take_over - whole_range,
                                       self.take_over - his_horizon) - start_interval
        start_time_list = [start_time]
        if end_interval - self.take_over > 2 * whole_range:
            start_time = np.random.randint(self.take_over + whole_range, end_interval - whole_range) - start_interval
            start_time_list.append(start_time)
        for cnt, start_time in enumerate(start_time_list):
            map_data = {}
            agent_data = raw_data['agent'][start_time: start_time + whole_range, ...]
            agent_data = process_sub_track(agent_data, 200)
            scenario_id = data_file + "_" + str(start_time)
            start_timestamp = 0
            end_timestamp = 110
            ego = agent_data[his_horizon, 0, :]
            agent = self.process_xiaoxin_agent(agent_data, scenario_id, start_timestamp, end_timestamp)
            sub_map = process_sublanes(raw_data['all_polylines'], ego, 0, 200)
            ref_line = process_subref(raw_data['refline'], ego, 0, 200)
            boundaries = process_subbounds(raw_data['boundaries'], ego, 0, 200)
            connections = process_connections(raw_data["connections"], ego, 0, 200)
            crosswalks = process_sub_crosswalks(raw_data["crosswalks"], ego, 0, 200)
            map_data["lane_segments"] = sub_map
            map_data["drivable_areas"] = boundaries
            map_data["connections"] = connections
            map_data["pedestrian_crossings"] = crosswalks
            save_path = os.path.join(output_path, scenario_id)
            scenario_take_over_time = (self.take_over - start_interval) - start_time
            if scenario_take_over_time >= 0:
                p = {"takeover_time": int(scenario_take_over_time),
                     "ref_line": ref_line}
            else:
                p = {"takeover_time": -1,
                     "ref_line": ref_line}
            if self.ceph is not None:
                self.ceph.save(agent, os.path.join(save_path, "scenario_{}.pkl".format(scenario_id)))
                self.ceph.save(map_data, os.path.join(save_path, "log_map_archive_{}.pkl".format(scenario_id)))
                self.ceph.save(p, os.path.join(save_path, "additional_info_{}.pkl".format(scenario_id)))
            else:
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                agent.to_parquet(os.path.join(save_path, "scenario_{}.parquet".format(scenario_id)))
                with open(os.path.join(save_path, "log_map_archive_{}.json".format(scenario_id)), 'w') as f:
                    json.dump(map_data, f)
                with open(os.path.join(save_path, "additional_info_{}.json".format(scenario_id)), 'w') as f:
                    json.dump(p, f)
                # self.draw_res(agent, map_data, cnt)

    def transfer_scenario_cache2argo(self, data_file, raw_data, output_path):
        time_slot = 50
        whole_range = 110
        prediction_horizon = 60
        clip_num = (raw_data['agent'].shape[0] - prediction_horizon) // time_slot
        for cnt in range(clip_num):
            map_data = {}
            agent_data = raw_data['agent'][cnt * time_slot: cnt * time_slot + whole_range, ...]
            agent_data = process_sub_track(agent_data, 150)
            scenario_id = data_file + "_" + str(cnt)
            start_timestamp = 0
            end_timestamp = 110
            # agents_array, scenario_id, start_timestamp, end_timestamp
            ego = agent_data[whole_range - prediction_horizon, 0, :]
            agent = self.process_xiaoxin_agent(agent_data, cnt, start_timestamp, end_timestamp)
            sub_map = process_sublanes(raw_data['all_polylines'], ego, 0, 200)
            ref_line = process_subref(raw_data['refline'], ego, 0, 150)
            boundaries = process_subbounds(raw_data['boundaries'], ego, 0, 200)
            connections = process_connections(raw_data["connections"], ego, 0, 200)
            crosswalks = process_sub_crosswalks(raw_data["crosswalks"], ego, 0, 200)
            # map_data["drivable_areas"] = sub_ref
            map_data["lane_segments"] = sub_map
            map_data["drivable_areas"] = boundaries
            map_data["connections"] = connections
            map_data["pedestrian_crossings"] = crosswalks
            save_path = os.path.join(output_path, scenario_id)
            if self.ceph is not None:
                p = {"ref_line": ref_line}
                self.ceph.save(agent, os.path.join(save_path, "scenario_{}.pkl".format(scenario_id)))
                self.ceph.save(map_data, os.path.join(save_path, "log_map_archive_{}.pkl".format(scenario_id)))
                self.ceph.save(p, os.path.join(save_path, "additional_info_{}.pkl".format(scenario_id)))
            else:
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                agent.to_pickle(os.path.join(save_path, "scenario_{}.pkl".format(scenario_id)))
                with open(os.path.join(save_path, "log_map_archive_{}.json".format(scenario_id)), 'w') as f:
                    json.dump(map_data, f)
                # self.draw_res(agent, map_data, cnt)

    def process_agent(self, agents, agents_p, start, end):
        type_hash = {
            "vehicle": 0,
            "pedestrain": 1,
            "bike": 2,
            "obstacle": 3,
            "unknown": 4
        }
        tracks = agents_p["track_id"].unique()
        if not isinstance(tracks[0], str):
            agents_p = agents_p.copy()
            agents_p["track_id"] = agents_p["track_id"].astype(str)
            tracks = agents_p["track_id"].unique()
        assert isinstance(tracks[0], str)
        agent_list = []
        for agent_id in agents.track_id.unique():
            assert isinstance(agent_id, str)
            if agent_id != "ego" and agent_id not in tracks:
                continue
            agent_copy = agents.loc[agents["track_id"] == agent_id]
            aget_start = agent_copy["interval"].min()
            agent_end = agent_copy["interval"].max()
            if agent_id != "ego":
                agent_p = np.array(agents_p.loc[agents_p["track_id"] == agent_id])[:, [2, 3, 4, 1, 1]]
                agent_p[:, 3] = type_hash[agent_p[:, 3].item()]
                agent_p[:, 4] = agent_id
            else:
                agent_p = np.ones([1, 5])
                agent_p[:, 0] = 2.2
                agent_p[:, 1] = 5.2
                agent_p[:, 2] = 2.3
                agent_p[:, 3] = int(0)
                agent_p[:, 4] = 0
            agent = np.array(agent_copy)[:, [2, 3, 4, 7, 8, 12]]
            valid = np.ones([agent.shape[0], 1])
            agent[:, 2] = 0
            agent_p = np.repeat(agent_p, repeats=agent.shape[0], axis=0)
            agent = np.concatenate([agent, agent_p, valid], axis=1)
            agent = np.pad(agent, ((aget_start - start, end - agent_end), (0, 0)))
            agent_list.append(agent)
        return np.stack(agent_list, axis=1)

    def topo_info(self, l2l_relation, lane_ids_hash):
        l2l_relation = l2l_relation.copy()
        l2l_relation["source"] = l2l_relation["source"].apply(lambda x: lane_ids_hash[x])
        l2l_relation["target"] = l2l_relation["target"].apply(lambda x: lane_ids_hash[x])
        lanes_topo = {}
        for lane_id in lane_ids_hash.values():
            relations = {}
            lane_topo = l2l_relation.loc[l2l_relation["source"] == lane_id]
            exits = lane_topo.loc[lane_topo["relation"].str.contains("next")]["target"].tolist()
            previous = lane_topo.loc[lane_topo["relation"].str.contains("previous")]["target"].tolist()
            left_neighbors = lane_topo.loc[lane_topo["relation"] == "left"]["target"].tolist()
            left_neighbors.extend(lane_topo.loc[lane_topo["relation"] == "junc_neighbor"]["target"].tolist())
            right_neighbors = lane_topo.loc[lane_topo["relation"] == "right"]["target"].tolist()
            relations["entry"] = previous
            relations["exits"] = exits
            relations["left_neighbors"] = left_neighbors
            relations["right_neighbors"] = right_neighbors
            lanes_topo[lane_id] = relations
        return lanes_topo

    def light_info(self, traffic_light, start_interval, end_interval):
        if traffic_light.shape[0] == 0:
            return None, None
        traffic_light = traffic_light.copy()
        traffic_light["interval"] -= start_interval
        slope = end_interval - start_interval
        traffic_light_ids = traffic_light["traffic_light_id"].unique()
        light_type_hash = {
            "green": 3,
            "green_flash": 3,
            "green_number": 3,
            "yellow": 2,
            "red": 1,
            "red_flash": 1,
            "black": 4,
            "yellow_flash": 2
        }
        traffic_light["to"] = traffic_light["to"].apply(lambda x: light_type_hash[x])
        all_lights_info = []
        for light_id in traffic_light_ids:
            light = traffic_light.loc[traffic_light["traffic_light_id"] == light_id]
            light_info = np.array(light)[:, [5, 1, 1, 1, 8, 4]]
            light_info[:, 2:4] = 0
            intervals = np.unique(light_info[:, 0])
            light_info_list = []
            for time_idx in range(len(intervals) - 1):
                interval = intervals[time_idx]
                next_interval = intervals[time_idx + 1]
                if next_interval < 0:
                    continue
                if interval > slope:
                    break
                if next_interval > slope:
                    next_interval = slope
                light_extent = light_info[light_info[:, 0] == interval]
                if interval < 0 and next_interval > 0:
                    interval = 0
                inteplate = np.arange(interval, next_interval, 1)
                repeat_nums = next_interval - interval
                light_extents = np.repeat(light_extent[:, np.newaxis, :], repeats=int(repeat_nums), axis=1)
                light_extents[:, :, 0] = inteplate
                light_extents = light_extents.reshape(-1, 6)
                light_info_list.append(light_extents)
            if len(light_info_list) != 0:
                light_info_list = np.concatenate(light_info_list, axis=0)
            else:
                continue
            all_lights_info.append(light_info_list)
        all_lights_info = np.concatenate(all_lights_info, axis=0)
        all_lights_info[:, 0] = all_lights_info[:, 0].astype(int)
        dynamic_map_infos = defaultdict(list)
        for idx in range(slope + 1):
            light = all_lights_info[all_lights_info[:, 0] == idx]
            if light.shape[0] == 0:
                dynamic_map_infos["lane_id"].append(np.array([]))
                dynamic_map_infos["state"].append(np.array([]))
            else:
                dynamic_map_infos["lane_id"].append(light[:, 1][np.newaxis, :])
                dynamic_map_infos["state"].append(light[:, 4][np.newaxis, :])
        if len(all_lights_info) != 0:
            return all_lights_info, dynamic_map_infos
        else:
            return None, None


