
from petrel_client.client import Client
import glob
import multiprocessing
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2

from waymo_types import lane_type, object_type, polyline_type, road_edge_type, road_line_type, signal_state
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

client = Client()


def decode_tracks_from_proto(tracks):
    track_infos = {
        'object_id': [],  # {0: unset, 1: vehicle, 2: pedestrian, 3: cyclist, 4: others}
        'object_type': [],
        'trajs': []
    }
    for cur_data in tracks:  # number of objects
        cur_traj = [np.array([x.center_x, x.center_y, x.center_z, x.length, x.width, x.height, x.heading,
                              x.velocity_x, x.velocity_y, x.valid], dtype=np.float32) for x in cur_data.states]
        cur_traj = np.stack(cur_traj, axis=0)  # (num_timestamp, 10)

        track_infos['object_id'].append(cur_data.id)
        track_infos['object_type'].append(object_type[cur_data.object_type])
        track_infos['trajs'].append(cur_traj)

    track_infos['trajs'] = np.stack(track_infos['trajs'], axis=0)  # (num_objects, num_timestamp, 9)
    return track_infos


def get_polyline_dir(polyline):
    polyline_pre = np.roll(polyline, shift=1, axis=0)
    polyline_pre[0] = polyline[0]
    diff = polyline - polyline_pre
    polyline_dir = diff / np.clip(np.linalg.norm(diff, axis=-1)[:, np.newaxis], a_min=1e-6, a_max=1000000000)
    return polyline_dir


def decode_map_features_from_proto(map_features, scenario_id):
    map_infos = {
        'lane': [],
        'road_line': [],
        'road_edge': [],
        'stop_sign': [],
        'crosswalk': [],
        'speed_bump': [],
        'lane_dict': {},
        'lane2other_dict': {}
    }
    polylines = []

    point_cnt = 0
    lane2other_dict = defaultdict(list)

    for cur_data in map_features:
        cur_info = {'id': cur_data.id}

        if cur_data.lane.ByteSize() > 0:
            cur_info['speed_limit_mph'] = cur_data.lane.speed_limit_mph
            cur_info['type'] = cur_data.lane.type + 1  # 0: undefined, 1: freeway, 2: surface_street, 3: bike_lane
            cur_info['left_neighbors'] = [lane.feature_id for lane in cur_data.lane.left_neighbors]
            cur_info['right_neighbors'] = [lane.feature_id for lane in cur_data.lane.right_neighbors]

            cur_info['interpolating'] = cur_data.lane.interpolating
            cur_info['entry_lanes'] = list(cur_data.lane.entry_lanes)
            cur_info['exit_lanes'] = list(cur_data.lane.exit_lanes)

            cur_info['left_boundary_type'] = [x.boundary_type + 5 for x in cur_data.lane.left_boundaries]
            cur_info['right_boundary_type'] = [x.boundary_type + 5 for x in cur_data.lane.right_boundaries]

            cur_info['left_boundary'] = [x.boundary_feature_id for x in cur_data.lane.left_boundaries]
            cur_info['right_boundary'] = [x.boundary_feature_id for x in cur_data.lane.right_boundaries]

            cur_info['left_boundary_start_index'] = [x.lane_start_index for x in cur_data.lane.left_boundaries]
            cur_info['right_boundary_start_index'] = [x.lane_start_index for x in cur_data.lane.right_boundaries]

            cur_info['left_boundary_end_index'] = [x.lane_end_index for x in cur_data.lane.left_boundaries]
            cur_info['right_boundary_end_index'] = [x.lane_end_index for x in cur_data.lane.right_boundaries]

            lane2other_dict[cur_data.id].extend(cur_info['left_boundary'])
            lane2other_dict[cur_data.id].extend(cur_info['right_boundary'])

            global_type = cur_info['type']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type, cur_data.id]) for point in cur_data.lane.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['lane'].append(cur_info)
            map_infos['lane_dict'][cur_data.id] = cur_info

        elif cur_data.road_line.ByteSize() > 0:
            cur_info['type'] = cur_data.road_line.type + 5

            global_type = cur_info['type']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type, cur_data.id]) for point in cur_data.road_line.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['road_line'].append(cur_info)

        elif cur_data.road_edge.ByteSize() > 0:
            cur_info['type'] = cur_data.road_edge.type + 14

            global_type = cur_info['type']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type, cur_data.id]) for point in cur_data.road_edge.polyline], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['road_edge'].append(cur_info)

        elif cur_data.stop_sign.ByteSize() > 0:
            cur_info['lane_ids'] = list(cur_data.stop_sign.lane)
            for i in cur_info['lane_ids']:
                lane2other_dict[i].append(cur_data.id)
            point = cur_data.stop_sign.position
            cur_info['position'] = np.array([point.x, point.y, point.z])

            global_type = polyline_type['TYPE_STOP_SIGN']
            cur_polyline = np.array([point.x, point.y, point.z, 0, 0, 0, global_type, cur_data.id]).reshape(1, 8)

            map_infos['stop_sign'].append(cur_info)
        elif cur_data.crosswalk.ByteSize() > 0:
            global_type = polyline_type['TYPE_CROSSWALK']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type, cur_data.id]) for point in cur_data.crosswalk.polygon], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['crosswalk'].append(cur_info)

        elif cur_data.speed_bump.ByteSize() > 0:
            global_type = polyline_type['TYPE_SPEED_BUMP']
            cur_polyline = np.stack([np.array([point.x, point.y, point.z, global_type, cur_data.id]) for point in cur_data.speed_bump.polygon], axis=0)
            cur_polyline_dir = get_polyline_dir(cur_polyline[:, 0:3])
            cur_polyline = np.concatenate((cur_polyline[:, 0:3], cur_polyline_dir, cur_polyline[:, 3:]), axis=-1)

            map_infos['speed_bump'].append(cur_info)

        else:
            # print(cur_data)
            continue

        polylines.append(cur_polyline)
        cur_info['polyline_index'] = (point_cnt, point_cnt + len(cur_polyline))
        point_cnt += len(cur_polyline)

    try:
        polylines = np.concatenate(polylines, axis=0).astype(np.float32)
    except Exception as e:
        polylines = np.zeros((0, 8), dtype=np.float32)
        # print(e)
        # print(scenario_id)
    map_infos['all_polylines'] = polylines
    map_infos['lane2other_dict'] = lane2other_dict
    return map_infos


def decode_dynamic_map_states_from_proto(dynamic_map_states):
    dynamic_map_infos = {
        'lane_id': [],
        'state': [],
        'stop_point': []
    }
    for cur_data in dynamic_map_states:  # (num_timestamp)
        lane_id, state, stop_point = [], [], []
        for cur_signal in cur_data.lane_states:  # (num_observed_signals)
            lane_id.append(cur_signal.lane)
            state.append(signal_state[cur_signal.state])
            stop_point.append([cur_signal.stop_point.x, cur_signal.stop_point.y, cur_signal.stop_point.z])

        dynamic_map_infos['lane_id'].append(np.array([lane_id]))
        dynamic_map_infos['state'].append(np.array([state]))
        dynamic_map_infos['stop_point'].append(np.array([stop_point]))

    return dynamic_map_infos


def process_waymo_data_with_scenario_proto(data_file, output_path=None, data_path=None):
    tmp_path = "/mnt/cache/gaoziyan/tmp_data/"
    dataset_file = client.get(data_path + data_file)
    f = open(tmp_path + data_file, 'wb')
    f.write(dataset_file)
    f.close()
    dataset = tf.data.TFRecordDataset(tmp_path+data_file, compression_type='')
    ret_infos = []

    # scenario_list = []
    for cnt, data in enumerate(dataset):
        info = {}
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(bytearray(data.numpy()))

        info['scenario_id'] = scenario.scenario_id
        info['timestamps_seconds'] = list(scenario.timestamps_seconds)  # list of int of shape (91)
        info['current_time_index'] = scenario.current_time_index  # int, 10
        info['sdc_track_index'] = scenario.sdc_track_index  # int
        info['objects_of_interest'] = list(scenario.objects_of_interest)  # list, could be empty list
        # if list(scenario.objects_of_interest) == []:
        #     print(len(scenario.tracks_to_predict))
        # else:
        #     print(len(list(scenario.objects_of_interest)))
    #     if len(info['objects_of_interest']) !=2 or len(set(info['objects_of_interest']))!=2:
    #         continue
        # if len(list(scenario.objects_of_interest)) != 2:
        #     print('pair', len(scenario.tracks_to_predict))

        # if len(scenario.tracks_to_predict) !=2:
        #     print(len(scenario.tracks_to_predict))
    #     scenario_list.append('sample_'+info['scenario_id']+'.pkl')
    # return scenario_list

        info['tracks_to_predict'] = {
            'track_index': [cur_pred.track_index for cur_pred in scenario.tracks_to_predict],
            'difficulty': [cur_pred.difficulty for cur_pred in scenario.tracks_to_predict]
        }  # for training: suggestion of objects to train on, for val/test: need to be predicted

        track_infos = decode_tracks_from_proto(scenario.tracks)
        info['tracks_to_predict']['object_type'] = [track_infos['object_type'][cur_idx] for cur_idx in info['tracks_to_predict']['track_index']]

        # decode map related data
        if not scenario.map_features:
            print(scenario.scenario_id)
        map_infos = decode_map_features_from_proto(scenario.map_features, data_file)
        dynamic_map_infos = decode_dynamic_map_states_from_proto(scenario.dynamic_map_states)

        save_infos = {
            'track_infos': track_infos,
            'dynamic_map_infos': dynamic_map_infos,
            'map_infos': map_infos
        }
        save_infos.update(info)

        # output_file = os.path.join(output_path, f'sample_{scenario.scenario_id}.pkl')
        # with open(output_file, 'wb') as f:
        #     pickle.dump(save_infos, f)
        output_file = output_path + f'sample_{scenario.scenario_id}.pkl'

        save_info_file = pickle.dumps(save_infos)
        client.put(output_file, save_info_file)

        ret_infos.append(info)
    print('finish processing.')
    if os.path.exists(tmp_path+data_file):
        os.remove(tmp_path+data_file)
        print("**************************************delete ", tmp_path+data_file)
    return ret_infos


def get_infos_from_protos(data_path, output_path=None, num_workers=36):
    from functools import partial
    # if output_path is not None:
    #     os.makedirs(output_path, exist_ok=True)

    func = partial(
        process_waymo_data_with_scenario_proto, output_path=output_path, data_path=data_path
    )

    src_files = client.list(data_path)
    print(data_path)
    src_files = [str(file_name) for file_name in src_files]
    print(len(src_files))
    # src_files = glob.glob(os.path.join(data_path, '*.tfrecord*'))
    # src_files.sort()
    # print(src_files)
    # all_infos = []
    # for i in tqdm(src_files):
    #     all_infos.append(func(i))

    with multiprocessing.Pool(num_workers) as p:
        data_infos = list(tqdm(p.imap_unordered(func, src_files), total=len(src_files)))
    all_infos = [item for infos in data_infos for item in infos]
    return all_infos


def create_infos_from_protos(raw_data_path, output_path, num_workers=48):
    train_infos = get_infos_from_protos(
        data_path=raw_data_path,
        output_path=output_path,
        num_workers=num_workers
    )
    train_filename = output_path + 'processed_scenarios_val_infos.pkl'
    train_info_file = pickle.dumps(train_infos)
    client.put(train_filename, train_info_file)
    print('----------------Waymo info train file is saved to %s----------------' % train_filename)
