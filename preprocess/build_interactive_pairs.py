from utils.cluster_reader import LoadScenarioFromCeph
from tqdm import tqdm
from multiprocessing import Pool
import torch
import os
from waymo_open_dataset.wdl_limited.sim_agents_metrics.interaction_features import compute_time_to_collision_with_object_in_front
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def process_scenario(path):
    file_path, dir_path, write_dir, src_dir_path = path
    loader = LoadScenarioFromCeph()
    data = loader.read(dir_path + file_path)
    objects_of_interest_src = loader.read(src_dir_path + file_path)
    objects_of_interest = find_interaction_pair(data, objects_of_interest_src['objects_of_interest'])
    if len(objects_of_interest) == 2 and isinstance(objects_of_interest[0], int):
        data['objects_of_interest'] = [objects_of_interest]
        loader.save(data, write_dir + file_path)
        return file_path

    if len(objects_of_interest) >= 1 and len(objects_of_interest[0]) == 2:
        if [] in data['objects_of_interest']:
            tmp = [i for i in data['objects_of_interest'] if i != []]
            data['objects_of_interest'] = tmp
        loader.save(data, write_dir + file_path)
        return file_path

    return None


def find_interaction_pair(data, objects_of_interest_src):
    agent = data['agent']
    eval_mask = agent['category'] == 3
    objects_of_interest = []
    if eval_mask.sum() >= 2:
        valid_index = torch.where(eval_mask)[0]
        index_pair = torch.combinations(valid_index)
        for i in range(index_pair.shape[0]):
            center_x = agent['position'][:, :, 0][index_pair[i]].numpy()
            center_y = agent['position'][:, :, 1][index_pair[i]].numpy()
            length = agent['shape'][:, :, 0][index_pair[i]].numpy()
            width = agent['shape'][:, :, 1][index_pair[i]].numpy()
            heading = agent['heading'][index_pair[i]].numpy()
            valid = agent['valid_mask'][index_pair[i]].numpy()
            evaluated_object_mask = eval_mask[index_pair[i]].numpy()
            agent_id = torch.tensor(agent['id'])[index_pair[i]]

            evaluated_object_mask[:] = True
            ttc = compute_time_to_collision_with_object_in_front(center_x=center_x, center_y=center_y, length=length, width=width, heading=heading, valid=valid, evaluated_object_mask=evaluated_object_mask, seconds_per_step=0.1)
            if (ttc.numpy() < 10).any():
                objects_of_interest.append(agent_id.int().tolist())
    if len(objects_of_interest_src) == 2 and objects_of_interest_src not in objects_of_interest:
        objects_of_interest.append(objects_of_interest_src)
    return objects_of_interest


def main(dir_path, write_dir, output_path, src_dir_path):
    loader = LoadScenarioFromCeph()
    file_names = [name for name in loader.list(dir_path)]
    # file_names = loader.read(output_path)

    with Pool(processes=48) as pool:
        interactive_list = list(tqdm(pool.imap_unordered(process_scenario, [(file, dir_path, write_dir, src_dir_path) for file in file_names]), total=len(file_names)))

    interactive_list = [i for i in interactive_list if i is not None]
    loader.save(interactive_list, output_path)
