

import os
import pickle
from typing import Callable, List, Optional, Tuple, Union
import random
import pandas as pd
import torch
from torch_geometric.data import Dataset
from utils.log import Logging
from waymo_open_dataset.wdl_limited.sim_agents_metrics.interaction_features import compute_time_to_collision_with_object_in_front


class WaymoDataset(Dataset):

    def __init__(self,
                 root: str,
                 split: str,
                 raw_dir: Optional[str] = None,
                 processed_dir: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 cluster: bool = False) -> None:
        self.logger = Logging().log(level='DEBUG')
        self.root = root
        self.cluster = cluster
        if split not in ('train', 'val', 'test'):
            raise ValueError(f'{split} is not a valid split')
        self.split = split
        if processed_dir is not None:
            processed_dir = os.path.expanduser(os.path.normpath(processed_dir))
            self._processed_dir = processed_dir
            if os.path.isdir(self._processed_dir):
                if self.split == "test":
                    self._processed_file_names = os.listdir(self._processed_dir)
                else:
                    self._processed_file_names = os.listdir(self._processed_dir)
            else:
                self._processed_file_names = []
        if self.root is not None:
            split_datainfo = os.path.join(root, "split_datainfo.pkl")
            with open(split_datainfo, 'rb+') as f:
                split_datainfo = pickle.load(f)
            if split == "test":
                split = "val"
            self._processed_file_names = split_datainfo[split]

        self._num_samples = len(self._processed_file_names) - 1
        self.logger.debug("The number of {} dataset is ".format(split) + str(self._num_samples))
        super(WaymoDataset, self).__init__(root=root, transform=transform, pre_transform=None, pre_filter=None)

    @property
    def processed_dir(self) -> str:
        return self._processed_dir

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names

    def len(self) -> int:
        return self._num_samples

    def get(self, idx: int):
        with open(self.processed_paths[idx], 'rb') as handle:
            data = pickle.load(handle)
            data["scenario_id"] = self.processed_paths[idx].split('/')[-1]
            if self.split == 'train':
                if len(data['objects_of_interest']) > 3:
                    print(len(data['objects_of_interest']))
                    data['objects_of_interest'][:-1] = random.sample(data['objects_of_interest'][:-1], 2)
                data['objects_of_interest'] = [list(i) for i in list(set([tuple(sorted(i)) for i in data['objects_of_interest']]))]
                # data['objects_of_interest'] = data['objects_of_interest'][-1:]
            else:
                data['objects_of_interest'] = data['objects_of_interest'][-1:]
            # self.find_interaction_pair(data)
            # self.build_pair(data)
        return data

    def build_pair(self, data):
        agent = data['agent']
        eval_mask = agent['category'] == 3
        if eval_mask.sum() >= 2:
            data['objects_of_interest'] = torch.tensor(agent['id'])[eval_mask].to(torch.int).tolist()[:2]
        elif eval_mask.sum() == 1:
            other_index = (agent['position'][:, 10, :2] - agent['position'][:, 10, :2][eval_mask]).norm(dim=1).topk(k=2, largest=False)[1][1]
            eval_mask[other_index] = True
            data['objects_of_interest'] = torch.tensor(agent['id'])[eval_mask].to(torch.int).tolist()[:2]
        else:
            eval_mask = agent['valid_mask'][:, 10]
            data['objects_of_interest'] = torch.tensor(agent['id'])[eval_mask].to(torch.int).tolist()[:2]

    def find_interaction_pair(self, data):
        agent = data['agent']
        eval_mask = agent['category'] == 3
        objects_of_interest = []

        if eval_mask.sum() >= 2:
            # valid_index = torch.arange(eval_mask.sum())
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
                # evaluated_object_mask[index_pair[i]] = True
                ttc = compute_time_to_collision_with_object_in_front(center_x=center_x, center_y=center_y, length=length, width=width, heading=heading, valid=valid, evaluated_object_mask=evaluated_object_mask, seconds_per_step=0.1)
                if (ttc.numpy() < 10).any():
                    objects_of_interest.append(agent_id.int().tolist())
