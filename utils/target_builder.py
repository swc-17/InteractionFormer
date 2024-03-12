
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import BaseTransform
from utils import wrap_angle
from augumentation.kinematic_history_agent_augmentation import KinematicHistoryAgentAugmentor


def to_16(data):
    if isinstance(data, dict):
        for key, value in data.items():
            new_value = to_16(value)
            data[key] = new_value
    if isinstance(data, torch.Tensor):
        if data.dtype == torch.float32:
            data = data.to(torch.float16)
    return data


class TargetBuilder(BaseTransform):

    def __init__(self,
                 num_historical_steps: int,
                 num_future_steps: int) -> None:
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.use_raw_heading = True

    def do_shit_easy(self, agent, mask):
        excluded = ["num_nodes", "av_index"]
        for key, val in agent.items():
            if key in excluded:
                continue
            if key == "id":
                val = list(np.array(val)[mask])
                agent[key] = val
                continue
            if len(val.size()) > 1:
                agent[key] = val[mask, ...]
            else:
                agent[key] = val[mask]
        return agent

    def kinematic_augmentation(self, ego, target):
        augment_prob = 1.0
        dt = 0.1
        mean = [0.1, 0.05, 0]
        std = [0.1, 0.05, 0.01]
        low = [-1.0, -0.5, -0.2]
        high = [1.0, 0.5, 0.2]
        self.gaussian_augmentor = KinematicHistoryAgentAugmentor(
            dt, mean, std, low, high, augment_prob, use_uniform_noise=True
        )
        aug_feature, aug_target = self.gaussian_augmentor.augment(ego, target)
        return aug_feature, aug_target

    def clip(self, agent, max_num=32):
        av_index = agent["av_index"]
        valid = agent['valid_mask']
        valid_current = valid[:, self.num_historical_steps:]
        valid_counts = valid_current.sum(1)
        sort_idx = valid_counts.sort()[1]
        mask = torch.zeros(valid.shape[0])
        mask[sort_idx[-max_num:]] = 1
        mask = mask.to(torch.bool)
        mask[av_index] = True
        new_av_index = mask[:av_index].sum()
        agent["num_nodes"] = int(mask.sum())
        agent["av_index"] = int(new_av_index)
        agent = self.do_shit_easy(agent, mask)
        return agent

    def __call__(self, data) -> HeteroData:
        # agent = data['agent']
        # if agent["num_nodes"] > 50:
        #     agent = self.clip(agent, 50)
        #     data['agent'] = agent

        origin = data['agent']['position'][:, self.num_historical_steps - 1]
        theta = data['agent']['heading'][:, self.num_historical_steps - 1]
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(data['agent']['num_nodes'], 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        data['agent']['target'] = origin.new_zeros(data['agent']['num_nodes'], self.num_future_steps, 4)
        data['agent']['target'][..., :2] = torch.bmm(data['agent']['position'][:, self.num_historical_steps:, :2] -
                                                     origin[:, :2].unsqueeze(1), rot_mat)
        if data['agent']['position'].size(2) == 3:
            data['agent']['target'][..., 2] = (data['agent']['position'][:, self.num_historical_steps:, 2] -
                                               origin[:, 2].unsqueeze(-1))
        data['agent']['target'][..., 3] = wrap_angle(data['agent']['heading'][:, self.num_historical_steps:] -
                                                     theta.unsqueeze(-1))
        return HeteroData(data)


class WaymoTargetBuilder(BaseTransform):

    def __init__(self,
                 num_historical_steps: int,
                 num_future_steps: int,
                 mode="train") -> None:
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.mode = mode
        self.num_features = 3

    def clip(self, agent, max_num=32):
        av_index = agent["av_index"]
        valid = agent['valid_mask']
        ego_pos = agent["position"][av_index]
        distance = torch.norm(agent["position"][:, self.num_historical_steps-1, :2] - ego_pos[self.num_historical_steps-1, :2], dim=-1)  # keep the closest 100 vehicles near the ego car
        sort_idx = distance.sort()[1]
        mask = torch.zeros(valid.shape[0])
        mask[sort_idx[:max_num]] = 1
        mask = mask.to(torch.bool)
        mask[av_index] = True
        new_av_index = mask[:av_index].sum()
        agent["num_nodes"] = int(mask.sum())
        agent["av_index"] = int(new_av_index)
        excluded = ["num_nodes", "av_index"]
        for key, val in agent.items():
            if key in excluded:
                continue
            if key == "id":
                val = list(np.array(val)[mask])
                agent[key] = val
                continue
            if len(val.size()) > 1:
                agent[key] = val[mask, ...]
            else:
                agent[key] = val[mask]
        return agent

    def score_ego_agent_only(self, agent):
        av_index = agent['av_index']
        agent["category"][av_index] = 6
        return agent

    def score_sensead_vehicle(self, agent, max_num=10):
        av_index = agent['av_index']
        agent["category"] = torch.zeros_like(agent["category"])
        pos = agent["position"][av_index, self.num_historical_steps, :2]
        distance = torch.norm(agent["position"][:, self.num_historical_steps, :2] - pos, dim=-1)
        distance_all_time = torch.norm(agent["position"][:, :, :2] - agent["position"][av_index, :, :2], dim=-1)
        invalid_mask = distance_all_time < 100
        agent["valid_mask"] = agent["valid_mask"] * invalid_mask
        closet_vehicle = distance < 50
        valid = agent['valid_mask']
        valid_current = valid[:, (self.num_historical_steps - 10):]
        valid_counts = valid_current.sum(1)
        counts_vehicle = valid_counts >= 61
        no_backgroud = agent['type'] != 3
        vehicle2pred = closet_vehicle & counts_vehicle & no_backgroud
        if vehicle2pred.sum() > max_num:
            inds_last = (valid * torch.arange(1, valid.size(-1) + 1, device=valid.device)).argmax(dim=-1)
            move_distances = agent["position"][torch.arange(valid.shape[0]), inds_last, :2] - agent["position"][:, self.num_historical_steps, :2]
            move_distance = torch.norm(move_distances, dim=-1)
            sort_dis, sort_idx = move_distance.sort()
            score_max = sort_idx[-max_num:]
            velocity_score = torch.zeros_like(counts_vehicle).to(bool)
            velocity_score[score_max] = True
            vehicle2pred = velocity_score & vehicle2pred
        agent["category"][vehicle2pred] = 3

    def kinematic_augmentation(self, ego, target):
        augment_prob = 1.0
        dt = 0.1
        mean = [0.1, 0.05, 0]
        std = [0.1, 0.05, 0.01]
        low = [-1.0, -0.5, -0.2]
        high = [1.0, 0.5, 0.2]
        self.gaussian_augmentor = KinematicHistoryAgentAugmentor(
            dt, mean, std, low, high, augment_prob, use_uniform_noise=True
        )
        aug_feature, aug_target = self.gaussian_augmentor.augment(ego, target)
        return aug_feature, aug_target

    def rotate_agents(self, position, heading, num_nodes, num_historical_steps, num_future_steps):
        origin = position[:, num_historical_steps - 1]
        theta = heading[:, num_historical_steps - 1]
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(num_nodes, 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        target = origin.new_zeros(num_nodes, num_future_steps, 3)
        target[..., :2] = torch.bmm(position[:, num_historical_steps:, :2] -
                                    origin[:, :2].unsqueeze(1), rot_mat)
        his = origin.new_zeros(num_nodes, num_historical_steps, 3)
        his[..., :2] = torch.bmm(position[:, :num_historical_steps, :2] -
                                 origin[:, :2].unsqueeze(1), rot_mat)
        if position.size(2) == 3:
            target[..., 2] = (position[:, num_historical_steps:, 2] -
                              origin[:, 2].unsqueeze(-1))
            his[..., 2] = (position[:, :num_historical_steps, 2] -
                           origin[:, 2].unsqueeze(-1))
            target[..., 3] = wrap_angle(heading[:, num_historical_steps:] -
                                        theta.unsqueeze(-1))
            his[..., 3] = wrap_angle(heading[:, :num_historical_steps] -
                                     theta.unsqueeze(-1))
        else:
            target[..., 2] = wrap_angle(heading[:, num_historical_steps:] -
                                        theta.unsqueeze(-1))
            his[..., 2] = wrap_angle(heading[:, :num_historical_steps] -
                                     theta.unsqueeze(-1))
        return his, target

    def __call__(self, data) -> HeteroData:
        agent = data['agent']
        av_index = agent["av_index"]
        data['agent']['position'] = data['agent']['position'][..., :2]
        coordinate = data['agent']['position'][av_index, 49, :2]
        data['agent']['position'] = data['agent']['position'][..., :2] - coordinate
        data['map_point']['position'] = data['map_point']['position'][..., :2] - coordinate
        data['map_polygon']['position'] = data['map_polygon']['position'][..., :2] - coordinate
        if agent["num_nodes"] > 50:
            agent = self.clip(agent, 50)
        self.score_ego_agent_only(agent)
        his, target = self.rotate_agents(data['agent']['position'],
                                         data["agent"]['heading'],
                                         data['agent']['num_nodes'], self.num_historical_steps, self.num_future_steps)
        data["agent"]["his"] = his
        data["agent"]["target"] = target

        av_index = agent["av_index"]
        ego_his, ego_target = self.rotate_agents(data['agent']['position'][av_index].unsqueeze(0),
                                                 data["agent"]['heading'][av_index].unsqueeze(0),
                                                 1, 50, 60)

        ego_valid_mask = data['agent']["valid_mask"][:, :][av_index]
        ego_his = ego_his[ego_valid_mask[:self.num_historical_steps].unsqueeze(0)]
        ego_target = ego_target[ego_valid_mask[self.num_historical_steps:].unsqueeze(0)]
        aug_feature, aug_target = self.kinematic_augmentation(ego_his.cpu().numpy(),
                                                              ego_target.cpu().numpy())

        smooth_ego = torch.from_numpy(aug_feature).to(ego_valid_mask.device).to(torch.float32)
        aug_target = torch.from_numpy(aug_target).to(ego_valid_mask.device).to(torch.float32)
        fully_ego = torch.cat([smooth_ego, aug_target], dim=0)
        origin = data["agent"]["position"][av_index][self.num_historical_steps - 1]
        theta = -data['agent']['heading'][[av_index], self.num_historical_steps - 1]
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(1, 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos
        fully_ego = fully_ego.unsqueeze(0)
        final_ego = torch.bmm(fully_ego[:, :, :2], rot_mat) + origin
        new_head = wrap_angle(fully_ego[:, :, 2] - theta)
        final_ego = final_ego[0]
        data['agent']['position'][av_index, ego_valid_mask] = final_ego
        data['agent']['heading'][av_index, ego_valid_mask] = new_head[0]
        data["agent"]["target"][av_index] = aug_target
        return HeteroData(data)
