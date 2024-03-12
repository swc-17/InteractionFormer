
import math
from typing import Dict, List, Mapping, Optional
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius
from torch_cluster import radius_graph
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
from torch_geometric.utils import dense_to_sparse

from layers import AttentionLayer
from layers import FourierEmbedding
from layers import MLPLayer
from utils import angle_between_2d_vectors
from utils import bipartite_dense_to_sparse
from utils import weight_init
from utils import wrap_angle


class InteractionDecoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 output_head: bool,
                 num_historical_steps: int,
                 num_future_steps: int,
                 num_modes: int,
                 num_recurrent_steps: int,
                 num_t2m_steps: Optional[int],
                 pl2m_radius: float,
                 a2m_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float) -> None:
        super(InteractionDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_head = output_head
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_modes = num_modes
        self.num_recurrent_steps = num_recurrent_steps
        self.num_t2m_steps = num_t2m_steps if num_t2m_steps is not None else num_historical_steps
        self.pl2m_radius = pl2m_radius
        self.a2m_radius = a2m_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        input_dim_r_t = 4
        input_dim_r_pl2m = 3
        input_dim_r_a2m = 3

        self.mode_emb = nn.Embedding(num_modes, hidden_dim)
        self.r_t2m_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pl2m_emb = FourierEmbedding(input_dim=input_dim_r_pl2m, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2m_emb = FourierEmbedding(input_dim=input_dim_r_a2m, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.y_emb = FourierEmbedding(input_dim=output_dim + output_head, hidden_dim=hidden_dim,
                                      num_freq_bands=num_freq_bands)
        self.traj_emb = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, bias=True,
                               batch_first=False, dropout=0.0, bidirectional=False)
        self.traj_emb_h0 = nn.Parameter(torch.zeros(1, hidden_dim))
        self.t2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_propose_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.m2m_propose_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                     dropout=dropout, bipartite=False, has_pos_emb=False)
        self.t2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pl2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2m_refine_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.m2m_refine_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                    dropout=dropout, bipartite=False, has_pos_emb=False)
        self.to_loc_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                           output_dim=num_future_steps * output_dim // num_recurrent_steps)
        self.to_scale_propose_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                             output_dim=num_future_steps * output_dim // num_recurrent_steps)
        self.to_loc_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                          output_dim=num_future_steps * output_dim)
        self.to_scale_refine_pos = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                            output_dim=num_future_steps * output_dim)
        if output_head:
            self.to_loc_propose_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=num_future_steps // num_recurrent_steps)
            self.to_conc_propose_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                 output_dim=num_future_steps // num_recurrent_steps)
            self.to_loc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=num_future_steps)
            self.to_conc_refine_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                                output_dim=num_future_steps)
        else:
            self.to_loc_propose_head = None
            self.to_conc_propose_head = None
            self.to_loc_refine_head = None
            self.to_conc_refine_head = None
        self.to_pi = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=1)
        self.sample_future_dynamic = 4

        self.matching = True
        self.multi_pair = True
        if self.matching:
            self.r_m2m_emb_dynamic = FourierEmbedding(input_dim=input_dim_r_pl2m, hidden_dim=hidden_dim,
                                                      num_freq_bands=num_freq_bands)
            self.cross_m2m_propose_attn_layer = AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                                               dropout=dropout, bipartite=False, has_pos_emb=False)

            self.matching_cls = nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            self.matching_cls.apply(weight_init)

            self.matching_cls_pi = nn.Sequential(
                nn.Linear(hidden_dim*2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            self.matching_cls_pi.apply(weight_init)

            self.r_m2m_emb_dynamic_refine = FourierEmbedding(input_dim=input_dim_r_pl2m, hidden_dim=hidden_dim,
                                                             num_freq_bands=num_freq_bands)
            self.cross_m2m_refine_attn_layer = nn.ModuleList(
                [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim,
                                dropout=dropout, bipartite=False, has_pos_emb=False) for _ in range(num_layers)]
            )

        self.apply(weight_init)

    def forward(self,
                data: HeteroData,
                scene_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pos_m = data['agent']['position'][:, self.num_historical_steps - 1, :self.input_dim]
        head_m = data['agent']['heading'][:, self.num_historical_steps - 1]
        head_vector_m = torch.stack([head_m.cos(), head_m.sin()], dim=-1)

        x_t = scene_enc['x_a'].reshape(-1, self.hidden_dim)
        x_pl = scene_enc['x_pl'][:, self.num_historical_steps - 1].repeat(self.num_modes, 1)
        x_a = scene_enc['x_a'][:, -1].repeat(self.num_modes, 1)

        if self.multi_pair and self.training:
            matching_num = [len(i) for i in data['objects_of_interest']]

            matching_ids_single = [i for k in data['objects_of_interest'] for i in k]
            matching_ids_single = [i for k in matching_ids_single for i in k]
            matching_ids_batch = [i for i in range(data['agent']['batch'].max()+1) for _ in range(matching_num[i]*2)]
            matching_ids = [str(matching_ids_single[i]) + '_'+str(matching_ids_batch[i]) for i in range(len(matching_ids_single))]
            matching_ids = list(OrderedDict.fromkeys(matching_ids))
            agent_id = np.array([i for k in data['agent']['id'] for i in k]).astype(np.int)
            agent_batch = data['agent']['batch'].tolist()
            agent_id = [str(agent_id[i]) + '_'+str(agent_batch[i]) for i in range(len(agent_batch))]
            matching_mask = np.isin(np.array(agent_id), np.array(matching_ids))
            matching_ids = np.array(agent_id)[matching_mask]
            matching_mask = torch.from_numpy(matching_mask).to(x_a.device)
            matching_index = torch.cat((torch.arange(6)[:, None].repeat_interleave(6, 0),
                                        torch.arange(6)[:, None].repeat(6, 1)), dim=-1)[None, ...].repeat(sum(matching_num), 1, 1).to(x_a.device)

            matching_ids_pair = [
                [f"{k}_{i}" for k in obj]
                for i, objs in enumerate(data['objects_of_interest'])
                for obj in objs
            ]
            matching_pair_indices = torch.tensor([[np.where(matching_ids == elem)[0][0] for elem in row]
                                                  for row in np.array(matching_ids_pair)]).to(x_a.device)
        else:
            matching_ids = [i for k in data['objects_of_interest'] for i in k]
            matching_ids_batch = [i for i in range(data['agent']['batch'].max()+1) for _ in range(2)]
            matching_ids = [str(matching_ids[i]) + '_'+str(matching_ids_batch[i]) for i in range(len(matching_ids))]
            agent_id = np.array([i for k in data['agent']['id'] for i in k]).astype(np.int)
            agent_batch = data['agent']['batch'].tolist()
            agent_id = [str(agent_id[i]) + '_'+str(agent_batch[i]) for i in range(len(agent_batch))]
            matching_mask = torch.from_numpy(np.isin(np.array(agent_id), np.array(matching_ids))).to(x_a.device)
            matching_index = torch.cat((torch.arange(6)[:, None].repeat_interleave(6, 0), torch.arange(6)[:, None].repeat(6, 1)), dim=-1)[None, ...].repeat(data['agent']['batch'].max()+1, 1, 1).to(x_a.device)
            matching_pair_indices = None

        matching_score = None
        matching_score_pi = None

        m = self.mode_emb.weight.repeat(scene_enc['x_a'].size(0), 1)

        mask_src = data['agent']['valid_mask'][:, :self.num_historical_steps].contiguous()
        mask_src[:, :self.num_historical_steps - self.num_t2m_steps] = False
        mask_dst = data['agent']['valid_mask'].any(dim=-1, keepdim=True)
        mask_dst[~matching_mask] = False
        mask_dst = mask_dst.repeat(1, self.num_modes)

        pos_t = data['agent']['position'][:, :self.num_historical_steps, :self.input_dim].reshape(-1, self.input_dim)
        head_t = data['agent']['heading'][:, :self.num_historical_steps].reshape(-1)
        edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst[:, -1:].unsqueeze(1))
        rel_pos_t2m = pos_t[edge_index_t2m[0]] - pos_m[edge_index_t2m[1]]
        rel_head_t2m = wrap_angle(head_t[edge_index_t2m[0]] - head_m[edge_index_t2m[1]])
        r_t2m = torch.stack(
            [torch.norm(rel_pos_t2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_t2m[1]], nbr_vector=rel_pos_t2m[:, :2]),
             rel_head_t2m,
             (edge_index_t2m[0] % self.num_historical_steps) - self.num_historical_steps + 1], dim=-1)
        r_t2m = self.r_t2m_emb(continuous_inputs=r_t2m, categorical_embs=None)
        edge_index_t2m = bipartite_dense_to_sparse(mask_src.unsqueeze(2) & mask_dst.unsqueeze(1))
        r_t2m = r_t2m.repeat_interleave(repeats=self.num_modes, dim=0)

        pos_pl = data['map_polygon']['position'][:, :self.input_dim]
        orient_pl = data['map_polygon']['orientation']

        edge_index_pl2m = radius(
            x=pos_m[:, :2],
            y=pos_pl[:, :2],
            r=self.pl2m_radius,
            batch_x=data['agent']['batch'] if isinstance(data, Batch) else None,
            batch_y=data['map_polygon']['batch'] if isinstance(data, Batch) else None,
            max_num_neighbors=300)
        edge_index_pl2m = edge_index_pl2m[:, mask_dst[edge_index_pl2m[1], 0]]
        rel_pos_pl2m = pos_pl[edge_index_pl2m[0]] - pos_m[edge_index_pl2m[1]]
        rel_orient_pl2m = wrap_angle(orient_pl[edge_index_pl2m[0]] - head_m[edge_index_pl2m[1]])
        r_pl2m = torch.stack(
            [torch.norm(rel_pos_pl2m[:, :2], p=2, dim=-1),
                angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_pl2m[1]], nbr_vector=rel_pos_pl2m[:, :2]),
                rel_orient_pl2m], dim=-1)
        r_pl2m = self.r_pl2m_emb(continuous_inputs=r_pl2m, categorical_embs=None)
        edge_index_pl2m = torch.cat([edge_index_pl2m + i * edge_index_pl2m.new_tensor(
            [[data['map_polygon']['num_nodes']], [data['agent']['num_nodes']]]) for i in range(self.num_modes)], dim=1)
        r_pl2m = r_pl2m.repeat(self.num_modes, 1)

        edge_index_a2m = radius_graph(
            x=pos_m[:, :2],
            r=self.a2m_radius,
            batch=data['agent']['batch'] if isinstance(data, Batch) else None,
            loop=False,
            max_num_neighbors=300)
        edge_index_a2m = edge_index_a2m[:, mask_src[:, -1][edge_index_a2m[0]] & mask_dst[edge_index_a2m[1], 0]]
        rel_pos_a2m = pos_m[edge_index_a2m[0]] - pos_m[edge_index_a2m[1]]
        rel_head_a2m = wrap_angle(head_m[edge_index_a2m[0]] - head_m[edge_index_a2m[1]])
        r_a2m = torch.stack(
            [torch.norm(rel_pos_a2m[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_m[edge_index_a2m[1]], nbr_vector=rel_pos_a2m[:, :2]),
             rel_head_a2m], dim=-1)
        r_a2m_src = self.r_a2m_emb(continuous_inputs=r_a2m, categorical_embs=None)
        edge_index_a2m = torch.cat(
            [edge_index_a2m + i * edge_index_a2m.new_tensor([data['agent']['num_nodes']]) for i in
             range(self.num_modes)], dim=1)
        r_a2m = r_a2m_src.repeat(self.num_modes, 1)

        edge_index_m2m = dense_to_sparse(mask_dst.unsqueeze(2) & mask_dst.unsqueeze(1))[0]

        locs_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        scales_propose_pos: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        locs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        concs_propose_head: List[Optional[torch.Tensor]] = [None] * self.num_recurrent_steps
        loc_propose_pos_t = None
        for t in range(self.num_recurrent_steps):
            for i in range(self.num_layers):
                m = m.reshape(-1, self.hidden_dim)
                m = self.t2m_propose_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
                m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
                m = self.pl2m_propose_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
                m = self.a2m_propose_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
                m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)

            m = self.m2m_propose_attn_layer(m, None, edge_index_m2m)
            m = m.reshape(-1, self.num_modes, self.hidden_dim)
            locs_propose_pos[t] = self.to_loc_propose_pos(m)
            scales_propose_pos[t] = self.to_scale_propose_pos(m)
            loc_propose_pos_t = torch.cumsum(
                torch.cat(locs_propose_pos[:t + 1], dim=-1).view(-1, self.num_modes,
                                                                 self.num_future_steps // self.num_recurrent_steps * (t + 1),
                                                                 self.output_dim), dim=-2)[:, :, -self.num_future_steps // self.num_recurrent_steps:, :]
            if self.matching:
                agent_batch = data['agent']['batch'].unsqueeze(1)
                unique_mask = ~torch.eye(agent_batch.shape[0], dtype=torch.bool).to(m.device)
                m2m_index = dense_to_sparse((mask_dst[:, 0].unsqueeze(1) & mask_dst[:, 0].unsqueeze(0)) & (agent_batch == agent_batch.t()) & unique_mask)[0]

                m2m_index = torch.cat(
                    [m2m_index + i * m2m_index.new_tensor([data['agent']['num_nodes']]) for i in
                     range(self.num_modes)], dim=1)
                sample_index = torch.linspace(0, loc_propose_pos_t.shape[2] - 1, self.sample_future_dynamic,
                                              dtype=torch.int32).to(loc_propose_pos_t.device)
                loc_propose_pos_t = torch.index_select(loc_propose_pos_t, dim=2, index=sample_index)
                num_future_steps = self.sample_future_dynamic
                theta = data['agent']['heading'][:, self.num_historical_steps - 1]
                cos, sin = theta.cos(), theta.sin()
                val_agent_num = loc_propose_pos_t.shape[0]
                rot_mat = theta.new_zeros(val_agent_num, 2, 2)
                rot_mat[:, 0, 0] = cos
                rot_mat[:, 0, 1] = sin
                rot_mat[:, 1, 0] = -sin
                rot_mat[:, 1, 1] = cos
                pos_m_dynamic = torch.bmm(loc_propose_pos_t.reshape(val_agent_num, -1, 2), rot_mat) + pos_m[:, :2].unsqueeze(1)
                diff_m = pos_m_dynamic.reshape(val_agent_num, self.num_modes, num_future_steps, 2)[:, :, -1, :] - pos_m_dynamic.reshape(val_agent_num, self.num_modes, num_future_steps, 2)[:, :, 0, :]
                head_m_motion = torch.atan2(diff_m[:, :, 1], diff_m[:, :, 0]).transpose(1, 0).reshape(-1)
                head_vector_m_motion = torch.stack([head_m_motion.cos(), head_m_motion.sin()], dim=-1)

                m_position = pos_m_dynamic.reshape(val_agent_num, self.num_modes, num_future_steps, 2)[:, :, -1, :].transpose(1, 0).reshape(-1, 2)

                rel_pos_m2m_dynamic = m_position[m2m_index[0]] - m_position[m2m_index[1]]
                rel_orient_m2m_dynamic = wrap_angle(head_m_motion[m2m_index[0]] - head_m_motion[m2m_index[1]])
                r_m2m_dynamic = torch.stack(
                    [torch.norm(rel_pos_m2m_dynamic[:, :2], p=2, dim=-1),
                        angle_between_2d_vectors(ctr_vector=head_vector_m_motion[m2m_index[1]], nbr_vector=rel_pos_m2m_dynamic[:, :2]),
                        rel_orient_m2m_dynamic], dim=-1)
                r_m2m_dynamic = self.r_m2m_emb_dynamic(continuous_inputs=r_m2m_dynamic, categorical_embs=None)
                m = m.transpose(1, 0).reshape(-1, self.hidden_dim)
                m = self.cross_m2m_propose_attn_layer(m, r_m2m_dynamic, m2m_index)
                m = m.reshape(self.num_modes, val_agent_num, self.hidden_dim).transpose(1, 0)
                locs_propose_pos[t] = self.to_loc_propose_pos(m)
                scales_propose_pos[t] = self.to_scale_propose_pos(m)

                matching_m = m[matching_mask]
                if self.multi_pair and self.training:
                    matching_m = torch.cat((matching_m[matching_pair_indices[:, 0]].repeat_interleave(6, 1),
                                            matching_m[matching_pair_indices[:, 1]].repeat(1, 6, 1)), dim=-1)
                else:
                    matching_m = torch.cat((matching_m[0::2].repeat_interleave(6, 1), matching_m[1::2].repeat(1, 6, 1)), dim=-1)

                matching_score = self.matching_cls(matching_m)

            if self.output_head:
                locs_propose_head[t] = self.to_loc_propose_head(m)
                concs_propose_head[t] = self.to_conc_propose_head(m)

        loc_propose_pos = torch.cumsum(
            torch.cat(locs_propose_pos, dim=-1).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
            dim=-2)
        scale_propose_pos = torch.cumsum(
            F.elu_(
                torch.cat(scales_propose_pos, dim=-1).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
                alpha=1.0) +
            1.0,
            dim=-2) + 0.1

        if self.output_head:
            loc_propose_head = torch.cumsum(torch.tanh(torch.cat(locs_propose_head, dim=-1).unsqueeze(-1)) * math.pi,
                                            dim=-2)
            conc_propose_head = 1.0 / (torch.cumsum(F.elu_(torch.cat(concs_propose_head, dim=-1).unsqueeze(-1)) + 1.0,
                                                    dim=-2) + 0.02)
            m = self.y_emb(torch.cat([loc_propose_pos.detach(),
                                      wrap_angle(loc_propose_head.detach())], dim=-1).view(-1, self.output_dim + 1))
        else:
            loc_propose_head = loc_propose_pos.new_zeros((loc_propose_pos.size(0), self.num_modes,
                                                          self.num_future_steps, 1))
            conc_propose_head = scale_propose_pos.new_zeros((scale_propose_pos.size(0), self.num_modes,
                                                             self.num_future_steps, 1))
            m = self.y_emb(loc_propose_pos.detach().view(-1, self.output_dim))

        m = m.reshape(-1, self.num_future_steps, self.hidden_dim).transpose(0, 1)
        m = self.traj_emb(m, self.traj_emb_h0.unsqueeze(1).repeat(1, m.size(1), 1))[1].squeeze(0)
        m = m.reshape(-1, self.num_modes, self.hidden_dim)

        if self.matching:
            if self.training:
                gt = data['agent']['target'][..., :self.output_dim]
                reg_mask = data['agent']['valid_mask'][:, self.num_historical_steps:]
                l2_norm = (torch.norm(loc_propose_pos[..., :self.output_dim] -
                                      gt.unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
                best_mode = l2_norm.argmin(dim=-1)
            # pair first
            edge_pair_first = []
            for i in range(self.num_modes):
                # pair_index = torch.nonzero(torch.all(matching_index==matching_index[matching_index[:,0]==i][torch.argmax(matching_score[matching_index[:,0]==i])], dim=1))[0]
                if not self.training:
                    max_score_index = torch.argmax(matching_score[matching_index[:, :, 0] == i].reshape(-1, 6), dim=1)
                    pair_index = matching_index[matching_index[:, :, 0] == i].reshape(-1, 6, 2)[torch.arange(max_score_index.shape[0]), max_score_index]
                else:
                    if self.multi_pair:
                        pair_index = torch.cat((torch.tensor([i]*matching_pair_indices.shape[0])[None, :].to(m.device), best_mode[torch.where(matching_mask)[0][matching_pair_indices[:, 1]]][None, :]), dim=0).transpose(1, 0)
                    else:
                        pair_index = torch.cat((torch.tensor([i]*best_mode[matching_mask][1::2].shape[0])[None, :].to(m.device), best_mode[matching_mask][1::2][None, :]), dim=0).transpose(1, 0)

                if self.multi_pair and self.training:
                    pair_index1 = torch.where(matching_mask)[0][matching_pair_indices[:, 0]]*6 + pair_index[:, 0]
                    pair_index2 = torch.where(matching_mask)[0][matching_pair_indices[:, 1]]*6 + pair_index[:, 1]
                else:
                    pair_index1 = torch.where(matching_mask)[0][0::2]*6 + pair_index[:, 0]
                    pair_index2 = torch.where(matching_mask)[0][1::2]*6 + pair_index[:, 1]

                edge_pair_first.append(torch.cat((pair_index2[None, :], pair_index1[None, :]), dim=0).to(m.device))

            # pair second
            edge_pair_second = []
            for i in range(self.num_modes):
                # pair_index = torch.nonzero(torch.all(matching_index==matching_index[matching_index[:,0]==i][torch.argmax(matching_score[matching_index[:,0]==i])], dim=1))[0]
                if not self.training:
                    max_score_index = torch.argmax(matching_score[matching_index[:, :, 1] == i].reshape(-1, 6), dim=1)
                    pair_index = matching_index[matching_index[:, :, 1] == i].reshape(-1, 6, 2)[torch.arange(max_score_index.shape[0]), max_score_index]
                else:
                    if self.multi_pair:
                        pair_index = torch.cat((best_mode[torch.where(matching_mask)[0][matching_pair_indices[:, 1]]][None, :], torch.tensor([i]*matching_pair_indices.shape[0])[None, :].to(m.device)), dim=0).transpose(1, 0)
                    else:
                        pair_index = torch.cat((best_mode[matching_mask][0::2][None, :], torch.tensor([i]*best_mode[matching_mask][0::2].shape[0])[None, :].to(m.device)), dim=0).transpose(1, 0)

                if self.multi_pair and self.training:
                    pair_index1 = torch.where(matching_mask)[0][matching_pair_indices[:, 0]]*6 + pair_index[:, 0]
                    pair_index2 = torch.where(matching_mask)[0][matching_pair_indices[:, 1]]*6 + pair_index[:, 1]
                else:
                    pair_index1 = torch.where(matching_mask)[0][0::2]*6 + pair_index[:, 0]
                    pair_index2 = torch.where(matching_mask)[0][1::2]*6 + pair_index[:, 1]

                edge_pair_second.append(torch.cat((pair_index1[None, :], pair_index2[None, :]), dim=0).to(m.device))

            edge_index_m2m_pair = torch.cat(edge_pair_first + edge_pair_second, dim=1)

            pos_m_pair = pos_m.repeat_interleave(self.num_modes, 0)
            head_m_pair = head_m.repeat_interleave(self.num_modes, 0)
            head_vector_m_pair = head_vector_m.repeat_interleave(self.num_modes, 0)

            rel_pos_m2m = pos_m_pair[edge_index_m2m_pair[0]] - pos_m_pair[edge_index_m2m_pair[1]]

            rel_head_m2m = wrap_angle(head_m_pair[edge_index_m2m_pair[0]] - head_m_pair[edge_index_m2m_pair[1]])
            r_m2m = torch.stack(
                [torch.norm(rel_pos_m2m[:, :2], p=2, dim=-1),
                 angle_between_2d_vectors(ctr_vector=head_vector_m_pair[edge_index_m2m_pair[1]], nbr_vector=rel_pos_m2m[:, :2]),
                 rel_head_m2m], dim=-1)
            r_m2m = self.r_a2m_emb(continuous_inputs=r_m2m, categorical_embs=None)
        # edge_index_m2m_pair = torch.cat(
        #     [edge_index_m2m_pair + i * edge_index_m2m_pair.new_tensor([data['agent']['num_nodes']]) for i in
        #      range(self.num_modes)], dim=1)

        m = m.reshape(-1, self.hidden_dim)
        for i in range(self.num_layers):
            m = self.t2m_refine_attn_layers[i]((x_t, m), r_t2m, edge_index_t2m)
            m = m.reshape(-1, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            m = self.pl2m_refine_attn_layers[i]((x_pl, m), r_pl2m, edge_index_pl2m)
            m = self.a2m_refine_attn_layers[i]((x_a, m), r_a2m, edge_index_a2m)
            m = m.reshape(self.num_modes, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            if self.matching:
                m = self.cross_m2m_refine_attn_layer[i](m, r_m2m, edge_index_m2m_pair)
        m = self.m2m_refine_attn_layer(m, None, edge_index_m2m)
        m = m.reshape(-1, self.num_modes, self.hidden_dim)
        loc_refine_pos = self.to_loc_refine_pos(m).view(-1, self.num_modes, self.num_future_steps, self.output_dim)
        loc_refine_pos = loc_refine_pos + loc_propose_pos.detach()
        scale_refine_pos = F.elu_(
            self.to_scale_refine_pos(m).view(-1, self.num_modes, self.num_future_steps, self.output_dim),
            alpha=1.0) + 1.0 + 0.1
        if self.output_head:
            loc_refine_head = torch.tanh(self.to_loc_refine_head(m).unsqueeze(-1)) * math.pi
            loc_refine_head = loc_refine_head + loc_propose_head.detach()
            conc_refine_head = 1.0 / (F.elu_(self.to_conc_refine_head(m).unsqueeze(-1)) + 1.0 + 0.02)
        else:
            loc_refine_head = loc_refine_pos.new_zeros((loc_refine_pos.size(0), self.num_modes, self.num_future_steps,
                                                        1))
            conc_refine_head = scale_refine_pos.new_zeros((scale_refine_pos.size(0), self.num_modes,
                                                           self.num_future_steps, 1))

        matching_m = m[matching_mask]
        if self.multi_pair and self.training:
            matching_m = torch.cat((matching_m[matching_pair_indices[:, 0]].repeat_interleave(6, 1),
                                    matching_m[matching_pair_indices[:, 1]].repeat(1, 6, 1)), dim=-1)
        else:
            matching_m = torch.cat((matching_m[0::2].repeat_interleave(6, 1), matching_m[1::2].repeat(1, 6, 1)), dim=-1)
        if self.matching:
            matching_score_pi = self.matching_cls_pi(matching_m)

        pi = self.to_pi(m).squeeze(-1)

        return {
            'loc_propose_pos': loc_propose_pos,
            'scale_propose_pos': scale_propose_pos,
            'loc_propose_head': loc_propose_head,
            'conc_propose_head': conc_propose_head,
            'loc_refine_pos': loc_refine_pos,
            'scale_refine_pos': scale_refine_pos,
            'loc_refine_head': loc_refine_head,
            'conc_refine_head': conc_refine_head,
            'pi': pi,
            'matching_score': matching_score,
            'matching_score_pi': matching_score_pi,
            'matching_mask': matching_mask,
            'matching_index': matching_index,
            'matching_pair_indices': matching_pair_indices
        }
