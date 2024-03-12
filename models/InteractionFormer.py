
from itertools import chain
from itertools import compress
import contextlib
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.data import HeteroData
import os
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from losses import MixtureNLLLoss
from losses import NLLLoss
from metrics import minADE
from metrics import minADE_src
from metrics import minFDE
from metrics import minFDE_src
from models import InteractionDecoder
from models import InteractionEncoder


def cal_polygon_contour(x, y, theta, width, length):
    left_front_x = x + 0.5 * length * math.cos(theta) - 0.5 * width * math.sin(theta)
    left_front_y = y + 0.5 * length * math.sin(theta) + 0.5 * width * math.cos(theta)
    left_front = (left_front_x, left_front_y)

    right_front_x = x + 0.5 * length * math.cos(theta) + 0.5 * width * math.sin(theta)
    right_front_y = y + 0.5 * length * math.sin(theta) - 0.5 * width * math.cos(theta)
    right_front = (right_front_x, right_front_y)

    right_back_x = x - 0.5 * length * math.cos(theta) + 0.5 * width * math.sin(theta)
    right_back_y = y - 0.5 * length * math.sin(theta) - 0.5 * width * math.cos(theta)
    right_back = (right_back_x, right_back_y)

    left_back_x = x - 0.5 * length * math.cos(theta) - 0.5 * width * math.sin(theta)
    left_back_y = y - 0.5 * length * math.sin(theta) + 0.5 * width * math.cos(theta)
    left_back = (left_back_x, left_back_y)
    polygon_contour = [left_front, right_front, right_back, left_back]

    return polygon_contour


class InteractionFormer(pl.LightningModule):

    def __init__(self, model_config) -> None:
        super(InteractionFormer, self).__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.dataset = model_config.dataset
        self.input_dim = model_config.input_dim
        self.hidden_dim = model_config.hidden_dim
        self.output_dim = model_config.output_dim
        self.output_head = model_config.output_head
        self.num_historical_steps = model_config.num_historical_steps
        self.num_future_steps = model_config.decoder.num_future_steps
        self.num_modes = model_config.decoder.num_modes
        self.num_recurrent_steps = model_config.decoder.num_recurrent_steps
        self.num_freq_bands = model_config.num_freq_bands
        self.m2m_attention = model_config.decoder.m2m_attention
        self.encoder = InteractionEncoder(
            dataset=model_config.dataset,
            input_dim=model_config.input_dim,
            hidden_dim=model_config.hidden_dim,
            num_historical_steps=model_config.num_historical_steps,
            pl2pl_radius=model_config.encoder.pl2pl_radius,
            pl2a_radius=model_config.encoder.pl2a_radius,
            a2a_radius=model_config.encoder.a2a_radius,
            num_freq_bands=model_config.num_freq_bands,
            num_map_layers=model_config.encoder.num_map_layers,
            num_agent_layers=model_config.encoder.num_agent_layers,
            num_heads=model_config.num_heads,
            head_dim=model_config.head_dim,
            dropout=model_config.dropout,
            time_span=model_config.encoder.time_span
        )
        self.decoder = InteractionDecoder(
            dataset=model_config.dataset,
            input_dim=model_config.input_dim,
            hidden_dim=model_config.hidden_dim,
            output_dim=model_config.output_dim,
            output_head=model_config.output_head,
            num_historical_steps=model_config.num_historical_steps,
            num_future_steps=model_config.decoder.num_future_steps,
            num_modes=model_config.decoder.num_modes,
            num_recurrent_steps=model_config.decoder.num_recurrent_steps,
            num_t2m_steps=model_config.decoder.num_t2m_steps,
            pl2m_radius=model_config.decoder.pl2m_radius,
            a2m_radius=model_config.decoder.a2m_radius,
            num_freq_bands=model_config.num_freq_bands,
            num_layers=model_config.decoder.num_dec_layers,
            num_heads=model_config.num_heads,
            head_dim=model_config.head_dim,
            dropout=model_config.dropout
        )

        self.reg_loss = NLLLoss(component_distribution=['laplace'] * model_config.output_dim + ['von_mises'] * model_config.output_head,
                                reduction='none')
        self.cls_loss = MixtureNLLLoss(component_distribution=['laplace'] * model_config.output_dim + ['von_mises'] * model_config.output_head,
                                       reduction='none')

        self.minADE = minADE(max_guesses=6)
        self.minFDE = minFDE(max_guesses=6)
        self.minADE_pi = minADE(max_guesses=6)
        self.minFDE_pi = minFDE(max_guesses=6)
        self.minADE_refine = minADE(max_guesses=6)
        self.minFDE_refine = minFDE(max_guesses=6)
        self.minADE_refine_nms = minADE(max_guesses=6)
        self.minFDE_refine_nms = minFDE(max_guesses=6)
        self.minADE_marginal = minADE_src(max_guesses=6)
        self.minFDE_marginal = minFDE_src(max_guesses=6)
        self.matching_loss = nn.BCELoss()

        self.test_predictions = dict()
        self.brier = []
        self.fde = []
        self.bfde = []
        self.vis = False
        self.batch_nms = True
        self.multi_pair = True

    def forward(self, data: HeteroData):
        scene_enc = self.encoder(data)
        pred = self.decoder(data, scene_enc)
        return pred

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def training_step(self,
                      data,
                      batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        reg_mask = data['agent']['valid_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['valid_mask'][:, -1]
        pred = self(data)
        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)

        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)
        l2_norm = (torch.norm(traj_propose[..., :self.output_dim] -
                              gt[..., :self.output_dim].unsqueeze(1), p=2, dim=-1) * reg_mask.unsqueeze(1)).sum(dim=-1)
        best_mode = l2_norm.argmin(dim=-1)

        matching_score = pred['matching_score'].squeeze(-1)
        matching_score_pi = pred['matching_score_pi'].squeeze(-1)
        matching_mask = pred['matching_mask']
        matching_index = pred['matching_index']
        matching_pair_indices = pred['matching_pair_indices']

        reg_mask = reg_mask & matching_mask[:, None]
        cls_mask = cls_mask & matching_mask

        if self.multi_pair:
            matching_gt = torch.cat((best_mode[torch.where(matching_mask)[0][matching_pair_indices[:, 0]]][:, None, None],
                                     best_mode[torch.where(matching_mask)[0][matching_pair_indices[:, 1]]][:, None, None]), dim=2)
        else:
            matching_gt = best_mode[matching_mask]
            matching_gt_1 = matching_gt[0::2][:, None, None]
            matching_gt_2 = matching_gt[1::2][:, None, None]
            matching_gt = torch.cat((matching_gt_1, matching_gt_2), dim=2)

        matching_target = (matching_index == matching_gt).all(-1).float()
        matching_loss = self.matching_loss(matching_score, matching_target)
        matching_loss_pi = self.matching_loss(matching_score_pi, matching_target)

        traj_propose_best = traj_propose[torch.arange(traj_propose.size(0)), best_mode]
        traj_refine_best = traj_refine[torch.arange(traj_refine.size(0)), best_mode]
        reg_loss_propose = self.reg_loss(traj_propose_best,
                                         gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_propose = reg_loss_propose.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_propose = reg_loss_propose.mean()
        reg_loss_refine = self.reg_loss(traj_refine_best,
                                        gt[..., :self.output_dim + self.output_head]).sum(dim=-1) * reg_mask
        reg_loss_refine = reg_loss_refine.sum(dim=0) / reg_mask.sum(dim=0).clamp_(min=1)
        reg_loss_refine = reg_loss_refine.mean()
        cls_loss = self.cls_loss(pred=traj_refine[:, :, -1:].detach(),
                                 target=gt[:, -1:, :self.output_dim + self.output_head],
                                 prob=pi,
                                 mask=reg_mask[:, -1:]) * cls_mask
        cls_loss = cls_loss.sum() / cls_mask.sum().clamp_(min=1) * 0.1

        self.log('train_reg_loss_propose', reg_loss_propose, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_reg_loss_refine', reg_loss_refine, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_cls_loss', cls_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_matching_loss', matching_loss, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_matching_loss_pi', matching_loss_pi, prog_bar=False, on_step=True, on_epoch=True, batch_size=1)

        loss = reg_loss_propose + reg_loss_refine + cls_loss + matching_loss + matching_loss_pi
        return loss

    def validation_step(self,
                        data,
                        batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        reg_mask = data['agent']['valid_mask'][:, self.num_historical_steps:]
        cls_mask = data['agent']['valid_mask'][:, -1]
        pred = self(data)
        if self.output_head:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['loc_propose_head'],
                                      pred['scale_propose_pos'][..., :self.output_dim],
                                      pred['conc_propose_head']], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_propose = torch.cat([pred['loc_propose_pos'][..., :self.output_dim],
                                      pred['scale_propose_pos'][..., :self.output_dim]], dim=-1)
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']
        gt = torch.cat([data['agent']['target'][..., :self.output_dim], data['agent']['target'][..., -1:]], dim=-1)

        matching_score = pred['matching_score']
        matching_score_pi = pred['matching_score_pi']
        matching_mask = pred['matching_mask']
        matching_index = pred['matching_index']
        if matching_score is None:
            matching = False
        else:
            matching = True

        batch_size = matching_index.shape[0]
        if matching:
            topk_num = 6
            topk_pair_index = matching_score.squeeze(-1).topk(dim=1, k=topk_num)[1]
            topk_pair_index_refine = matching_score_pi.squeeze(-1).topk(dim=1, k=topk_num)[1]
            topk_matching_score_refine = matching_score_pi.squeeze(-1).topk(dim=1, k=topk_num)[0]
            topk_pair_index = matching_index[torch.arange(batch_size)[:, None].repeat(1, topk_num), topk_pair_index]
            topk_pair_index_refine = matching_index[torch.arange(batch_size)[:, None].repeat(1, topk_num), topk_pair_index_refine]

        if self.batch_nms:
            pair_index_nms = torch.cat((torch.arange(6)[:, None].repeat_interleave(6, 0),
                                        torch.arange(6)[:, None].repeat(6, 1)),
                                       dim=-1)[None, ...].repeat(batch_size, 1, 1).to(matching_score.device)
            pair_traj1_nms = traj_refine[matching_mask][0::2][torch.arange(batch_size)[:, None].repeat(1, pair_index_nms.shape[1]), pair_index_nms[:, :, 0]]
            pair_traj2_nms = traj_refine[matching_mask][1::2][torch.arange(batch_size)[:, None].repeat(1, pair_index_nms.shape[1]), pair_index_nms[:, :, 1]]
            pair_trajs_nms = torch.cat((pair_traj1_nms[:, :, None, ...], pair_traj2_nms[:, :, None, ...]), dim=2)
            ret_idxs = []
            for i in range(matching_score_pi.shape[0]):
                score_matrix = matching_score_pi[i].cpu().numpy().reshape(6, 6)
                row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)
                ret_idxs.append(torch.tensor([np.arange(36).reshape(6, 6)[i] for i in list(zip(row_ind, col_ind))]).to(matching_score.device)[None, ...])

            ret_idxs = torch.cat(ret_idxs, dim=0)
            pair_score_nms = matching_score_pi[torch.arange(batch_size)[:, None], ret_idxs]
            pair_trajs_nms = pair_trajs_nms[torch.arange(batch_size)[:, None], ret_idxs]

            topk_pair_index_refine_nms = matching_index[torch.arange(batch_size)[:, None].repeat(1, topk_num), ret_idxs]
            topk_matching_score_refine_nms = pair_score_nms

        # select with pi topk pair
        pi_eval = F.softmax(pi[matching_mask], dim=-1)
        pi_eval_new = pi_eval[0::2][:, None, ...] * pi_eval[1::2][..., None]
        pi_topk_index = pi_eval_new.view(batch_size, -1).topk(k=6, dim=1)[1]
        topk_pair_index_pi = torch.cat((torch.arange(6)[:, None].repeat_interleave(6, 0),
                                        torch.arange(6)[:, None].repeat(6, 1)),
                                       dim=-1)[None, ...].repeat(batch_size, 1, 1).to(pi_eval.device)
        topk_pair_index_pi = topk_pair_index_pi[torch.arange(batch_size)[:, None].repeat(1, 6), pi_topk_index]

        if matching:
            pair_traj1 = traj_refine[matching_mask][0::2][torch.arange(batch_size)[:, None].repeat(1, topk_pair_index.shape[1]), topk_pair_index[:, :, 0]]
            pair_traj2 = traj_refine[matching_mask][1::2][torch.arange(batch_size)[:, None].repeat(1, topk_pair_index.shape[1]), topk_pair_index[:, :, 1]]

            pair_traj1_refine = traj_refine[matching_mask][0::2][torch.arange(batch_size)[:, None].repeat(1, topk_pair_index_refine.shape[1]), topk_pair_index_refine[:, :, 0]]
            pair_traj2_refine = traj_refine[matching_mask][1::2][torch.arange(batch_size)[:, None].repeat(1, topk_pair_index_refine.shape[1]), topk_pair_index_refine[:, :, 1]]

            if self.batch_nms:
                pair_traj1_refine_nms = traj_refine[matching_mask][0::2][torch.arange(batch_size)[:, None].repeat(1, topk_pair_index_refine_nms.shape[1]), topk_pair_index_refine_nms[:, :, 0]]
                pair_traj2_refine_nms = traj_refine[matching_mask][1::2][torch.arange(batch_size)[:, None].repeat(1, topk_pair_index_refine_nms.shape[1]), topk_pair_index_refine_nms[:, :, 1]]

        # eval with pi topk pair
        pair_traj1_pi = traj_refine[matching_mask][0::2][torch.arange(batch_size)[:, None].repeat(1, topk_pair_index_pi.shape[1]), topk_pair_index_pi[:, :, 0]]
        pair_traj2_pi = traj_refine[matching_mask][1::2][torch.arange(batch_size)[:, None].repeat(1, topk_pair_index_pi.shape[1]), topk_pair_index_pi[:, :, 1]]

        reg_mask = reg_mask & matching_mask[:, None]

        if matching:
            pair_trajs = torch.cat((pair_traj1[:, :, None, ...], pair_traj2[:, :, None, ...]), dim=2)
            pair_trajs_refine = torch.cat((pair_traj1_refine[:, :, None, ...], pair_traj2_refine[:, :, None, ...]), dim=2)
            if self.batch_nms:
                pair_trajs_refine_nms = torch.cat((pair_traj1_refine_nms[:, :, None, ...], pair_traj2_refine_nms[:, :, None, ...]), dim=2)

        pair_trajs_pi = torch.cat((pair_traj1_pi[:, :, None, ...], pair_traj2_pi[:, :, None, ...]), dim=2)

        gt_eval = gt[matching_mask]
        gt_eval_1 = gt_eval[0::2][:, None, ...].repeat(1, 6, 1, 1)
        gt_eval_2 = gt_eval[1::2][:, None, ...].repeat(1, 6, 1, 1)
        gt_eval_trajs = torch.cat((gt_eval_1[:, :, None, ...], gt_eval_2[:, :, None, ...]), dim=2)

        eval_mask = reg_mask[matching_mask]
        eval_mask_1 = eval_mask[0::2]
        eval_mask_2 = eval_mask[1::2]
        eval_mask = torch.cat((eval_mask_1[:, None, ...], eval_mask_2[:, None, ...]), dim=1)
        if matching:
            self.minADE.update(pred=pair_trajs[..., :self.output_dim], target=gt_eval_trajs[..., :self.output_dim],
                               valid_mask=eval_mask)
            self.minFDE.update(pred=pair_trajs[..., :self.output_dim], target=gt_eval_trajs[..., :self.output_dim],
                               valid_mask=eval_mask)
            self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
            self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))

            self.minADE_refine.update(pred=pair_trajs_refine[..., :self.output_dim], target=gt_eval_trajs[..., :self.output_dim],
                                      valid_mask=eval_mask)
            self.minFDE_refine.update(pred=pair_trajs_refine[..., :self.output_dim], target=gt_eval_trajs[..., :self.output_dim],
                                      valid_mask=eval_mask)
            self.log('val_minADE_refine', self.minADE_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
            self.log('val_minFDE_refine', self.minFDE_refine, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
            if self.batch_nms:
                self.minADE_refine_nms.update(pred=pair_trajs_refine_nms[..., :self.output_dim], target=gt_eval_trajs[..., :self.output_dim],
                                              valid_mask=eval_mask)
                self.minFDE_refine_nms.update(pred=pair_trajs_refine_nms[..., :self.output_dim], target=gt_eval_trajs[..., :self.output_dim],
                                              valid_mask=eval_mask)
                self.log('val_minADE_refine_nms', self.minADE_refine_nms, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
                self.log('val_minFDE_refine_nms', self.minFDE_refine_nms, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))

        self.minADE_pi.update(pred=pair_trajs_pi[..., :self.output_dim], target=gt_eval_trajs[..., :self.output_dim],
                              valid_mask=eval_mask)
        self.minFDE_pi.update(pred=pair_trajs_pi[..., :self.output_dim], target=gt_eval_trajs[..., :self.output_dim],
                              valid_mask=eval_mask)
        self.log('val_minADE_pi', self.minADE_pi, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFDE_pi', self.minFDE_pi, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))

        valid_mask_eval = reg_mask[matching_mask]
        traj_eval = traj_refine[matching_mask, :, :, :self.output_dim + self.output_head]

        pi_eval = F.softmax(pi[matching_mask], dim=-1)
        gt_eval = gt[matching_mask]
        self.minADE_marginal.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                                    valid_mask=valid_mask_eval)
        self.minFDE_marginal.update(pred=traj_eval[..., :self.output_dim], target=gt_eval[..., :self.output_dim], prob=pi_eval,
                                    valid_mask=valid_mask_eval)
        self.log('val_minADE_marginal', self.minADE_marginal, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))
        self.log('val_minFDE_marginal', self.minFDE_marginal, prog_bar=True, on_step=False, on_epoch=True, batch_size=gt_eval.size(0))

        if self.batch_nms:
            pair_trajs_refine = pair_trajs_refine_nms
            topk_matching_score_refine = topk_matching_score_refine_nms

    def test_step(self,
                  data,
                  batch_idx):
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        pred = self(data)
        if self.output_head:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['loc_refine_head'],
                                     pred['scale_refine_pos'][..., :self.output_dim],
                                     pred['conc_refine_head']], dim=-1)
        else:
            traj_refine = torch.cat([pred['loc_refine_pos'][..., :self.output_dim],
                                     pred['scale_refine_pos'][..., :self.output_dim]], dim=-1)
        pi = pred['pi']

        eval_mask = data['agent']['category'] == 3

        origin_eval = data['agent']['position'][eval_mask, self.num_historical_steps - 1]
        theta_eval = data['agent']['heading'][eval_mask, self.num_historical_steps - 1]
        cos, sin = theta_eval.cos(), theta_eval.sin()
        rot_mat = torch.zeros(eval_mask.sum(), 2, 2, device=self.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        traj_eval = torch.matmul(traj_refine[eval_mask, :, :, :2],
                                 rot_mat.unsqueeze(1)) + origin_eval[:, :2].reshape(-1, 1, 1, 2)
        pi_eval = F.softmax(pi[eval_mask], dim=-1)

        traj_eval = traj_eval.cpu().numpy()
        pi_eval = pi_eval.cpu().numpy()
        if self.dataset == 'argoverse_v2':
            eval_id = list(compress(list(chain(*data['agent']['id'])), eval_mask))
            if isinstance(data, Batch):
                for i in range(data.num_graphs):
                    self.test_predictions[data['scenario_id'][i]] = (pi_eval[i], {eval_id[i]: traj_eval[i]})
            else:
                self.test_predictions[data['scenario_id']] = (pi_eval[0], {eval_id[0]: traj_eval[0]})
        else:
            raise ValueError('{} is not a valid dataset'.format(self.dataset))

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.MultiheadAttention, nn.LSTM,
                                    nn.LSTMCell, nn.GRU, nn.GRUCell)
        blacklist_weight_modules = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.Embedding)
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = '%s.%s' % (module_name, param_name) if module_name else param_name
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {param_name: param for param_name, param in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        assert len(param_dict.keys() - union_params) == 0
        optimizer_para = self.model_config.optimizer
        optim_groups = [
            {"params": [param_dict[param_name] for param_name in sorted(list(decay))],
             "weight_decay": optimizer_para.weight_decay},
            {"params": [param_dict[param_name] for param_name in sorted(list(no_decay))],
             "weight_decay": 0.0},
        ]
        # from deepspeed.ops.adam import DeepSpeedCPUAdam
        # optimizer = DeepSpeedCPUAdam(optim_groups, lr=self.lr, weight_decay=self.weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=optimizer_para.lr, weight_decay=optimizer_para.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=optimizer_para.T_max, eta_min=0.00001)
        return [optimizer], [scheduler]

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['state_dict']

        version = checkpoint.get("version", None)
        if version is not None:
            logger.info('==> Checkpoint trained from version: %s' % version)

        logger.info(f'The number of disk ckpt keys: {len(model_state_disk)}')
        model_state = self.state_dict()
        model_state_disk_filter = {}
        for key, val in model_state_disk.items():
            if key in model_state and model_state_disk[key].shape == model_state[key].shape:
                model_state_disk_filter[key] = val
            else:
                if key not in model_state:
                    print(f'Ignore key in disk (not found in model): {key}, shape={val.shape}')
                else:
                    print(f'Ignore key in disk (shape does not match): {key}, load_shape={val.shape}, model_shape={model_state[key].shape}')

        model_state_disk = model_state_disk_filter

        missing_keys, unexpected_keys = self.load_state_dict(model_state_disk, strict=False)

        logger.info(f'Missing keys: {missing_keys}')
        logger.info(f'The number of missing keys: {len(missing_keys)}')
        logger.info(f'The number of unexpected keys: {len(unexpected_keys)}')
        logger.info('==> Done (total keys %d)' % (len(model_state)))

        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        return it, epoch
