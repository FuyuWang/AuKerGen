import math

from torch.distributions import Categorical
from torch.distributions import Bernoulli
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import Transformer


class Actor(nn.Module):
    def __init__(self, d_model, d_inner, n_layers, n_head, d_k, d_v, buf_spmap_cstr, buffer_size_list, steps_per_level,
                 problem_instance, prime2idx):
        super(Actor, self).__init__()

        self.prime2idx = prime2idx
        self.idx2prime = {value: key for key, value in prime2idx.items()}
        self.num_primes = len(self.prime2idx.keys())
        self.transformer = Transformer(d_word_vec=d_model, d_model=d_model, d_inner=d_inner,
                                       n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v, dropout=0,
                                       n_position=100, trg_emb_prj_weight_sharing=True,
                                       scale_emb_or_prj='prj', num_primes=len(prime2idx.keys()))
        self.buffer_size_list = buffer_size_list
        self.buf_spmap_cstr = buf_spmap_cstr
        self.steps_per_level = steps_per_level
        self.problem_instance = problem_instance
        self.finished_levels = []

    def reset(self):
        self.finished_levels = []

    def get_remain_buffer_size(self, cur_buffer_level, trg_seq_disorder, order_action, is_cur):
        buffer_size = self.buffer_size_list[f'l{cur_buffer_level}']
        batch_size = trg_seq_disorder.size(0)
        tiles = trg_seq_disorder.new_ones(batch_size, self.steps_per_level)
        for buffer_idx in range(1, cur_buffer_level + 1):
            start_ind = (buffer_idx - 1) * self.steps_per_level
            end_ind = buffer_idx * self.steps_per_level
            level_trg_seq_disorder = copy.deepcopy(trg_seq_disorder[:, start_ind:end_ind])
            for k, v in self.prime2idx.items():
                tiles *= torch.pow(int(k), level_trg_seq_disorder[:, :, v + 1])

        # print(tiles.size(), tiles)
        H, M, K, N = torch.unbind(tiles, dim=1)
        if cur_buffer_level == 1:   # LRF
            N = trg_seq_disorder.new_zeros(batch_size)
        elif cur_buffer_level == 2:     # RF
            K = trg_seq_disorder.new_zeros(batch_size)

        # input_tile = N * (P + R - 1) * (Q + S - 1) * C
        input_tile = H * M * K
        weight_tile = H * K * N
        output_tile = H * M * N

        H_sub = trg_seq_disorder.new_zeros(batch_size).float()
        M_sub = weight_tile
        K_sub = output_tile
        N_sub = input_tile

        H_coef = input_tile + weight_tile + output_tile
        M_coef = M*(H * K + H * N)
        K_coef = K*(H * M + H * N)
        N_coef = N*(H * M + H * K)

        coef_arr = torch.stack([H_coef, M_coef, K_coef, N_coef], dim=1)[
            np.arange(batch_size), order_action]
        sub_arr = torch.stack([H_sub, M_sub, K_sub, N_sub], dim=1)[
            np.arange(batch_size), order_action]

        remain_buffer_size = (buffer_size - sub_arr.float()) / coef_arr.float()

        return remain_buffer_size

    def get_max_temporal_size(self, cur_buffer_level, tile2_remain_dimension_budgets, remain_buf_spmap):
        '''
        param: tile2_remain_budget [batch, 7]
        '''
        # print(tile2_remain_budget.size(), tile2_remain_budget[0])

        max_temporal_tile2 = tile2_remain_dimension_budgets - torch.log2(torch.clamp(remain_buf_spmap, min=1))

        for level in range(1, len(self.buffer_size_list) + 1):
            buf_spmap_cstr = self.buf_spmap_cstr[f'l{level}']
            if level not in self.finished_levels and level != cur_buffer_level:
                max_temporal_tile2 -= math.log2(buf_spmap_cstr)
        # if cur_buffer_level == 5:
        #     print(tile2_remain_budget[0], remain_buf_spmap[0], max_temporal_tile2[0])
        return torch.clamp(max_temporal_tile2, min=0).long()

    def forward(self, trg_seq, trg_mask, order_mask, tile_remain_budgets, tile_masks, parallel_mask,
                mode, cur_buffer_level, trg_seq_disorder):

        # trg_seq    [batch, level*7, num_primes+2]
        # trg_seq_disorder [batch, level*7, 2*num_primes+2] NKCPQRS tiles

        tile_logits, sp_tile2_logit = self.transformer(trg_seq)
        tile2_logit = tile_logits[:, 0]
        batch_size = trg_seq.size(0)

        # _, order_action = torch.min(tile2_remain_budget + order_mask[:, :-1], dim=-1)
        # print(tile2_remain_budget, order_mask, order_action)

        if mode%self.steps_per_level == 0:
            order_action = trg_seq.new_ones(batch_size).fill_(0)
        elif mode%self.steps_per_level == 1:
            order_action = trg_seq.new_ones(batch_size).fill_(1)
        elif mode%self.steps_per_level == 2:
            order_action = trg_seq.new_ones(batch_size).fill_(3)
        elif mode%self.steps_per_level == 3:
            order_action = trg_seq.new_ones(batch_size).fill_(2)

        # order_action = self.dim_orders[:, mode % self.steps_per_level]
        # order_action = torch.from_numpy(order_action).to(tile_logits.device)

        log_probs = tile2_logit.new_zeros(batch_size, self.num_primes+1)
        log_prob_masks = tile2_logit.new_zeros(batch_size, self.num_primes+1)

        if cur_buffer_level == len(self.buffer_size_list):
            return (order_action, None, None, None, None), log_probs, log_prob_masks

        # predict tiles
        tile2_remain_budget = tile_remain_budgets[:, :, 0]
        tile2_remain_budget_dimensions = copy.deepcopy(tile2_remain_budget)
        tile2_remain_budget = tile2_remain_budget[torch.arange(0, batch_size), order_action]
        tile2_mask = tile_masks[torch.arange(0, batch_size), order_action, 0]

        remain_buffer_size = self.get_remain_buffer_size(cur_buffer_level, trg_seq_disorder, order_action, is_cur=True)

        for later_level in range(cur_buffer_level + 1, len(self.buffer_size_list) + 1):
            remain_buffer_size = torch.minimum(remain_buffer_size,
                                               self.get_remain_buffer_size(later_level, trg_seq_disorder,
                                                                           order_action, is_cur=False))

        tile2_max = torch.log2(torch.clamp(remain_buffer_size, min=1))
        tile2_max = torch.clamp(tile2_max.long(), min=0, max=tile2_mask.size(-1)-1)
        tile2_max = torch.minimum(tile2_max, tile2_remain_budget)

        parallel_mask = parallel_mask[torch.arange(0, batch_size), order_action]
        # print(order_action, parallel_mask)
        buf_spmap_cstr = self.buf_spmap_cstr[f'l{cur_buffer_level}']
        start_ind = (cur_buffer_level - 1) * self.steps_per_level
        end_ind = cur_buffer_level * self.steps_per_level
        level_trg_seq_disorder = copy.deepcopy(trg_seq_disorder[:, start_ind:end_ind])
        used_buf_spmap = trg_seq.new_ones(batch_size)
        # print(buf_spmap_cstr)
        for i in range(self.steps_per_level):
            parallel, sp_tile2 = torch.unbind(level_trg_seq_disorder[:, i, self.num_primes + 1: self.num_primes + 3], dim=-1)
            used_buf_spmap *= torch.clamp(parallel * torch.pow(2, sp_tile2), min=1)
        remain_buf_spmap = buf_spmap_cstr / used_buf_spmap.float()

        # print("remain_buf_spmap:  ", cur_buffer_level, remain_buf_spmap)
        sp_tile2_max = torch.log2(torch.clamp(remain_buf_spmap, min=1))
        sp_tile2_max = torch.clamp(sp_tile2_max.long(), min=0, max=tile2_mask.size(-1) - 1)
        sp_tile2_max = sp_tile2_max * (parallel_mask[:, 1] == 0).long()
        sp_tile2_max = torch.minimum(sp_tile2_max, tile2_max)

        if mode % self.steps_per_level == self.steps_per_level - 1:
            sp_tile2_action = sp_tile2_max
            sp_tile2_log_prob = tile2_logit.new_zeros(batch_size)
            sp_tile2_log_prob_mask = order_mask.new_zeros(batch_size)
            self.finished_levels.append(cur_buffer_level)
        else:
            tile2_remain_dimension_budgets = tile2_remain_budget_dimensions - order_mask[:, :self.steps_per_level]
            tile2_remain_dimension_budgets[tile2_remain_dimension_budgets < 0] = 0
            tile2_remain_dimension_budgets = tile2_remain_dimension_budgets.sum(dim=-1) - tile2_remain_budget
            sp_tile2_min = torch.clamp(torch.log2(torch.clamp(remain_buf_spmap, min=1)) - tile2_remain_dimension_budgets, min=0)
            sp_tile2_min = torch.minimum(sp_tile2_min.long(), sp_tile2_max)

            sp_tile2_mask_tmp = torch.cat([tile2_mask, torch.zeros_like(tile2_mask)], dim=-1)
            for i in range(1, tile2_mask.size(-1) + 1):
                sp_tile2_mask_tmp[np.arange(batch_size), sp_tile2_max + i] = float('-inf')
            sp_tile2_mask_tmp = sp_tile2_mask_tmp[:, :tile2_mask.size(-1)]

            sp_tile2_mask_tmp = torch.cat([torch.zeros_like(tile2_mask), sp_tile2_mask_tmp], dim=-1)
            for i in range(1, tile2_mask.size(-1) + 1):
                sp_tile2_mask_tmp[np.arange(batch_size), sp_tile2_min + tile2_mask.size(-1) - i] = float('-inf')
            sp_tile2_mask = sp_tile2_mask_tmp[:, tile2_mask.size(-1):]

            sp_tile2_score = sp_tile2_logit + sp_tile2_mask
            sp_tile2_prob = F.softmax(sp_tile2_score, dim=-1)
            sp_tile2_density = Categorical(sp_tile2_prob)
            sp_tile2_action = sp_tile2_density.sample()
            sp_tile2_log_prob = sp_tile2_density.log_prob(sp_tile2_action)
            sp_tile2_log_prob_mask = ((sp_tile2_mask == 0).sum(dim=-1) > 1).float()

        # test_sol = np.array([[0, 0, 0, 0],
        #                      [1, 0, 0, 0],
        #                      [3, 3, 0, 0],
        #                      [2, 0, 0, 0],
        #                      [0, 2, 0, 1],
        #                      [1, 5, 1, 5],
        #                      [3, 1, 1, 0],
        #                      [2, 3, 1, 2],
        #                      [0, 2, 1, 1],
        #                      [1, 1, 0, 0],
        #                      [3, 4, 0, 0],
        #                      [2, 1, 1, 1],
        #                      [0, 0, 0, 0],
        #                      [1, 3, 0, 0],
        #                      [3, 1, 0, 0],
        #                      [2, 2, 0, 0]])

        # test_sol = torch.from_numpy(test_sol).to(sp_tile2_action).long()
        # start_ind = (cur_buffer_level - 1) * self.steps_per_level
        # end_ind = cur_buffer_level * self.steps_per_level
        # level_sol = test_sol[start_ind:end_ind]
        # sp_tile2_action[0:] = level_sol[mode % self.steps_per_level, self.num_primes+2]
        # print(mode, level_sol, sp_tile2_action[0])

        tile2_min = sp_tile2_action
        # tile2_remain_dimension_budgets = tile2_remain_budget_dimensions.sum(dim=-1)
        # max_temporal_tile2 = self.get_max_temporal_size(cur_buffer_level, tile2_remain_dimension_budgets, remain_buf_spmap)
        # tile2_max = torch.minimum(tile2_max, tile2_min+max_temporal_tile2.long())

        tile2_mask_tmp = torch.cat([tile2_mask, torch.zeros_like(tile2_mask)], dim=-1)
        for i in range(1, tile2_mask.size(-1) + 1):
            tile2_mask_tmp[np.arange(batch_size), tile2_max + i] = float('-inf')
        tile2_mask_tmp = tile2_mask_tmp[:, :tile2_mask.size(-1)]

        tile2_mask_tmp = torch.cat([torch.zeros_like(tile2_mask), tile2_mask_tmp], dim=-1)
        for i in range(1, tile2_mask.size(-1) + 1):
            tile2_mask_tmp[np.arange(batch_size), tile2_min + tile2_mask.size(-1) - i] = float('-inf')
        tile2_mask = tile2_mask_tmp[:, tile2_mask.size(-1):]

        tile2_score = tile2_logit + tile2_mask
        tile2_prob = F.softmax(tile2_score, dim=-1)
        tile2_density = Categorical(tile2_prob)
        tile2_action = tile2_density.sample()
        tile2_log_prob = tile2_density.log_prob(tile2_action)
        tile2_log_prob_mask = ((tile2_mask == 0).sum(dim=-1) > 1).float()

        # tile2_action[0:] = level_sol[mode % self.steps_per_level, 1]

        tile_action = tile2_action
        tile_actions = []
        sp_tile_actions = []
        log_probs = []
        log_prob_masks = []

        tile_actions.append(tile2_action)
        sp_tile_actions.append(sp_tile2_action)
        log_probs.append(tile2_log_prob)
        log_prob_masks.append(tile2_log_prob_mask)

        # if cur_buffer_level == 2 or cur_buffer_level == 3:
        #     print(mode, sp_tile2_action[0], sp_tile2_max[0],  tile2_action[0], tile2_min[0],tile2_max[0])

        for p in range(1, self.num_primes):
            remain_buffer_size = remain_buffer_size / torch.pow(int(self.idx2prime[p - 1]), tile_action).float()
            tile_max = torch.log2(torch.clamp(remain_buffer_size, min=1)) / math.log2(int(self.idx2prime[p]))
            tile_max = torch.clamp(tile_max.long(), min=0, max=tile_masks.size(-1)-1)
            tile_max = torch.minimum(tile_max, tile_remain_budgets[torch.arange(0, batch_size), order_action, p])

            tile_mask = tile_masks[torch.arange(0, batch_size), order_action, p]
            tile_mask_tmp = torch.cat([tile_mask, torch.zeros_like(tile_mask)], dim=-1)
            for i in range(1, tile_mask.size(-1) + 1):
                tile_mask_tmp[np.arange(batch_size), tile_max + i] = float('-inf')
            tile_mask = tile_mask_tmp[:, :tile_mask.size(-1)]

            tile_logit = tile_logits[:, p]
            tile_score = tile_logit + tile_mask
            tile_prob = F.softmax(tile_score, dim=-1)
            tile_density = Categorical(tile_prob)
            tile_action = tile_density.sample()
            tile_log_prob = tile_density.log_prob(tile_action)
            tile_log_prob_mask = ((tile_mask == 0).sum(dim=-1) > 1).float()

            tile_actions.append(tile_action)
            log_probs.append(tile_log_prob)
            log_prob_masks.append(tile_log_prob_mask)
            sp_tile_actions.append(trg_seq.new_zeros(batch_size))

        log_probs.append(sp_tile2_log_prob)
        log_prob_masks.append(sp_tile2_log_prob_mask)

        parallel_action = sp_tile2_action
        parallel_action = torch.clamp(parallel_action, max=1)

        # order_log_prob_mask = ((order_mask == 0).sum(dim=-1) > 1).float()
        tile_actions = torch.stack(tile_actions, dim=1)
        sp_tile_actions = torch.stack(sp_tile_actions, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        log_prob_masks = torch.stack(log_prob_masks, dim=1)


        return (order_action, tile_actions, parallel_action, sp_tile_actions), log_probs, log_prob_masks
