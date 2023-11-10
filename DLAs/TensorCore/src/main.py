import os
import psutil
import yaml
import pickle

import torch
import torch.optim as optim
import argparse
import numpy as np
import random
from datetime import datetime

from env import Environment
from actor import Actor
from transformer.Optim import ScheduledOptim



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #  torch.backends.cudnn.deterministic = True


def compute_policy_loss(rewards, log_probs, log_prob_masks):
    dis_rewards = []
    gamma = 0.99
    batch_size = log_probs.size(1)
    batch_masks = log_probs.new_ones(batch_size)
    success_idx = []
    fail_idx = []
    for i in range(batch_size):
        if rewards[-1, i] > 0:
            success_idx.append(i)
        else:
            fail_idx.append(i)
    if len(fail_idx) > 3*len(success_idx):
        fail_idx = random.sample(fail_idx, 3*len(success_idx))
    print(len(success_idx), len(fail_idx), rewards[-1, :])

    rewards = rewards[7:]
    log_probs = log_probs[:-7]
    log_prob_masks = log_prob_masks[:-7]

    R = np.zeros(batch_size)
    for r in rewards[::-1]:
        R = r + gamma * R
        dis_rewards.insert(0, R)
    dis_rewards = torch.from_numpy(np.array(dis_rewards)).to(log_probs.device)
    # print(dis_rewards.size(), log_prob_masks[:,7,:], log_prob_masks[:,6,:])
    policy_loss = dis_rewards * (-1 * log_probs * log_prob_masks).sum(dim=-1)
    policy_loss = policy_loss.sum(dim=0) * batch_masks
    policy_loss = policy_loss.sum() / batch_masks.sum()

    return policy_loss


def run():
    opt = parser.parse_args()
    fitness = [opt.fitness1]
    fitness.append(opt.fitness2) if opt.fitness2 is not None else None
    fitness.append(opt.fitness3) if opt.fitness3 is not None else None
    print(f'Fitness Objective: {fitness}')
    density = opt.density.split(',')
    density = {'Inputs': float(density[0]), 'Outputs': float(density[1]), 'Weights': float(density[2])}
    problem_dir = '../../../LLMs'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    architectures = ['tensorcore']
    for architecture in architectures:
        llms = ['bertlarge']
        for llm in llms:
            with open(os.path.join(problem_dir, '{}_problems/layers.yaml'.format(llm)), 'r') as fd:
                layers = yaml.load(fd, Loader=yaml.SafeLoader)
            fd.close()
            problem = {'problem': {'instance': {'H': 1, 'M': 512, 'K': 1024, 'N': 1024},
                                   'shape': {'data-spaces':
                                                 [{'name': 'Weights', 'projection': [[['H']], [['K']], [['N']]]},
                                                  {'name': 'Inputs', 'projection': [[['H']], [['M']], [['K']]]},
                                                  {'name': 'Outputs', 'projection': [[['H']], [['M']], [['N']]],
                                                   'read-write': True}],
                                             'dimensions': ['H', 'M', 'K', 'N'], 'name': 'bmm'}}}
            input_sizes = [1]

            for input_size in input_sizes:
                actor_state_dict = None
                layer2chkpt = {}
                for i, layer in enumerate(layers[0:1]):
                    print(architecture, llm, input_size, i, layer)
                    set_seed(opt.seed)
                    report_dir = os.path.join(opt.report_dir, architecture, 'fitness_{}'.format(opt.fitness1),
                                              'sampled_episodes_{}'.format(opt.batch_size),
                                              '{}_input{}'.format(llm, input_size), 'layer-{}'.format(i))
                    with open(os.path.join(problem_dir, '{}_problems/{}.yaml'.format(llm, layer)), 'r') as fd:
                        layer_problem = yaml.load(fd, Loader=yaml.SafeLoader)
                        problem['problem']['instance']['H'] = layer_problem['problem']['H'] * input_size
                        problem['problem']['instance']['M'] = layer_problem['problem']['M']
                        problem['problem']['instance']['K'] = layer_problem['problem']['K']
                        problem['problem']['instance']['N'] = layer_problem['problem']['N']
                    fd.close()

                    layer_to_key = ''
                    for key in ['H', 'M', 'K', 'N']:
                        layer_to_key += str(problem['problem']['instance'][key]) + ' '

                    if layer_to_key in layer2chkpt:
                        print(layer_to_key, 'repeated')
                        chkpt = layer2chkpt[layer_to_key]
                        os.makedirs(report_dir, exist_ok=True)
                        with open(os.path.join(report_dir, 'env_chkpt.plt'), 'wb') as fd:
                            pickle.dump(chkpt, fd)
                        fd.close()
                        continue
                    else:
                        with open('../in_config/problem.yaml', 'w') as fd:
                            yaml.dump(problem, fd)
                        fd.close()

                        env = Environment(fitness_obj=fitness, report_dir=report_dir, use_pool=True, use_IO=True,
                                          debug=False, in_config_dir=opt.config_path, arch_file=architecture,
                                          density=density, save_chkpt=opt.save_chkpt,
                                          use_sparse=opt.use_sparse, explore_bypass=opt.explore_bypass,
                                          batch_size=opt.batch_size)
                        actor = Actor(opt.d_model, opt.d_inner, opt.n_layers, opt.n_head, opt.d_k, opt.d_v,
                                      env.buf_spmap_cstr, env.buffer_size_list, env.steps_per_level,
                                      problem['problem']['instance'], env.prime2idx).to(device)
                        actor.train()

                        optimizer = ScheduledOptim(
                            optim.Adam(actor.parameters(), betas=(0.9, 0.98), eps=1e-09),
                            opt.lr_mul, opt.d_model, opt.n_warmup_steps)

                        for ep in range(opt.epochs):
                            print('Epoch {}'.format(ep))
                            print('epoch start : ', datetime.now())
                            state_info = env.reset()
                            actor.reset()
                            total_log_probs = []
                            total_log_prob_masks = []
                            total_rewards = []
                            optimizer.zero_grad()
                            for step in range(env.total_steps):
                                trg_seq = torch.from_numpy(state_info[0]).type(torch.LongTensor).to(device)
                                trg_mask = torch.from_numpy(state_info[1]).type(torch.FloatTensor).to(device)
                                order_mask = torch.from_numpy(state_info[2]).type(torch.FloatTensor).to(device)
                                tile_remain_budgets = torch.from_numpy(state_info[3]).type(torch.LongTensor).to(device)
                                tile_masks = torch.from_numpy(state_info[4]).type(torch.FloatTensor).to(device)
                                parallel_mask = torch.from_numpy(state_info[5]).type(torch.FloatTensor).to(device)
                                mode = state_info[6]
                                cur_buffer_level = state_info[7]
                                trg_seq_disorder = torch.from_numpy(state_info[8]).type(torch.LongTensor).to(device)
                                step_actions, step_log_probs, step_log_prob_masks = actor(trg_seq, trg_mask, order_mask,
                                                                                          tile_remain_budgets,
                                                                                          tile_masks,
                                                                                          parallel_mask, mode,
                                                                                          cur_buffer_level,
                                                                                          trg_seq_disorder)
                                state_info, reward, done, info = env.step(step_actions)
                                total_rewards.append(reward)
                                total_log_probs.append(step_log_probs)
                                total_log_prob_masks.append(step_log_prob_masks)

                                if done:
                                    break

                            total_rewards = np.stack(total_rewards, axis=0)
                            total_log_probs = torch.stack(total_log_probs, dim=0)
                            total_log_prob_masks = torch.stack(total_log_prob_masks, dim=0)
                            policy_loss = compute_policy_loss(total_rewards, total_log_probs, total_log_prob_masks)
                            policy_loss.backward()
                            optimizer.step_and_update_lr()

                            chkpt = env.record_chkpt(ep == opt.epochs - 1)
                        layer2chkpt[layer_to_key] = chkpt

                        env.clean_timeloop_output_files()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fitness1', type=str, default="latency", help='1st order fitness objective')
    parser.add_argument('--fitness2', type=str, default=None, help='2nd order fitness objective')
    parser.add_argument('--fitness3', type=str, default=None, help='3rd order fitness objective')
    parser.add_argument('--epochs', type=int, default=10, help='number of generations/epochs')
    parser.add_argument('--config_path', type=str, default='../in_config',
                        help='Configuration path, should include arch.yaml, problem.yaml, (and sparse.yaml if sparsity is considered)')
    parser.add_argument('--report_dir', type=str, default='../report', help='The report directory')
    parser.add_argument('--density', type=str, default='0.5,1,1', help='The density of Input, Output, Weight Tenor')
    parser.add_argument('--save_chkpt', action='store_true', default=True, help='Create a checkpoint when finished')
    parser.add_argument('--use_sparse', action='store_true', default=False, help='Execute Map Space Exploration on sparse accelerator')
    parser.add_argument('--explore_bypass', action='store_true', default=False,
                        help='Enable it can add bypass buffer option in to the search space')

    parser.add_argument('--llm', type=str, default=None, help='large language model')
    parser.add_argument('--input_size', type=int, default=1, help='the size of dimension N')
    parser.add_argument('--architecture', type=str, default='arch', help='accelerator architecture')

    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_inner', type=int, default=1024)
    parser.add_argument('--d_k', type=int, default=64)
    parser.add_argument('--d_v', type=int, default=64)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=4)

    parser.add_argument('--n_warmup_steps', type=int, default=4000)
    parser.add_argument('--lr_mul', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)

    run()

