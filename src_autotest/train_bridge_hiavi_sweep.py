import os
import json
import argparse
import sys

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algorithms import HIAVI_B


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_states', type=int, required=True)
    parser.add_argument('--num_actions', type=int, required=True)
    parser.add_argument('--ll_filename', type=str, default='ll.csv')
    parser.add_argument('--discount', type=float, default=0.97)
    parser.add_argument('--num_repeats', type=int, default=1)
    parser.add_argument('--num_latents', type=int, default=4)
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--group_id', type=str, default='default')
    parser.add_argument('--save_npy', type=int, default=1)

    parser.add_argument('--num_traj_steps', type=int, default=5,
                        help='Number of evenly spaced num_trajs values')
    parser.add_argument('--num_trajs_list', type=str, default=None,
                        help='Comma-separated explicit num_trajs values (overrides num_traj_steps)')

    args = parser.parse_args()

    num_folds = 5
    num_states = args.num_states
    num_actions = args.num_actions

    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.rand_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{device}')

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join('outputs', 'bridge_hiavi', f'ns_{num_states}_na_{num_actions}')
    run_dir = os.path.join(output_dir, args.group_id)
    os.makedirs(run_dir, exist_ok=True)
    output_df = pd.DataFrame(columns=['num_trajs', 'fold', 'train_ll', 'test_ll'])

    trans_path = os.path.join(args.data_dir, f'trans_probs_{num_states}_{num_actions}.npy')
    trajs_path = os.path.join(args.data_dir, f'trajs_{num_states}_{num_actions}.json')
    with open(trans_path, 'rb') as f:
        P = np.load(f)
    with open(trajs_path) as f:
        trajs = json.load(f)

    len_trajs = len(trajs)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10042)
    if args.num_trajs_list is not None:
        traj_steps = [int(x) for x in args.num_trajs_list.split(',')]
    else:
        traj_steps = [len_trajs * i // args.num_traj_steps for i in range(1, args.num_traj_steps + 1)]

    for num_trajs in traj_steps:
        for kf_idx, (train_idxes, test_idxes) in enumerate(kf.split(trajs[:num_trajs])):
            train_trajs = [trajs[train_idx] for train_idx in train_idxes]
            test_trajs = [trajs[test_idx] for test_idx in test_idxes]

            best_test_ll = -np.inf
            best_ll = None
            best_state = None
            for _ in range(args.num_repeats):
                model = HIAVI_B(
                    num_latents=args.num_latents,
                    num_states=num_states, num_actions=num_actions,
                    train_trajs=train_trajs, test_trajs=test_trajs,
                    P=P, discount=args.discount,
                )
                ll, logp_init, logp_tr, agents = model.fit()
                if ll['test'] > best_test_ll:
                    best_test_ll = ll['test']
                    best_ll = ll
                    best_state = (logp_init, logp_tr, agents)

            if args.save_npy and num_trajs == len_trajs:
                logp_init, logp_tr, agents = best_state
                param_dir = os.path.join(run_dir, f'{num_trajs}/fold_{kf_idx}')
                os.makedirs(param_dir, exist_ok=True)
                with open(os.path.join(param_dir, 'train_idxes.json'), 'w') as fout:
                    json.dump(train_idxes.tolist(), fout)
                with open(os.path.join(param_dir, 'test_idxes.json'), 'w') as fout:
                    json.dump(test_idxes.tolist(), fout)
                np.save(os.path.join(param_dir, 'logp_init.npy'), logp_init)
                np.save(os.path.join(param_dir, 'logp_tr.npy'), logp_tr)
                for agent_idx, agent in enumerate(agents):
                    np.save(os.path.join(param_dir, f'r_{agent_idx}.npy'), agent.get_rewards())
                    np.save(os.path.join(param_dir, f'q_{agent_idx}.npy'), agent.get_q_values())

            output_df.loc[len(output_df)] = [num_trajs, kf_idx, best_ll['train'], best_ll['test']]
            output_df.to_csv(os.path.join(run_dir, args.ll_filename), index=False)
