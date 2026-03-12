import os
import json
import argparse
import sys

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import KFold

# Prefer algorithms from same directory (src_test) when running as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algorithms import PGIAVI


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_states', type=int, required=True, help='NS (number of states)')
    parser.add_argument('--num_actions', type=int, required=True, help='NA (number of actions)')
    parser.add_argument('--ll_filename', type=str, default='ll_pgiql.csv')
    parser.add_argument('--discount', type=float, default=0.97)
    parser.add_argument('--num_repeats', type=int, default=1)
    parser.add_argument('--num_latents', type=int, default=5)
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--data_dir', type=str, default='data', help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: outputs/bridge_train/ns_NS_na_NA)')
    parser.add_argument('--group_id', type=str, default='default')

    parser.add_argument('--model_type', type=str, default='IntentionRNN',
                        choices=['IntentionRNN', 'IntentionLSTM', 'IntentionTransformer'])
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--rnn_hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--reg_type', type=str, default='l1', choices=['l1', 'kl'])
    parser.add_argument('--reg_weight', type=float, default=0.)

    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--loss_threshold', type=float, default=1e-2)
    parser.add_argument('--max_iterations', type=int, default=150)

    args = parser.parse_args()

    num_folds = 5
    num_repeats = args.num_repeats
    num_states = args.num_states
    num_actions = args.num_actions
    num_latents = args.num_latents

    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.rand_seed)
    print(f'{device}')

    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join('outputs', 'bridge_train', f'ns_{num_states}_na_{num_actions}')
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
    for num_trajs in [len_trajs]:
        for kf_idx, (train_idxes, test_idxes) in enumerate(kf.split(trajs[:num_trajs])):
            train_trajs = [trajs[train_idx] for train_idx in train_idxes]
            test_trajs = [trajs[test_idx] for test_idx in test_idxes]

            best_test_ll = -np.inf
            best_ll = None
            for repeats in range(num_repeats):
                model = PGIAVI(num_latents=num_latents, num_states=num_states, num_actions=num_actions,
                                train_trajs=train_trajs, test_trajs=test_trajs, P=P, discount=args.discount,
                                model_type=args.model_type, hidden_dim=args.hidden_dim,
                                rnn_hidden_dim=args.rnn_hidden_dim, num_layers=args.num_layers,
                                dropout=args.dropout, nhead=args.nhead, lr=args.lr,
                                reg_type=args.reg_type, reg_weight=args.reg_weight,
                                num_epochs=args.num_epochs, loss_threshold=args.loss_threshold,
                                max_iterations=args.max_iterations)
                ll, f, mask, agents = model.fit()
                if ll['test'] > best_test_ll:
                    best_test_ll = ll['test']
                    best_ll = ll
                    if num_trajs == len_trajs:
                        param_dir = os.path.join(run_dir, f'{num_trajs}/fold_{kf_idx}')
                        os.makedirs(param_dir, exist_ok=True)
                        np.save(os.path.join(param_dir, 'f_train.npy'), f['train'])
                        np.save(os.path.join(param_dir, 'mask_train.npy'), mask['train'])
                        np.save(os.path.join(param_dir, 'f_test.npy'), f['test'])
                        np.save(os.path.join(param_dir, 'mask_test.npy'), mask['test'])
                        for agent_idx, agent in enumerate(agents):
                            np.save(os.path.join(param_dir, f'r_{agent_idx}.npy'), agent.r.cpu().numpy())
                            np.save(os.path.join(param_dir, f'q_{agent_idx}.npy'), agent.q.cpu().numpy())
            output_df.loc[len(output_df)] = [num_trajs, kf_idx, best_ll['train'], best_ll['test']]
            output_df.to_csv(os.path.join(run_dir, args.ll_filename), index=False)
