import os
import json
import argparse

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import KFold

from src.algorithms import PGIAVI


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ll_filename', type=str, default='ll_pgiql.csv')
    parser.add_argument('--discount', type=float, default=0.97)
    parser.add_argument('--num_repeats', type=int, default=1)
    parser.add_argument('--num_latents', type=int, default=5)
    parser.add_argument('--rand_seed', type=int, default=42)
    args = parser.parse_args()

    num_folds = 5
    num_repeats = args.num_repeats
    num_states = 512
    num_actions = 32
    num_latents = args.num_latents

    np.random.seed(args.rand_seed)
    torch.manual_seed(args.rand_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.rand_seed)
    print(f'{device}')

    output_dir = f'outputs/bridge_train'
    os.makedirs(output_dir, exist_ok=True)
    output_df = pd.DataFrame(columns=['num_trajs', 'fold', 'train_ll', 'test_ll'])

    with open('data/trans_probs.npy', 'rb') as f:
        P = np.load(f)
    with open('data/trajs.json') as f:
        trajs = json.load(f)

    len_trajs = len(trajs)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10042)
    for num_trajs in [len_trajs, len_trajs//2, len_trajs//3]:
        for kf_idx, (train_idxes, test_idxes) in enumerate(kf.split(trajs[:num_trajs])):
            train_trajs = [trajs[train_idx] for train_idx in train_idxes]
            test_trajs = [trajs[test_idx] for test_idx in test_idxes]

            best_test_ll = -np.inf
            best_ll = None
            for repeats in range(num_repeats):
                model = PGIAVI(num_latents=num_latents, num_states=num_states, num_actions=num_actions,
                                train_trajs=train_trajs, test_trajs=test_trajs, P=P, discount=args.discount)
                ll, f, agents = model.fit()
                if ll['test'] > best_test_ll:
                    best_test_ll = ll['test']
                    best_ll = ll
                    if num_trajs == len_trajs - 1:
                        param_dir = os.path.join(output_dir, f'pgiql/{num_trajs}/fold_{kf_idx}')
                        os.makedirs(param_dir, exist_ok=True)
                        np.save(os.path.join(param_dir, f'f_train.npy'), f['train'])
                        np.save(os.path.join(param_dir, f'f_test.npy'), f['test'])
                        for agent_idx, agent in enumerate(agents):
                            np.save(os.path.join(param_dir, f'r_{agent_idx}.npy'), agent.r)
                            np.save(os.path.join(param_dir, f'q_{agent_idx}.npy'), agent.q)
            output_df.loc[len(output_df)] = [num_trajs, kf_idx, best_ll['train'], best_ll['test']]
            output_df.to_csv(os.path.join(output_dir, args.ll_filename), index=False)
