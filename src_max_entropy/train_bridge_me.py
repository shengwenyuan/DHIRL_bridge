"""
Bridge experiment: Maximum Entropy IRL (state-visitation matching).
Uses data/ inputs (trans_probs.npy, trajs.json) and mirrors src/train_bridge.py
KFold/splits; runs MaxEntropyIRL from src_max_entropy.
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src_max_entropy.max_entropy_irl import MaxEntropyIRL


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ll_filename', type=str, default='ll_max_entropy.csv')
    parser.add_argument('--discount', type=float, default=0.97)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='outputs/bridge_train')
    parser.add_argument('--rand_seed', type=int, default=42)
    args = parser.parse_args()

    num_folds = 5
    num_states = 768
    num_actions = 32

    np.random.seed(args.rand_seed)

    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_df = pd.DataFrame(columns=['num_trajs', 'fold', 'train_ll', 'test_ll'])

    with open(os.path.join(data_dir, 'trans_probs.npy'), 'rb') as f:
        P = np.load(f)
    with open(os.path.join(data_dir, 'trajs.json')) as f:
        trajs = json.load(f)

    # P is (S, A, S') from data/build_trans.py
    assert P.shape == (num_states, num_actions, num_states), P.shape

    len_trajs = len(trajs)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10042)

    for num_trajs in [len_trajs, len_trajs // 2, len_trajs // 3]:
        for kf_idx, (train_idxes, test_idxes) in enumerate(kf.split(trajs[:num_trajs])):
            train_trajs = [trajs[i] for i in train_idxes]
            test_trajs = [trajs[i] for i in test_idxes]

            # Expert state visitation counts from train (max entropy matches state visitation)
            expert_s = np.zeros(num_states)
            for traj in train_trajs:
                for step in traj:
                    s = int(step[0])
                    expert_s[s] += 1

            agent = MaxEntropyIRL(
                num_states=num_states,
                num_actions=num_actions,
                P=P,
                expert_s_count=expert_s,
                discount=args.discount,
            )
            agent.train(trajs=train_trajs)

            pi_hat = agent.get_policy()
            ll_train_list = []
            for traj in train_trajs:
                likes = [pi_hat[int(s), int(a)] for s, a, _ in traj]
                ll_train_list.append(np.mean(np.log(np.array(likes) + 1e-8)))
            ll_test_list = []
            for traj in test_trajs:
                likes = [pi_hat[int(s), int(a)] for s, a, _ in traj]
                ll_test_list.append(np.mean(np.log(np.array(likes) + 1e-8)))

            ll_train = np.mean(ll_train_list)
            ll_test = np.mean(ll_test_list)

            if num_trajs == len_trajs:
                param_dir = os.path.join(output_dir, f'max_entropy/{num_trajs}/fold_{kf_idx}')
                os.makedirs(param_dir, exist_ok=True)
                np.save(os.path.join(param_dir, 'r.npy'), agent.get_rewards())
                np.save(os.path.join(param_dir, 'q.npy'), agent.get_q_values())

            output_df.loc[len(output_df)] = [num_trajs, kf_idx, ll_train, ll_test]
            output_df.to_csv(os.path.join(output_dir, args.ll_filename), index=False)

    print(f'Done. Log-likelihoods saved to {os.path.join(output_dir, args.ll_filename)}')
