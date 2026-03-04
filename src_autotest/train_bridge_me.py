"""
Bridge experiment: Maximum Entropy IRL (state-visitation matching).
Autotest-compatible: accepts --num_states and --num_actions; reads
trajs_NS_NA.json and trans_probs_NS_NA.npy from data_dir (e.g. data_autotest).
"""

import os
import json
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Prefer max_entropy_irl from same directory (src_autotest) when running as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from max_entropy_irl import MaxEntropyIRL


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_states', type=int, required=True, help='NS (number of states)')
    parser.add_argument('--num_actions', type=int, required=True, help='NA (number of actions)')
    parser.add_argument('--ll_filename', type=str, default='ll_max_entropy.csv')
    parser.add_argument('--discount', type=float, default=0.97)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: outputs/bridge_train/ns_NS_na_NA)')
    parser.add_argument('--rand_seed', type=int, default=42)
    args = parser.parse_args()

    num_folds = 5
    num_states = args.num_states
    num_actions = args.num_actions

    np.random.seed(args.rand_seed)

    data_dir = args.data_dir
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join('outputs', 'bridge_train', f'ns_{num_states}_na_{num_actions}')
    os.makedirs(output_dir, exist_ok=True)
    output_df = pd.DataFrame(columns=['num_trajs', 'fold', 'train_ll', 'test_ll'])

    trans_path = os.path.join(data_dir, f'trans_probs_{num_states}_{num_actions}.npy')
    trajs_path = os.path.join(data_dir, f'trajs_{num_states}_{num_actions}.json')
    with open(trans_path, 'rb') as f:
        P = np.load(f)
    with open(trajs_path) as f:
        trajs = json.load(f)

    # P is (S, A, S') from data/build_trans.py
    assert P.shape == (num_states, num_actions, num_states), P.shape

    len_trajs = len(trajs)
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10042)

    # for num_trajs in [len_trajs, len_trajs // 2, len_trajs // 3]:
    for num_trajs in [len_trajs]:
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
