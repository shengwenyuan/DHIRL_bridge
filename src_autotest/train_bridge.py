import os
import hashlib
import json
import argparse
import platform
import sys

import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import KFold

# Prefer algorithms from same directory (src_test) when running as script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algorithms import PGIAVI


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as fin:
        for block in iter(lambda: fin.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pad_step_scores(step_scores, mask):
    padded = np.full(mask.shape, np.nan, dtype=np.float32)
    for idx, scores in enumerate(step_scores):
        padded[idx, :len(scores)] = scores
    return padded


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
    parser.add_argument('--gate_mode', type=str, default='retrospective',
                        choices=['retrospective', 'causal', 'state_only'])
    parser.add_argument('--fold_idx', type=int, default=-1,
                        help='Fold to run for smoke testing. -1 runs all folds.')
    parser.add_argument('--max_trajs', type=int, default=0,
                        help='Prefix size for smoke testing. 0 uses all trajectories.')
    parser.add_argument('--paired_fold_seeds', type=int, default=0,
                        help='Reset a deterministic seed before each fold. 1=on, 0=legacy stream.')
    parser.add_argument('--p0_artifacts', type=int, default=0,
                        help='Save explicit gate/responsibility/score arrays. 1=on, 0=legacy f arrays.')

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

    parser.add_argument('--save_npy', type=int, default=1,
                        help='Save per-fold arrays. 1=on, 0=off.')

    args = parser.parse_args()

    num_folds = 5
    num_repeats = args.num_repeats
    num_states = args.num_states
    num_actions = args.num_actions
    num_latents = args.num_latents
    if num_repeats < 1:
        raise ValueError('num_repeats must be at least one.')
    if args.gate_mode != 'retrospective' and num_repeats != 1:
        raise ValueError('P0 predictive runs require num_repeats=1; use separate seed jobs.')
    if args.fold_idx < -1 or args.fold_idx >= num_folds:
        raise ValueError(f'fold_idx must be -1 or in [0, {num_folds - 1}].')

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
    output_df = pd.DataFrame(columns=[
        'num_trajs', 'fold', 'train_ll', 'test_ll', 'seed', 'fold_seed',
        'gate_mode', 'score_type', 'train_step_ll', 'test_step_ll',
        'train_steps', 'test_steps', 'iterations', 'stop_reason', 'final_loss',
        'status',
    ])

    trans_path = os.path.join(args.data_dir, f'trans_probs_{num_states}_{num_actions}.npy')
    trajs_path = os.path.join(args.data_dir, f'trajs_{num_states}_{num_actions}.json')
    with open(trans_path, 'rb') as f:
        P = np.load(f)
    with open(trajs_path) as f:
        trajs = json.load(f)

    len_trajs = len(trajs)
    total_steps = sum(len(traj) for traj in trajs)
    if args.p0_artifacts and (len_trajs != 60064 or total_steps != 2195527):
        raise ValueError(
            f'Expected canonical Bridge size 60064/2195527, got {len_trajs}/{total_steps}.'
        )
    num_trajs = len_trajs if args.max_trajs <= 0 else min(args.max_trajs, len_trajs)
    if num_trajs < num_folds:
        raise ValueError(f'Need at least {num_folds} trajectories, got {num_trajs}.')
    expected_p_shape = (num_states, num_actions, num_states)
    if P.shape != expected_p_shape:
        raise ValueError(f'Expected transition shape {expected_p_shape}, got {P.shape}.')
    if not np.isfinite(P).all() or (P < 0).any():
        raise ValueError('Transition tensor must be finite and nonnegative.')
    transition_row_sums = P.sum(axis=2)
    valid_row_sums = np.isclose(transition_row_sums, 0.0) | np.isclose(
        transition_row_sums, 1.0, atol=1e-6
    )
    if not valid_row_sums.all():
        raise ValueError('Transition rows must sum to zero or one.')

    manifest = {
        'domain': 'bridge',
        'gate_mode': args.gate_mode,
        'model_type': args.model_type,
        'num_latents': num_latents,
        'num_states': num_states,
        'num_actions': num_actions,
        'rand_seed': args.rand_seed,
        'num_repeats': num_repeats,
        'num_folds': num_folds,
        'fold_idx': args.fold_idx,
        'fold_random_state': 10042,
        'fold_seed_rule': (
            'rand_seed + 1009 * fold_idx + repeat_idx'
            if args.paired_fold_seeds
            else 'legacy continuous RNG stream'
        ),
        'fold_seeds': (
            [args.rand_seed + 1009 * idx for idx in range(num_folds)]
            if args.paired_fold_seeds
            else None
        ),
        'num_trajs_total': len_trajs,
        'num_trajs_used': num_trajs,
        'num_steps_total': total_steps,
        'transition_shape': list(P.shape),
        'zero_transition_rows': int(np.isclose(transition_row_sums, 0.0).sum()),
        'transition_path': os.path.abspath(trans_path),
        'trajectory_path': os.path.abspath(trajs_path),
        'transition_bytes': os.path.getsize(trans_path),
        'trajectory_bytes': os.path.getsize(trajs_path),
        'transition_sha256': sha256_file(trans_path),
        'trajectory_sha256': sha256_file(trajs_path),
        'score_timestep_rule': 'all valid observed actions selected by mask',
        'score_type': (
            'retrospective_compatibility_score'
            if args.gate_mode == 'retrospective'
            else 'predictive_action_log_likelihood'
        ),
        'legacy_padded_loss_scaling': True,
        'status': 'running',
        'resolved_args': vars(args),
        'python': platform.python_version(),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'device': str(device),
    }
    with open(os.path.join(run_dir, 'run_manifest.json'), 'w') as fout:
        json.dump(manifest, fout, indent=2)

    kf = KFold(n_splits=num_folds, shuffle=True, random_state=10042)
    completed_folds = []
    for kf_idx, (train_idxes, test_idxes) in enumerate(kf.split(trajs[:num_trajs])):
        if args.fold_idx >= 0 and kf_idx != args.fold_idx:
            continue
        train_trajs = [trajs[train_idx] for train_idx in train_idxes]
        test_trajs = [trajs[test_idx] for test_idx in test_idxes]

        best_test_ll = -np.inf
        best_ll = None
        for repeats in range(num_repeats):
            model_seed = args.rand_seed + 1009 * kf_idx + repeats
            if args.paired_fold_seeds:
                set_seed(model_seed)
            model = PGIAVI(
                num_latents=num_latents, num_states=num_states, num_actions=num_actions,
                train_trajs=train_trajs, test_trajs=test_trajs, P=P, discount=args.discount,
                model_type=args.model_type, hidden_dim=args.hidden_dim,
                rnn_hidden_dim=args.rnn_hidden_dim, num_layers=args.num_layers,
                dropout=args.dropout, nhead=args.nhead, lr=args.lr,
                reg_type=args.reg_type, reg_weight=args.reg_weight,
                num_epochs=args.num_epochs, loss_threshold=args.loss_threshold,
                max_iterations=args.max_iterations, gate_mode=args.gate_mode,
            )
            ll, f, mask, batched_iavi = model.fit()
            if ll['test'] > best_test_ll:
                best_test_ll = ll['test']
                best_ll = ll
                if args.save_npy:
                    param_dir = os.path.join(run_dir, f'{num_trajs}/fold_{kf_idx}')
                    os.makedirs(param_dir, exist_ok=True)
                    with open(os.path.join(param_dir, 'train_idxes.json'), 'w') as fout:
                        json.dump(train_idxes.tolist(), fout)
                    with open(os.path.join(param_dir, 'test_idxes.json'), 'w') as fout:
                        json.dump(test_idxes.tolist(), fout)
                    np.save(os.path.join(param_dir, 'mask_train.npy'), mask['train'])
                    np.save(os.path.join(param_dir, 'mask_test.npy'), mask['test'])
                    if args.p0_artifacts:
                        np.save(os.path.join(param_dir, 'gate_train.npy'),
                                np.asarray(f['gate_train']))
                        np.save(os.path.join(param_dir, 'gate_test.npy'),
                                np.asarray(f['gate_test']))
                        np.save(os.path.join(param_dir, 'responsibility_train.npy'),
                                np.asarray(f['responsibility_train']))
                        np.save(os.path.join(param_dir, 'responsibility_test.npy'),
                                np.asarray(f['responsibility_test']))
                        np.save(os.path.join(param_dir, 'step_log_score_train.npy'),
                                pad_step_scores(f['step_log_score_train'], mask['train']))
                        np.save(os.path.join(param_dir, 'step_log_score_test.npy'),
                                pad_step_scores(f['step_log_score_test'], mask['test']))
                    else:
                        np.save(os.path.join(param_dir, 'f_train.npy'), f['train'])
                        np.save(os.path.join(param_dir, 'f_test.npy'), f['test'])
                    for agent_idx in range(batched_iavi.K):
                        np.save(os.path.join(param_dir, f'r_{agent_idx}.npy'),
                                batched_iavi.r[agent_idx].cpu().numpy())
                        np.save(os.path.join(param_dir, f'q_{agent_idx}.npy'),
                                batched_iavi.q[agent_idx].cpu().numpy())
        output_df.loc[len(output_df)] = [
            num_trajs, kf_idx, best_ll['train'], best_ll['test'], args.rand_seed,
            args.rand_seed + 1009 * kf_idx if args.paired_fold_seeds else np.nan,
            args.gate_mode, best_ll['score_type'],
            best_ll['train_step_mean'], best_ll['test_step_mean'],
            best_ll['train_steps'], best_ll['test_steps'], best_ll['iterations'],
            best_ll['stop_reason'], best_ll['final_loss'], 'complete',
        ]
        output_df.to_csv(os.path.join(run_dir, args.ll_filename), index=False)
        completed_folds.append(kf_idx)
        del model, batched_iavi, f, mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    manifest['status'] = 'complete'
    manifest['completed_folds'] = completed_folds
    with open(os.path.join(run_dir, 'run_manifest.json'), 'w') as fout:
        json.dump(manifest, fout, indent=2)
