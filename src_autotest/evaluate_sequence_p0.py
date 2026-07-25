"""Evaluate out-of-fold P0 sequence artifacts without rerunning a model."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score


BOOTSTRAP_SEED = 20260725
BOOTSTRAP_SAMPLES = 2000
PROBABILITY_ATOL = 1e-5


def read_json(path):
    with path.open() as fin:
        return json.load(fin)


def parse_experiment(value):
    label, separator, path = value.partition('=')
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError(
            '--experiment must have the form LABEL=PATH'
        )
    return label, Path(path).expanduser()


def load_array(fold_dir, filename):
    path = fold_dir / filename
    if not path.is_file():
        raise ValueError(f'Missing artifact: {path}')
    return np.load(path, allow_pickle=False)


def validate_mask(mask, fold_dir):
    if mask.ndim != 2:
        raise ValueError(f'{fold_dir}: mask_test.npy must be two-dimensional.')
    if mask.dtype != np.bool_:
        if not np.isin(mask, [0, 1]).all():
            raise ValueError(f'{fold_dir}: mask_test.npy is not boolean.')
        mask = mask.astype(bool)
    lengths = mask.sum(axis=1)
    if (lengths <= 0).any():
        raise ValueError(f'{fold_dir}: every test trajectory needs a valid step.')
    prefix_mask = np.arange(mask.shape[1])[None, :] < lengths[:, None]
    if not np.array_equal(mask, prefix_mask):
        raise ValueError(f'{fold_dir}: each mask row must be one valid prefix.')
    return mask, lengths


def validate_probabilities(name, values, mask, num_latents, fold_dir):
    if values.ndim != 3 or values.shape[:2] != mask.shape:
        raise ValueError(
            f'{fold_dir}: {name} must have shape (trajectory, step, latent).'
        )
    if values.shape[2] != num_latents:
        raise ValueError(
            f'{fold_dir}: {name} has {values.shape[2]} latents, '
            f'expected {num_latents}.'
        )
    valid = values[mask]
    if not np.isfinite(valid).all():
        raise ValueError(f'{fold_dir}: {name} has non-finite valid values.')
    if (valid < -PROBABILITY_ATOL).any():
        raise ValueError(f'{fold_dir}: {name} has negative valid values.')
    if not np.allclose(
        valid.sum(axis=1), 1.0, atol=PROBABILITY_ATOL, rtol=PROBABILITY_ATOL
    ):
        raise ValueError(f'{fold_dir}: {name} is not normalized on valid steps.')
    if not np.allclose(values[~mask], 0.0, atol=PROBABILITY_ATOL, rtol=0.0):
        raise ValueError(f'{fold_dir}: {name} must be zero on padded steps.')
    return valid


def validate_step_scores(values, mask, fold_dir):
    if values.ndim != 2 or values.shape != mask.shape:
        raise ValueError(
            f'{fold_dir}: step_log_score_test.npy must match mask_test.npy.'
        )
    if not np.isfinite(values[mask]).all():
        raise ValueError(f'{fold_dir}: valid step scores must be finite.')
    if not np.isnan(values[~mask]).all():
        raise ValueError(f'{fold_dir}: padded step scores must be NaN.')


def empty_assignment_stats(num_latents):
    return {
        'num_latents': num_latents,
        'num_steps': 0,
        'entropy_sum': 0.0,
        'occupancy': np.zeros(num_latents, dtype=np.int64),
        'transitions': 0,
        'adjacent_pairs': 0,
        'duration_histogram': {},
    }


def update_assignment_stats(stats, probabilities):
    labels = np.argmax(probabilities, axis=1).astype(np.int16)
    positive = probabilities > 0
    entropy = -np.sum(
        np.where(positive, probabilities * np.log(np.maximum(probabilities, 1e-300)), 0.0),
        axis=1,
    )
    if stats['num_latents'] > 1:
        entropy /= np.log(stats['num_latents'])
    else:
        entropy[:] = 0.0
    stats['entropy_sum'] += float(entropy.sum())
    stats['num_steps'] += int(labels.size)
    stats['occupancy'] += np.bincount(
        labels, minlength=stats['num_latents']
    )
    if labels.size > 1:
        stats['transitions'] += int(np.count_nonzero(labels[1:] != labels[:-1]))
        stats['adjacent_pairs'] += int(labels.size - 1)
    boundaries = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    durations = np.diff(np.concatenate(([0], boundaries, [labels.size])))
    unique, counts = np.unique(durations, return_counts=True)
    for duration, count in zip(unique, counts):
        duration = int(duration)
        stats['duration_histogram'][duration] = (
            stats['duration_histogram'].get(duration, 0) + int(count)
        )
    return labels


def histogram_quantile(histogram, quantile):
    items = sorted(histogram.items())
    count = sum(value for _, value in items)

    def order_statistic(index):
        cumulative = 0
        for duration, frequency in items:
            cumulative += frequency
            if index < cumulative:
                return float(duration)
        raise RuntimeError('Invalid segment-duration histogram.')

    position = quantile * (count - 1)
    lower = int(np.floor(position))
    upper = int(np.ceil(position))
    fraction = position - lower
    return (
        order_statistic(lower) * (1.0 - fraction)
        + order_statistic(upper) * fraction
    )


def summarize_assignment(stats):
    steps = stats['num_steps']
    occupancy = stats['occupancy'].astype(float) / steps
    duration_histogram = stats['duration_histogram']
    segment_count = sum(duration_histogram.values())
    duration_sum = sum(
        duration * count for duration, count in duration_histogram.items()
    )
    durations = sorted(duration_histogram)
    return {
        'mean_normalized_entropy': stats['entropy_sum'] / steps,
        'occupancy_raw': occupancy.tolist(),
        'occupancy_sorted': np.sort(occupancy)[::-1].tolist(),
        'transition_rate': (
            stats['transitions'] / stats['adjacent_pairs']
            if stats['adjacent_pairs']
            else 0.0
        ),
        'num_transitions': stats['transitions'],
        'num_adjacent_pairs': stats['adjacent_pairs'],
        'segment_duration': {
            'count': segment_count,
            'mean': duration_sum / segment_count,
            'min': durations[0],
            'p25': histogram_quantile(duration_histogram, 0.25),
            'median': histogram_quantile(duration_histogram, 0.5),
            'p75': histogram_quantile(duration_histogram, 0.75),
            'max': durations[-1],
        },
    }


def load_experiment(label, experiment_dir):
    experiment_dir = experiment_dir.resolve()
    manifest_path = experiment_dir / 'run_manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'{label}: missing {manifest_path}')
    manifest = read_json(manifest_path)
    num_trajs = int(manifest['num_trajs_used'])
    num_latents = int(manifest['num_latents'])
    artifact_root = experiment_dir / str(num_trajs)
    fold_dirs = sorted(
        artifact_root.glob('fold_*'),
        key=lambda path: int(path.name.removeprefix('fold_')),
    )
    if not fold_dirs:
        raise ValueError(f'{label}: no fold artifacts under {artifact_root}')

    records = [None] * num_trajs
    gate_stats = empty_assignment_stats(num_latents)
    responsibility_stats = empty_assignment_stats(num_latents)
    fold_occupancies = {'gate': [], 'responsibility': []}
    observed_ids = set()

    for fold_dir in fold_dirs:
        fold_idx = int(fold_dir.name.removeprefix('fold_'))
        fold_gate_stats = empty_assignment_stats(num_latents)
        fold_responsibility_stats = empty_assignment_stats(num_latents)
        test_idxes = np.asarray(
            read_json(fold_dir / 'test_idxes.json'), dtype=np.int64
        )
        if test_idxes.ndim != 1 or len(np.unique(test_idxes)) != len(test_idxes):
            raise ValueError(f'{fold_dir}: test_idxes.json must contain unique IDs.')
        if ((test_idxes < 0) | (test_idxes >= num_trajs)).any():
            raise ValueError(f'{fold_dir}: test trajectory ID is out of range.')
        duplicate_ids = observed_ids.intersection(test_idxes.tolist())
        if duplicate_ids:
            raise ValueError(
                f'{label}: test trajectory IDs occur in multiple folds: '
                f'{sorted(duplicate_ids)[:10]}'
            )

        mask, lengths = validate_mask(
            load_array(fold_dir, 'mask_test.npy'), fold_dir
        )
        if mask.shape[0] != len(test_idxes):
            raise ValueError(f'{fold_dir}: test IDs and artifact rows disagree.')
        gate = load_array(fold_dir, 'gate_test.npy')
        responsibility = load_array(fold_dir, 'responsibility_test.npy')
        step_scores = load_array(fold_dir, 'step_log_score_test.npy')
        validate_probabilities(
            'gate_test.npy', gate, mask, num_latents, fold_dir
        )
        validate_probabilities(
            'responsibility_test.npy', responsibility, mask, num_latents, fold_dir
        )
        validate_step_scores(step_scores, mask, fold_dir)

        for row, trajectory_id in enumerate(test_idxes):
            length = int(lengths[row])
            gate_labels = update_assignment_stats(
                gate_stats, gate[row, :length]
            )
            update_assignment_stats(fold_gate_stats, gate[row, :length])
            responsibility_labels = update_assignment_stats(
                responsibility_stats, responsibility[row, :length]
            )
            update_assignment_stats(
                fold_responsibility_stats, responsibility[row, :length]
            )
            records[int(trajectory_id)] = {
                'fold': fold_idx,
                'gate': gate_labels,
                'responsibility': responsibility_labels,
                'step_score': step_scores[row, :length].astype(np.float64, copy=True),
            }
        observed_ids.update(test_idxes.tolist())
        fold_occupancies['gate'].append({
            'fold': fold_idx,
            'raw': (
                fold_gate_stats['occupancy'] / fold_gate_stats['num_steps']
            ).tolist(),
            'sorted': np.sort(
                fold_gate_stats['occupancy'] / fold_gate_stats['num_steps']
            )[::-1].tolist(),
        })
        fold_occupancies['responsibility'].append({
            'fold': fold_idx,
            'raw': (
                fold_responsibility_stats['occupancy']
                / fold_responsibility_stats['num_steps']
            ).tolist(),
            'sorted': np.sort(
                fold_responsibility_stats['occupancy']
                / fold_responsibility_stats['num_steps']
            )[::-1].tolist(),
        })

    expected_ids = set(range(num_trajs))
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        raise ValueError(
            f'{label}: OOF coverage is not exact; '
            f'missing={missing[:10]}, extra={extra[:10]}.'
        )

    trajectory_means = np.asarray(
        [record['step_score'].mean() for record in records], dtype=np.float64
    )
    step_sum = sum(float(record['step_score'].sum()) for record in records)
    step_count = sum(len(record['step_score']) for record in records)
    gate_summary = summarize_assignment(gate_stats)
    responsibility_summary = summarize_assignment(responsibility_stats)
    for name, assignment_summary in (
        ('gate', gate_summary),
        ('responsibility', responsibility_summary),
    ):
        assignment_summary.pop('occupancy_raw')
        assignment_summary.pop('occupancy_sorted')
        sorted_occupancy = np.asarray([
            fold['sorted'] for fold in fold_occupancies[name]
        ])
        assignment_summary['occupancy_by_fold'] = fold_occupancies[name]
        assignment_summary['occupancy_sorted_fold_mean'] = (
            sorted_occupancy.mean(axis=0).tolist()
        )
        assignment_summary['occupancy_sorted_fold_std'] = (
            sorted_occupancy.std(axis=0).tolist()
        )

    summary = {
        'path': str(experiment_dir),
        'manifest': manifest,
        'validation': {
            'exact_oof_coverage': True,
            'num_trajectories': num_trajs,
            'num_folds': len(fold_dirs),
            'num_valid_steps': step_count,
            'normalized_probabilities': True,
            'prefix_masks': True,
            'zero_probability_padding': True,
            'nan_step_score_padding': True,
        },
        'gate': gate_summary,
        'responsibility': responsibility_summary,
        'step_score': {
            'score_type': manifest.get('score_type'),
            'trajectory_macro_mean': float(trajectory_means.mean()),
            'valid_step_micro_mean': step_sum / step_count,
            'num_trajectories': num_trajs,
            'num_valid_steps': step_count,
        },
    }
    return {'manifest': manifest, 'records': records, 'summary': summary}


def bootstrap_score_delta(left_records, right_records):
    left_sums = np.asarray(
        [record['step_score'].sum() for record in left_records], dtype=np.float64
    )
    right_sums = np.asarray(
        [record['step_score'].sum() for record in right_records], dtype=np.float64
    )
    counts = np.asarray(
        [len(record['step_score']) for record in left_records], dtype=np.int64
    )
    trajectory_deltas = right_sums / counts - left_sums / counts
    sum_deltas = right_sums - left_sums
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    macro_samples = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    micro_samples = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for sample_idx in range(BOOTSTRAP_SAMPLES):
        selected = rng.integers(0, len(counts), size=len(counts))
        macro_samples[sample_idx] = trajectory_deltas[selected].mean()
        micro_samples[sample_idx] = (
            sum_deltas[selected].sum() / counts[selected].sum()
        )
    return {
        'delta_definition': 'right_minus_left',
        'bootstrap_unit': 'trajectory',
        'bootstrap_seed': BOOTSTRAP_SEED,
        'bootstrap_samples': BOOTSTRAP_SAMPLES,
        'trajectory_macro_delta': float(trajectory_deltas.mean()),
        'trajectory_macro_95_ci': np.quantile(
            macro_samples, [0.025, 0.975]
        ).tolist(),
        'valid_step_micro_delta': float(sum_deltas.sum() / counts.sum()),
        'valid_step_micro_95_ci': np.quantile(
            micro_samples, [0.025, 0.975]
        ).tolist(),
    }


def comparable_dataset(left, right):
    left_manifest = left['manifest']
    right_manifest = right['manifest']
    keys = (
        'domain', 'num_trajs_used', 'num_latents',
        'trajectory_bytes', 'trajectory_sha256',
        'transition_bytes', 'transition_sha256',
    )
    differences = [
        key for key in keys
        if (
            key not in left_manifest
            or key not in right_manifest
            or left_manifest[key] != right_manifest[key]
        )
    ]
    return differences


def compare_pair(left_label, left, right_label, right):
    result = {'left': left_label, 'right': right_label}
    differences = comparable_dataset(left, right)
    if differences:
        result.update({
            'comparable': False,
            'reason': f'dataset manifest fields differ: {", ".join(differences)}',
        })
        return result

    if len(left['records']) != len(right['records']):
        raise ValueError(
            f'{left_label}/{right_label}: trajectory counts differ.'
        )
    left_records = left['records']
    right_records = right['records']
    for trajectory_id, (left_record, right_record) in enumerate(
        zip(left_records, right_records)
    ):
        if left_record['fold'] != right_record['fold']:
            raise ValueError(
                f'{left_label}/{right_label}: trajectory {trajectory_id} '
                'belongs to different test folds.'
            )
        if len(left_record['gate']) != len(right_record['gate']):
            raise ValueError(
                f'{left_label}/{right_label}: trajectory {trajectory_id} '
                'has different valid lengths.'
            )

    fold_comparisons = []
    fold_ids = sorted({record['fold'] for record in left_records})
    for fold_idx in fold_ids:
        paired_records = [
            (left_record, right_record)
            for left_record, right_record in zip(left_records, right_records)
            if left_record['fold'] == fold_idx
        ]
        gate_left = np.concatenate([pair[0]['gate'] for pair in paired_records])
        gate_right = np.concatenate([pair[1]['gate'] for pair in paired_records])
        responsibility_left = np.concatenate([
            pair[0]['responsibility'] for pair in paired_records
        ])
        responsibility_right = np.concatenate([
            pair[1]['responsibility'] for pair in paired_records
        ])
        fold_comparisons.append({
            'fold': fold_idx,
            'num_trajectories': len(paired_records),
            'num_steps': int(gate_left.size),
            'gate_ari': float(adjusted_rand_score(gate_left, gate_right)),
            'responsibility_ari': float(
                adjusted_rand_score(responsibility_left, responsibility_right)
            ),
        })
    fold_step_counts = np.asarray([
        comparison['num_steps'] for comparison in fold_comparisons
    ])
    result.update({
        'comparable': True,
        'num_trajectories': len(left_records),
        'num_steps': int(fold_step_counts.sum()),
        'ari_by_fold': fold_comparisons,
        'gate_ari_fold_mean': float(np.mean([
            comparison['gate_ari'] for comparison in fold_comparisons
        ])),
        'gate_ari_step_weighted_mean': float(np.average(
            [comparison['gate_ari'] for comparison in fold_comparisons],
            weights=fold_step_counts,
        )),
        'responsibility_ari_fold_mean': float(np.mean([
            comparison['responsibility_ari'] for comparison in fold_comparisons
        ])),
        'responsibility_ari_step_weighted_mean': float(np.average(
            [comparison['responsibility_ari'] for comparison in fold_comparisons],
            weights=fold_step_counts,
        )),
    })
    predictive_type = 'predictive_action_log_likelihood'
    left_type = left['manifest'].get('score_type')
    right_type = right['manifest'].get('score_type')
    if left_type == predictive_type and right_type == predictive_type:
        result['step_score_comparison'] = bootstrap_score_delta(
            left_records, right_records
        )
        result['step_score_comparison']['computed'] = True
    else:
        result['step_score_comparison'] = {
            'computed': False,
            'reason': (
                'requires predictive_action_log_likelihood for both experiments'
            ),
            'left_score_type': left_type,
            'right_score_type': right_type,
        }
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate saved P0 test-fold sequence artifacts.'
    )
    parser.add_argument(
        '--experiment', action='append', required=True, type=parse_experiment,
        metavar='LABEL=PATH',
    )
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    labels = [label for label, _ in args.experiment]
    if len(labels) != len(set(labels)):
        raise ValueError('Experiment labels must be unique.')
    experiments = {
        label: load_experiment(label, path)
        for label, path in args.experiment
    }
    report = {
        'schema': 'p0_sequence_evaluation_v1',
        'experiments': {
            label: experiment['summary']
            for label, experiment in experiments.items()
        },
        'pairwise': [
            compare_pair(left_label, experiments[left_label],
                         right_label, experiments[right_label])
            for left_label, right_label in itertools.combinations(labels, 2)
        ],
    }
    output_path = args.output.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as fout:
        json.dump(report, fout, indent=2)
        fout.write('\n')
    print(f'Wrote {output_path}')


if __name__ == '__main__':
    main()
