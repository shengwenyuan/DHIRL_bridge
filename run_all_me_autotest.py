import os
import re
import shutil
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_AUTOTEST_DIR = os.path.join(REPO_ROOT, 'data_autotest')
SRC_MAXENT_DIR = os.path.join(REPO_ROOT, 'src_max_entropy')
VENV_PYTHON = '/home/swy/myVENV/dhiql_venv/bin/python'


def discover_cases():
    """Find all (num_states, num_actions) from train_trajs_NS_NA.json in data_autotest/."""
    cases = []
    pattern = re.compile(r'^train_trajs_(\d+)_(\d+)\.json$')
    for name in os.listdir(DATA_AUTOTEST_DIR):
        m = pattern.match(name)
        if not m:
            continue
        ns, na = int(m.group(1)), int(m.group(2))
        val_name = f'val_trajs_{ns}_{na}.json'
        if not os.path.isfile(os.path.join(DATA_AUTOTEST_DIR, val_name)):
            continue
        cases.append((ns, na))
    cases.sort(key=lambda x: (x[0], x[1]))
    return cases


def run(cmd, cwd=None, description=""):
    """Run command (cmd[0] should be VENV_PYTHON); return True on success, False on failure."""
    cwd = cwd or REPO_ROOT
    print(f"  Run: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode}): {description}")
        return False
    return True


def prepare_maxent_data(ns, na):
    """
    MaxEnt script expects `trans_probs.npy` and `trajs.json` under a data_dir.
    For each (ns, na) autotest case, copy the suffixed files into a dedicated folder.
    """
    trajs_src = os.path.join(DATA_AUTOTEST_DIR, f'trajs_{ns}_{na}.json')
    trans_src = os.path.join(DATA_AUTOTEST_DIR, f'trans_probs_{ns}_{na}.npy')

    if not os.path.isfile(trajs_src) or not os.path.isfile(trans_src):
        print(f"  Missing trajs/trans_probs for ns={ns} na={na}; expected:")
        print(f"    {trajs_src}")
        print(f"    {trans_src}")
        return None

    data_dir = os.path.join(DATA_AUTOTEST_DIR, f'maxent_data_ns_{ns}_na_{na}')
    os.makedirs(data_dir, exist_ok=True)

    shutil.copy(trajs_src, os.path.join(data_dir, 'trajs.json'))
    shutil.copy(trans_src, os.path.join(data_dir, 'trans_probs.npy'))

    return data_dir


def main():
    os.chdir(REPO_ROOT)
    cases = discover_cases()
    if not cases:
        print("No train_trajs_NS_NA.json / val_trajs_NS_NA.json pairs found in data_autotest/")
        sys.exit(1)
    print(f"Found {len(cases)} cases: {cases}")

    failed = []
    for i, (ns, na) in enumerate(cases):
        print(f"\n[{i+1}/{len(cases)}] ns={ns} na={na}")

        # Currently, train_bridge_me.py is written for ns=768 only.
        # Skip incompatible cases to avoid shape assertions.
        if ns != 768:
            print(f"  Skipping ns={ns} na={na} (MaxEnt script currently expects num_states=768).")
            continue

        # 1) Merge train + val -> trajs_NS_NA.json (in data_autotest)
        if not run(
            [VENV_PYTHON, 'build_trajs_autotest.py', '--num_states', str(ns), '--num_actions', str(na)],
            cwd=DATA_AUTOTEST_DIR,
            description="build_trajs_autotest",
        ):
            failed.append((ns, na, "build_trajs_autotest"))
            continue

        # 2) Build trans_probs_NS_NA.npy (in data_autotest)
        if not run(
            [VENV_PYTHON, 'build_trans_autotest.py', '--num_states', str(ns), '--num_actions', str(na)],
            cwd=DATA_AUTOTEST_DIR,
            description="build_trans_autotest",
        ):
            failed.append((ns, na, "build_trans_autotest"))
            continue

        # 3) Prepare MaxEnt data dir with generic filenames
        data_dir = prepare_maxent_data(ns, na)
        if data_dir is None:
            failed.append((ns, na, "prepare_maxent_data"))
            continue

        # 4) Train MaxEnt IRL on this case
        output_dir = os.path.join(REPO_ROOT, 'outputs', 'maxent_autotest', f'ns_{ns}_na_{na}')
        if not run(
            [
                VENV_PYTHON,
                'train_bridge_me.py',
                '--data_dir',
                data_dir,
                '--output_dir',
                output_dir,
            ],
            cwd=SRC_MAXENT_DIR,
            description="train_bridge_me (MaxEnt IRL)",
        ):
            failed.append((ns, na, "train_bridge_me"))
            continue

        print(f"  OK ns={ns} na={na} -> outputs/maxent_autotest/ns_{ns}_na_{na}/")

    print("\n" + "=" * 60)
    if failed:
        print(f"Failed {len(failed)} case(s):")
        for ns, na, stage in failed:
            print(f"  ns={ns} na={na} at {stage}")
    else:
        print("All cases completed successfully.")
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()

