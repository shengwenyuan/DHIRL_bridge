import unittest

import numpy as np
import torch

from model import intention
from src_autotest.algorithms import PGIAVI


MODEL_CLASSES = ('IntentionRNN', 'IntentionLSTM', 'IntentionTransformer')


def make_model(class_name, gate_mode=None):
    model_class = getattr(intention, class_name)
    kwargs = {
        'num_states': 9,
        'num_actions': 4,
        'num_latents': 3,
        'num_layers': 1,
        'dropout': 0.0,
    }
    if class_name == 'IntentionTransformer':
        kwargs.update(d_model=8, nhead=2)
    else:
        kwargs.update(hidden_dim=8, rnn_hidden_dim=8)
    if gate_mode is not None:
        kwargs['gate_mode'] = gate_mode
    return model_class(**kwargs).eval()


def legacy_forward(model, class_name, states, actions):
    x = model.state_embed(states) + model.action_embed(actions)
    if class_name == 'IntentionTransformer':
        return model.fc_out(model.transformer(model.pos_encoding(x)))
    recurrent_output, _ = model.rnn(x)
    return model.output_proj(recurrent_output)


def toy_problem():
    num_states = 3
    num_actions = 2
    transition = np.zeros((num_states, num_actions, num_states), dtype=np.float64)
    for state in range(num_states):
        for action in range(num_actions):
            transition[state, action, (state + action + 1) % num_states] = 1.0
    train_trajs = [
        [(0, 0, 1), (1, 1, 0), (0, 1, 2)],
        [(1, 0, 2), (2, 0, 0)],
        [(2, 1, 1), (1, 1, 0), (0, 0, 1), (1, 0, 2)],
    ]
    test_trajs = [
        [(0, 1, 2), (2, 0, 0)],
        [(1, 1, 0), (0, 0, 1), (1, 0, 2)],
    ]
    return transition, train_trajs, test_trajs


class FakeBatchedIAVI:
    def __init__(self, q):
        self.q = q

    def get_policies(self):
        return torch.softmax(self.q, dim=-1)


class CausalGateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.states = torch.tensor([[0, 1, 2, 3, 4]])
        self.actions = torch.tensor([[0, 1, 2, 3, 0]])

    def test_legacy_default_is_unchanged(self):
        for class_name in MODEL_CLASSES:
            default_model = make_model(class_name)
            explicit_model = make_model(class_name, 'retrospective')
            explicit_model.load_state_dict(default_model.state_dict())
            default_output = default_model(self.states, self.actions)
            explicit_output = explicit_model(self.states, self.actions)
            manual_output = legacy_forward(
                default_model, class_name, self.states, self.actions
            )
            self.assertEqual(
                list(default_model.state_dict()), list(explicit_model.state_dict())
            )
            torch.testing.assert_close(default_output, explicit_output)
            torch.testing.assert_close(default_output, manual_output)

    def test_causal_and_state_history_information_sets(self):
        changed_states = self.states.clone()
        changed_states[:, 3:] = torch.tensor([[7, 8]])
        changed_actions = self.actions.clone()
        changed_actions[:, 2:] = torch.tensor([[1, 0, 3]])

        causal_x = intention._combine_gate_inputs(
            torch.zeros(1, 5, 8), torch.ones(1, 5, 8), 'causal'
        )
        self.assertTrue(torch.equal(causal_x[:, 0], torch.zeros(1, 8)))

        for class_name in MODEL_CLASSES:
            causal_model = make_model(class_name, 'causal')
            original = causal_model(self.states, self.actions)
            changed = causal_model(changed_states, changed_actions)
            torch.testing.assert_close(original[:, :3], changed[:, :3])

            state_model = make_model(class_name, 'state_only')
            changed_all_actions = (self.actions + 1) % 4
            torch.testing.assert_close(
                state_model(self.states, self.actions),
                state_model(self.states, changed_all_actions),
            )

    def test_padding_does_not_change_valid_outputs(self):
        states = torch.tensor([[0, 1, 2, 0, 0]])
        actions = torch.tensor([[0, 1, 2, 0, 0]])
        changed_states = torch.tensor([[0, 1, 2, 7, 8]])
        changed_actions = torch.tensor([[0, 1, 2, 3, 1]])
        mask = torch.tensor([[True, True, True, False, False]])
        for class_name in MODEL_CLASSES:
            model = make_model(class_name, 'causal')
            output = model(states, actions, mask=mask, total_length=5)
            changed = model(
                changed_states, changed_actions, mask=mask, total_length=5
            )
            torch.testing.assert_close(output[:, :3], changed[:, :3])

    def test_predictive_policy_is_normalized(self):
        q = torch.tensor([
            [[20.0, -20.0, 0.0, 1.0]],
            [[-10.0, 10.0, 0.0, -1.0]],
        ])
        scorer = PGIAVI.__new__(PGIAVI)
        scorer.num_latents = 2
        scorer.gate_mode = 'causal'
        log_gate = torch.log_softmax(torch.tensor([0.3, -0.7]), dim=0)
        iavi = FakeBatchedIAVI(q)
        log_action_probs = []
        for action in range(4):
            log_policy = scorer.get_batch_log_pi(
                [[(0, action, 0)]], iavi
            )[0, 0]
            log_action_probs.append(torch.logsumexp(log_gate + log_policy, dim=0))
            torch.testing.assert_close(
                log_policy,
                torch.log_softmax(q, dim=-1)[:, 0, action],
            )
        torch.testing.assert_close(
            torch.logsumexp(torch.stack(log_action_probs), dim=0),
            torch.tensor(0.0),
            atol=1e-6,
            rtol=0,
        )

    def test_fit_contract(self):
        transition, train_trajs, test_trajs = toy_problem()
        for gate_mode in ('retrospective', 'causal', 'state_only'):
            np.random.seed(17)
            torch.manual_seed(17)
            model = PGIAVI(
                num_latents=2,
                num_states=3,
                num_actions=2,
                train_trajs=train_trajs,
                test_trajs=test_trajs,
                P=transition,
                discount=0.9,
                hidden_dim=8,
                rnn_hidden_dim=8,
                dropout=0.0,
                reg_weight=0.0,
                num_epochs=1,
                loss_threshold=0.0,
                max_iterations=1,
                gate_mode=gate_mode,
            )
            scores, outputs, masks, _ = model.fit()
            self.assertEqual(scores['iterations'], 1)
            self.assertEqual(scores['stop_reason'], 'max_iterations')
            for split in ('train', 'test'):
                legacy_gate = np.asarray(outputs[split])
                gate = np.asarray(outputs[f'gate_{split}'])
                responsibility = np.asarray(outputs[f'responsibility_{split}'])
                mask = masks[split]
                np.testing.assert_allclose(gate[mask].sum(axis=-1), 1.0, atol=1e-6)
                np.testing.assert_allclose(
                    responsibility[mask].sum(axis=-1), 1.0, atol=1e-6
                )
                np.testing.assert_allclose(gate[~mask], 0.0)
                np.testing.assert_allclose(responsibility[~mask], 0.0)
                np.testing.assert_allclose(legacy_gate[~mask], 1.0)
                if gate_mode != 'retrospective':
                    self.assertLessEqual(
                        max(np.max(x) for x in outputs[f'step_log_score_{split}']),
                        1e-6,
                    )

    def test_invalid_gate_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            make_model('IntentionRNN', 'future_action')


if __name__ == '__main__':
    unittest.main(verbosity=2)
