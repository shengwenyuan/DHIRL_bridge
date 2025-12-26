import numpy as np
import torch
import time

from scipy.special import logsumexp
from model.intention import IntentionNet, StatesRNN, IntentionTransformer
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
old_settings = np.seterr(over='raise')

class IAVI:
    def __init__(self, num_states, num_actions, P, expert_policy, discount, threshold=1e-3):
        self.num_states = num_states
        self.num_actions = num_actions
        self.P = P
        self.expert_policy = expert_policy
        self.discount = discount
        self.threshold = threshold
        self.epsilon = 1e-6

        self.r = np.random.randn(self.num_states, self.num_actions)
        self.q = np.random.randn(self.num_states, self.num_actions)

    def train(self):
        X = np.ones((self.num_actions, self.num_actions))
        X *= -1 / (self.num_actions - 1)
        for i in range(self.num_actions):
            X[i, i] = 1

        e = 0
        while True:
            e += 1
            delta = 0
            for s in range(self.num_states):
                tp = self.P[s, :, :]
                # eta = np.log(self.expert_policy[s, :] + self.epsilon) - self.discount * np.matmul(
                #     tp.T, logsumexp(self.q, axis=1).reshape(-1, 1)).reshape(-1)
                try:
                    eta = np.log(self.expert_policy[s, :] + self.epsilon) - self.discount * np.matmul(
                        tp.T, np.max(self.q, axis=1).reshape(-1, 1)).reshape(-1)
                except:
                    print("Overflow detected in matmul!")
                finally:
                    np.seterr(**old_settings)
                Y = np.zeros(self.num_actions)
                for a in range(self.num_actions):
                    eta_a = eta[a]
                    action_b = [b for b in range(self.num_actions) if b != a]
                    eta_b = eta[action_b]
                    Y[a] = eta_a - 1 / (self.num_actions - 1) * np.sum(eta_b)

                r = np.linalg.lstsq(X, Y, rcond=None)[0]

                delta = max(delta, np.max(np.abs(self.r[s, :] - r)))

                self.r[s, :] = r
                # self.q[s, :] = r + self.discount * np.matmul(tp.T, logsumexp(self.q, axis=1).reshape(-1, 1)).reshape(-1)
                self.q[s, :] = r + self.discount * np.matmul(tp.T, np.max(self.q, axis=1).reshape(-1, 1)).reshape(-1)

            if delta < self.threshold:
                break


class PGIAVI:
    def __init__(self, num_latents, num_states, num_actions, P, train_trajs, test_trajs, discount):
        self.num_latents = num_latents  # K
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_phis = 80              # φ
        self.P = P                      # env trans
        self.discount = discount
        self.train_trajs = train_trajs
        self.test_trajs = test_trajs

        # self.intention_net = IntentionTransformer(phi_dim=self.num_phis, 
        #                                num_latents=self.num_latents, 
        #                                d_model=128, 
        #                                nhead=4,
        #                                num_layers=1,
        #                                dropout=0.2)
        # self.target_intention_net = IntentionTransformer(phi_dim=self.num_phis, 
        #                                num_latents=self.num_latents, 
        #                                d_model=128, 
        #                                nhead=4,
        #                                num_layers=1,
        #                                dropout=0.2)
        self.intention_net = StatesRNN(phi_dim=self.num_phis, 
                                       num_latents=self.num_latents, 
                                       hidden_dim=128, 
                                       rnn_hidden_dim=128, 
                                       num_layers=1,
                                       dropout=0.3)
        self.target_intention_net = StatesRNN(phi_dim=self.num_phis, 
                                       num_latents=self.num_latents, 
                                       hidden_dim=128, 
                                       rnn_hidden_dim=128, 
                                       num_layers=1,
                                       dropout=0.3)
        self.target_intention_net.load_state_dict(self.intention_net.state_dict())
        self.target_intention_net.eval()
        self.optimizer = torch.optim.Adam(self.intention_net.parameters(), lr=5e-3)

        self.state_emb = torch.nn.Embedding(self.num_states, 64)
        self.action_emb = torch.nn.Embedding(self.num_actions, 16)

    def intention_mapping(self, phis, log_pi):
        f_logits = self.target_intention_net(phis.unsqueeze(0)).squeeze(0)              # (T, K)
        log_f = torch.log_softmax(f_logits, dim=-1) # TODO: MDP? set explicit prior: intention transition dynamics
        log_joint = log_f + log_pi.T                # (T, K), log(f_k * π_k) = log P(z_t=k, a_t | s_t, phi_t)
        log_p_gamma = log_joint - torch.logsumexp(log_joint, dim=-1, keepdim=True)  # (T, K)

        return log_p_gamma, log_f, log_joint
    
    def get_log_pi(self, traj, agents):
        log_pi = torch.zeros((self.num_latents, len(traj)))

        for latent_idx, agent in enumerate(agents):
            q = torch.as_tensor(agent.q, dtype=torch.float32)
            pi = torch.softmax(q, dim=-1)
            for t, (s, a, ns) in enumerate(traj):
                log_pi[latent_idx, t] = torch.log(pi[s, a] + 1e-8)

        return log_pi

    def encode_session_traj(self, traj):
        states = torch.tensor([s for s, a, ns in traj], dtype=torch.long)
        actions = torch.tensor([a for s, a, ns in traj], dtype=torch.long)
        s_emb = self.state_emb(states)  # (T, )
        a_emb = self.action_emb(actions)  # (T, )
        phis = torch.cat([s_emb, a_emb], dim=-1)  # (T, )
        return phis
    
    def train_batched(self, batch_phis, batch_target_gamma, num_epochs=1):
        """
        :param agents: List of IAVI agents
        :param num_epochs: Number of passes through the data
        """
        total_loss = 0
        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            
            pred_logits = self.intention_net(batch_phis)  # (B, T, K)
            pred_logf = torch.log_softmax(pred_logits, dim=-1)  # (B, T, K)
            
            # Compute loss: negative log-likelihood
            loss = -(batch_target_gamma * pred_logf).sum(dim=-1).mean()
            # ce_loss = -(batch_target_gamma * pred_logf).sum(-1).mean()
            # entropy = -(pred_logf * torch.exp(pred_logf)).sum(-1).mean()
            # loss = ce_loss - 0.02 * entropy
            
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / num_epochs
    
    def fit(self):
        uniform_policy = np.full((self.num_states, self.num_actions), 1.0 / self.num_actions)
        agents = []
        for _ in range(self.num_latents):
            agent = IAVI(
                num_states=self.num_states,
                num_actions=self.num_actions,
                P=self.P,
                expert_policy=uniform_policy,
                discount=self.discount
            )
            agent.train()
            agents.append(agent)

        logger_cnt = 0
        total_q_time = 0
        total_other_time = 0
        iteration_start_time = time.time()

        while True:
            logger_cnt += 1
            
            # * * * E-step: compute posterior * * *
            log_p_gammas = []
            batch_phis = []
            batch_target_gamma = []
            for traj_idx, traj in enumerate(self.train_trajs):
                phis = self.encode_session_traj(traj)
                log_pi = self.get_log_pi(traj, agents)
                with torch.no_grad():
                    log_p_gamma, *_ = self.intention_mapping(phis, log_pi)
                log_p_gammas.append(log_p_gamma)

                batch_phis.append(phis)
                batch_target_gamma.append(torch.exp(log_p_gamma))
            
            # Pad sequences to same length for RNN input
            max_len = max(phi.shape[0] for phi in batch_phis)
            batch_phis_padded = torch.zeros(len(batch_phis), max_len, self.num_phis)
            batch_target_gamma_padded = torch.zeros(len(batch_target_gamma), max_len, self.num_latents)
            
            for i, (phi, gamma) in enumerate(zip(batch_phis, batch_target_gamma)):
                seq_len = phi.shape[0]
                batch_phis_padded[i, :seq_len] = phi
                batch_target_gamma_padded[i, :seq_len] = gamma
            
            batch_phis = batch_phis_padded  # (B, T, phi_dim)
            batch_target_gamma = batch_target_gamma_padded  # (B, T, K)

            # * * * Update Q-value & policies * * *
            q_start_time = time.time()
            for latent_idx in range(self.num_latents):
                expert_pi = torch.zeros((self.num_states, self.num_actions))
                for traj_idx, traj in enumerate(self.train_trajs):
                    weights = batch_target_gamma[traj_idx][:, latent_idx]
                    for t, (s, a, ns) in enumerate(traj):
                        expert_pi[s, a] += weights[t]
                mask = expert_pi.sum(dim=1) == 0
                expert_pi[mask] = 1e-6
                expert_pi /= expert_pi.sum(dim=1, keepdim=True)

                agent = IAVI(
                    num_states=self.num_states,
                    num_actions=self.num_actions,
                    P=self.P,
                    expert_policy=expert_pi.numpy(),
                    discount=self.discount
                )
                agent.train()
                agents[latent_idx] = agent
            q_time = time.time() - q_start_time
            total_q_time += q_time

            # * * * Update intention network * * *
            other_start_time = time.time()
            total_loss = self.train_batched(batch_phis, batch_target_gamma, num_epochs=1)
            other_time = time.time() - other_start_time
            total_other_time += other_time

            self.target_intention_net.load_state_dict(self.intention_net.state_dict())

            if logger_cnt % 1 == 0:
                iteration_time = time.time() - iteration_start_time
                print(f'Iteration {logger_cnt}, Loss: {total_loss:.4f}, Q-update: {total_q_time:.2f}s, NN: {total_other_time:.2f}s, Total: {iteration_time:.2f}s')
                total_q_time = 0
                total_other_time = 0

            if abs(total_loss) < 5e-3 or logger_cnt >= 100:
                final_iteration_time = time.time() - iteration_start_time
                print(f'Iteration {logger_cnt}, Converged with Loss: {total_loss:.4f}, Total time: {final_iteration_time:.2f}s')
                break

        f = {}
        ll = {}
        for ds in ['train', 'test']:
            trajs = eval(f'self.{ds}_trajs')
            fs = []
            lls = []
            for traj_idx, traj in enumerate(trajs):
                phis = self.encode_session_traj(traj)
                log_pi = self.get_log_pi(traj, agents)
                with torch.no_grad():
                    _, log_f, log_p_joint = self.intention_mapping(phis, log_pi)
                    log_f = log_f.numpy()
                    log_p_joint = log_p_joint.numpy()
                fs.append(np.exp(log_f))
                # lls.append(logsumexp(log_p_joint, axis=-1).sum()) # whole trajectory LL
                lls.append(np.mean(logsumexp(log_p_joint, axis=-1))) # per-step LL
            lls = np.mean(np.hstack(lls))
            ll[ds] = np.mean(lls)
            f[ds] = fs

        return ll, f, agents

    def predict(self, trajs, agents):
        fs = []
        lls = []
        for traj_idx, traj in enumerate(trajs):
            phis = self.encode_session_traj(traj)
            log_pi = self.get_log_pi(traj, agents)
            with torch.no_grad():
                _, log_f, log_p_joint = self.intention_mapping(phis, log_pi)
                log_f = log_f.numpy()
                log_p_joint = log_p_joint.numpy()
            fs.append(np.exp(log_f))
            lls.append(logsumexp(log_p_joint, axis=-1).sum()) # whole trajectory LL
        lls = np.mean(np.hstack(lls))
        
        return lls, fs