import numpy as np
import torch
import time

from scipy.special import logsumexp
from model.intention import IntentionNet, StatesRNN, IntentionTransformer
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class IAVI:
    def __init__(self, num_states, num_actions, P, expert_policy, discount, threshold=1e-3):
        self.num_states = num_states
        self.num_actions = num_actions
        self.P = np.transpose(P, (0, 2, 1))
        self.expert_policy = expert_policy
        self.discount = discount
        self.threshold = threshold
        self.epsilon = 1e-6

        self.r = np.random.randn(self.num_states, self.num_actions)
        self.q = np.random.randn(self.num_states, self.num_actions)

        X = np.full((self.num_actions, self.num_actions), -1 / (self.num_actions - 1))
        np.fill_diagonal(X, 1.0)
        self.X = X

    def train(self):
        e = 0
        while True:
            e += 1
            delta = 0
            for s in range(self.num_states):
                tp = self.P[s, :, :]
                opt_nextv = self.discount * np.matmul(tp.T, np.max(self.q, axis=1).reshape(-1, 1)).reshape(-1)
                eta = np.log(self.expert_policy[s, :] + self.epsilon) - opt_nextv
                if not np.all(np.isfinite(eta)):
                    print("Non-finite eta detected!")
                
                # Y = np.zeros(self.num_actions)
                # for a in range(self.num_actions):
                #     eta_a = eta[a]
                #     action_b = [b for b in range(self.num_actions) if b != a]
                #     eta_b = eta[action_b]
                #     Y[a] = eta_a - 1 / (self.num_actions - 1) * np.sum(eta_b)
                eta_sum = eta.sum(axis=0, keepdims=True)
                Y = eta - (eta_sum - eta) / (self.num_actions - 1) 

                r = np.linalg.lstsq(self.X, Y, rcond=None)[0]

                delta = max(delta, np.max(np.abs(self.r[s, :] - r)))

                alpha = 0.0
                self.r[s, :] = alpha * self.r[s, :] + (1 - alpha) * r
                self.q[s, :] = alpha * self.q[s, :] + (1 - alpha) * (self.r[s, :] + opt_nextv)

            if delta < self.threshold:
                break

        return delta


class IAVI_GPU:
    def __init__(self, num_states, num_actions, P, expert_policy, discount, threshold=1e-3, alpha=0.1, device='cuda'):
        self.num_states = num_states
        self.num_actions = num_actions
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.P = torch.as_tensor(P, dtype=torch.float64, device=self.device)
        self.expert_policy = torch.as_tensor(expert_policy, dtype=torch.float64, device=self.device)
        self.discount = discount
        self.threshold = threshold
        self.epsilon = 1e-6
        self.alpha = alpha
        self.batch_size = num_states

        self.r = torch.randn(self.num_states, self.num_actions, dtype=torch.float64, device=self.device)
        self.q = torch.randn(self.num_states, self.num_actions, dtype=torch.float64, device=self.device)

        X = torch.full((self.num_actions, self.num_actions), -1 / (self.num_actions - 1), dtype=torch.float64, device=self.device)
        X.fill_diagonal_(1.0)
        self.X = X

    def train(self):
        e = 0
        while e < 1e5:
            e += 1
            delta = 0
            # sampled_indices = torch.randperm(self.num_states, device=self.device)[:self.batch_size]
            for start_idx in range(0, self.num_states, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.num_states)
                sampled_indices = torch.arange(start_idx, end_idx, device=self.device)

                tp_batch = self.P[sampled_indices]  # (batch_size, num_actions, num_states)
                expert_policy_batch = self.expert_policy[sampled_indices]  # (batch_size, num_actions)
                max_q = torch.max(self.q, dim=1).values
                opt_nextv_batch = self.discount * torch.matmul(tp_batch, max_q.unsqueeze(-1)).squeeze(-1)
                eta_batch = torch.log(expert_policy_batch + self.epsilon) - opt_nextv_batch  # (batch_size, num_actions)
                if not torch.all(torch.isfinite(eta_batch)):
                    print("Non-finite eta detected!")

                eta_sum = eta_batch.sum(dim=1, keepdim=True)
                Y_batch = eta_batch - (eta_sum - eta_batch) / (self.num_actions - 1)  # (batch_size, num_actions)

                # must convert to numpy for lstsq
                tX = self.X.cpu().numpy()
                tY = Y_batch.cpu().numpy()
                tr = np.linalg.lstsq(tX, tY.T, rcond=None)[0]

                r_new_batch = torch.as_tensor(tr, dtype=torch.float64, device=self.device).T 
                delta_batch = torch.max(torch.abs(self.r[sampled_indices] - r_new_batch)).item()
                delta = max(delta, delta_batch)

                self.r[sampled_indices] = self.alpha * self.r[sampled_indices] + (1 - self.alpha) * r_new_batch
                self.q[sampled_indices] = self.alpha * self.q[sampled_indices] + (1 - self.alpha) * (self.r[sampled_indices] + opt_nextv_batch)

            if delta < self.threshold:
                break

        if e >= 1e5:
            raise RuntimeError("IAVI_GPU did not converge within the maximum number of iterations.")
        
        return delta
    
    def get_policy(self):
        return torch.softmax(self.q, dim=-1).cpu().numpy()
    
    def get_q_values(self):
        return self.q.cpu().numpy()
    
    def get_rewards(self):
        return self.r.cpu().numpy()


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
                                       num_layers=2,
                                       dropout=0.3)
        self.target_intention_net = StatesRNN(phi_dim=self.num_phis, 
                                       num_latents=self.num_latents, 
                                       hidden_dim=128, 
                                       rnn_hidden_dim=128, 
                                       num_layers=2,
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
        s_emb = self.state_emb(states).detach()  # (T, )
        a_emb = self.action_emb(actions).detach()  # (T, )
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

    def train_minibatch(self, loader, num_epochs=1):
        """
        :param agents: List of IAVI agents
        :param num_epochs: Number of passes through the data
        """
        loss_list = []
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_phis, batch_target_gamma in loader:
                self.optimizer.zero_grad()
                
                pred_logits = self.intention_net(batch_phis)  # (B, T, K)
                pred_logf = torch.log_softmax(pred_logits, dim=-1)  # (B, T, K)
                
                # Compute loss: negative log-likelihood
                loss = -(batch_target_gamma * pred_logf).sum(dim=-1).mean()
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            loss_list.append(total_loss)
        
        return np.mean(loss_list)
    
    def fit(self):
        uniform_policy = np.full((self.num_states, self.num_actions), 1.0 / self.num_actions)
        agents = []
        for _ in range(self.num_latents):
            # agent = IAVI(
            agent = IAVI_GPU(
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
                batch_target_gamma.append(torch.exp(log_p_gamma).detach())
            
            # Pad sequences to same length for RNN input
            max_len = max(phi.shape[0] for phi in batch_phis)
            batch_phis_padded = torch.zeros(len(batch_phis), max_len, self.num_phis)
            batch_target_gamma_padded = torch.zeros(len(batch_target_gamma), max_len, self.num_latents)
            
            for i, (phi, gamma) in enumerate(zip(batch_phis, batch_target_gamma)):
                seq_len = phi.shape[0]
                batch_phis_padded[i, :seq_len] = phi
                batch_target_gamma_padded[i, :seq_len] = gamma
            
            batch_phis = batch_phis_padded  # (B, T, phi_dim)
            batch_target_gamma = batch_target_gamma_padded.detach().clone()  # (B, T, K)
            dataset = TensorDataset(batch_phis, batch_target_gamma)
            loader = DataLoader(dataset, batch_size=256, shuffle=True)

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

                # agent = IAVI(
                agent = IAVI_GPU(
                    num_states=self.num_states,
                    num_actions=self.num_actions,
                    P=self.P,
                    expert_policy=expert_pi.numpy(),
                    discount=self.discount
                )
                latent_delta = agent.train()
                agents[latent_idx] = agent
            q_time = time.time() - q_start_time
            total_q_time += q_time

            # * * * Update intention network * * *
            other_start_time = time.time()
            # total_loss = self.train_batched(batch_phis, batch_target_gamma, num_epochs=1)
            total_loss = self.train_minibatch(loader, num_epochs=1)
            other_time = time.time() - other_start_time
            total_other_time += other_time

            self.target_intention_net.load_state_dict(self.intention_net.state_dict())

            if logger_cnt % 1 == 0:
                iteration_time = time.time() - iteration_start_time
                print(f'Iteration {logger_cnt}, Loss: {total_loss:.4f}, Q-update: {total_q_time:.2f}s, NN: {total_other_time:.2f}s, Total: {iteration_time:.2f}s')
                total_q_time = 0
                total_other_time = 0

            if (abs(total_loss) < 5e-3) or (logger_cnt >= 60):
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