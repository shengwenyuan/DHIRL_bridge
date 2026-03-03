import numpy as np
import torch
import time

from scipy.special import logsumexp
from model.intention import IntentionNet, StatesRNN, IntentionTransformer
from torch.utils.data import DataLoader, TensorDataset


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


class IAVI_B:
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
        while e < 500:
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

        if e >= 500:
            raise RuntimeError("IAVI did not converge within the maximum number of iterations.")
        
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.intention_net = IntentionTransformer(phi_dim=self.num_phis, 
        #                                num_latents=self.num_latents, 
        #                                d_model=128, 
        #                                nhead=4,
        #                                num_layers=2,
        #                                dropout=0.2).to(self.device)
        # self.target_intention_net = IntentionTransformer(phi_dim=self.num_phis, 
        #                                num_latents=self.num_latents, 
        #                                d_model=128, 
        #                                nhead=4,
        #                                num_layers=2,
        #                                dropout=0.2).to(self.device)
        self.intention_net = StatesRNN(phi_dim=self.num_phis, 
                                       num_latents=self.num_latents, 
                                       hidden_dim=128, 
                                       rnn_hidden_dim=128, 
                                       num_layers=2,
                                       dropout=0.3).to(self.device)
        self.target_intention_net = StatesRNN(phi_dim=self.num_phis, 
                                       num_latents=self.num_latents, 
                                       hidden_dim=128, 
                                       rnn_hidden_dim=128, 
                                       num_layers=2,
                                       dropout=0.3).to(self.device)
        self.target_intention_net.load_state_dict(self.intention_net.state_dict())
        self.target_intention_net.eval()
        self.optimizer = torch.optim.Adam(self.intention_net.parameters(), lr=3e-3)

        self.state_emb = torch.nn.Embedding(self.num_states, 64)
        self.action_emb = torch.nn.Embedding(self.num_actions, 16)

    def intention_batch_mapping(self, e_loader, total_length):
        log_p_gammas = []
        log_fs = []
        log_joints = []
        with torch.no_grad():
            for batch_phis, batch_log_pi, bphis_mask in e_loader:
                f_logits = self.target_intention_net(batch_phis.to(self.device), 
                                                     mask=bphis_mask.to(self.device),
                                                     total_length=total_length).to('cpu')       # (B, T_max, K)
                log_f = torch.log_softmax(f_logits, dim=-1)
                log_joint = log_f + batch_log_pi
                log_p_gamma = log_joint - torch.logsumexp(log_joint, dim=-1, keepdim=True) # (B, T_max, K)

                if bphis_mask is not None:
                    log_p_gamma = log_p_gamma * bphis_mask.unsqueeze(-1)
                    log_f = log_f * bphis_mask.unsqueeze(-1)
                    log_joint = log_joint * bphis_mask.unsqueeze(-1)

                log_p_gammas.append(log_p_gamma)
                log_fs.append(log_f)
                log_joints.append(log_joint)

        return torch.concatenate(log_p_gammas, axis=0).to('cpu'), \
               torch.concatenate(log_fs, axis=0).to('cpu'), \
               torch.concatenate(log_joints, axis=0).to('cpu')

    def get_batch_log_pi(self, trajs, agents):
        agent_policies = []
        for agent in agents:
            pi = torch.softmax(agent.q.to('cpu'), dim=-1)  # Boltzmann policy
            agent_policies.append(pi)
        agent_policies = torch.stack(agent_policies, dim=0)  # (K, num_states, num_actions)

        log_pi_list = []
        phis_lens = []
        for traj in trajs:
            T = len(traj)
            phis_lens.append(T)

            states = torch.tensor([s for s, a, ns in traj], dtype=torch.long, device='cpu')
            actions = torch.tensor([a for s, a, ns in traj], dtype=torch.long, device='cpu')
            log_pi = torch.log(agent_policies[:, states, actions] + 1e-8)  # (K, T)
            log_pi_list.append(log_pi.T)  # (T, K)
        
        # Padding
        max_len = max(phis_lens)
        batch_log_pi = torch.zeros((len(trajs), max_len, self.num_latents), dtype=torch.float32, device='cpu')
        for i, log_pi in enumerate(log_pi_list):
            batch_log_pi[i, :phis_lens[i], :] = log_pi
        
        return batch_log_pi

    def encode_session_traj(self, traj):
        states = torch.tensor([s for s, a, ns in traj], dtype=torch.long, device='cpu')
        actions = torch.tensor([a for s, a, ns in traj], dtype=torch.long, device='cpu')

        s_emb = self.state_emb(states).detach()  # (T, E_S)
        a_emb = self.action_emb(actions).detach()  # (T, E_A)
        phis = torch.cat([s_emb, a_emb], dim=-1)  # (T, E_S + E_A)

        return phis
    
    def encode_batch_trajs(self, trajs):
        batch_phis = []
        phis_lens = []
        for traj in trajs:
            phis = self.encode_session_traj(traj)
            batch_phis.append(phis)
            phis_lens.append(phis.shape[0])
        
        max_len = max(phis_lens)
        batch_phis_padded = torch.zeros((len(batch_phis), max_len, self.num_phis), dtype=torch.float32, device='cpu')
        mask = torch.zeros((len(trajs), max_len), dtype=torch.bool, device='cpu')
        for i, (phis, seq_len) in enumerate(zip(batch_phis, phis_lens)):
            batch_phis_padded[i, :seq_len] = phis
            mask[i, :seq_len] = 1
        batch_phis = batch_phis_padded  # (B, T, phi_dim)

        return batch_phis, mask
    
    def train_minibatch(self, m_loader, total_length, num_epochs=1):
        """
        :param agents: List of IAVI agents
        :param num_epochs: Number of passes through the data
        """
        loss_list = []
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_phis, batch_target_gamma, batch_mask in m_loader:
                self.optimizer.zero_grad()

                pred_logits = self.intention_net(batch_phis.to(self.device), mask=batch_mask.to(self.device), total_length=total_length)  # (B, T, K)
                pred_logf = torch.log_softmax(pred_logits, dim=-1)  # (B, T, K)
                
                # Compute loss: negative log-likelihood
                loss = -(batch_target_gamma.to(self.device) * pred_logf * batch_mask.to(self.device).unsqueeze(-1)).sum(dim=-1).mean()

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            loss_list.append(total_loss)
        
        return np.mean(loss_list)
    
    def fit(self):
        # * * * Initialize agents with uniform policy * * *
        uniform_policy = np.full((self.num_states, self.num_actions), 1.0 / self.num_actions)
        agents = []
        for _ in range(self.num_latents):
            agent = IAVI_B(
                num_states=self.num_states,
                num_actions=self.num_actions,
                P=self.P,
                expert_policy=uniform_policy,
                discount=self.discount,
                device=self.device
            )
            agent.train()
            agents.append(agent)

        # * * * Encode train trajs * * *
        batch_phis, bphis_mask = self.encode_batch_trajs(self.train_trajs)
        max_len = batch_phis.shape[1]

        logger_cnt = 0
        logstep_exp_time = 0
        logstep_q_time = 0
        logstep_intention_time = 0
        iteration_start_time = time.time()

        while True:
            logger_cnt += 1


            # * * * E-step: compute posterior * * *
            expectation_start_time = time.time()
            batch_log_pi = self.get_batch_log_pi(self.train_trajs, agents) # (B, K, T)

            e_dataset = TensorDataset(batch_phis, batch_log_pi, bphis_mask)
            e_loader = DataLoader(e_dataset, batch_size=1024, shuffle=False)
            log_p_gammas, *_ = self.intention_batch_mapping(e_loader, max_len)
            batch_target_gamma = torch.exp(log_p_gammas).detach()
            expectation_time = time.time() - expectation_start_time
            logstep_exp_time += expectation_time


            # * * * Update Q-value & policies * * *
            m_dataset = TensorDataset(batch_phis, batch_target_gamma, bphis_mask)
            m_loader = DataLoader(m_dataset, batch_size=256, shuffle=True)

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

                agent = IAVI_B(
                    num_states=self.num_states,
                    num_actions=self.num_actions,
                    P=self.P,
                    expert_policy=expert_pi.numpy(),
                    discount=self.discount,
                    device=self.device
                )
                agent.train()
                agents[latent_idx] = agent
            q_time = time.time() - q_start_time
            logstep_q_time += q_time

            # * * * Update intention network * * *
            intention_start_time = time.time()
            # total_loss = self.train_batched(batch_phis, batch_target_gamma, num_epochs=1)
            total_loss = self.train_minibatch(m_loader, max_len, num_epochs=1)
            intention_time = time.time() - intention_start_time
            logstep_intention_time += intention_time

            self.target_intention_net.load_state_dict(self.intention_net.state_dict())

            if logger_cnt % 2 == 0:
                iteration_time = time.time() - iteration_start_time
                print(f'Iteration {logger_cnt}, Loss: {total_loss:.4f}, \n\
                       \tExpectation: {logstep_exp_time:.2f}s, Q-update: {logstep_q_time:.2f}s, Intention: {logstep_intention_time:.2f}s, \n\
                       \tTotal: {int(iteration_time // 60):d}m, {int(iteration_time % 60):d}s')
                logstep_exp_time = 0
                logstep_q_time = 0
                logstep_intention_time = 0

            if (abs(total_loss) < 1e-2) or (logger_cnt >= 50):
                final_iteration_time = time.time() - iteration_start_time
                print(f'Iteration {logger_cnt}, Converged with Loss: {total_loss:.4f}, Total time: {final_iteration_time:.2f}s')
                break

        f = {}
        ll = {}
        mask = {}
        for ds in ['train', 'test']:
            trajs = eval(f'self.{ds}_trajs')
            batch_phis_eval, mask_eval = self.encode_batch_trajs(trajs)
            max_len_eval = batch_phis_eval.shape[1]
            batch_log_pi_eval = self.get_batch_log_pi(trajs, agents)

            eval_dataset = TensorDataset(batch_phis_eval, batch_log_pi_eval, mask_eval)
            eval_loader = DataLoader(eval_dataset, batch_size=1024, shuffle=False)
            _, log_f, log_p_joint = self.intention_batch_mapping(eval_loader, max_len_eval)
            
            # Get per-trajectory results
            fs = []
            lls = []
            for i, seq_len in enumerate(mask_eval.sum(dim=1)):
                f_i = torch.exp(log_f[i, :, :]).cpu().numpy()
                log_joint_i = log_p_joint[i, :seq_len, :].cpu().numpy()
                fs.append(f_i)
                lls.append(np.mean(logsumexp(log_joint_i, axis=-1)))
            lls = np.mean(np.hstack(lls))
            ll[ds] = np.mean(lls)
            f[ds] = fs
            mask[ds] = mask_eval.cpu().numpy()

        return ll, f, mask, agents

    def predict(self, trajs, agents):
        batch_phis, mask = self.encode_batch_trajs(trajs)
        max_len = batch_phis.shape[1]
        batch_log_pi, _ = self.get_batch_log_pi(trajs, agents)

        eval_dataset = TensorDataset(batch_phis, batch_log_pi, mask)
        eval_loader = DataLoader(eval_dataset, batch_size=1024, shuffle=False)
        _, log_f, log_p_joint = self.intention_batch_mapping(eval_loader, max_len)

        fs = []
        lls = []
        for i, seq_len in enumerate(mask.sum(dim=1)):
            f_i = torch.exp(log_f[i, :seq_len, :]).cpu().numpy()
            log_joint_i = log_p_joint[i, :seq_len, :].cpu().numpy()
            fs.append(f_i)
            lls.append(np.mean(logsumexp(log_joint_i, axis=-1)))
        lls = np.mean(np.hstack(lls))
        
        return np.mean(lls), fs