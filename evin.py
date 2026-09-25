import copy
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cornac.models import Recommender
from tqdm import tqdm
from scipy.sparse import csr_matrix


class EVINModule(nn.Module):
    def __init__(
        self,
        num_users,
        num_items,
        latent_dim,
        vocab_size,
        max_snippets=10,
        temperature=0.5,
        padding_idx=0,
        min_rating=1.0,
        max_rating=5.0,
        dropout=0.4,
        alpha=0.5,
    ):
        super(EVINModule, self).__init__()
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.max_snippets = max_snippets
        self.alpha = alpha

        self.user_emb = nn.Embedding(num_users, latent_dim)
        self.item_emb = nn.Embedding(num_items, latent_dim)
        self.word_embeddings = nn.Embedding(
            vocab_size, latent_dim, padding_idx=padding_idx
        )

        self.snippet_encoder_gru = nn.GRU(
            input_size=latent_dim, hidden_size=latent_dim, batch_first=True
        )

        self.text_projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Dropout(p=dropout)
        )

        self.selector_score_head = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )

        self.evidence_norm = nn.LayerNorm(latent_dim)

        self.rating_predictor = nn.Sequential(
            nn.Linear(latent_dim * 3, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1),
        )

        self.dropout = nn.Dropout(p=dropout)
        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.dim() > 1:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.normal_(self.rating_predictor[-1].weight, mean=0.0, std=0.01)

    def forward(
        self,
        user_indices,
        item_indices,
        item_snippet_tensor=None,
        pre_encoded_snippets=None,
        snippet_times=None,
        interaction_times=None,
        custom_z_mask=None,
    ):
        u_f = self.dropout(self.user_emb(user_indices))
        i_f = self.dropout(self.item_emb(item_indices))

        if pre_encoded_snippets is not None:
            encoded_snippets = pre_encoded_snippets
            M = encoded_snippets.size(1)
        else:
            B_s, M, L = item_snippet_tensor.shape
            flat_snippets = item_snippet_tensor.view(B_s * M, L)
            snippet_word_embs = self.dropout(self.word_embeddings(flat_snippets))
            _, h_n = self.snippet_encoder_gru(snippet_word_embs)
            encoded_snippets = h_n.squeeze(0).view(B_s, M, self.latent_dim)

        if snippet_times is not None:
            is_metadata = (snippet_times == 0).unsqueeze(-1).float()
            is_review = (snippet_times > 0).unsqueeze(-1).float()
            source_weights = (self.alpha * is_metadata) + (
                (1.0 - self.alpha) * is_review
            )
            encoded_snippets = encoded_snippets * source_weights

        projected_snippets = self.text_projection(encoded_snippets)

        u_f_expanded = u_f.unsqueeze(1).expand(-1, M, -1)
        i_f_expanded = i_f.unsqueeze(1).expand(-1, M, -1)

        combined_context = torch.cat(
            [u_f_expanded, i_f_expanded, projected_snippets], dim=-1
        )
        snippet_logits = self.selector_score_head(combined_context).squeeze(dim=-1)

        modified_logits = snippet_logits.clone()

        if self.training:
            eps = 1e-7
            gumbels = -torch.empty_like(snippet_logits).exponential_().add_(eps).log()
            modified_logits = modified_logits + gumbels

        if snippet_times is not None and interaction_times is not None:
            tx_expanded = interaction_times.unsqueeze(1).expand(-1, M)
            future_mask = snippet_times > tx_expanded
            modified_logits = modified_logits.masked_fill(future_mask, float("-inf"))

        scaled_logits = modified_logits / self.temperature

        is_all_masked = torch.all(
            modified_logits == float("-inf"), dim=-1, keepdim=True
        )

        max_logits = torch.max(scaled_logits, dim=-1, keepdim=True).values
        max_logits = torch.where(
            is_all_masked, torch.zeros_like(max_logits), max_logits
        )
        shifted_logits = scaled_logits - max_logits

        exp_logits = torch.exp(shifted_logits)
        exp_logits = torch.where(
            modified_logits == float("-inf"), torch.zeros_like(exp_logits), exp_logits
        )

        sum_exp = torch.sum(exp_logits, dim=-1, keepdim=True)

        z = exp_logits / torch.where(is_all_masked, torch.ones_like(sum_exp), sum_exp)
        z = torch.where(is_all_masked.expand_as(z), torch.zeros_like(z), z)

        if custom_z_mask is not None:
            z = z * custom_z_mask.float()

        aggregated_evidence = torch.bmm(z.unsqueeze(1), projected_snippets).squeeze(1)
        normalized_evidence = self.evidence_norm(aggregated_evidence)

        refined_user = u_f * normalized_evidence
        refined_item = i_f * normalized_evidence
        prediction_input = torch.cat(
            [refined_user, refined_item, normalized_evidence], dim=-1
        )
        predicted_ratings = self.rating_predictor(prediction_input).squeeze(-1)
        raw_outputs = predicted_ratings.clone()

        if not self.training:
            predicted_ratings = torch.clamp(
                predicted_ratings, min=self.min_rating, max=self.max_rating
            )

        return predicted_ratings, snippet_logits, raw_outputs


class EVIN(Recommender):
    def __init__(
        self,
        name="EVIN",
        latent_dim=32,
        learning_rate=0.0002,
        weight_decay=1e-5,
        dropout=0.4,
        epochs=20,
        batch_size=256,
        lambda_sup=0.01,
        rating_weight=1.0,
        max_snippets=10,
        alpha=0.5,
        patience=5,
        min_delta=1e-4,
        init_temperature=1.0,
        min_temperature=0.2,
        tau_select=0.15,
        trainable=True,
        verbose=False,
        device="cpu",
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_sup = lambda_sup
        self.rating_weight = rating_weight
        self.init_temperature = init_temperature
        self.min_temperature = min_temperature
        self.max_snippets = max_snippets
        self.alpha = alpha
        self.patience = patience
        self.min_delta = min_delta
        self.tau_select = tau_select
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.best_weights = None

        self.cached_gpu_snippets = None
        self.cached_gpu_timestamps = None

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)

        if not hasattr(train_set, "joint_text") or train_set.joint_text is None:
            raise ValueError("EVIN requires an operational text modality module.")

        modality = train_set.joint_text

        if self.seed is not None:
            os.environ["PYTHONHASHSEED"] = str(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True

        self.network = EVINModule(
            num_users=train_set.num_users,
            num_items=train_set.num_items,
            latent_dim=self.latent_dim,
            vocab_size=modality.vocab.size,
            max_snippets=self.max_snippets,
            dropout=self.dropout,
            padding_idx=0,
            alpha=self.alpha,
        ).to(self.device)

        optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-6
        )

        self.precompute_evaluation_artifacts()

        num_ratings = train_set.num_ratings
        best_val_mse = float("inf")
        patience_counter = 0
        self.best_weights = copy.deepcopy(self.network.state_dict())

        if self.verbose:
            print(f"🚀 Training EVIN via Weak Supervision on: {self.device}")

        for epoch in range(1, self.epochs + 1):
            self.network.train()

            decay_progress = (epoch - 1) / max(1, self.epochs - 1)
            current_temp = self.init_temperature - decay_progress * (
                self.init_temperature - self.min_temperature
            )
            self.network.temperature = max(self.min_temperature, current_temp)

            accumulated_loss = 0.0
            idx_shuffle = np.random.permutation(num_ratings)
            total_batches = int(np.ceil(num_ratings / self.batch_size))

            pbar = tqdm(
                range(total_batches),
                desc=f"Epoch {epoch:02d}/{self.epochs:02d}",
                ncols=100,
            )

            warmup_epochs = 8
            current_lambda = (
                self.lambda_sup * (epoch / warmup_epochs)
                if epoch <= warmup_epochs
                else self.lambda_sup
            )

            for b in pbar:
                start_idx = b * self.batch_size
                end_idx = min(start_idx + self.batch_size, num_ratings)
                batch_slices = idx_shuffle[start_idx:end_idx]

                batch_uid = train_set.uir_tuple[0][batch_slices]
                batch_iid = train_set.uir_tuple[1][batch_slices]
                batch_ratings = train_set.uir_tuple[2][batch_slices]

                u_t = torch.tensor(batch_uid, dtype=torch.long, device=self.device)
                i_t = torch.tensor(batch_iid, dtype=torch.long, device=self.device)
                y_t = torch.tensor(
                    batch_ratings, dtype=torch.float32, device=self.device
                )

                c_snippets = self.cached_gpu_snippets[i_t]
                c_times = self.cached_gpu_timestamps[i_t]

                interaction_times = torch.tensor(
                    train_set.timestamps[batch_slices],
                    dtype=torch.long,
                    device=self.device,
                )
                optimizer.zero_grad()

                preds, logits, _ = self.network(
                    u_t,
                    i_t,
                    item_snippet_tensor=c_snippets,
                    snippet_times=c_times,
                    interaction_times=interaction_times,
                )

                rating_loss = F.mse_loss(preds, y_t) * self.rating_weight

                log_p_selection = F.log_softmax(logits, dim=-1)
                M = logits.size(-1)
                uniform_target = torch.full_like(log_p_selection, fill_value=(1.0 / M))
                kl_divergence = F.kl_div(
                    log_p_selection, uniform_target, reduction="batchmean"
                )

                total_loss = rating_loss + (current_lambda * kl_divergence)

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=5.0)
                optimizer.step()

                current_loss = total_loss.item()
                accumulated_loss += current_loss

                pbar.set_postfix(loss=f"{current_loss:.4f}", refresh=False)

            pbar.close()

            scheduler.step()

            avg_loss = accumulated_loss / total_batches
            val_status_str = ""

            if val_set is not None:
                val_mse, val_mae = self._evaluate_performance(val_set)
                val_status_str = f" | Val MSE: {val_mse:.4f} | Val MAE: {val_mae:.4f}"

                if val_mse < (best_val_mse - self.min_delta):
                    best_val_mse = val_mse
                    self.best_weights = copy.deepcopy(self.network.state_dict())
                    patience_counter = 0
                    val_status_str += " ⭐ [New Best Model]"
                else:
                    patience_counter += 1
                    val_status_str += (
                        f" ⚠️ [Patience: {patience_counter}/{self.patience}]"
                    )

            if self.verbose:
                print(
                    f"Epoch {epoch:02d}/{self.epochs:02d} | Loss: {avg_loss:.4f}{val_status_str}"
                )

            if val_set is not None and patience_counter >= self.patience:
                if self.verbose:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch}.")
                break

        if self.best_weights is not None:
            self.network.load_state_dict(self.best_weights)

        return self

    @torch.no_grad()
    def _evaluate_performance(self, dataset):
        self.network.eval()
        user_ids, item_ids, targets = dataset.uir_tuple

        total_samples = len(user_ids)
        accumulated_mse, accumulated_mae = 0.0, 0.0
        eval_batch_size = self.batch_size * 4

        for start_idx in range(0, total_samples, eval_batch_size):
            end_idx = min(start_idx + eval_batch_size, total_samples)

            u_batch = torch.tensor(
                user_ids[start_idx:end_idx], dtype=torch.long, device=self.device
            )
            i_batch = torch.tensor(
                item_ids[start_idx:end_idx], dtype=torch.long, device=self.device
            )
            y_batch = torch.tensor(
                targets[start_idx:end_idx], dtype=torch.float32, device=self.device
            )

            c_snippets = self.cached_gpu_snippets[i_batch]
            c_times = self.cached_gpu_timestamps[i_batch]
            interaction_times = torch.tensor(
                dataset.timestamps[start_idx:end_idx],
                dtype=torch.long,
                device=self.device,
            )

            preds, _, _ = self.network(
                u_batch,
                i_batch,
                item_snippet_tensor=c_snippets,
                snippet_times=c_times,
                interaction_times=interaction_times,
            )

            accumulated_mse += F.mse_loss(preds, y_batch, reduction="sum").item()
            accumulated_mae += F.l1_loss(preds, y_batch, reduction="sum").item()

        return (
            (accumulated_mse / total_samples, accumulated_mae / total_samples)
            if total_samples > 0
            else (float("inf"), float("inf"))
        )

    @torch.no_grad()
    def evaluate_threshold_fidelity(
        self, dataset, metrics, tau_select=None, user_based=False
    ):
        if len(metrics) == 0:
            return {}, {}

        batch_size = self.batch_size * 4
        tau = tau_select if tau_select is not None else self.tau_select
        self.network.eval()

        user_ids, item_ids, r_values = dataset.uir_tuple
        total_samples = len(user_ids)
        gt_mat = dataset.csr_matrix

        baselines = ["EVIN", "Random", "EVIN-Inverse"]

        all_preds_base = []
        all_preds_suff = {b: [] for b in baselines}
        all_preds_necc = {b: [] for b in baselines}

        out_lengths = []
        out_allocation_weights = []
        out_ext_age_vals = []
        out_ext_age_u_idx = []
        out_disc_age_vals = []
        out_disc_age_u_idx = []

        for start_idx in range(0, total_samples, batch_size):
            end_idx = min(start_idx + batch_size, total_samples)
            batch_u_ids = user_ids[start_idx:end_idx]
            batch_i_ids = item_ids[start_idx:end_idx]

            u_b = torch.tensor(batch_u_ids, dtype=torch.long, device=self.device)
            i_b = torch.tensor(batch_i_ids, dtype=torch.long, device=self.device)

            c_snippets = self.cached_gpu_snippets[i_b]
            c_times = self.cached_gpu_timestamps[i_b]

            i_times = torch.full(
                (end_idx - start_idx,),
                fill_value=999999999999,
                dtype=torch.long,
                device=self.device,
            )

            preds_base, logits, _ = self.network(
                u_b,
                i_b,
                item_snippet_tensor=c_snippets,
                snippet_times=c_times,
                interaction_times=i_times,
            )
            all_preds_base.append(preds_base.cpu().numpy())

            B_size, M = logits.shape
            tx_expanded = i_times.unsqueeze(1).expand(-1, M)
            future_mask = c_times > tx_expanded
            valid_counts = (~future_mask).sum(dim=-1, keepdim=True)

            logits_masked = logits.clone().masked_fill(future_mask, float("-inf"))
            all_masked_rows = valid_counts.squeeze(-1) == 0
            if all_masked_rows.any():
                logits_masked[all_masked_rows, 0] = 0.0

            z_distribution = torch.softmax(
                logits_masked / self.network.temperature, dim=-1
            )
            if all_masked_rows.any():
                fallback_distribution = torch.zeros_like(
                    z_distribution[all_masked_rows]
                )
                fallback_distribution[..., 0] = 1.0
                z_distribution[all_masked_rows] = fallback_distribution

            out_allocation_weights.append(z_distribution.detach().cpu())

            threshold_mask = z_distribution >= tau
            any_valid = threshold_mask.any(dim=-1, keepdim=True)

            fallback_idx = torch.argmax(logits_masked, dim=-1, keepdim=True)
            fallback_mask = torch.zeros_like(threshold_mask, dtype=torch.bool).scatter_(
                dim=-1, index=fallback_idx, value=True
            )

            evin_suff_mask = torch.where(any_valid, threshold_mask, fallback_mask) & (
                ~future_mask
            )
            evin_lengths = (
                edin_suff_mask.sum(dim=-1)
                if "edin_suff_mask" in locals()
                else evin_suff_mask.sum(dim=-1)
            )
            out_lengths.append(evin_lengths.cpu())

            k_clamped = torch.clamp(evin_lengths.unsqueeze(-1), max=valid_counts)

            # --- Random Baseline Selection ---
            rand_weights = torch.rand(B_size, M, device=self.device).masked_fill(
                future_mask, -1e9
            )
            _, rand_top_indices = torch.topk(rand_weights, k=M, dim=-1)
            seq_matrix = (
                torch.arange(M, device=self.device).view(1, M).expand(B_size, M)
            )
            rand_selection_rank_mask = seq_matrix < k_clamped
            random_suff_mask = torch.zeros_like(evin_suff_mask).scatter_(
                dim=-1, index=rand_top_indices, src=rand_selection_rank_mask
            )
            random_suff_mask = torch.where(
                valid_counts.expand_as(evin_suff_mask) > 0,
                random_suff_mask,
                evin_suff_mask,
            )

            # --- EVIN-Inverse Baseline Selection ---
            sorted_attn_indices = torch.argsort(logits_masked, dim=-1, descending=True)
            anti_rank_indices = torch.clamp(valid_counts - k_clamped, min=0)
            anti_selection_rank_mask = (seq_matrix >= anti_rank_indices) & (
                seq_matrix < valid_counts
            )
            anti_suff_mask = torch.zeros_like(evin_suff_mask).scatter_(
                dim=-1, index=sorted_attn_indices, src=anti_selection_rank_mask
            )
            anti_suff_mask = torch.where(
                valid_counts.expand_as(evin_suff_mask) > 0,
                anti_suff_mask,
                evin_suff_mask,
            )

            masks_suff = {
                "EVIN": evin_suff_mask,
                "Random": random_suff_mask,
                "EVIN-Inverse": anti_suff_mask,
            }

            for b_name in baselines:
                m_suff = masks_suff[b_name]
                m_necc = (~m_suff) & (~future_mask)

                p_suff, _, _ = self.network(
                    u_b,
                    i_b,
                    item_snippet_tensor=c_snippets,
                    snippet_times=c_times,
                    interaction_times=i_times,
                    custom_z_mask=m_suff,
                )
                p_necc, _, _ = self.network(
                    u_b,
                    i_b,
                    item_snippet_tensor=c_snippets,
                    snippet_times=c_times,
                    interaction_times=i_times,
                    custom_z_mask=m_necc,
                )

                all_preds_suff[b_name].append(p_suff.cpu().numpy())
                all_preds_necc[b_name].append(p_necc.cpu().numpy())

            # Age Tracking Calculations
            is_review_mask = (c_times > 0) & (~future_mask)
            age_days = (tx_expanded - c_times).float() / (1000.0 * 60.0 * 60.0 * 24.0)
            ext_reviews_mask = evin_suff_mask & is_review_mask
            disc_reviews_mask = (~evin_suff_mask) & is_review_mask

            ext_batch_indices, ext_snippet_indices = torch.where(ext_reviews_mask)
            if ext_batch_indices.numel() > 0:
                out_ext_age_vals.append(
                    age_days[ext_batch_indices, ext_snippet_indices].cpu()
                )
                out_ext_age_u_idx.append(u_b[ext_batch_indices].cpu())

            disc_batch_indices, disc_snippet_indices = torch.where(disc_reviews_mask)
            if disc_batch_indices.numel() > 0:
                out_disc_age_vals.append(
                    age_days[disc_batch_indices, disc_snippet_indices].cpu()
                )
                out_disc_age_u_idx.append(u_b[disc_batch_indices].cpu())

        # 2. Sparse Transformation Matrices
        r_preds_base = np.concatenate(all_preds_base)
        pd_mat_base = csr_matrix(
            (r_preds_base, (user_ids, item_ids)), shape=gt_mat.shape
        )

        pd_mat_suff = {}
        pd_mat_necc = {}
        for b in baselines:
            pd_mat_suff[b] = csr_matrix(
                (np.concatenate(all_preds_suff[b]), (user_ids, item_ids)),
                shape=gt_mat.shape,
            )
            pd_mat_necc[b] = csr_matrix(
                (np.concatenate(all_preds_necc[b]), (user_ids, item_ids)),
                shape=gt_mat.shape,
            )

        # 3. Native Cornac Reduction Suite
        test_user_indices = set(user_ids)
        avg_results = {
            "base": [],
            "suff": {b: [] for b in baselines},
            "necc": {b: [] for b in baselines},
        }
        user_results = {
            "base": [],
            "suff": {b: [] for b in baselines},
            "necc": {b: [] for b in baselines},
        }

        for mt in metrics:
            if user_based:
                u_res_base = {}
                u_res_suff = {b: {} for b in baselines}
                u_res_necc = {b: {} for b in baselines}

                for u_idx in test_user_indices:
                    # Create a boolean selection mask tracking the current user's item locations
                    user_mask = user_ids == u_idx

                    # Extract the true test values explicitly grouped for this user
                    gt_data = r_values[user_mask]
                    if len(gt_data) == 0:
                        continue

                    # Direct indexing via the user mask completely bypasses the CSR sorting artifact
                    pd_data_base = r_preds_base[user_mask]
                    u_res_base[u_idx] = mt.compute(
                        gt_ratings=gt_data, pd_ratings=pd_data_base
                    ).item()

                    for b in baselines:
                        pd_data_suff = np.concatenate(all_preds_suff[b])[user_mask]
                        pd_data_necc = np.concatenate(all_preds_necc[b])[user_mask]

                        u_res_suff[b][u_idx] = mt.compute(
                            gt_ratings=gt_data, pd_ratings=pd_data_suff
                        ).item()
                        u_res_necc[b][u_idx] = mt.compute(
                            gt_ratings=gt_data, pd_ratings=pd_data_necc
                        ).item()

                user_results["base"].append(u_res_base)
                avg_results["base"].append(
                    sum(u_res_base.values()) / len(u_res_base) if u_res_base else 0.0
                )

                for b in baselines:
                    user_results["suff"][b].append(u_res_suff[b])
                    user_results["necc"][b].append(u_res_necc[b])
                    avg_results["suff"][b].append(
                        sum(u_res_suff[b].values()) / len(u_res_suff[b])
                        if u_res_suff[b]
                        else 0.0
                    )
                    avg_results["necc"][b].append(
                        sum(u_res_necc[b].values()) / len(u_res_necc[b])
                        if u_res_necc[b]
                        else 0.0
                    )
            else:
                # Global Reduction
                user_results["base"].append({})
                avg_results["base"].append(
                    mt.compute(gt_ratings=r_values, pd_ratings=r_preds_base).item()
                )

                for b in baselines:
                    user_results["suff"][b].append({})
                    user_results["necc"][b].append({})
                    avg_results["suff"][b].append(
                        mt.compute(
                            gt_ratings=r_values,
                            pd_ratings=np.concatenate(all_preds_suff[b]),
                        ).item()
                    )
                    avg_results["necc"][b].append(
                        mt.compute(
                            gt_ratings=r_values,
                            pd_ratings=np.concatenate(all_preds_necc[b]),
                        ).item()
                    )

        return {
            "avg_results": avg_results,
            "user_results": user_results,
            "diagnostics": {
                "lengths": torch.cat(out_lengths) if out_lengths else torch.tensor([]),
                "evidence_weights": (
                    torch.cat(out_allocation_weights)
                    if out_allocation_weights
                    else torch.tensor([])
                ),
                "ext_age_vals": (
                    torch.cat(out_ext_age_vals)
                    if out_ext_age_vals
                    else torch.tensor([])
                ),
                "ext_age_u_idx": (
                    torch.cat(out_ext_age_u_idx)
                    if out_ext_age_u_idx
                    else torch.tensor([])
                ),
                "disc_age_vals": (
                    torch.cat(out_disc_age_vals)
                    if out_disc_age_vals
                    else torch.tensor([])
                ),
                "disc_age_u_idx": (
                    torch.cat(out_disc_age_u_idx)
                    if out_disc_age_u_idx
                    else torch.tensor([])
                ),
            },
        }

    @torch.no_grad()
    def precompute_evaluation_artifacts(self):
        self.network.eval()
        modality = self.train_set.joint_text
        num_items = self.train_set.num_items
        _, M, L = modality.item_snippets.shape

        aligned_snippets = np.zeros((num_items, M, L), dtype=np.int64)
        aligned_timestamps = np.zeros((num_items, M), dtype=np.int64)

        for internal_idx in self.train_set.iid_map.values():
            if internal_idx < len(modality.item_snippets):
                aligned_snippets[internal_idx] = modality.item_snippets[internal_idx]
                aligned_timestamps[internal_idx] = modality.snippet_timestamps[
                    internal_idx
                ]

        if self.cached_gpu_snippets is not None:
            del self.cached_gpu_snippets
        if self.cached_gpu_timestamps is not None:
            del self.cached_gpu_timestamps

        self.cached_gpu_snippets = torch.from_numpy(aligned_snippets).to(self.device)
        self.cached_gpu_timestamps = torch.from_numpy(aligned_timestamps).to(
            self.device
        )

    def score(self, user_idx, item_idx=None):
        self.network.eval()
        with torch.no_grad():
            if item_idx is not None:
                item_idx_arr = np.atleast_1d(item_idx)
                i_t = torch.tensor(item_idx_arr, dtype=torch.long, device=self.device)
                u_t = torch.tensor(
                    [user_idx], dtype=torch.long, device=self.device
                ).expand(len(item_idx_arr))

                c_snippets = self.cached_gpu_snippets[i_t]
                c_times = self.cached_gpu_timestamps[i_t]
                interaction_times = torch.full_like(
                    u_t, fill_value=999999999999, dtype=torch.long, device=self.device
                )

                preds, _, _ = self.network(
                    u_t,
                    i_t,
                    item_snippet_tensor=c_snippets,
                    snippet_times=c_times,
                    interaction_times=interaction_times,
                )
                return (
                    preds.cpu().item() if np.isscalar(item_idx) else preds.cpu().numpy()
                )

            i_t = torch.arange(
                self.train_set.num_items, dtype=torch.long, device=self.device
            )
            u_t = torch.tensor([user_idx], dtype=torch.long, device=self.device).expand(
                self.train_set.num_items
            )

            c_snippets = self.cached_gpu_snippets[i_t]
            c_times = self.cached_gpu_timestamps[i_t]
            interaction_times = torch.full_like(
                u_t, fill_value=999999999999, dtype=torch.long, device=self.device
            )

            preds, _, _ = self.network(
                u_t,
                i_t,
                item_snippet_tensor=c_snippets,
                snippet_times=c_times,
                interaction_times=interaction_times,
            )
            return preds.cpu().numpy()

    def produce_case_study(
        self, user_id_str, item_id_str, interaction_time=None, pprint=print
    ):
        self.network.eval()
        if (
            user_id_str not in self.train_set.uid_map
            or item_id_str not in self.train_set.iid_map
        ):
            pprint(
                f"⚠️ Map Error: Key combination ({user_id_str}, {item_id_str}) not in training mappings."
            )
            return

        u_idx = self.train_set.uid_map[user_id_str]
        i_idx = self.train_set.iid_map[item_id_str]

        u_t = torch.tensor([u_idx], dtype=torch.long, device=self.device)
        i_t = torch.tensor([i_idx], dtype=torch.long, device=self.device)
        c_snippets = self.cached_gpu_snippets[i_t]
        c_times = self.cached_gpu_timestamps[i_t]

        i_time_val = interaction_time if interaction_time is not None else 999999999999
        i_times = torch.tensor([i_time_val], dtype=torch.long, device=self.device)

        with torch.no_grad():
            preds, logits, _ = self.network(
                u_t,
                i_t,
                item_snippet_tensor=c_snippets,
                snippet_times=c_times,
                interaction_times=i_times,
            )
            M = logits.size(-1)
            tx_expanded = i_times.unsqueeze(1).expand(-1, M)
            future_mask = c_times > tx_expanded
            logits_masked = logits.clone().masked_fill(future_mask, float("-inf"))
            probs = (
                torch.softmax(logits_masked / self.network.temperature, dim=-1)
                .squeeze(0)
                .cpu()
                .numpy()
            )

        pprint(f"🔮 Predicted Evidential Rating: {preds.item():.4f}")
        pprint("⚡ Extractive Segment Attention Probability Weights Hierarchy:")

        modality = self.train_set.joint_text

        for s_idx in range(min(len(probs), self.max_snippets)):
            p_val = probs[s_idx]
            t_stamp = c_times[0, s_idx].item()

            token_ids = modality.item_snippets[i_idx][s_idx]
            valid_indices = [t.item() for t in token_ids if t.item() != 0]
            text_snippet = modality.vocab.to_text(valid_indices)

            marker = "🔴 [FUTURE HIDDEN]" if t_stamp > i_time_val else "🟢 [VISIBLE]"
            pprint(
                f'  └─ Segment [{s_idx:02d}] Prob: {p_val:.4f} | Time: {t_stamp:<12} {marker} -> "{text_snippet}"'
            )
