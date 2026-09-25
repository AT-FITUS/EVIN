import copy
import math
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cornac.exception import ScoreException
from cornac.models import Recommender
from tqdm.auto import tqdm


class HRDRTextProcessor(nn.Module):
    """
    CNN-based text processor to capture contextual review representations.
    """

    def __init__(
        self, max_text_length, embedding_dim, n_filters, kernel_sizes, dropout_rate
    ):
        super(HRDRTextProcessor, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, n_filters, (k, embedding_dim)) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, training=True):
        # x shape: (batch_size, max_reviews, max_text_length, embedding_dim)
        batch_size, max_reviews, max_text_length, embedding_dim = x.size()
        x = x.view(batch_size * max_reviews, 1, max_text_length, embedding_dim)

        pooled_outputs = []
        for conv in self.convs:
            c = F.relu(conv(x)).squeeze(
                3
            )  # (batch_size * max_reviews, n_filters, height)
            p = F.max_pool1d(c, c.size(2)).squeeze(
                2
            )  # (batch_size * max_reviews, n_filters)
            pooled_outputs.append(p)

        out = torch.cat(
            pooled_outputs, dim=-1
        )  # (batch_size * max_reviews, n_filters * len(kernel_sizes))
        if training:
            out = self.dropout(out)
        return out.view(batch_size, max_reviews, -1)


class HRDRReviewAttention(nn.Module):
    """
    Review-level attention mechanism mapping review-text summaries
    using the rating-based vector matrix representation as the query token.
    """

    def __init__(self, text_dim, attention_dim):
        super(HRDRReviewAttention, self).__init__()
        self.a_dense = nn.Sequential(
            nn.Linear(text_dim, attention_dim, bias=True),
            nn.ReLU(),
            nn.Linear(attention_dim, 1, bias=True),
        )

    def forward(self, text_features, rating_query, counts, max_reviews):
        # text_features: (batch_size, max_reviews, text_dim)
        # rating_query: (batch_size, text_dim)

        combined = text_features * rating_query.unsqueeze(1)
        weights = self.a_dense(combined).squeeze(-1)

        # Mask padded elements
        mask = torch.arange(max_reviews, device=text_features.device).unsqueeze(
            0
        ) < counts.unsqueeze(1)

        # FIX: Dynamic lower bound assignment based on exact floating-point precision
        min_value = torch.finfo(weights.dtype).min
        weights = weights.masked_fill(~mask, min_value)

        alpha = F.softmax(weights, dim=-1)
        context = torch.bmm(alpha.unsqueeze(1), text_features).squeeze(1)
        return context, alpha


class HRDRModule(nn.Module):
    """
    Paper-accurate Neural Network Layer Framework matching Liu et al. (2020).
    Fuses MLP Rating Tower patterns with Query-Attentive Review CNN Embeddings.
    """

    def __init__(
        self,
        n_users,
        n_items,
        n_vocab,
        global_mean,
        n_factors=32,
        embedding_dim=100,
        id_embedding_dim=32,
        attention_dim=16,
        kernel_sizes=[3],
        n_filters=64,
        n_user_mlp_factors=128,
        n_item_mlp_factors=128,
        dropout_rate=0.5,
        max_text_length=50,
    ):
        super(HRDRModule, self).__init__()

        self.l_user_review_embedding = nn.Embedding(
            n_vocab, embedding_dim, padding_idx=0
        )
        self.l_item_review_embedding = nn.Embedding(
            n_vocab, embedding_dim, padding_idx=0
        )

        self.l_user_embedding = nn.Embedding(n_users, id_embedding_dim)
        self.l_item_embedding = nn.Embedding(n_items, id_embedding_dim)
        nn.init.uniform_(self.l_user_embedding.weight, -0.05, 0.05)
        nn.init.uniform_(self.l_item_embedding.weight, -0.05, 0.05)

        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        nn.init.constant_(self.user_bias.weight, 0.1)
        nn.init.constant_(self.item_bias.weight, 0.1)

        self.global_bias = nn.Parameter(
            torch.tensor([global_mean], dtype=torch.float32)
        )

        self.user_text_processor = HRDRTextProcessor(
            max_text_length, embedding_dim, n_filters, kernel_sizes, dropout_rate
        )
        self.item_text_processor = HRDRTextProcessor(
            max_text_length, embedding_dim, n_filters, kernel_sizes, dropout_rate
        )

        total_cnn_features = n_filters * len(kernel_sizes)

        # Rating MLP architectures
        self.l_user_mlp = nn.Sequential(
            nn.Linear(n_items, n_user_mlp_factors),
            nn.ReLU(),
            nn.Linear(n_user_mlp_factors, n_user_mlp_factors // 2),
            nn.ReLU(),
            nn.Linear(n_user_mlp_factors // 2, total_cnn_features),
            nn.ReLU(),
            nn.BatchNorm1d(total_cnn_features),
        )
        self.l_item_mlp = nn.Sequential(
            nn.Linear(n_users, n_item_mlp_factors),
            nn.ReLU(),
            nn.Linear(n_item_mlp_factors, n_item_mlp_factors // 2),
            nn.ReLU(),
            nn.Linear(n_item_mlp_factors // 2, total_cnn_features),
            nn.ReLU(),
            nn.BatchNorm1d(total_cnn_features),
        )

        self.user_review_attention = HRDRReviewAttention(
            text_dim=total_cnn_features, attention_dim=attention_dim
        )
        self.item_review_attention = HRDRReviewAttention(
            text_dim=total_cnn_features, attention_dim=attention_dim
        )

        self.ou_dropout = nn.Dropout(dropout_rate)
        self.oi_dropout = nn.Dropout(dropout_rate)

        self.ou = nn.Linear(total_cnn_features, n_factors)
        self.oi = nn.Linear(total_cnn_features, n_factors)

        final_repr_dim = total_cnn_features + n_factors + id_embedding_dim
        self.W1 = nn.Linear(final_repr_dim, 1, bias=False)

    def forward(self, inputs, training=True):
        (
            i_user_id,
            i_item_id,
            i_user_rating,
            i_user_review,
            i_user_num_reviews,
            i_item_rating,
            i_item_review,
            i_item_num_reviews,
        ) = inputs

        user_review_h = self.user_text_processor(
            self.l_user_review_embedding(i_user_review), training=training
        )
        item_review_h = self.item_text_processor(
            self.l_item_review_embedding(i_item_review), training=training
        )

        user_rating_h = self.l_user_mlp(i_user_rating)
        item_rating_h = self.l_item_mlp(i_item_rating)

        ou_context, _ = self.user_review_attention(
            user_review_h, user_rating_h, i_user_num_reviews, i_user_review.size(1)
        )
        oi_context, _ = self.item_review_attention(
            item_review_h, item_rating_h, i_item_num_reviews, i_item_review.size(1)
        )

        if training:
            ou_context = self.ou_dropout(ou_context)
            oi_context = self.oi_dropout(oi_context)

        ou = self.ou(ou_context)
        oi = self.oi(oi_context)

        pu = torch.cat([user_rating_h, ou, self.l_user_embedding(i_user_id)], dim=-1)
        qi = torch.cat([item_rating_h, oi, self.l_item_embedding(i_item_id)], dim=-1)

        h0 = pu * qi
        r = (
            self.W1(h0)
            + self.user_bias(i_user_id)
            + self.item_bias(i_item_id)
            + self.global_bias
        )
        return r.squeeze(1)


class HRDR(Recommender):
    """Unified Deep Dual Recommendation Mechanism with Hybrid Relation Development in PyTorch."""

    def __init__(
        self,
        name="HRDR",
        embedding_dim=100,
        id_embedding_dim=32,
        n_factors=32,
        n_user_mlp_factors=128,
        n_item_mlp_factors=128,
        attention_dim=16,
        kernel_sizes=[3],
        n_filters=64,
        dropout_rate=0.5,
        max_text_length=50,
        max_num_review=32,
        batch_size=512,
        max_iter=20,
        learning_rate=0.001,
        model_selection="best",
        trainable=True,
        verbose=True,
        seed=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        if "embedding_size" in kwargs:
            embedding_dim = kwargs.pop("embedding_size")

        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.seed = seed
        self.embedding_dim = embedding_dim
        self.id_embedding_dim = id_embedding_dim
        self.n_factors = n_factors
        self.n_user_mlp_factors = n_user_mlp_factors
        self.n_item_mlp_factors = n_item_mlp_factors
        self.attention_dim = attention_dim
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout_rate
        self.max_text_length = max_text_length
        self.max_num_review = max_num_review
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.model_selection = model_selection
        self.device = device

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

    def _precompute_hrdr_tensors(self, train_set):
        """Precomputes tracking matrices using PyTorch sparse tensors for ratings

        to prevent Out-Of-Memory (OOM) errors and runtime pipeline stalls.
        """
        if self.verbose:
            print("Pre-computing HRDR tracking matrices memory-efficiently...")

        review_modality = train_set.review_text

        u_indices = []
        i_indices = []
        rating_values = []

        for u in range(train_set.num_users):
            jds, ratings = train_set.user_data.get(u, ([], []))
            for jdx, r in zip(jds, ratings):
                u_indices.append(u)
                i_indices.append(jdx)
                rating_values.append(r)

        coords = torch.tensor([u_indices, i_indices], dtype=torch.long)
        values = torch.tensor(rating_values, dtype=torch.float32)
        with torch.sparse.check_sparse_tensor_invariants(enable=False):
            self.u_ratings = torch.sparse_coo_tensor(
                indices=coords,
                values=values,
                size=(train_set.num_users, train_set.num_items),
                device=self.device,
                check_invariants=False,
            ).coalesce()

            reversed_coords = torch.stack([coords[1], coords[0]])
            self.i_ratings = torch.sparse_coo_tensor(
                indices=reversed_coords,
                values=values,
                size=(train_set.num_items, train_set.num_users),
                device=self.device,
                check_invariants=False,
            ).coalesce()

        del u_indices, i_indices, rating_values, coords, reversed_coords, values

        self.u_reviews = torch.zeros(
            (train_set.num_users, self.max_num_review, self.max_text_length),
            dtype=torch.long,
            device=self.device,
        )
        self.u_counts = torch.zeros(
            (train_set.num_users,), dtype=torch.long, device=self.device
        )

        u_reviews_np = np.zeros(
            (train_set.num_users, self.max_num_review, self.max_text_length),
            dtype=np.int64,
        )
        u_counts_np = np.zeros((train_set.num_users,), dtype=np.int64)

        for u in range(train_set.num_users):
            u_rev_dict = review_modality.user_review.get(u, {})
            review_ids = list(u_rev_dict.values())[: self.max_num_review]
            if review_ids:
                seqs = review_modality.batch_seq(
                    review_ids, max_length=self.max_text_length
                )
                n = len(seqs)
                u_reviews_np[u, :n, :] = seqs
                u_counts_np[u] = n

        self.u_reviews.copy_(torch.from_numpy(u_reviews_np))
        self.u_counts.copy_(torch.from_numpy(u_counts_np))
        del u_reviews_np, u_counts_np

        self.i_reviews = torch.zeros(
            (train_set.num_items, self.max_num_review, self.max_text_length),
            dtype=torch.long,
            device=self.device,
        )
        self.i_counts = torch.zeros(
            (train_set.num_items,), dtype=torch.long, device=self.device
        )

        i_reviews_np = np.zeros(
            (train_set.num_items, self.max_num_review, self.max_text_length),
            dtype=np.int64,
        )
        i_counts_np = np.zeros((train_set.num_items,), dtype=np.int64)

        for i in range(train_set.num_items):
            i_rev_dict = review_modality.item_review.get(i, {})
            review_ids = list(i_rev_dict.values())[: self.max_num_review]
            if review_ids:
                seqs = review_modality.batch_seq(
                    review_ids, max_length=self.max_text_length
                )
                n = len(seqs)
                i_reviews_np[i, :n, :] = seqs
                i_counts_np[i] = n

        self.i_reviews.copy_(torch.from_numpy(i_reviews_np))
        self.i_counts.copy_(torch.from_numpy(i_counts_np))
        del i_reviews_np, i_counts_np

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)
        if not self.trainable:
            return self

        review_modality = train_set.review_text
        global_mean = float(train_set.global_mean)

        self._precompute_hrdr_tensors(train_set)

        self.model = HRDRModule(
            train_set.num_users,
            train_set.num_items,
            review_modality.vocab.size,
            global_mean,
            self.n_factors,
            self.embedding_dim,
            self.id_embedding_dim,
            self.attention_dim,
            self.kernel_sizes,
            self.n_filters,
            self.n_user_mlp_factors,
            self.n_item_mlp_factors,
            self.dropout_rate,
            self.max_text_length,
        ).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss()
        num_batches = math.ceil(train_set.num_ratings / self.batch_size)

        best_val_loss = float("inf")
        best_model_states = None

        is_cuda = "cuda" in str(self.device)
        scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)

        for epoch in range(self.max_iter):
            self.model.train()
            sum_train_loss = 0.0

            progress_bar = tqdm(
                train_set.uir_iter(batch_size=self.batch_size, shuffle=True),
                total=num_batches,
                desc=f"HRDR Epoch {epoch+1}/{self.max_iter}",
                disable=not self.verbose,
            )

            for batch_users, batch_items, batch_ratings in progress_bar:
                optimizer.zero_grad()

                u_ids = (
                    torch.from_numpy(batch_users)
                    .long()
                    .to(self.device, non_blocking=True)
                )
                i_ids = (
                    torch.from_numpy(batch_items)
                    .long()
                    .to(self.device, non_blocking=True)
                )
                ratings = (
                    torch.from_numpy(batch_ratings)
                    .float()
                    .to(self.device, non_blocking=True)
                )

                u_rev = self.u_reviews[u_ids]
                u_cnt = self.u_counts[u_ids]
                u_rat = torch.index_select(self.u_ratings, dim=0, index=u_ids).to_dense()

                i_rev = self.i_reviews[i_ids]
                i_cnt = self.i_counts[i_ids]
                i_rat = torch.index_select(self.i_ratings, dim=0, index=i_ids).to_dense()

                inputs = (u_ids, i_ids, u_rat, u_rev, u_cnt, i_rat, i_rev, i_cnt)

                with torch.amp.autocast("cuda", enabled=is_cuda):
                    predictions = self.model(inputs, training=True)
                    loss = criterion(predictions, ratings)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                sum_train_loss += loss.item() * len(batch_users)

            final_train_loss = sum_train_loss / train_set.num_ratings

            if val_set is not None:
                val_loss = self._evaluate_loss(val_set, criterion)
                if self.verbose:
                    print(
                        f"Epoch {epoch+1:02d} | Train MSE: {final_train_loss:.4f} | Val MSE: {val_loss:.4f}"
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_states = copy.deepcopy(self.model.state_dict())
            else:
                if self.verbose:
                    print(f"Epoch {epoch+1:02d} | Train MSE: {final_train_loss:.4f}")

        if self.model_selection == "best" and best_model_states is not None:
            self.model.load_state_dict(best_model_states)

        self._sync_weights_to_numpy_optimized()
        return self

    def _evaluate_loss(self, eval_set, criterion):
        self.model.eval()
        sum_eval_loss = 0.0
        is_cuda = "cuda" in str(self.device)
        with torch.no_grad():
            for batch_users, batch_items, batch_ratings in eval_set.uir_iter(
                batch_size=self.batch_size, shuffle=False
            ):
                u_ids = (
                    torch.from_numpy(batch_users)
                    .long()
                    .to(self.device, non_blocking=True)
                )
                i_ids = (
                    torch.from_numpy(batch_items)
                    .long()
                    .to(self.device, non_blocking=True)
                )
                ratings = (
                    torch.from_numpy(batch_ratings)
                    .float()
                    .to(self.device, non_blocking=True)
                )

                inputs = (
                    u_ids,
                    i_ids,
                    torch.index_select(self.u_ratings, dim=0, index=u_ids).to_dense(),
                    self.u_reviews[u_ids],
                    self.u_counts[u_ids],
                    torch.index_select(self.i_ratings, dim=0, index=i_ids).to_dense(),
                    self.i_reviews[i_ids],
                    self.i_counts[i_ids],
                )
                with torch.amp.autocast("cuda", enabled=is_cuda):
                    predictions = self.model(inputs, training=False)
                    loss = criterion(predictions, ratings)
                sum_eval_loss += loss.item() * len(batch_users)
        return sum_eval_loss / eval_set.num_ratings

    def _sync_weights_to_numpy_optimized(self):
        """Vectorized parameter serialization using cached device matrices."""
        self.model.eval()
        with torch.no_grad():
            total_dim = (
                (self.n_filters * len(self.kernel_sizes))
                + self.n_factors
                + self.id_embedding_dim
            )
            self.P = np.zeros((self.u_reviews.size(0), total_dim))
            self.Q = np.zeros((self.i_reviews.size(0), total_dim))

            for u_start in range(0, self.u_reviews.size(0), self.batch_size):
                u_end = min(u_start + self.batch_size, self.u_reviews.size(0))
                u_ids = torch.arange(
                    u_start, u_end, dtype=torch.long, device=self.device
                )

                user_review_h = self.model.user_text_processor(
                    self.model.l_user_review_embedding(self.u_reviews[u_ids]),
                    training=False,
                )
                user_rating_h = self.model.l_user_mlp(torch.index_select(self.u_ratings, dim=0, index=u_ids).to_dense())
                ou_context, _ = self.model.user_review_attention(
                    user_review_h,
                    user_rating_h,
                    self.u_counts[u_ids],
                    self.u_reviews.size(1),
                )
                ou = self.model.ou(ou_context)

                pu = torch.cat(
                    [user_rating_h, ou, self.model.l_user_embedding(u_ids)], dim=-1
                )
                self.P[u_start:u_end] = pu.cpu().numpy()

            for i_start in range(0, self.i_reviews.size(0), self.batch_size):
                i_end = min(i_start + self.batch_size, self.i_reviews.size(0))
                i_ids = torch.arange(
                    i_start, i_end, dtype=torch.long, device=self.device
                )

                item_review_h = self.model.item_text_processor(
                    self.model.l_item_review_embedding(self.i_reviews[i_ids]),
                    training=False,
                )
                item_rating_h = self.model.l_item_mlp(torch.index_select(self.i_ratings, dim=0, index=i_ids).to_dense())
                oi_context, _ = self.model.item_review_attention(
                    item_review_h,
                    item_rating_h,
                    self.i_counts[i_ids],
                    self.i_reviews.size(1),
                )
                oi = self.model.oi(oi_context)

                qi = torch.cat(
                    [item_rating_h, oi, self.model.l_item_embedding(i_ids)], dim=-1
                )
                self.Q[i_start:i_end] = qi.cpu().numpy()

            self.W1 = self.model.W1.weight.data.cpu().numpy().flatten()
            self.bu = self.model.user_bias.weight.data.cpu().numpy().flatten()
            self.bi = self.model.item_bias.weight.data.cpu().numpy().flatten()
            self.mu = float(self.model.global_bias.data.cpu().item())

    def score(self, user_idx, item_idx=None):
        if self.is_unknown_user(user_idx):
            raise ScoreException(f"Unknown user {user_idx}")

        if item_idx is None:
            pu = self.P[user_idx]  # (total_dim,)
            h0 = self.Q * pu  # (num_items, total_dim)
            scores = np.dot(h0, self.W1) + self.bu[user_idx] + self.bi + self.mu
            return np.nan_to_num(scores.ravel(), nan=0.0)
        else:
            if self.is_unknown_item(item_idx):
                raise ScoreException(f"Unknown item {item_idx}")
            h0 = self.P[user_idx] * self.Q[item_idx]
            score = (
                np.dot(h0, self.W1) + self.bu[user_idx] + self.bi[item_idx] + self.mu
            )
            return float(score)
