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
from cornac.models.recommender import ANNMixin, MEASURE_DOT
from tqdm.auto import tqdm


class TextProcessor(nn.Module):
    """Processes textual content matching Keras-style parallel 2D Convolutions."""

    def __init__(
        self,
        max_text_length,
        embedding_dim,
        filters=64,
        kernel_sizes=[3],
        dropout_rate=0.5,
    ):
        super(TextProcessor, self).__init__()
        self.max_text_length = max_text_length
        self.filters = filters
        self.kernel_sizes = kernel_sizes

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=filters,
                    kernel_size=(k, embedding_dim),
                    padding=(k // 2, 0),
                )
                for k in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, training=False):
        batch_size, num_reviews, max_len, embed_dim = x.shape
        x = x.view(batch_size * num_reviews, 1, max_len, embed_dim)

        pooled_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            pooled = F.max_pool2d(conv_out, kernel_size=(conv_out.size(2), 1))
            pooled_outputs.append(pooled)

        text_h = torch.cat(pooled_outputs, dim=1)
        text_h = text_h.view(batch_size, num_reviews, -1)

        if training:
            text_h = self.dropout(text_h)
        return text_h


class MaskedAttentionAggregation(nn.Module):
    """Applies sequence-masked attention over review features."""

    def __init__(self, feature_dim, id_embedding_dim, attention_dim):
        super(MaskedAttentionAggregation, self).__init__()
        self.attn_network = nn.Sequential(
            nn.Linear(feature_dim + id_embedding_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, review_features, id_embeddings, num_reviews, max_reviews):
        combined = torch.cat([review_features, id_embeddings], dim=-1)
        attn_scores = self.attn_network(combined)

        mask = torch.arange(max_reviews, device=num_reviews.device).view(
            1, -1
        ) < num_reviews.view(-1, 1)
        mask = mask.unsqueeze(-1)

        fill_value = torch.finfo(attn_scores.dtype).min
        attn_scores = attn_scores.masked_fill(~mask, fill_value)
        attn_weights = F.softmax(attn_scores, dim=1)

        aggregated = torch.sum(attn_weights * review_features, dim=1)
        return aggregated, attn_weights


class NARREModule(nn.Module):
    """
    Isolated PyTorch parameter graph execution layer.
    Fully validated against the NARRE (WWW 2018) specification.
    """

    def __init__(
        self,
        n_users,
        n_items,
        n_vocab,
        global_mean,
        n_factors=32,  # Latent space dimension (d)
        embedding_dim=100,  # Word embedding dimension (c)
        id_embedding_dim=32,  # ID embedding dimension (d_id)
        attention_dim=16,  # Attention hidden layer dimension
        kernel_sizes=[3],
        n_filters=64,
        dropout_rate=0.5,
        max_text_length=50,
    ):
        super(NARREModule, self).__init__()

        # Text Feature Extraction Layers (Section 3.2)
        self.l_user_review_embedding = nn.Embedding(
            n_vocab, embedding_dim, padding_idx=0
        )
        self.l_item_review_embedding = nn.Embedding(
            n_vocab, embedding_dim, padding_idx=0
        )

        self.user_text_processor = TextProcessor(
            max_text_length, embedding_dim, n_filters, kernel_sizes, dropout_rate
        )
        self.item_text_processor = TextProcessor(
            max_text_length, embedding_dim, n_filters, kernel_sizes, dropout_rate
        )

        total_cnn_features = n_filters * len(kernel_sizes)

        # Attention Aggregation Layers (Section 3.3)
        # Paper targets the partner ID dimension to evaluate review usefulness
        self.l_user_iid_embedding = nn.Embedding(n_items, id_embedding_dim)
        self.l_item_uid_embedding = nn.Embedding(n_users, id_embedding_dim)

        self.user_attention_block = MaskedAttentionAggregation(
            total_cnn_features, id_embedding_dim, attention_dim
        )
        self.item_attention_block = MaskedAttentionAggregation(
            total_cnn_features, id_embedding_dim, attention_dim
        )

        # Dimension Projection Layers: Projects review features to the Latent Factor Dimension (d)
        self.user_Oi_dropout = nn.Dropout(dropout_rate)
        self.Xu = nn.Linear(total_cnn_features, n_factors)

        self.item_Oi_dropout = nn.Dropout(dropout_rate)
        self.Yi = nn.Linear(total_cnn_features, n_factors)

        # Pure Collaborative Latent Embeddings (Section 3.1) - Enforced to match n_factors (d)
        self.l_user_embedding = nn.Embedding(n_users, n_factors)
        self.l_item_embedding = nn.Embedding(n_items, n_factors)

        # Biases (Section 3.4, Eq. 12)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(
            torch.tensor([global_mean], dtype=torch.float32)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def compute_user_text_features(self, user_reviews):
        embeds = self.l_user_review_embedding(user_reviews)
        return self.user_text_processor(embeds, training=False)

    def compute_item_text_features(self, item_reviews):
        embeds = self.l_item_review_embedding(item_reviews)
        return self.item_text_processor(embeds, training=False)

    def forward(self, inputs, training=True):
        (
            i_user_id,
            i_item_id,
            user_text_h,
            i_user_iid_review,
            i_user_num_reviews,
            item_text_h,
            i_item_uid_review,
            i_item_num_reviews,
        ) = inputs

        max_u_reviews = i_user_iid_review.size(1)
        max_i_reviews = i_item_uid_review.size(1)

        # 1. User Stream Attention Aggregation
        u_partner_embeds = self.l_user_iid_embedding(i_user_iid_review)
        user_Oi, _ = self.user_attention_block(
            user_text_h, u_partner_embeds, i_user_num_reviews, max_u_reviews
        )
        user_Oi = self.user_Oi_dropout(user_Oi) if training else user_Oi
        xu_features = self.Xu(user_Oi)

        # 2. Item Stream Attention Aggregation
        i_partner_embeds = self.l_item_uid_embedding(i_item_uid_review)
        item_Oi, _ = self.item_attention_block(
            item_text_h, i_partner_embeds, i_item_num_reviews, max_i_reviews
        )
        item_Oi = self.item_Oi_dropout(item_Oi) if training else item_Oi
        yi_features = self.Yi(item_Oi)

        # 3. Combined Latent Representations (Eq. 11)
        user_combined = self.l_user_embedding(i_user_id) + xu_features
        item_combined = self.l_item_embedding(i_item_id) + yi_features

        # 4. Strict Matrix Factorization Interaction Space (Eq. 12)
        # Calculates dot product strictly without a linear projection layer deformation
        interaction_score = torch.sum(
            user_combined * item_combined, dim=1, keepdim=True
        )

        b_u = self.user_bias(i_user_id)
        b_i = self.item_bias(i_item_id)

        return (interaction_score + b_u + b_i + self.global_bias).squeeze(1)


class NARRE(Recommender, ANNMixin):
    """Unified Neural Attentional Rating Regression framework."""

    def __init__(
        self,
        name="NARRE",
        embedding_dim=100,
        id_embedding_dim=32,
        n_factors=32,
        attention_dim=16,
        kernel_sizes=[3],
        n_filters=64,
        dropout_rate=0.5,
        max_text_length=50,
        max_num_review=32,
        batch_size=1024,
        max_iter=20,
        tune_text_epochs=2,  # Number of initial epochs to train text processing layers
        learning_rate=0.001,
        model_selection="best",
        trainable=True,
        verbose=True,
        seed=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.seed = seed
        self.embedding_dim = embedding_dim
        self.id_embedding_dim = id_embedding_dim
        self.n_factors = n_factors
        self.attention_dim = attention_dim
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout_rate
        self.max_text_length = max_text_length
        self.max_num_review = max_num_review
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.tune_text_epochs = tune_text_epochs
        self.learning_rate = learning_rate
        self.model_selection = model_selection
        self.device = device

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

    def _precompute_reviews(self, train_set, review_modality):
        if self.verbose:
            print("Pre-computing partition tracking matrices...")

        self.u_reviews = np.zeros(
            (train_set.num_users, self.max_num_review, self.max_text_length),
            dtype=np.int64,
        )
        self.u_partners = np.zeros(
            (train_set.num_users, self.max_num_review), dtype=np.int64
        )
        self.u_counts = np.zeros((train_set.num_users,), dtype=np.int64)

        self.i_reviews = np.zeros(
            (train_set.num_items, self.max_num_review, self.max_text_length),
            dtype=np.int64,
        )
        self.i_partners = np.zeros(
            (train_set.num_items, self.max_num_review), dtype=np.int64
        )
        self.i_counts = np.zeros((train_set.num_items,), dtype=np.int64)

        for u in range(train_set.num_users):
            p_dict = review_modality.user_review.get(u, {})
            interacted_items, _ = train_set.user_data.get(u, ([], []))
            valid_items = [i for i in interacted_items if i in p_dict][
                : self.max_num_review
            ]
            if valid_items:
                seqs = review_modality.batch_seq(
                    [p_dict[i] for i in valid_items], max_length=self.max_text_length
                )
                n = len(seqs)
                self.u_reviews[u, :n, :] = seqs
                self.u_partners[u, :n] = valid_items
                self.u_counts[u] = n

        for i in range(train_set.num_items):
            p_dict = review_modality.item_review.get(i, {})
            interacted_users, _ = train_set.item_data.get(i, ([], []))
            valid_users = [u for u in interacted_users if u in p_dict][
                : self.max_num_review
            ]
            if valid_users:
                seqs = review_modality.batch_seq(
                    [p_dict[u] for u in valid_users], max_length=self.max_text_length
                )
                n = len(seqs)
                self.i_reviews[i, :n, :] = seqs
                self.i_partners[i, :n] = valid_users
                self.i_counts[i] = n

        self.u_reviews = torch.from_numpy(self.u_reviews).to(self.device)
        self.u_partners = torch.from_numpy(self.u_partners).to(self.device)
        self.u_counts = torch.from_numpy(self.u_counts).to(self.device)
        self.i_reviews = torch.from_numpy(self.i_reviews).to(self.device)
        self.i_partners = torch.from_numpy(self.i_partners).to(self.device)
        self.i_counts = torch.from_numpy(self.i_counts).to(self.device)

    def fit(self, train_set, val_set=None):
        Recommender.fit(self, train_set, val_set)
        if not self.trainable:
            return self

        review_modality = train_set.review_text
        self._precompute_reviews(train_set, review_modality)
        global_mean = float(train_set.global_mean)

        # Initialize the validated paper-compliant architecture
        self.model = NARREModule(
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
            self.dropout_rate,
            self.max_text_length,
        ).to(self.device)

        optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        criterion = torch.nn.MSELoss()

        is_cuda = "cuda" in str(self.device)
        scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)
        num_batches = math.ceil(train_set.num_ratings / self.batch_size)

        # Early Stopping & Model Selection Trackers
        patience = 3
        min_delta = 0.001
        patience_counter = 0
        best_val_loss = float("inf")
        best_model_states = None
        total_cnn_features = self.n_filters * len(self.kernel_sizes)

        for epoch in range(self.max_iter):
            sum_train_loss = 0.0
            is_tuning_text = epoch < self.tune_text_epochs

            # -----------------------------------------------------------------
            # CACHING PASS: Safe detached intermediate state generation
            # -----------------------------------------------------------------
            self.model.eval()
            with torch.no_grad():
                cached_user_features = torch.zeros(
                    (train_set.num_users, self.max_num_review, total_cnn_features),
                    dtype=torch.float32,
                    device=self.device,
                )
                cached_item_features = torch.zeros(
                    (train_set.num_items, self.max_num_review, total_cnn_features),
                    dtype=torch.float32,
                    device=self.device,
                )

                with torch.amp.autocast("cuda", enabled=is_cuda):
                    for u_start in range(0, train_set.num_users, self.batch_size):
                        u_end = min(u_start + self.batch_size, train_set.num_users)
                        cached_user_features[u_start:u_end] = (
                            self.model.compute_user_text_features(
                                self.u_reviews[u_start:u_end]
                            )
                        )
                    for i_start in range(0, train_set.num_items, self.batch_size):
                        i_end = min(i_start + self.batch_size, train_set.num_items)
                        cached_item_features[i_start:i_end] = (
                            self.model.compute_item_text_features(
                                self.i_reviews[i_start:i_end]
                            )
                        )

            # -----------------------------------------------------------------
            # BATCH GRADIENT DESCENT OPTIMIZATION LOOP
            # -----------------------------------------------------------------
            self.model.train()
            progress_bar = tqdm(
                train_set.uir_iter(batch_size=self.batch_size, shuffle=True),
                total=num_batches,
                desc=f"Epoch {epoch+1}/{self.max_iter} [{'End-to-End' if is_tuning_text else 'Fast-Latent'}]",
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

                if is_tuning_text:
                    # Use the epoch-level precomputed cache directly!
                    # Simply clone them to detach from the caching graph, but keep gradients enabled for this step
                    u_feats = cached_user_features[u_ids].clone().requires_grad_(True)
                    i_feats = cached_item_features[i_ids].clone().requires_grad_(True)

                    with torch.amp.autocast("cuda", enabled=is_cuda):
                        inputs = (u_ids, i_ids, u_feats, self.u_partners[u_ids], self.u_counts[u_ids],
                                i_feats, self.i_partners[i_ids], self.i_counts[i_ids])
                        predictions = self.model(inputs, training=True)
                        loss = criterion(predictions, ratings)
                else:
                    with torch.amp.autocast("cuda", enabled=is_cuda):
                        inputs = (
                            u_ids,
                            i_ids,
                            cached_user_features[u_ids],
                            self.u_partners[u_ids],
                            self.u_counts[u_ids],
                            cached_item_features[i_ids],
                            self.i_partners[i_ids],
                            self.i_counts[i_ids],
                        )
                        predictions = self.model(inputs, training=True)
                        loss = criterion(predictions, ratings)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                sum_train_loss += loss.item() * len(batch_users)

            final_train_loss = sum_train_loss / train_set.num_ratings

            # -----------------------------------------------------------------
            # VALIDATION AND EARLY STOPPING EVALUATION
            # -----------------------------------------------------------------
            if val_set is not None:
                val_loss = self._evaluate_loss_cached(
                    val_set, cached_user_features, cached_item_features, criterion
                )
                if self.verbose:
                    print(
                        f"Epoch {epoch+1:02d} | Train MSE: {final_train_loss:.4f} | Val MSE: {val_loss:.4f}"
                    )

                # Check performance improvements
                if val_loss < (best_val_loss - min_delta):
                    best_val_loss = val_loss
                    best_model_states = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0  # Reset counter on meaningful improvement
                else:
                    # Guard: Only increment early stopping counters after the initial text tuning window completes
                    if not is_tuning_text:
                        patience_counter += 1
                        if self.verbose:
                            print(
                                f" Early stopping patience incremented: {patience_counter}/{patience}"
                            )

                # Break loop if patience threshold is broken
                if patience_counter >= patience:
                    if self.verbose:
                        print(
                            f" Early stopping triggered. Training halted at epoch {epoch+1}."
                        )
                    break
            else:
                if self.verbose:
                    print(f"Epoch {epoch+1:02d} | Train MSE: {final_train_loss:.4f}")

        # Rollback parameters to the documented optimal validation point
        if self.model_selection == "best" and best_model_states is not None:
            if self.verbose:
                print(
                    f"Restoring best checkpoint model state (Val MSE: {best_val_loss:.4f})..."
                )
            self.model.load_state_dict(best_model_states)

        self._sync_weights_to_numpy(train_set)
        return self

    def _evaluate_loss_cached(self, eval_set, cached_u, cached_i, criterion):
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
                    cached_u[u_ids],
                    self.u_partners[u_ids],
                    self.u_counts[u_ids],
                    cached_i[i_ids],
                    self.i_partners[i_ids],
                    self.i_counts[i_ids],
                )
                with torch.amp.autocast("cuda", enabled=is_cuda):
                    predictions = self.model(inputs, training=False)
                    loss = criterion(predictions, ratings)
                sum_eval_loss += loss.item() * len(batch_users)
        return sum_eval_loss / eval_set.num_ratings

    def _sync_weights_to_numpy(self, train_set):
        self.model.eval()
        with torch.no_grad():
            self.X = np.zeros((train_set.num_users, self.n_factors))
            self.Y = np.zeros((train_set.num_items, self.n_factors))

            for u_start in range(0, train_set.num_users, self.batch_size):
                u_end = min(u_start + self.batch_size, train_set.num_users)
                u_ids = torch.arange(
                    u_start, u_end, dtype=torch.long, device=self.device
                )
                u_features = self.model.compute_user_text_features(
                    self.u_reviews[u_ids]
                )
                user_Oi, _ = self.model.user_attention_block(
                    u_features,
                    self.model.l_user_iid_embedding(self.u_partners[u_ids]),
                    self.u_counts[u_ids],
                    self.u_reviews.size(1),
                )
                self.X[u_start:u_end] = self.model.Xu(user_Oi).cpu().numpy()

            for i_start in range(0, train_set.num_items, self.batch_size):
                i_end = min(i_start + self.batch_size, train_set.num_items)
                i_ids = torch.arange(
                    i_start, i_end, dtype=torch.long, device=self.device
                )
                i_features = self.model.compute_item_text_features(
                    self.i_reviews[i_ids]
                )
                item_Oi, _ = self.model.item_attention_block(
                    i_features,
                    self.model.l_item_uid_embedding(self.i_partners[i_ids]),
                    self.i_counts[i_ids],
                    self.i_reviews.size(1),
                )
                self.Y[i_start:i_end] = self.model.Yi(item_Oi).cpu().numpy()

            self.user_embedding = self.model.l_user_embedding.weight.data.cpu().numpy()
            self.item_embedding = self.model.l_item_embedding.weight.data.cpu().numpy()
            self.bu = self.model.user_bias.weight.data.cpu().numpy().flatten()
            self.bi = self.model.item_bias.weight.data.cpu().numpy().flatten()
            self.mu = float(self.model.global_bias.data.cpu().item())

    def score(self, user_idx, item_idx=None):
        if self.is_unknown_user(user_idx):
            raise ScoreException(f"Unknown user {user_idx}")

        if item_idx is None:
            # Reverted to compliant Matrix Factorization prediction space
            user_vec = self.user_embedding[user_idx] + self.X[user_idx]
            item_vecs = self.item_embedding + self.Y
            scores = np.dot(item_vecs, user_vec) + self.bu[user_idx] + self.bi + self.mu
            return np.nan_to_num(scores.ravel(), nan=0.0)
        else:
            if self.is_unknown_item(item_idx):
                raise ScoreException(f"Unknown item {item_idx}")
            user_vec = self.user_embedding[user_idx] + self.X[user_idx]
            item_vec = self.item_embedding[item_idx] + self.Y[item_idx]
            score = (
                np.dot(user_vec, item_vec)
                + self.bu[user_idx]
                + self.bi[item_idx]
                + self.mu
            )
            return float(score)

    def get_vector_measure(self):
        return MEASURE_DOT

    def get_user_vectors(self):
        return np.concatenate(
            (self.user_embedding + self.X, np.ones([self.user_embedding.shape[0], 1])),
            axis=1,
        )

    def get_item_vectors(self):
        return np.concatenate(
            (self.item_embedding + self.Y, self.bi.reshape((-1, 1))), axis=1
        )
