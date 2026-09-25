import os
import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from cornac.models import Recommender
from tqdm import tqdm

logger = logging.getLogger("ARRR")


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        hidden1_size = input_size // 20
        hidden2_size = max(hidden1_size // 2, 2)
        hidden3_size = max(hidden2_size // 2, 2)

        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, hidden3_size)
        self.fc4 = nn.Linear(hidden3_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)


class AspectImp(nn.Module):
    def __init__(self, logger, args):
        super(AspectImp, self).__init__()
        self.logger = logger
        self.args = args

        self.W_a = nn.Parameter(
            torch.Tensor(self.args.h1, self.args.h1), requires_grad=True
        )
        self.W_u = nn.Parameter(
            torch.Tensor(self.args.h2, self.args.h1), requires_grad=True
        )
        self.w_hu = nn.Parameter(torch.Tensor(self.args.h2, 1), requires_grad=True)
        self.W_i = nn.Parameter(
            torch.Tensor(self.args.h2, self.args.h1), requires_grad=True
        )
        self.w_hi = nn.Parameter(torch.Tensor(self.args.h2, 1), requires_grad=True)

        self.W_a.data.uniform_(-0.01, 0.01)
        self.W_u.data.uniform_(-0.01, 0.01)
        self.w_hu.data.uniform_(-0.01, 0.01)
        self.W_i.data.uniform_(-0.01, 0.01)
        self.w_hi.data.uniform_(-0.01, 0.01)

    def forward(self, userAspRep, itemAspRep, verbose=0):
        userAspRepTrans = torch.transpose(userAspRep, 1, 2)
        itemAspRepTrans = torch.transpose(itemAspRep, 1, 2)

        diff = torch.abs(userAspRep[:, :, None, :] - itemAspRep[:, None, :, :])
        similarityMatrix = 1 / (1 + diff.sum(dim=-1))

        H_u_1 = torch.matmul(self.W_u, userAspRepTrans)
        H_u_2 = torch.matmul(self.W_i, itemAspRepTrans)
        H_u_2 = torch.matmul(H_u_2, torch.transpose(similarityMatrix, 1, 2))
        H_u = F.relu(H_u_1 + H_u_2)

        userAspImpt = torch.matmul(torch.transpose(self.w_hu, 0, 1), H_u)
        userAspImpt = torch.transpose(userAspImpt, 1, 2)
        userAspImpt = F.softmax(userAspImpt, dim=1)
        userAspImpt = torch.squeeze(userAspImpt, 2)

        H_i_1 = torch.matmul(self.W_i, itemAspRepTrans)
        H_i_2 = torch.matmul(self.W_u, userAspRepTrans)
        H_i_2 = torch.matmul(H_i_2, similarityMatrix)
        H_i = F.relu(H_i_1 + H_i_2)

        itemAspImpt = torch.matmul(torch.transpose(self.w_hi, 0, 1), H_i)
        itemAspImpt = torch.transpose(itemAspImpt, 1, 2)
        itemAspImpt = F.softmax(itemAspImpt, dim=1)
        itemAspImpt = torch.squeeze(itemAspImpt, 2)

        return userAspImpt, itemAspImpt


class AspectRep(nn.Module):
    def __init__(self, logger, args, num_users, num_items, device):
        super(AspectRep, self).__init__()
        self.logger = logger
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.device = device

        self.aspProj = nn.Parameter(
            torch.Tensor(self.args.num_aspects, self.args.word_embed_dim, self.args.h1),
            requires_grad=True,
        )
        self.aspProj.data.uniform_(-0.01, 0.01)

        self.l1 = nn.Linear(self.args.d1, self.args.h1)
        self.user_embedding = nn.Embedding(num_users, self.args.d1)
        self.item_embedding = nn.Embedding(num_items, self.args.d1)

        self.user_embedding.weight.data.uniform_(-0.01, 0.01)
        self.item_embedding.weight.data.uniform_(-0.01, 0.01)

    def forward(self, args, batch_docIn, batch_id, nums, verbose=0):
        lst_batch_aspAttn = []
        lst_batch_aspRep = []
        batch_docIn = batch_docIn.to(self.device)
        if nums == self.num_items:
            id_vector = self.item_embedding(batch_id)
        else:
            id_vector = self.user_embedding(batch_id)

        qu = F.relu(self.l1(id_vector)).unsqueeze(-1)

        for a in range(self.args.num_aspects):
            batch_aspProjDoc = torch.matmul(batch_docIn, self.aspProj[a])

            if self.args.ctx_win_size == 1:
                attention_scores = torch.matmul(batch_aspProjDoc, qu).squeeze(-1)
                attention_weights = F.softmax(attention_scores, dim=1)
            else:
                qu_rep = qu.repeat(1, self.args.ctx_win_size, 1)
                pad_size = int((self.args.ctx_win_size - 1) / 2)
                batch_aspProjDoc_padded = F.pad(
                    batch_aspProjDoc, (0, 0, pad_size, pad_size), "constant", 0
                )
                batch_aspProjDoc_padded = batch_aspProjDoc_padded.unfold(
                    1, self.args.ctx_win_size, 1
                )
                batch_aspProjDoc_padded = torch.transpose(batch_aspProjDoc_padded, 2, 3)
                batch_aspProjDoc_padded = batch_aspProjDoc_padded.contiguous().view(
                    -1, self.args.max_doc_len, self.args.ctx_win_size * self.args.h1
                )

                attention_scores = torch.matmul(
                    batch_aspProjDoc_padded, qu_rep
                ).squeeze(-1)
                attention_weights = F.softmax(attention_scores, dim=1)

            attention_weights = attention_weights.unsqueeze(2)
            batch_aspRep = batch_aspProjDoc * attention_weights.expand_as(
                batch_aspProjDoc
            )
            batch_aspRep = torch.sum(batch_aspRep, dim=1)

            lst_batch_aspAttn.append(torch.transpose(attention_weights, 1, 2))
            lst_batch_aspRep.append(torch.unsqueeze(batch_aspRep, 1))

        batch_aspAttn = torch.cat(lst_batch_aspAttn, dim=1)
        batch_aspRep = torch.cat(lst_batch_aspRep, dim=1)

        return batch_aspAttn, batch_aspRep, id_vector


class ANet(nn.Module):
    def __init__(self, logger, args, num_users, num_items):
        super(ANet, self).__init__()
        self.logger = logger
        self.args = args
        self.num_users = num_users
        self.num_items = num_items

    def forward(self, userAspRep, itemAspRep, userAspImpt, itemAspImpt):
        lstAsp = []
        userAspRep = torch.transpose(userAspRep, 0, 1)
        itemAspRep = torch.transpose(itemAspRep, 0, 1)

        for k in range(self.args.num_aspects):
            userAspImpt_k = torch.unsqueeze(userAspImpt[:, k], 1)
            userANetLF_k = userAspImpt_k * userAspRep[k]

            itemAspImpt_k = torch.unsqueeze(itemAspImpt[:, k], 1)
            itemANetLF_k = itemAspImpt_k * itemAspRep[k]

            lstAsp.append((userANetLF_k, itemANetLF_k))

        userANetLF = torch.sum(
            torch.stack([user for user, item in lstAsp], dim=1), dim=1
        )
        itemANetLF = torch.sum(
            torch.stack([item for user, item in lstAsp], dim=1), dim=1
        )

        return userANetLF, itemANetLF


class RNet(nn.Module):
    def __init__(self, logger, args, num_users, num_items, device):
        super(RNet, self).__init__()
        self.logger = logger
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.device = device

        self.ratings_matrix = None
        self.ratings_matrix_T = None
        self.user_mlp = MLP(self.num_items, self.args.h1).to(self.device)
        self.item_mlp = MLP(self.num_users, self.args.h1).to(self.device)

    def forward(self, args, batch_uid, batch_iid):
        user_idx_list = list(batch_uid.cpu().numpy())
        item_idx_list = list(batch_iid.cpu().numpy())

        user_ratings = (
            torch.from_numpy(self.ratings_matrix[user_idx_list].toarray())
            .float()
            .to(self.device)
        )
        user_ratings = self.normalize(user_ratings)
        userlatent_vector = self.user_mlp(user_ratings)

        item_ratings = (
            torch.from_numpy(self.ratings_matrix_T[:, item_idx_list].toarray())
            .float()
            .t()
            .to(self.device)
        )
        item_ratings = self.normalize(item_ratings)
        itemlatent_vector = self.item_mlp(item_ratings)

        return userlatent_vector, itemlatent_vector

    @staticmethod
    def normalize(tensor, eps=1e-6):
        norm = torch.norm(tensor, p=2, dim=-1, keepdim=True)
        return tensor / (norm + eps)


class ARRRModule(nn.Module):
    def __init__(self, logger, args, num_users, num_items, device):
        super(ARRRModule, self).__init__()
        self.logger = logger
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.device = device

        if self.args.dropout_rate > 0.0:
            self.userAspRepDropout = nn.Dropout(p=self.args.dropout_rate)
            self.itemAspRepDropout = nn.Dropout(p=self.args.dropout_rate)

        self.globalOffset = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.uid_userOffset = nn.Embedding(self.num_users, 1)
        self.iid_itemOffset = nn.Embedding(self.num_items, 1)

        self.globalOffset.data.fill_(0)
        self.uid_userOffset.weight.data.fill_(0)
        self.iid_itemOffset.weight.data.fill_(0)

        self.shared_ANet = ANet(logger, args, num_users, num_items)
        self.shared_RNet = RNet(logger, args, num_users, num_items, device)

        self.uid_userDoc = nn.Embedding(self.num_users, self.args.max_doc_len)
        self.uid_userDoc.weight.requires_grad = False

        self.iid_itemDoc = nn.Embedding(self.num_items, self.args.max_doc_len)
        self.iid_itemDoc.weight.requires_grad = False

        self.wid_wEmbed = nn.Embedding(self.args.vocab_size, self.args.word_embed_dim)
        self.wid_wEmbed.weight.requires_grad = False

        self.shared_AspectRep = AspectRep(
            logger, args, self.num_users, self.num_items, device
        )
        self.shared_AspectImp = AspectImp(logger, args)

        self.weight_matrix = nn.Parameter(
            torch.Tensor(self.args.h1 * 2 + self.args.d1, 1), requires_grad=True
        )
        self.weight_matrix.data.uniform_(-0.01, 0.01)

    def forward(self, args, batch_uid, batch_iid):
        batch_userOffset = self.uid_userOffset(batch_uid)
        batch_itemOffset = self.iid_itemOffset(batch_iid)

        batch_userDoc = self.uid_userDoc(batch_uid.cpu()).long()
        batch_itemDoc = self.iid_itemDoc(batch_iid.cpu()).long()

        batch_userDocEmbed = self.wid_wEmbed(batch_userDoc)
        batch_itemDocEmbed = self.wid_wEmbed(batch_itemDoc)

        userAspAttn, userAspRep, uidvector = self.shared_AspectRep(
            args, batch_userDocEmbed, batch_uid, self.num_users
        )
        itemAspAttn, itemAspRep, iidvector = self.shared_AspectRep(
            args, batch_itemDocEmbed, batch_iid, self.num_items
        )

        userAspImpt, itemAspImpt = self.shared_AspectImp(userAspRep, itemAspRep)

        if self.args.dropout_rate > 0.0:
            userAspRep = self.userAspRepDropout(userAspRep)
            itemAspRep = self.itemAspRepDropout(itemAspRep)

        userANetLF, itemANetLF = self.shared_ANet(
            userAspRep, itemAspRep, userAspImpt, itemAspImpt
        )

        user_latent_vector, item_latent_vector = self.shared_RNet(
            args, batch_uid, batch_iid
        )

        final_user_embedding = torch.cat(
            (uidvector, user_latent_vector, userANetLF), dim=1
        )
        final_item_embedding = torch.cat(
            (iidvector, item_latent_vector, itemANetLF), dim=1
        )

        raw_prediction_scores = torch.matmul(
            (final_user_embedding * final_item_embedding), self.weight_matrix
        )
        prediction_scores = (
            raw_prediction_scores
            + batch_userOffset
            + batch_itemOffset
            + self.globalOffset
        )

        return prediction_scores


class ARRR(Recommender):
    def __init__(
        self,
        name="ARRR",
        epochs=20,
        batch_size=256,
        lr=0.001,
        dropout_rate=0.0,
        num_aspects=5,
        word_embed_dim=100,
        vocab_size=20000,
        max_doc_len=50,
        h1=64,
        h2=32,
        d1=50,
        ctx_win_size=1,
        patience=5,
        use_cuda=True,
        seed=None,
        verbose=False,
    ):
        super().__init__(name=name, trainable=True, verbose=verbose)

        class Args:
            pass

        self.args = Args()
        self.args.epochs = epochs
        self.args.batch_size = batch_size
        self.args.lr = lr
        self.args.dropout_rate = dropout_rate
        self.args.num_aspects = num_aspects
        self.args.word_embed_dim = word_embed_dim
        self.args.vocab_size = vocab_size
        self.args.max_doc_len = max_doc_len
        self.args.h1 = h1
        self.args.h2 = h2
        self.args.d1 = d1
        self.args.ctx_win_size = ctx_win_size
        self.args.patience = patience
        self.args.use_cuda = use_cuda

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        )

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)

        self.num_users = train_set.num_users
        self.num_items = train_set.num_items

        self.ratings_matrix = train_set.matrix
        self.ratings_matrix_T = train_set.matrix.tocsc()
        review_modality = train_set.review_text

        user_docs = np.zeros((self.num_users, self.args.max_doc_len), dtype=np.int32)
        item_docs = np.zeros((self.num_items, self.args.max_doc_len), dtype=np.int32)

        for u_idx, item_interaction_dict in review_modality.user_review.items():
            u_tokens = []
            for i_idx, seq_idx in item_interaction_dict.items():
                u_tokens.extend(review_modality.sequences[seq_idx])
            u_tokens = u_tokens[: self.args.max_doc_len]
            user_docs[u_idx, : len(u_tokens)] = u_tokens

        for i_idx, user_interaction_dict in review_modality.item_review.items():
            i_tokens = []
            for u_idx, seq_idx in user_interaction_dict.items():
                i_tokens.extend(review_modality.sequences[seq_idx])
            i_tokens = i_tokens[: self.args.max_doc_len]
            item_docs[i_idx, : len(i_tokens)] = i_tokens

        self.model = ARRRModule(
            logger, self.args, self.num_users, self.num_items, self.device
        ).to(self.device)

        self.model.uid_userDoc.cpu()
        self.model.iid_itemDoc.cpu()
        self.model.wid_wEmbed.cpu()

        self.model.uid_userDoc.weight.data.copy_(torch.from_numpy(user_docs))
        self.model.iid_itemDoc.weight.data.copy_(torch.from_numpy(item_docs))
        self.model.shared_RNet.ratings_matrix = self.ratings_matrix
        self.model.shared_RNet.ratings_matrix_T = self.ratings_matrix_T

        torch.cuda.empty_cache()
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        criterion = torch.nn.MSELoss()

        patience = self.args.patience
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_states = None

        num_batches = train_set.num_batches(self.args.batch_size)

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            sum_loss = 0.0
            count = 0

            progress_bar = tqdm(
                train_set.uir_iter(batch_size=self.args.batch_size, shuffle=True),
                total=num_batches,
                desc=f"Epoch [{epoch}/{self.args.epochs}]",
                disable=not self.verbose,
            )

            for batch_u, batch_i, batch_r in progress_bar:
                uid_tensor = torch.LongTensor(batch_u).to(self.device)
                iid_tensor = torch.LongTensor(batch_i).to(self.device)
                ratings_tensor = torch.FloatTensor(batch_r).view(-1, 1).to(self.device)

                optimizer.zero_grad()
                predictions = self.model(self.args, uid_tensor, iid_tensor)
                loss = criterion(predictions, ratings_tensor)
                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                sum_loss += batch_loss * len(batch_u)
                count += len(batch_u)

                progress_bar.set_postfix({"Train Loss": f"{batch_loss:.4f}"})

            train_avg_loss = sum_loss / count

            if val_set is not None:
                self.model.eval()
                val_loss = 0.0
                val_count = 0

                with torch.no_grad():
                    val_users, val_items, val_ratings = val_set.uir_tuple

                    for i in range(0, len(val_users), self.args.batch_size):
                        b_u = torch.LongTensor(
                            val_users[i : i + self.args.batch_size]
                        ).to(self.device)
                        b_i = torch.LongTensor(
                            val_items[i : i + self.args.batch_size]
                        ).to(self.device)
                        b_r = (
                            torch.FloatTensor(val_ratings[i : i + self.args.batch_size])
                            .view(-1, 1)
                            .to(self.device)
                        )

                        preds = self.model(self.args, b_u, b_i)
                        val_loss += criterion(preds, b_r).item() * len(b_u)
                        val_count += len(b_u)

                val_avg_loss = val_loss / val_count
                if self.verbose:
                    print(
                        f"Epoch [{epoch}/{self.args.epochs}] Metrics -> Train Loss: {train_avg_loss:.4f} | Val Loss: {val_avg_loss:.4f}"
                    )

                if val_avg_loss < best_val_loss:
                    best_val_loss = val_avg_loss
                    patience_counter = 0
                    best_model_states = copy.deepcopy(self.model.state_dict())
                else:
                    patience_counter += 1
                    if self.verbose:
                        print(
                            f"EarlyStopping counter: {patience_counter} out of {patience}"
                        )

                    if patience_counter >= patience:
                        if self.verbose:
                            print(
                                f"Early stopping triggered at epoch {epoch}. Restoring weights."
                            )
                        break
            else:
                if self.verbose:
                    print(
                        f"Epoch [{epoch}/{self.args.epochs}] Completed -> Train Loss: {train_avg_loss:.4f}"
                    )

        if best_model_states is not None:
            self.model.load_state_dict(best_model_states)

        return self

    def score(self, user_idx, item_idx=None):
        self.model.eval()
        with torch.no_grad():
            if item_idx is None:
                uid_tensor = torch.full(
                    (self.num_items,), user_idx, dtype=torch.long, device=self.device
                )
                iid_tensor = torch.arange(
                    self.num_items, dtype=torch.long, device=self.device
                )
            else:
                uid_tensor = torch.LongTensor([user_idx]).to(self.device)
                iid_tensor = torch.LongTensor([item_idx]).to(self.device)

            preds = self.model(self.args, uid_tensor, iid_tensor)
            return preds.cpu().numpy().flatten()
