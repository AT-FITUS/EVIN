import numpy as np
from cornac.data import TextModality
from cornac.data.text import BaseTokenizer, CountVectorizer


class JointReviewTextModality(TextModality):
    def __init__(
        self,
        review_data,
        item_text_corpus,
        item_text_ids,
        max_vocab=5000,
        max_snippets=10,
        max_snippet_len=20,
        tokenizer=None,
    ):
        active_tokenizer = tokenizer if tokenizer is not None else BaseTokenizer()

        super().__init__(
            tokenizer=active_tokenizer,
            max_vocab=max_vocab,
        )

        self.raw_review_data = review_data
        self.item_text_corpus = item_text_corpus
        self.item_text_ids = item_text_ids

        self.max_snippets = max_snippets
        self.max_snippet_len = max_snippet_len

    def _build_corpus(self, uid_map, iid_map, dok_matrix, train_timestamps=None):
        num_items = len(iid_map)

        # 1. Structural allocation for phrase matrices and snippet creation times
        self.item_snippets = np.zeros(
            (num_items, self.max_snippets, self.max_snippet_len), dtype=np.int32
        )
        self.raw_snippets_corpus = [[] for _ in range(num_items)]
        self.snippet_timestamps = np.zeros(
            (num_items, self.max_snippets), dtype=np.int64
        )

        # Temporary pools for processing
        temp_metadata_snippets = {idx: [] for idx in range(num_items)}
        temp_review_snippets = {idx: [] for idx in range(num_items)}

        # 2. Extract Static Metadata Features (Timestamp = 0 so they are always visible)
        metadata_id_map = {str_id: idx for idx, str_id in enumerate(self.item_text_ids)}
        for item_str_id, internal_item_idx in iid_map.items():
            meta_corpus_idx = metadata_id_map.get(item_str_id, None)
            if meta_corpus_idx is not None:
                text = self.item_text_corpus[meta_corpus_idx]
                if text:
                    sentences = [
                        s.strip()
                        for s in text.replace("!", ".").replace("?", ".").split(".")
                        if s.strip()
                    ]
                    valid_meta = [(s, 0) for s in sentences if len(s.split()) >= 3]
                    temp_metadata_snippets[internal_item_idx].extend(valid_meta)

        # 3. Create a Fast Training Interaction Mapping from the Dataset Matrix
        # We build a lookup dictionary to link interaction timestamps cleanly
        interaction_time_lookup = {}
        if train_timestamps is not None:
            for u_idx, i_idx, ts in zip(*train_timestamps):
                if (u_idx, i_idx) not in interaction_time_lookup:
                    interaction_time_lookup[(u_idx, i_idx)] = []
                interaction_time_lookup[(u_idx, i_idx)].append(ts)

            # Sort lists chronologically to match historical review ordering
            for key in interaction_time_lookup:
                interaction_time_lookup[key].sort()

        # 4. Extract Review Phrases and Inject Aligned Timestamps
        for raw_uid, raw_iid, review_text in self.raw_review_data:
            user_idx = uid_map.get(raw_uid, None)
            item_idx = iid_map.get(raw_iid, None)

            if (
                user_idx is None
                or item_idx is None
                or dok_matrix[user_idx, item_idx] == 0
            ):
                continue

            if isinstance(review_text, str) and review_text:
                # Retrieve the timestamp sequence for this interaction pair
                ts_list = interaction_time_lookup.get((user_idx, item_idx), None)

                if ts_list and len(ts_list) > 0:
                    # Safely consume the earliest remaining timestamp
                    timestamp = int(ts_list.pop(0))
                else:
                    timestamp = 0

                sentences = [
                    s.strip()
                    for s in review_text.replace("!", ".").replace("?", ".").split(".")
                    if s.strip()
                ]
                valid_reviews = [
                    (s, timestamp) for s in sentences if len(s.split()) >= 3
                ]
                temp_review_snippets[item_idx].extend(valid_reviews)

        # 5. Fit unified Vocabulary Space
        all_sentences_for_vocab = []
        for item_idx in range(num_items):
            all_sentences_for_vocab.extend(
                [item[0] for item in temp_metadata_snippets[item_idx]]
            )
            all_sentences_for_vocab.extend(
                [item[0] for item in temp_review_snippets[item_idx]]
            )

        vectorizer = CountVectorizer(
            tokenizer=self.tokenizer, max_features=self.max_vocab
        )
        vectorizer.fit(all_sentences_for_vocab)
        self.vocab = vectorizer.vocab

        # 6. Stratify, Allocate, and Tokenize Phrases
        for item_idx in range(num_items):
            meta_pool = temp_metadata_snippets[item_idx]
            review_pool = temp_review_snippets[item_idx]

            max_meta_slots = min(3, self.max_snippets // 3)

            review_pool = sorted(
                review_pool, key=lambda x: len(x[0].split()), reverse=True
            )
            meta_pool = sorted(meta_pool, key=lambda x: len(x[0].split()), reverse=True)

            selected_meta = meta_pool[:max_meta_slots]
            remaining_slots = self.max_snippets - len(selected_meta)
            selected_reviews = review_pool[:remaining_slots]

            selected_slots = selected_meta + selected_reviews

            if (
                len(selected_slots) < self.max_snippets
                and len(meta_pool) > max_meta_slots
            ):
                extra_meta = meta_pool[
                    max_meta_slots : max_meta_slots
                    + (self.max_snippets - len(selected_slots))
                ]
                selected_slots.extend(extra_meta)

            self.raw_snippets_corpus[item_idx] = [item[0] for item in selected_slots]

            for s_idx, (sentence, ts) in enumerate(selected_slots):
                self.snippet_timestamps[item_idx, s_idx] = ts

                tokens = self.tokenizer.tokenize(sentence)
                token_ids = [
                    self.vocab.tok2idx.get(t) for t in tokens if t in self.vocab.tok2idx
                ]

                actual_len = min(self.max_snippet_len, len(token_ids))
                if actual_len > 0:
                    self.item_snippets[item_idx, s_idx, :actual_len] = token_ids[
                        :actual_len
                    ]

        dummy_corpus = [" ".join(self.raw_snippets_corpus[i]) for i in range(num_items)]
        return dummy_corpus, iid_map

    def build(self, uid_map=None, iid_map=None, dok_matrix=None, **kwargs):
        if uid_map is None or iid_map is None or dok_matrix is None:
            raise ValueError("uid_map, iid_map, and dok_matrix are required")

        # Intercept and pass the dataset's uir structural vectors and timestamps
        # through the kwargs execution pipeline
        train_timestamps = None
        if "extra_data" in kwargs and kwargs["extra_data"] is not None:
            dataset_obj = kwargs["extra_data"]
            if getattr(dataset_obj, "timestamps", None) is not None:
                train_timestamps = (
                    dataset_obj.uir_tuple[0],  # user_indices
                    dataset_obj.uir_tuple[1],  # item_indices
                    dataset_obj.timestamps,  # aligned training timestamps
                )

        self.corpus, id_map = self._build_corpus(
            uid_map, iid_map, dok_matrix, train_timestamps=train_timestamps
        )
        super(TextModality, self).build(id_map=id_map)
        return self
