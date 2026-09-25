from cornac.eval_methods import BaseMethod
from cornac.data import Dataset


class JointTextMethod(BaseMethod):
    """
    Custom Cornac Evaluation Method extending BaseMethod to natively process,
    synchronize, and bind a JointReviewTextModality into training and validation splits.
    """

    def __init__(
        self,
        data=None,
        fmt="UIRT",
        rating_threshold=1.0,
        seed=None,
        exclude_unknowns=True,
        verbose=False,
        joint_text=None,
        **kwargs
    ):
        super().__init__(
            data=data,
            fmt=fmt,
            rating_threshold=rating_threshold,
            seed=seed,
            exclude_unknowns=exclude_unknowns,
            verbose=verbose,
            **kwargs
        )
        self.joint_text = joint_text

    def _build_modalities(self):
        """Bypassed to prevent premature processing before Dataset arrays are instantiated."""
        pass

    def build(self, train_data, val_data=None, test_data=None, **kwargs):
        """
        Main lifecycle entrypoint executing textual analysis after basic
        interaction maps are structurally built by the parent framework.
        """
        # 1. Construct standard dataset objects and indexing maps
        super().build(
            train_data=train_data, val_data=val_data, test_data=test_data, **kwargs
        )

        if self.joint_text is None:
            return self

        if self.verbose:
            print("🔄 Processing dual-source structural vocabulary matrices...")

        # 2. Dynamically carry model parameters into the text preprocessing pipeline
        if hasattr(self, "alpha"):
            self.joint_text.alpha = self.alpha

        # 3. Compile textual modalities with access to finalized train indices
        train_pairs = set()
        for [u_idx], [i_idx], _ in self.train_set.uir_iter():
            raw_u = self.train_set.user_ids[u_idx]
            raw_i = self.train_set.item_ids[i_idx]
            train_pairs.add((raw_u, raw_i))

        # Restrict text ingestion to training pairs only
        safe_train_reviews = [
            rec for rec in self.joint_text.raw_review_data
            if (rec[0], rec[1]) in train_pairs
        ]

        # Override the text reference within the modality before building
        self.joint_text.raw_review_data = safe_train_reviews
        self.joint_text.build(
            uid_map=self.train_set.uid_map,
            iid_map=self.train_set.iid_map,
            dok_matrix=self.train_set.dok_matrix,
            extra_data=self.train_set,
        )

        # 4. Inject compiled modality back into evaluation splits bypassing filters
        setattr(self.train_set, "joint_text", self.joint_text)
        if self.val_set is not None:
            setattr(self.val_set, "joint_text", self.joint_text)
        if self.test_set is not None:
            setattr(self.test_set, "joint_text", self.joint_text)

        if self.verbose:
            print("✅ Dual-source modality successfully bound to datasets.")

        return self

    @classmethod
    def from_splits(
        cls,
        train_data,
        val_data=None,
        test_data=None,
        fmt="UIRT",
        rating_threshold=1.0,
        exclude_unknowns=True,
        verbose=False,
        joint_text=None,
        **kwargs
    ):
        """Factory method to process raw benchmark data splits directly."""
        method = cls(
            fmt=fmt,
            rating_threshold=rating_threshold,
            exclude_unknowns=exclude_unknowns,
            verbose=verbose,
            joint_text=joint_text,
            **kwargs
        )
        method.build(train_data=train_data, val_data=val_data, test_data=test_data)
        return method
