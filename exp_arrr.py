import os
import torch

os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
torch.backends.cudnn.enabled = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cornac
from cornac.data import Reader
from cornac.data.text import BaseTokenizer, ReviewModality
from cornac.eval_methods import BaseMethod
from narre import NARRE
from cornac.metrics import MAE, RMSE, MSE
from arrr import ARRR

DATASET = {
    "baby": "Baby_Products",
    "musical": "Musical_Instruments",
    "cellphone": "Cell_Phones_and_Accessories",
}


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--dataset",
        type=str,
        choices=["musical", "baby", "cellphone"],
        default="musical",
    )
    parser.add_argument("-a", "--n_aspects", type=int, default=5)
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-e", "--n_epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    return args


args = parse_arguments()

rating_train = Reader().read(
    f"data/amazon_2023/benchmark/5core/timestamp_w_his/{DATASET[args.dataset]}.train.csv",
    fmt="UIRT",
    sep=",",
    skip_lines=1,
)
rating_valid = Reader().read(
    f"data/amazon_2023/benchmark/5core/timestamp_w_his/{DATASET[args.dataset]}.valid.csv",
    fmt="UIRT",
    sep=",",
    skip_lines=1,
)
rating_test = Reader().read(
    f"data/amazon_2023/benchmark/5core/timestamp_w_his/{DATASET[args.dataset]}.test.csv",
    fmt="UIRT",
    sep=",",
    skip_lines=1,
)
reviews = Reader().read(
    f"data/amazon_2023/{args.dataset}/review.txt", fmt="UIReview", sep="\t"
)

review_modality = ReviewModality(
    data=reviews,
    tokenizer=BaseTokenizer(stop_words="english"),
    max_vocab=5000,
    max_doc_freq=0.5,
)

eval_method = BaseMethod.from_splits(
    train_data=rating_train,
    val_data=rating_valid,
    test_data=rating_test,
    rating_threshold=4.0,
    review_text=review_modality,
    verbose=True,
    seed=args.seed,
    exclude_unknowns=True,
)


model = ARRR(
    epochs=args.n_epochs,
    batch_size=args.batch_size,
    num_aspects=args.n_aspects,
    vocab_size=eval_method.review_text.vocab.size,
    verbose=True,
)

cornac.Experiment(
    eval_method=eval_method,
    models=[model],
    metrics=[RMSE(), MAE(), MSE()],
).run()
