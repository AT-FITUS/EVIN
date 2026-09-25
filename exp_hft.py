import cornac
from cornac.data import Reader
from cornac.data.text import ReviewModality, BaseTokenizer
from cornac.eval_methods import BaseMethod
from cornac.metrics import MAE, RMSE, MSE
from cornac.data.reader import read_text

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

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--debug", action="store_true")
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
    f"data/amazon_2023/{args.dataset}/review.txt",
    fmt="UIReview",
    sep="\t",
)


item_text_modality = ReviewModality(
    data=reviews,
    group_by="item",
    tokenizer=BaseTokenizer(stop_words="english"),
    max_vocab=5000,
    max_doc_freq=0.5,
)

data_split = BaseMethod.from_splits(
    train_data=rating_train,
    val_data=rating_valid,
    test_data=rating_test,
    rating_threshold=4.0,
    item_text=item_text_modality,
    verbose=True,
)

models = [
    cornac.models.HFT(
        name=f"HFT_{args.dataset}_",
        k=10,
        max_iter=40,
        grad_iter=5,
        l2_reg=0.001,
        lambda_text=0.01,
        vocab_size=5000,
        seed=123,
        verbose=True,
    )
]

cornac.Experiment(
    eval_method=data_split,
    models=models,
    metrics=[RMSE(), MAE(), MSE()],
    user_based=True,
).run()
