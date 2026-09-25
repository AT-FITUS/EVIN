import cornac
from cornac.data import Reader
from eval_method import JointTextMethod
from cornac.metrics import MAE, RMSE, MSE

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
    parser.add_argument("-k", "--n_factors", type=int, default=32)
    parser.add_argument("-e", "--n_epochs", type=int, default=20)

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

# 3. Call your Custom Joint Evaluation Split Handler
data_split = JointTextMethod.from_splits(
    train_data=rating_train,
    val_data=rating_valid,
    test_data=rating_test,
    rating_threshold=4.0,
    verbose=True,
)

models = [
    cornac.models.MF(
        name=f"MF_{args.dataset}_k_{args.n_factors}_e_{args.n_epochs}",
        k=args.n_factors,
        max_iter=args.n_epochs,
        learning_rate=0.01,
        verbose=True,
    )
]

# 5. Execute unified experimental validation run
cornac.Experiment(
    eval_method=data_split,
    models=models,
    metrics=[RMSE(), MAE(), MSE()],
    user_based=True,
).run()
