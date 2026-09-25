import argparse
import random
import numpy as np
import torch

import cornac
from scipy.sparse import csr_matrix
from cornac.data import Reader
from cornac.data.reader import read_text
from cornac.data.text import BaseTokenizer
from cornac.metrics import MAE, RMSE, MSE

from experiment import Experiment
from eval_method import JointTextMethod
from evin import EVIN
from modality import JointReviewTextModality

DATASET = {
    "baby": "Baby_Products",
    "musical": "Musical_Instruments",
    "cellphone": "Cell_Phones_and_Accessories",
}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--dataset",
        type=str,
        choices=["musical", "baby", "cellphone"],
        default="musical",
    )
    parser.add_argument("-k", "--latent_dim", type=int, default=64)
    parser.add_argument("-b", "--batch_size", type=int, default=512)
    parser.add_argument("-sup", "--lambda_sup", type=float, default=0.01)
    parser.add_argument(
        "-alpha",
        "--alpha",
        type=float,
        default=0.1,
        help="Budget allocation weight assigned to Metadata vs Reviews.",
    )
    parser.add_argument("-rate", "--rating_weight", type=float, default=5.0)
    parser.add_argument("-lr", "--lr", type=float, default=0.0003)
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-5)
    parser.add_argument("-dropout", "--dropout", type=float, default=0.4)
    parser.add_argument("-it", "--init_temperature", type=float, default=0.8)
    parser.add_argument("-mt", "--min_temperature", type=float, default=0.04)
    parser.add_argument("-e", "--n_epochs", type=int, default=20)
    parser.add_argument("--max_snippets", type=int, default=10)
    parser.add_argument("--max_snippet_len", type=int, default=20)
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
    f"data/amazon_2023/{args.dataset}/review.txt",
    fmt="UIReview",
    sep="\t",
)
item_text, ids = read_text(f"data/amazon_2023/{args.dataset}/item_text.txt", sep="\t")

joint_modality = JointReviewTextModality(
    review_data=reviews,
    item_text_corpus=item_text,
    item_text_ids=ids,
    tokenizer=BaseTokenizer(stop_words="english"),
    max_vocab=5000,
    max_snippets=args.max_snippets,
    max_snippet_len=args.max_snippet_len,
)

data_split = JointTextMethod.from_splits(
    train_data=rating_train,
    val_data=rating_valid,
    test_data=rating_test,
    rating_threshold=4.0,
    joint_text=joint_modality,
    verbose=True,
    seed=args.seed,
)

evin_model = EVIN(
    name=(
        f"EVIN_{args.dataset}"
        f"_k_{args.latent_dim}"
        f"_e_{args.n_epochs}"
        f"_b_{args.batch_size}"
        f"_sup_{args.lambda_sup}"
        f"_alpha_{args.alpha}"
        f"_rate_{args.rating_weight}"
        f"_lr_{args.lr}"
        f"_wd_{args.weight_decay}"
        f"_drop_{args.dropout}"
        f"_it_{args.init_temperature}"
        f"_mt_{args.min_temperature}"
        f"_ms_{args.max_snippets}"
        f"_msl_{args.max_snippet_len}"
        f"_seed_{args.seed}"
    ),
    latent_dim=args.latent_dim,
    epochs=args.n_epochs,
    batch_size=args.batch_size,
    lambda_sup=args.lambda_sup,
    alpha=args.alpha,
    init_temperature=args.init_temperature,
    min_temperature=args.min_temperature,
    learning_rate=args.lr,
    weight_decay=args.weight_decay,
    dropout=args.dropout,
    max_snippets=args.max_snippets,
    rating_weight=args.rating_weight,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
    seed=args.seed,
    verbose=True,
)

eval_metrics = [RMSE(), MAE(), MSE()]

exp = Experiment(
    eval_method=data_split,
    models=[evin_model],
    metrics=eval_metrics,
    user_based=True,
)
exp.run()

test_set = data_split.test_set
tau_sweep = [0.01, 0.03, 0.05, 0.08, 0.12]

output_file = open(exp.output_file, "a")


def log_and_print(message):
    """Helper to write to both stdout and the target log file concurrently."""
    print(message)
    output_file.write(message + "\n")


log_and_print("-" * 110)
log_and_print(f"📊 COMPARATIVE BASELINE SWEEP MATRIX (USER-BASED MACRO MSE)")
log_and_print("-" * 110)
log_and_print(
    f"{'Tau':<6} | {'Method':<14} | {'Avg Len':<7} | {'Base MSE':<8} | {'Suff MSE':<8} | {'Necc MSE':<8}"
)
log_and_print("-" * 110)

# 2. Extract standard display header configurations
metric_names = [m.name for m in eval_metrics]
header_metrics = " | ".join(
    [f"{name:<8} | {name + '_Suff':<9} | {name + '_Necc':<9}" for name in metric_names]
)
header_line = f"{'Tau':<6} | {'Method':<14} | {'AvgLen':<7} | {header_metrics}"

log_and_print("📊 COMPARATIVE BASELINE SWEEP MATRIX (USER-BASED MACRO ALIGNED)")
log_and_print("-" * len(header_line))
log_and_print(header_line)
log_and_print("-" * len(header_line))

for tau in tau_sweep:
    # Run the updated threshold evaluation step
    res = evin_model.evaluate_threshold_fidelity(
        test_set, metrics=eval_metrics, tau_select=tau, user_based=True
    )

    avg_results = res["avg_results"]
    user_results = res["user_results"]
    diagnostics = res["diagnostics"]
    avg_len = diagnostics["lengths"].float().mean().item()

    # --- ALIGNMENT FIX ---
    # Cornac model instances track active users via self.train_set.uid_map.
    # We restrict macro calculations strictly to users the model officially knows.
    valid_model_users = set(evin_model.train_set.uid_map.values())

    for m_idx, method in enumerate(["EVIN", "Random", "EVIN-Inverse"]):
        metric_cells = []

        for idx in range(len(eval_metrics)):
            # Extract raw per-user dictionaries computed in your function
            u_dict_base = user_results["base"][idx]
            u_dict_suff = user_results["suff"][method][idx]
            u_dict_necc = user_results["necc"][method][idx]

            # Filter user dictionaries to match Cornac's active evaluation mask
            filtered_base = [
                v for u, v in u_dict_base.items() if u in valid_model_users
            ]
            filtered_suff = [
                v for u, v in u_dict_suff.items() if u in valid_model_users
            ]
            filtered_necc = [
                v for u, v in u_dict_necc.items() if u in valid_model_users
            ]

            # Compute macro-averages over the verified intersection set
            base_score = (
                sum(filtered_base) / len(filtered_base) if filtered_base else 0.0
            )
            suff_score = (
                sum(filtered_suff) / len(filtered_suff) if filtered_suff else 0.0
            )
            necc_score = (
                sum(filtered_necc) / len(filtered_necc) if filtered_necc else 0.0
            )

            metric_cells.append(
                f"{base_score:<8.4f} | {suff_score:<9.4f} | {necc_score:<9.4f}"
            )

        metrics_str = " | ".join(metric_cells)

        if m_idx == 0:
            log_and_print(
                f"{tau:<6.2f} | {method:<14} | {avg_len:<7.2f} | {metrics_str}"
            )
        else:
            log_and_print(f"{'':<6} | {method:<14} | {avg_len:<7.2f} | {metrics_str}")

    log_and_print("-" * len(header_line))

log_and_print("Building text alignment dictionary from raw review records...")
review_lookup = {}
for record in reviews:
    if len(record) >= 3:
        u_id_str = str(record[0]).strip()
        i_id_str = str(record[1]).strip()
        review_lookup[(u_id_str, i_id_str)] = record[2]

log_and_print(
    "\nSampling active test instances for extractive evaluation case studies..."
)

cases_printed = 0

for idx, (uid, iid, actual_rating) in enumerate(test_set.uir_iter()):
    if cases_printed >= 2:
        break

    u_idx = int(uid.item()) if hasattr(uid, "item") else int(uid)
    i_idx = int(iid.item()) if hasattr(iid, "item") else int(iid)

    raw_uid = str(test_set.user_ids[u_idx])
    raw_iid = str(test_set.item_ids[i_idx])
    if not raw_uid or not raw_iid:
        continue

    true_timestamp = test_set.timestamps[idx]

    log_and_print(
        f"\n📊 CASE STUDY [{cases_printed + 1}/2] — Ground Truth Record Verification"
    )
    log_and_print(f"🔗 Query Key Pair  : (User: '{raw_uid}', Item: '{raw_iid}')")
    log_and_print(
        f"📝 Actual Rating   : {actual_rating.item() if hasattr(actual_rating, 'item') else actual_rating:.1f} | Timestamp Vector: {true_timestamp}"
    )

    ground_truth_text = review_lookup.get((raw_uid, raw_iid), "None Linked")
    log_and_print(f'💬 Raw Ground-Truth Text: "{ground_truth_text}"')

    evin_model.produce_case_study(
        raw_uid, raw_iid, interaction_time=true_timestamp, pprint=log_and_print
    )
    cases_printed += 1

output_file.close()
