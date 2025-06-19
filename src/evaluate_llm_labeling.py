import os
import argparse
import texttable

import polars as pl


parser = argparse.ArgumentParser()
parser.add_argument(
    "llm_labeled_ingredients_files",
    nargs="+",
    help="Specify the file path of the labeled ingredients",
)
parser.add_argument(
    "--truth_labels_file",
    default="accepted_mturk_df_with_fixed_collisions_update.csv",
    help="Specify the file path of the truth labels",
)
args = parser.parse_args()

script_filepath = os.path.dirname(os.path.realpath(__file__))

accepted_answers_dfs = {}
for split in ["valid", "test"]:
    accepted_answers_filepath = os.path.join(
        script_filepath, f"{os.path.splitext(args.truth_labels_file)[0]}_{split}.csv"
    )
    accepted_answers_df = pl.read_csv(accepted_answers_filepath, separator="\t")
    accepted_answers_df = accepted_answers_df.with_columns(
        pl.col("answer_path").str.replace('"', "")
    )
    accepted_answers_df = accepted_answers_df.rename({"answer_path": "label"})
    accepted_answers_df = accepted_answers_df.unique(
        ["ingredient"]
    )  # it should already be unique, but done for safety
    accepted_answers_dfs[split] = accepted_answers_df

model_names_list = []
labeled_answers_dfs = []
for labeled_answers_filepath in args.llm_labeled_ingredients_files:
    model_name = (
        os.path.basename(labeled_answers_filepath)
        .replace(".csv", "")
        .replace("labeled_ingredients_", "")
    )
    model_names_list.append(model_name)

    labeled_answers_df = pl.read_csv(labeled_answers_filepath, separator="\t")
    labeled_answers_df = labeled_answers_df.with_columns(
        pl.col("answer_path").str.replace('"', "")
    )
    labeled_answers_df = labeled_answers_df.unique(["ingredient"])
    labeled_answers_dfs.append(labeled_answers_df)

min_threshold = 1
max_threshold = 5
for split, split_accepted_answers_df in accepted_answers_dfs.items():
    ingr_accepted_answers_df = split_accepted_answers_df.select(
        pl.col(["ingredient_index", "ingredient", "label"])
    )

    perfect_matches = []
    matches_tail_cut = []
    matches_head_levels = []
    for labeled_answers_df in labeled_answers_dfs:
        joined_answers = ingr_accepted_answers_df.join(
            labeled_answers_df, on=["ingredient"], how="inner"
        )
        perfect_matches.append(
            joined_answers.filter(pl.col("label") == pl.col("answer_path"))
        )
        model_matches_tail_cut = {}
        for i in range(min_threshold, max_threshold):
            model_matches_tail_cut[i] = (
                joined_answers.with_columns(pl.col("answer_path").str.split(" -> "))
                .with_columns(
                    pl.col("answer_path")
                    .list.slice(0, pl.col("answer_path").list.len() - i)
                    .list.join(" -> ")
                )
                .with_columns(pl.col("label").str.split(" -> "))
                .with_columns(
                    pl.col("label")
                    .list.slice(0, pl.col("label").list.len() - i)
                    .list.join(" -> ")
                )
                .filter(
                    (pl.col("label") == pl.col("answer_path")).or_(
                        pl.col("answer_path").str.len_chars() == 0
                    )
                )
            )

        model_matches_head_levels = {}
        for i in range(min_threshold, max_threshold):
            model_matches_head_levels[i] = (
                joined_answers.with_columns(pl.col("answer_path").str.split(" -> "))
                .with_columns(pl.col("answer_path").list.head(i).list.join(" -> "))
                .with_columns(pl.col("label").str.split(" -> "))
                .with_columns(pl.col("label").list.head(i).list.join(" -> "))
                .filter(
                    (pl.col("label") == pl.col("answer_path")).or_(
                        pl.col("answer_path").str.len_chars() == 0
                    )
                )
            )

        matches_tail_cut.append(model_matches_tail_cut)
        matches_head_levels.append(model_matches_head_levels)

    header = ["Match Type", *model_names_list]
    table_rows = [
        ["Perfect Matches"]
        + [
            f"{(model_perfect_matches.shape[0] / split_accepted_answers_df.shape[0]) * 100:.2f}%"
            for model_perfect_matches in perfect_matches
        ],
        *[
            [f"Matches with {i} tail node removed"]
            + [
                f"{(model_matches_tail_cut[i].shape[0] / split_accepted_answers_df.shape[0]) * 100:.2f}%"
                for model_matches_tail_cut in matches_tail_cut
            ]
            for i in range(min_threshold, max_threshold)
        ],
        *[
            [f"Matches after {i} levels"]
            + [
                f"{(model_matches_head_levels[i].shape[0] / split_accepted_answers_df.shape[0]) * 100:.2f}%"
                for model_matches_head_levels in matches_head_levels
            ]
            for i in range(min_threshold, max_threshold)
        ],
    ]

    table = texttable.Texttable()
    table.set_cols_align(["l"] + ["r"] * len(model_names_list))
    table.set_cols_dtype(["t"] + ["t"] * len(model_names_list))
    table.add_rows([header, *table_rows])
    print(f"######################### {split.upper()} #########################")
    print(table.draw())
    print(f"Comparison example of {model_names_list[-1]}:")
    with pl.Config(tbl_width_chars=400, fmt_str_lengths=1000):
        print(
            (
                joined_answers.select(pl.col(["ingredient", "answer_path", "label"]))
                .rename(
                    {
                        "answer_path": "answer_path (prediction)",
                        "label": "label (ground truth)",
                    }
                )
                .sample(4)
            )
        )


valid_ingr_accepted_answers_df = accepted_answers_dfs["valid"].select(
    pl.col(["ingredient", "label"])
)
mismatches_head = {}
for model_name, labeled_answers_df in zip(model_names_list, labeled_answers_dfs):
    for i in range(min_threshold, max_threshold):
        if i not in mismatches_head:
            valid_correct_answers = valid_ingr_accepted_answers_df.join(
                labeled_answers_df.select(pl.all().exclude("index")),
                on=["ingredient"],
                how="inner",
            )
            mismatches_head[i] = (
                valid_correct_answers.with_columns(
                    pl.col("answer_path").str.split(" -> ")
                )
                .with_columns(pl.col("answer_path").list.head(i).list.join(" -> "))
                .rename({"answer_path": f"answer_path_{model_name}"})
                .with_columns(pl.col("label").str.split(" -> "))
                .with_columns(pl.col("label").list.head(i).list.join(" -> "))
            )
        else:
            mismatches_head[i] = (
                mismatches_head[i]
                .join(
                    labeled_answers_df.select(pl.all().exclude("index"))
                    .with_columns(pl.col("answer_path").str.split(" -> "))
                    .with_columns(pl.col("answer_path").list.head(i).list.join(" -> "))
                    .rename({"answer_path": f"answer_path_{model_name}"}),
                    on=["ingredient"],
                    how="inner",
                )
                .with_columns(pl.col("label").str.split(" -> "))
                .with_columns(pl.col("label").list.head(i).list.join(" -> "))
            )

for i in range(min_threshold, 4):
    mismatches_head[i] = mismatches_head[i].filter(
        pl.lit(False).or_(
            *[
                (pl.col("label") == pl.col(f"answer_path_{model_name}")).not_()
                for model_name in model_names_list
            ]
        )
    )
    with pl.Config(fmt_str_lengths=100, tbl_rows=-1):
        print(
            f"######################### VALID MISMATCHES HEAD {i} #########################"
        )
        print(f"Number of mismatches: {mismatches_head[i].shape[0]}")
        print((mismatches_head[i].rename({"label": "label (ground truth)"})))

valid_ingr_accepted_answers_df = accepted_answers_dfs["valid"].select(
    pl.col(["ingredient", "label"])
)
mismatches_tail_cut = {}
for model_name, labeled_answers_df in zip(model_names_list, labeled_answers_dfs):
    for i in range(min_threshold, max_threshold):
        if i not in mismatches_tail_cut:
            valid_correct_answers = valid_ingr_accepted_answers_df.join(
                labeled_answers_df.select(pl.all().exclude("index")),
                on=["ingredient"],
                how="inner",
            )
            mismatches_tail_cut[i] = (
                valid_correct_answers.with_columns(
                    pl.col("answer_path").str.split(" -> ")
                )
                .with_columns(
                    pl.col("answer_path")
                    .list.slice(0, pl.col("answer_path").list.len() - i)
                    .list.join(" -> ")
                )
                .rename({"answer_path": f"answer_path_{model_name}"})
                .with_columns(pl.col("label").str.split(" -> "))
                .with_columns(
                    pl.col("label")
                    .list.slice(0, pl.col("label").list.len() - i)
                    .list.join(" -> ")
                )
            )
        else:
            mismatches_tail_cut[i] = (
                mismatches_tail_cut[i]
                .join(
                    labeled_answers_df.select(pl.all().exclude("index"))
                    .with_columns(pl.col("answer_path").str.split(" -> "))
                    .with_columns(
                        pl.col("answer_path")
                        .list.slice(0, pl.col("answer_path").list.len() - i)
                        .list.join(" -> ")
                    )
                    .rename({"answer_path": f"answer_path_{model_name}"}),
                    on=["ingredient"],
                    how="inner",
                )
                .with_columns(pl.col("label").str.split(" -> "))
                .with_columns(
                    pl.col("label")
                    .list.slice(0, pl.col("label").list.len() - i)
                    .list.join(" -> ")
                )
            )

for i in range(min_threshold, max_threshold):
    mismatches_tail_cut[i] = mismatches_tail_cut[i].filter(
        pl.lit(False).or_(
            *[
                (pl.col("label") == pl.col(f"answer_path_{model_name}")).not_()
                for model_name in model_names_list
            ]
        )
    )
    with pl.Config(fmt_str_lengths=100, tbl_rows=-1):
        print(
            f"######################### VALID MISMATCHES TAIL CUT {i} #########################"
        )
        print(f"Number of mismatches: {mismatches_tail_cut[i].shape[0]}")
        print((mismatches_tail_cut[i].rename({"label": "label (ground truth)"})))
