import os
import argparse
import texttable

import polars as pl


parser = argparse.ArgumentParser()
parser.add_argument('labeled_ingredients_file', help="Specify the file path of the labeled ingredients")
args = parser.parse_args()

script_filepath = os.path.dirname(os.path.realpath(__file__))

accepted_answers_filepath = os.path.join(script_filepath, 'accepted_mturk_df_with_fixed_collisions.csv')
accepted_answers_df = pl.read_csv(accepted_answers_filepath, separator="\t")
accepted_answers_df = accepted_answers_df.with_columns(pl.col('answer_path').str.replace('"', ''))
accepted_answers_df = accepted_answers_df.rename({'answer_path': 'label'})

labeled_answers_filepath = os.path.join(args.labeled_ingredients_file)
labeled_answers_df = pl.read_csv(labeled_answers_filepath, separator="\t")
labeled_answers_df = labeled_answers_df.with_columns(pl.col('answer_path').str.replace('"', ''))

# Calculate the number of correct answers
ingr_accepted_answers_df = accepted_answers_df.select(pl.col(['ingredient_index', 'ingredient', 'label']))
correct_answers = ingr_accepted_answers_df.join(labeled_answers_df, on=['ingredient'], how='inner')
perfect_matches = correct_answers.filter(pl.col('label') == pl.col('answer_path'))
matches_tail_cut = {}
for i in range(1, 4):
    matches_tail_cut[i] = (
        correct_answers
        .with_columns(pl.col('answer_path').str.split(' -> '))
        .with_columns(pl.col('answer_path').list.slice(0, pl.col('answer_path').list.len() - i).list.join(' -> '))
        .filter(
            pl.col('label').str.contains(pl.col('answer_path'))
            .and_(pl.col('answer_path').str.len_chars() > 0)
        )
    )

matches_head_levels = {}
for i in range(1, 4):
    matches_head_levels[i] = (
        correct_answers
        .with_columns(pl.col('answer_path').str.split(' -> '))
        .with_columns(pl.col('answer_path').list.head(i).list.join(' -> '))
        .filter(
            pl.col('label').str.contains(pl.col('answer_path'))
            .and_(pl.col('answer_path').str.len_chars() > 0)
        )
    )

table_rows = [
    ["Perfect Matches", f"{(perfect_matches.shape[0] / accepted_answers_df.shape[0])*100:.2f}%"],
    *[[f"Matches with {i} tail node removed", f"{(matches_tail_cut[i].shape[0] / accepted_answers_df.shape[0])*100:.2f}%"] for i in range(1, 4)],
    *[[f"Matches after {i} levels", f"{(matches_head_levels[i].shape[0] / accepted_answers_df.shape[0])*100:.2f}%"] for i in range(1, 4)],
]

table = texttable.Texttable()
table.set_cols_align(["l", "r"])
table.set_cols_dtype(["t", "t"])
table.add_rows([["Match Type", "Match Percentage"], *table_rows])
print(f"Results with {os.path.basename(labeled_answers_filepath).replace('.csv', '').replace('labeled_ingredients_', '')}")
print(table.draw())
print("Match examples:")
with pl.Config(tbl_width_chars=400, fmt_str_lengths=1000):
    print((
        correct_answers
        .select(pl.col(['ingredient', 'answer_path', 'label']))
        .rename({'answer_path': 'answer_path (prediction)', 'label': 'label (ground truth)'})
        .sample(5)
    ))
