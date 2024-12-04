import os
import argparse

import polars as pl


parser = argparse.ArgumentParser()
parser.add_argument('labeled_ingredients_file', help="Specify the file path of the labeled ingredients")
args = parser.parse_args()

script_filepath = os.path.dirname(os.path.realpath(__file__))

accepted_answers_filepath = os.path.join(script_filepath, 'accepted_mturk_df_with_fixed_collisions.csv')
accepted_answers_df = pl.read_csv(accepted_answers_filepath, separator="\t")
accepted_answers_df = accepted_answers_df.with_columns(pl.col('answer_path').str.replace('"', ''))
accepted_answers_df = accepted_answers_df.rename({'answer_path': 'label'})

ingredients_list = pl.read_csv(os.path.join(script_filepath, os.pardir, 'ingredient_food_kg_names.csv'))

labeled_answers_filepath = os.path.join(args.labeled_ingredients_file)
labeled_answers_df = pl.read_csv(labeled_answers_filepath, separator="\t")
labeled_answers_df = labeled_answers_df.with_columns(pl.col('answer_path').str.replace('"', ''))

# Calculate the number of correct answers
correct_answers = accepted_answers_df.join(labeled_answers_df, on=['ingredient'], how='inner')
correct_answers = correct_answers.filter(pl.col('answer_path') == pl.col('label'))
correct_answers_percentage = correct_answers.shape[0] / len(accepted_answers_df)
print(f"Number of correct answers: {correct_answers.shape[0]}/{len(accepted_answers_df)} ({correct_answers_percentage*100:.2f}%)")

