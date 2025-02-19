import os
import json
import polars as pl
from sentence_transformers import SentenceTransformer, util
from utils import ingredient_tree_from_json

#!/usr/bin/env python3

# Parameters
TOP_K = 5
STRICT_THRESHOLD = 0.98  # for exact matching between tree leaves and csv items
TOPK_THRESHOLD = 0.95    # for top-k similar ingredients
TOPK_THRESHOLD2 = 0.90    # for top-k similar ingredients

# Helper: recursively get the leaves of the hierarchical tree.
def get_tree_leaves(tree):
    leaves = []
    for key, value in tree.items():
        if not value['children']:
            leaves.append(key)
    return leaves

def main():
    # Discover the file paths relative to this script file.
    script_filepath = os.path.realpath(__file__)

    # Load the ingredient tree from JSON
    json_path = os.path.join(os.path.dirname(script_filepath), os.pardir, 'json_melted.json')
    with open(json_path, 'r') as f:
        sueatable_db = json.load(f)
    ingredient_tree = ingredient_tree_from_json(sueatable_db)
    tree_leaves = get_tree_leaves(ingredient_tree)

    # Load CSV ingredients using Polars, assuming there is one column (ingredient names)
    csv_path = os.path.join(os.path.dirname(script_filepath), os.pardir, 'ingredient_food_kg_names.csv')
    df = pl.read_csv(csv_path)
    # Assuming the csv column name is 'ingredient' or use the first column
    if 'ingredient' in df.columns:
        csv_ingredients = df['ingredient'].to_list()
    else:
        csv_ingredients = df[df.columns[0]].to_list()

    # Load the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode both sets of texts
    tree_embeddings = model.encode(tree_leaves, convert_to_tensor=True)
    csv_embeddings = model.encode(csv_ingredients, convert_to_tensor=True)

    # Compute cosine similarity between each tree leaf and csv ingredient:
    cos_scores = util.cos_sim(tree_embeddings, csv_embeddings)

    # 1. Find exact matches for which cosine similarity is higher than STRICT_THRESHOLD
    print("Exact matches (cosine similarity > {:.2f}):".format(STRICT_THRESHOLD))
    for i, leaf in enumerate(tree_leaves):
        # find indices with score higher than STRICT_THRESHOLD
        matching_indices = (cos_scores[i] > STRICT_THRESHOLD).nonzero(as_tuple=False)
        if matching_indices.numel() > 0:
            matches = []
            for idx in matching_indices:
                idx = int(idx)
                score = cos_scores[i][idx]
                matches.append((csv_ingredients[idx], float(score)))
            print(f"Tree ingredient '{leaf}': {matches}")

    # 2. For each tree ingredient, get top-k similar csv ingredients (with similarity >= TOPK_THRESHOLD)
    print("\nTop-k matches ({:.2f} <= cosine similarity < {:.2f}):".format(TOPK_THRESHOLD, STRICT_THRESHOLD))
    for i, leaf in enumerate(tree_leaves):
        # Get scores and filter based on threshold
        scores = cos_scores[i]
        # Get all indices where score exceeds threshold
        valid_indices = [idx for idx, score in enumerate(scores) if TOPK_THRESHOLD <= score < STRICT_THRESHOLD]
        if not valid_indices:
            continue
        # Sort valid indices by descending score
        valid_indices = sorted(valid_indices, key=lambda j: scores[j], reverse=True)[:TOP_K]
        matches = [(csv_ingredients[j], float(scores[j])) for j in valid_indices]
        print(f"Tree ingredient '{leaf}': {matches}")

    # 3. For each tree ingredient, get top-k similar csv ingredients (with similarity >= TOPK_THRESHOLD2)
    print("\nTop-k matches ({:.2f} <= cosine similarity < {:.2f}):".format(TOPK_THRESHOLD2, TOPK_THRESHOLD))
    for i, leaf in enumerate(tree_leaves):
        # Get scores and filter based on threshold
        scores = cos_scores[i]
        # Get all indices where score exceeds threshold
        valid_indices = [idx for idx, score in enumerate(scores) if TOPK_THRESHOLD2 <= score < TOPK_THRESHOLD]
        if not valid_indices:
            continue
        # Sort valid indices by descending score
        valid_indices = sorted(valid_indices, key=lambda j: scores[j], reverse=True)[:TOP_K]
        matches = [(csv_ingredients[j], float(scores[j])) for j in valid_indices]
        print(f"Tree ingredient '{leaf}': {matches}")

if __name__ == '__main__':
    main()