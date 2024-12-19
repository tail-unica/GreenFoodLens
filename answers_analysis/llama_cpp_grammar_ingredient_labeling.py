import argparse
import json
import os
import time
import tqdm

import llama_cpp
import polars as pl

from utils import ingredient_tree_from_json, gbnf_grammar_choice
from prompt_templates_guidance import prompt_templates


# Set up argument parser to specify model name and context length
parser = argparse.ArgumentParser()
parser.add_argument('gguf_path', help="Specify the path to the LLM GGUF file")
parser.add_argument('version_tag', type=str, help="Specify the version tag for the output file")
parser.add_argument('--context_len', '-ctx_len', type=int, default=0, help="Specify the context length for the model")
parser.add_argument('--temperature', type=float, default=0, help="Temperature for the LLM")
parser.add_argument('--top-p', type=float, default=1, help="Top-p sampling for the LLM")
parser.add_argument('--top-k', type=float, default=1, help="Top-k sampling for the LLM")
parser.add_argument('--split_grammar_chars', '-sp_gr', action='store_true', help="Split the grammar choices into individual characters")
parser.add_argument('--verbose', action='store_true', help="Enables llama-cpp verbose mode")
args = parser.parse_args()

script_filepath = os.path.dirname(os.path.realpath(__file__))

ingredients_file = pl.read_csv(os.path.join(script_filepath, os.pardir, 'ingredient_food_kg_names.csv'))
ingredients = ingredients_file.select(pl.col('ingredient_food_kg_names').str.replace('"', ''))['ingredient_food_kg_names'].to_list()

output_path = os.path.join(script_filepath, 'LLM Ingredient Labeling', "llama_cpp_grammar")
output_file = os.path.join(
    output_path,
    f"labeled_ingredients_{os.path.basename(args.gguf_path).replace('.gguf', '')}-{args.version_tag}.csv"
)
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(os.path.join(script_filepath, os.pardir, 'json_melted.json'), 'r') as f:
    sueatable_db = json.load(f)
    ingredient_tree = ingredient_tree_from_json(sueatable_db)

llm = llama_cpp.Llama(
    args.gguf_path,
    n_gpu_layers=-1,
    n_ctx=args.context_len,
    echo=False,
    compute_log_probs=True,
    verbose=args.verbose,
    flash_attn=True
)
gen_params = dict(
    temperature=args.temperature,
    top_p=args.top_p,
    top_k=args.top_k
)

system_messages = [
    {"role": "system", "content": prompt_templates["system"] + '\n' + prompt_templates["few_shot_examples"]},
]

if os.path.exists(output_file):
    labeled_ingredients_df = pl.read_csv(output_file, separator='\t')
    start_index = labeled_ingredients_df.select(pl.max('index')).item() + 1
    labeled_ingredients = list(map(tuple, labeled_ingredients_df.to_numpy()))
else:
    labeled_ingredients = []
    start_index = 0

for i in tqdm.tqdm(range(start_index, len(ingredients)), desc="Ingredients Labeling"):
    ingr = ingredients[i]
    start_time = time.time()

    # Generate ingredient description
    boostrap_messages = [
        *system_messages,
        {"role": "user", "content": prompt_templates["bootstrap_description"].format(ingredient=ingr)}
    ]

    bootstrap_llm_output = llm.create_chat_completion(boostrap_messages, max_tokens=200, **gen_params)
    labeler_messages = [
        *boostrap_messages,
        bootstrap_llm_output['choices'][0]['message']
    ]
    
    partial_path = ""
    path_candidates = list(ingredient_tree['-']['children']) + ["I DON'T KNOW"]
    while True:
        partial_path_messages = [
            *labeler_messages,
            {"role": "user", "content": prompt_templates["instruction_with_candidates"].format(
                ingredient=ingr,
                partial_path_instruction=prompt_templates["partial_path_instruction"].format(path=partial_path),
                candidates=' | '.join(path_candidates)
            )}
        ]

        path_candidates_grammar = gbnf_grammar_choice(
            path_candidates, as_string=True, split_chars=args.split_grammar_chars
        )
        path_candidates_grammar = llama_cpp.LlamaGrammar.from_string(path_candidates_grammar)

        partial_path_output = llm.create_chat_completion(
            partial_path_messages,
            grammar=path_candidates_grammar,
            **gen_params
        )
        generated_path_node = partial_path_output['choices'][0]['message']['content'].replace('`', '')

        assert generated_path_node in path_candidates, f"llama-cpp grammar not working as expected, generated path node: {generated_path_node}, path candidates: {path_candidates}"
        if "I DON'T KNOW" in generated_path_node:
            partial_path += "I DON'T KNOW"
            break
        elif ingredient_tree[generated_path_node]['children']:
            partial_path += generated_path_node + ' -> '
            path_candidates = list(ingredient_tree[generated_path_node]['children']) + ["I DON'T KNOW"]
        else:
            partial_path += generated_path_node
            break

    generated_path = partial_path
    print(f"{ingr:<50}:", generated_path)

    elapsed_time = time.time() - start_time
    print(f"Time taken: {int(elapsed_time // 3600):02d}:{int((elapsed_time % 3600) // 60):02d}:{int(elapsed_time % 60):02d}")
    labeled_ingredients.append((i, ingr, generated_path))

    pl.DataFrame(labeled_ingredients, schema=['index', 'ingredient', 'answer_path']).write_csv(output_file, separator='\t')

print(f"LLM output saved to {output_file}")
