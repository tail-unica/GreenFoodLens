import argparse
import json
import os
import re
import time

import llama_cpp
import polars as pl
import tqdm

from prompt_templates_guidance import prompt_templates
from utils import ingredient_tree_from_json, gbnf_grammar_choice


# Set up argument parser to specify model name and context length
parser = argparse.ArgumentParser()
parser.add_argument('gguf_path', help="Specify the path to the LLM GGUF file")
parser.add_argument('version_tag', type=str, help="Specify the version tag for the output file")
parser.add_argument('--truth_labels_file', '-lf', help="Specify the path to the labeled ingredients file")
parser.add_argument('--context_len', '-ctx_len', type=int, default=0, help="Specify the context length for the model")
parser.add_argument('--temperature', type=float, default=0, help="Temperature for the LLM")
parser.add_argument('--top-p', type=float, default=1, help="Top-p sampling for the LLM")
parser.add_argument('--top-k', type=float, default=1, help="Top-k sampling for the LLM")
parser.add_argument('--split_grammar_chars', '-sp_gr', action='store_true', help="Split the grammar choices into individual characters")
parser.add_argument('--verbose', action='store_true', help="Enables llama-cpp verbose mode")
parser.add_argument('--use_all_ingredients', action='store_true', help="Use all ingredients for labeling")
parser.add_argument('--validation_split', type=float, default=0.5, help="Validation split for the labeled ingredients")
parser.add_argument('--gpu_id', type=int, default=None, help="Specify the GPU ID to use for the LLM")
args = parser.parse_args()

script_filepath = os.path.dirname(os.path.realpath(__file__))

if re.match(r'v\d+', args.version_tag) is None:
    raise argparse.ArgumentError("Version tag must be in the format 'vX', where X is an integer")
else:
    version_id = int(args.version_tag.replace('v', ''))

if args.use_all_ingredients:
    ingredients_file = pl.read_csv(os.path.join(script_filepath, os.pardir, 'ingredient_food_kg_names.csv'))
    ingredients = ingredients_file.select(pl.col('ingredient_food_kg_names').str.replace('"', ''))['ingredient_food_kg_names'].to_list()
else:
    valid_answers_filepath = os.path.join(script_filepath, f"{os.path.splitext(args.truth_labels_file)[0]}_valid.csv")
    test_answers_filepath = os.path.join(script_filepath, f"{os.path.splitext(args.truth_labels_file)[0]}_test.csv")
    if not os.path.exists(valid_answers_filepath) or not os.path.exists(test_answers_filepath):
        truth_labels_file = os.path.join(script_filepath, args.truth_labels_file)
        truth_labels_df = pl.read_csv(truth_labels_file, separator='\t')
        valid_size = int(len(truth_labels_df) * args.validation_split)
        truth_labels_df = truth_labels_df.sample(fraction=1, shuffle=True)
        valid_answers_df = truth_labels_df.head(valid_size)
        test_answers_df = truth_labels_df.tail(len(truth_labels_df) - valid_size)
        valid_answers_df.write_csv(valid_answers_filepath, separator='\t')
        test_answers_df.write_csv(test_answers_filepath, separator='\t')
    else:
        valid_answers_df = pl.read_csv(valid_answers_filepath, separator='\t')
        test_answers_df = pl.read_csv(test_answers_filepath, separator='\t')
    ingredients = (
        pl.concat([valid_answers_df, test_answers_df], how='vertical')
        .select(pl.col('ingredient').str.replace('"', ''))['ingredient'].to_list()
    )

output_path = os.path.join(script_filepath, 'LLM Ingredient Labeling', "llama_cpp_grammar")
output_file = os.path.join(
    output_path,
    f"labeled_ingredients_{os.path.basename(args.gguf_path).replace('.gguf', '')}-{args.version_tag}.csv"
)
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(os.path.join(script_filepath, os.pardir, 'revised_su-eatable-life_taxonomy.json'), 'r') as f:
    sueatable_db = json.load(f)
    ingredient_tree = ingredient_tree_from_json(sueatable_db)

if args.gpu_id is not None:
    gpu_kwargs = dict(
        split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE,
        main_gpu=args.gpu_id
    )
else:
    gpu_kwargs = dict()

llm = llama_cpp.Llama(
    args.gguf_path,
    n_gpu_layers=-1,
    n_ctx=args.context_len,
    echo=True,
    compute_log_probs=True,
    verbose=args.verbose,
    flash_attn=True,
    **gpu_kwargs
)
gen_params = dict(
    temperature=args.temperature,
    top_p=args.top_p,
    top_k=args.top_k
)

system_messages = [
    {"role": "system", "content": prompt_templates["system"] + '\n' + prompt_templates["few_shot_examples"]},
]
if version_id > 1:
    system_messages.append({"role": "system", "content": prompt_templates["labeling_notes"]})

if os.path.exists(output_file):
    labeled_ingredients_df = pl.read_csv(output_file, separator='\t')
    labeled_ingredients = list(map(tuple, labeled_ingredients_df.to_numpy()))

    for i, ingr, _ in labeled_ingredients:
        ingredients.remove(ingr)
else:
    labeled_ingredients = []

for i in tqdm.tqdm(range(len(ingredients)), desc="Ingredients Labeling"):
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
