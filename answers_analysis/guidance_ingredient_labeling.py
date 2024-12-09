import argparse
import json
import os
import time

import guidance
import polars as pl

from utils import ingredient_tree_from_json
from prompt_templates_guidance import prompt_templates


# Set up argument parser to specify model name and context length
parser = argparse.ArgumentParser()
parser.add_argument('gguf_path', help="Specify the path to the LLM GGUF file")
parser.add_argument('--context_len', '-ctx_len', type=int, default=0, help="Specify the context length for the model")
parser.add_argument('--temperature', type=float, default=0, help="Temperature for the LLM")
parser.add_argument('--top-p', type=float, default=1, help="Top-p sampling for the LLM")
parser.add_argument(
    '--use_constrained_generation', '-cg', choices=["full", "only_gen", "only_select"],
    default="full", help="Use constrained generation for structured output"
)
args = parser.parse_args()

script_filepath = os.path.dirname(os.path.realpath(__file__))

ingredients_file = pl.read_csv(os.path.join(script_filepath, os.pardir, 'ingredient_food_kg_names.csv'))
ingredients = ingredients_file.select(pl.col('ingredient_food_kg_names').str.replace('"', ''))['ingredient_food_kg_names'].to_list()

output_path = os.path.join(script_filepath, 'LLM Ingredient Labeling', "guidance")
output_file = os.path.join(output_path, f"labeled_ingredients_{os.path.basename(args.gguf_path).replace('.gguf', '')}.txt")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(os.path.join(script_filepath, os.pardir, 'json_melted.json'), 'r') as f:
    sueatable_db = json.load(f)
    ingredient_tree = ingredient_tree_from_json(sueatable_db)

llm = guidance.models.LlamaCpp(
    args.gguf_path,
    n_gpu_layers=-1,
    n_ctx=args.context_len,
    echo=False,
    compute_log_probs=True
)
gen_params = dict(
    temperature=args.temperature,
    top_p=args.top_p
)

with guidance.system():
    llm_system = llm + prompt_templates["system"]
    llm_system += prompt_templates["few_shot_examples"]

labeled_ingredients = []
for i, ingr in enumerate(ingredients):
    start_time = time.time()

    # Generate ingredient description
    with guidance.user():
        llm_bootstrap = llm_system + prompt_templates["bootstrap_description"].format(ingredient=ingr)
    with guidance.assistant():
        llm_labeler = llm_bootstrap + guidance.gen(max_tokens=100, name="description")
    
    partial_path = ""
    path_candidates = list(ingredient_tree['-']['children']) + ["I DON'T KNOW"]
    if args.use_constrained_generation == "only_select":
        while True:
            with guidance.user():
                llm_partial_path = llm_labeler + prompt_templates["partial_path"].format(
                    ingredient=ingr,
                    path=partial_path
                )

            with guidance.assistant():
                llm_partial_path += guidance.select(path_candidates, name='path_node')
                generated_path_node = llm_partial_path["path_node"]
                assert generated_path_node in path_candidates, "Structured generation not working as expected"
                if "I DON'T KNOW" in generated_path_node:
                    partial_path += "I DON'T KNOW"
                    break
                elif ingredient_tree[generated_path_node]['children']:
                    partial_path += llm_partial_path["path_node"] + ' -> '
                    path_candidates = list(ingredient_tree[generated_path_node]['children']) + ["I DON'T KNOW"]
                else:
                    partial_path += llm_partial_path["path_node"]
                    break
    elif args.use_constrained_generation == "only_gen":
        while True:
            with guidance.user():
                if partial_path:
                    format_kwargs = dict(
                        ingredient=ingr,
                        candidates=path_candidates,
                        partial_path_instruction=prompt_templates["partial_path_instruction"].format(path=partial_path)
                    )
                else:
                    format_kwargs = dict(
                        ingredient=ingr,
                        candidates=prompt_templates["macrogroups_info"],
                        partial_path_instruction=""
                    )

                llm_partial_path = llm_labeler + prompt_templates["instruction_with_candidates"].format(**format_kwargs)

            with guidance.assistant():
                llm_partial_path += guidance.gen(name='path_node')
                generated_path_node = llm_partial_path["path_node"].strip()
                print(generated_path_node)
                assert generated_path_node in path_candidates, "Structured generation not working as expected"
                if "I DON'T KNOW" in generated_path_node:
                    partial_path += "I DON'T KNOW"
                    break
                elif ingredient_tree[generated_path_node]['children']:
                    partial_path += llm_partial_path["path_node"] + ' -> '
                    path_candidates = list(ingredient_tree[generated_path_node]['children']) + ["I DON'T KNOW"]
                else:
                    partial_path += llm_partial_path["path_node"]
                    break
    elif args.use_constrained_generation == "full":
        while True:
            with guidance.user():
                llm_partial_path = llm_labeler + prompt_templates["partial_path_with_candidates"].format(
                    ingredient=ingr,
                    candidates=path_candidates,
                    path=partial_path
                )

            with guidance.assistant():
                llm_partial_path_candidates = llm_partial_path + guidance.gen(name='candidates_knowledge')
                candidates_knowledge = llm_partial_path_candidates["candidates_knowledge"]
                import pdb; pdb.set_trace()
                # TODO: study how path_candidate work inside. Maybe attach the text of candidates_knowledge before the select
                # like:
                # candidates_knowledge = guidance.gen(name='candidates_knowledge')['candidates_knowledge']
                # llm_partial_path += candidates_knowledge + guidance.select(path_candidates, name='path_node')
                llm_partial_path_candidates += guidance.select(path_candidates, name='path_node')
                generated_path_node = llm_partial_path_candidates["path_node"]
                print(llm_partial_path_candidates["candidates_knowledge"])
                print(generated_path_node)
                assert generated_path_node in path_candidates, "Structured generation not working as expected"
                if "I DON'T KNOW" in generated_path_node:
                    partial_path += "I DON'T KNOW"
                    break
                elif ingredient_tree[generated_path_node]['children']:
                    partial_path += llm_partial_path["path_node"] + ' -> '
                    path_candidates = list(ingredient_tree[generated_path_node]['children']) + ["I DON'T KNOW"]
                else:
                    partial_path += llm_partial_path["path_node"]
                    break
    else:
        raise NotImplementedError(f"Constrained generation method {args.use_constrained_generation} not implemented")

    generated_path = partial_path
    print(f"{ingr:>50}", generated_path)

    elapsed_time = time.time() - start_time
    print(f"Time taken: {int(elapsed_time // 3600):02d}:{int((elapsed_time % 3600) // 60):02d}:{int(elapsed_time % 60):02d}")
    labeled_ingredients.append((ingr, generated_path))

with open(output_file, 'w') as output_stream:
    output_stream.write('\n'.join(labeled_ingredients))

print(f"LLM output saved to {output_file}")

# clean_and_save_llm_output(output_file, output_file.replace('.txt', '.csv'))
