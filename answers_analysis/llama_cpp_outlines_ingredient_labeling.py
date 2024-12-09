import argparse
import json
import os
import time

import llama_cpp
import polars as pl
import outlines

from utils import ingredient_tree_from_json
from prompt_templates import prompt_templates


# Set up argument parser to specify model name and context length
parser = argparse.ArgumentParser()
parser.add_argument('gguf_path', help="Specify the path to the LLM GGUF file")
parser.add_argument('--context_len', '-ctx_len', type=int, default=0, help="Specify the context length for the model")
parser.add_argument('--temperature', type=float, default=0, help="Temperature for the LLM")
parser.add_argument('--top-p', type=float, default=0.9, help="Top-p sampling for the LLM")
parser.add_argument('--top-k', type=int, default=1, help="Top-k sampling for the LLM")
parser.add_argument('--prompt_type', '-pt', type=str, default="chatgpt_prompt_engineering_prompt_v1", help="Prompt sent to the LLM")
args = parser.parse_args()

print(f"Using prompt type: {args.prompt_type}")

script_filepath = os.path.dirname(os.path.realpath(__file__))

ingredients_file = pl.read_csv(os.path.join(script_filepath, os.pardir, 'ingredient_food_kg_names.csv'))
ingredients = ingredients_file.select(pl.col('ingredient_food_kg_names').str.replace('"', ''))['ingredient_food_kg_names'].to_list()
prompt_tmpl = prompt_templates["user"][args.prompt_type]

output_path = os.path.join(script_filepath, 'LLM Ingredient Labeling', args.prompt_type)
output_file = os.path.join(output_path, f"labeled_ingredients_{args.hf_repo_id.replace('/', '_')}_{args.hf_repo_filename.replace('/', '_')}.txt")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(os.path.join(script_filepath, os.pardir, 'json_melted.json'), 'r') as f:
    sueatable_db = json.load(f)
    ingredient_tree = ingredient_tree_from_json(sueatable_db)

llm = llama_cpp.LLama(
    args.gguf_path,
    n_gpu_layers=-1
)

model = outlines.models.transformers(
    args.hf_repo_id,
    model_kwargs=dict(
        n_ctx=args.context_len,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )
)
generator = generate.json(
    model,
    schema_object=path_schema
)

labeled_ingredients = []
for i, ingr in enumerate(ingredients):
    prompt = prompt_tmpl.format(
        context=llm_path_context,
        ingredient=ingr
    )

    if args.context_len > 0:
        tokens = model.tokenizer.encode(prompt)
        token_count = len(tokens)
        print(f"Token count: {token_count}")

        # Adjust batch size if token count exceeds the limit
        if token_count > args.context_len:
            raise ValueError(f"Ingredient #{i+1} = {ingr} exceeds token limit: {token_count} > {args.context_len}")

    start_time = time.time()
    generated_text = generator(prompt)
    print(f"{ingr:>50}", generated_text)

    elapsed_time = time.time() - start_time
    print(f"Time taken: {int(elapsed_time // 3600):02d}:{int((elapsed_time % 3600) // 60):02d}:{int(elapsed_time % 60):02d}")
    labeled_ingredients.append(generated_text)

    if i == 20:
        import pdb; pdb.set_trace()

with open(output_file, 'w') as output_stream:
    output_stream.write('\n'.join(labeled_ingredients))

print(f"LLM output saved to {output_file}")

clean_and_save_llm_output(output_file, output_file.replace('.txt', '.csv'))
