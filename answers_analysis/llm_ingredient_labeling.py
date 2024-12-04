import os
import json
import time
import argparse

import ollama
import tiktoken
import polars as pl

from utils import build_path_context_from_json, clean_and_save_llm_output
from prompt_templates import prompt_templates


modelfile = """
FROM {llm}

PARAMETER temperature {temperature}
PARAMETER num_ctx {num_ctx}
PARAMETER top_k {top_k}
PARAMETER top_p {top_p}

# set the system message
SYSTEM \"""
{system_prompt}.
\"""
"""

# Set up argument parser to specify model name and context length
parser = argparse.ArgumentParser()
parser.add_argument('llm', help="Specify the Ollama model name to use (e.g., llama2)")
parser.add_argument('--context_len', '-ctx_len', type=int, default=1024, help="Specify the context length for the model")
parser.add_argument('--temperature', type=float, default=0.2, help="Temperature for the LLM")
parser.add_argument('--top-p', type=float, default=0.9, help="Top-p sampling for the LLM")
parser.add_argument('--top-k', type=int, default=10, help="Top-k sampling for the LLM")
parser.add_argument('--prompt_type', '-pt', type=str, default="chatgpt_prompt_engineering_prompt_v1", help="Prompt sent to the LLM")
parser.add_argument('--batch_size', '-bs', type=int, default=200, help="Specify the batch size for the ingredients in the prompt")
args = parser.parse_args()

print(f"Using prompt type: {args.prompt_type}")

script_filepath = os.path.dirname(os.path.realpath(__file__))

ingredients_file = pl.read_csv(os.path.join(script_filepath, os.pardir, 'ingredient_food_kg_names.csv'))
ingredients = ingredients_file.select(pl.col('ingredient_food_kg_names').str.replace('"', ''))['ingredient_food_kg_names'].to_list()

output_path = os.path.join(script_filepath, 'LLM Ingredient Labeling', args.prompt_type)

path_sueatable_db_filepath = os.path.join(script_filepath, 'path_sueatable_db.txt')
if not os.path.exists(path_sueatable_db_filepath):
    with open(os.path.join(script_filepath, os.pardir, 'json_melted.json'), 'r') as f:
        sueatable_db = json.load(f)
    llm_path_context = build_path_context_from_json(sueatable_db)

    with open(path_sueatable_db_filepath, 'w') as f:
        f.write(llm_path_context)
else:
    with open(path_sueatable_db_filepath, 'r') as f:
        llm_path_context = f.read()

# Initialize tokenizer for your model (e.g., 'gpt-3.5-turbo' or 'gpt-4')
tokenizer = tiktoken.encoding_for_model('gpt-4')

ingredient_batches = [ingredients[i:i + args.batch_size] for i in range(0, len(ingredients), args.batch_size)]

prompt_tmpl = prompt_templates["user"][args.prompt_type]

output_file = os.path.join(output_path, f"labeled_ingredients_{args.llm.replace('/', '_')}.txt")
os.makedirs(os.path.dirname(output_file), exist_ok=True)

if os.path.exists(output_file):
    os.remove(output_file)

custom_llm_name = "IngredientLabeler-" + args.llm.replace('/', '_')

ollama.create(
    model=custom_llm_name,
    modelfile=modelfile.format(
        llm=args.llm,
        temperature=args.temperature,
        num_ctx=args.context_len,
        top_k=args.top_k,
        top_p=args.top_p,
        system_prompt=prompt_templates['system']
    )
)

with open(output_file, 'a') as output_stream:
    for i, batch in enumerate(ingredient_batches):
        prompt = prompt_tmpl.format(
            context=llm_path_context,
            ingredients='\n'.join(batch)
        )

        tokens = tokenizer.encode(prompt)
        token_count = len(tokens)
        print(f"Batch {i+1} token count: {token_count}")
        
        # Adjust batch size if token count exceeds the limit
        if token_count > args.context_len: 
            raise ValueError(f"Batch {i+1} exceeds token limit: {token_count} > {args.context_len}")

        start_time = time.time()
        generated_text = ""
        stream = ollama.generate(
            model=custom_llm_name,
            prompt=prompt,
            stream=True
        )

        for chunk in stream:
            print(chunk['response'], end='', flush=True)
            generated_text += chunk['response']
        
        # Split generated text and add structured data back to the list
        batch_labeled_ingredients = generated_text.split('\n')
        labeled_ingredients = [labeled_ingr for labeled_ingr in batch_labeled_ingredients if '->' in labeled_ingr or labeled_ingr.strip() == 'I DON\'T KNOW']

        elapsed_time = time.time() - start_time
        print(f"Time taken: {int(elapsed_time // 3600):02d}:{int((elapsed_time % 3600) // 60):02d}:{int(elapsed_time % 60):02d}")

        output_stream.write('\n'.join(labeled_ingredients))

print(f"LLM output saved to {output_file}")

clean_and_save_llm_output(output_file, output_file.replace('.txt', '.csv'))
